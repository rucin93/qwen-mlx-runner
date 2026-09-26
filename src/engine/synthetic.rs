//! A shape-correct, deliberately synthetic Qwen3.8-27B decode workload.
//!
//! All layers of a given type reuse the same immutable packed matrices. State
//! and attention caches are private to each layer. No checkpoint is read, and
//! the zeros preceding `history` are synthetic KV entries, not prompt tokens.

use anyhow::{Context, Result, ensure};
use metal::MTLResourceOptions;

use super::{AttentionLayer, DeltaLayer, Engine, Layer, Matrix, Mixer, Scratch};
use crate::{config::ModelConfig, gpu::Gpu};

const GROUP: usize = 64;
const SCALE: f32 = 0.0005;
const BIAS: f32 = -0.00375;

// Buffer handles are retained Metal objects. This duplicates only references;
// the immutable bytes are shared by all 64 layer instances.
fn share(m: &Matrix) -> Matrix {
    Matrix {
        weight: m.weight.clone(),
        scales: m.scales.clone(),
        biases: m.biases.clone(),
        rows: m.rows,
        cols: m.cols,
        bits: m.bits,
        group: m.group,
        metadata_bf16: m.metadata_bf16,
    }
}

fn packed_q4(g: &Gpu, rows: usize, cols: usize) -> Result<Matrix> {
    ensure!(
        cols % GROUP == 0,
        "synthetic matrix width must divide by 64"
    );
    let weight_bytes = rows
        .checked_mul(cols / 2)
        .context("synthetic packed matrix size overflow")?;
    ensure!(
        weight_bytes <= g.device.max_buffer_length() as usize,
        "synthetic matrix exceeds Metal's maximum buffer length"
    );
    let groups = rows
        .checked_mul(cols / GROUP)
        .context("synthetic affine parameter size overflow")?;
    // Allocate the packed storage directly in unified memory. In particular,
    // the large embedding and head never require a second CPU staging copy.
    let weight = g
        .device
        .new_buffer(weight_bytes as u64, MTLResourceOptions::StorageModeShared);
    let row_bytes = cols / 2;
    unsafe {
        let data = weight.contents().cast::<u8>();
        for row in 0..rows {
            // Alternating nibbles encode small positive/negative weights.
            // Flipping by row avoids an identical output for every row.
            std::ptr::write_bytes(
                data.add(row * row_bytes),
                if row & 1 == 0 { 0x78 } else { 0x87 },
                row_bytes,
            );
        }
    }
    let scales = g.alloc_f32(groups)?;
    let biases = g.alloc_f32(groups)?;
    unsafe {
        std::slice::from_raw_parts_mut(scales.contents().cast::<f32>(), groups).fill(SCALE);
        std::slice::from_raw_parts_mut(biases.contents().cast::<f32>(), groups).fill(BIAS);
    }
    Ok(Matrix {
        weight,
        scales: Some(scales),
        biases: Some(biases),
        rows,
        cols,
        bits: 4,
        group: GROUP,
        metadata_bf16: false,
    })
}

fn packed_bytes(rows: usize, cols: usize) -> u64 {
    // All fixed Qwen3.8-27B dimensions are multiples of 64. Q4 weights use
    // 1/2 byte per element; scale and bias add 8 bytes per 64 elements.
    (rows as u64) * (cols as u64) * 5 / 8
}

impl Engine {
    /// Build the complete decode graph at Qwen3.8-27B dimensions without
    /// downloading a checkpoint. `history` is a zero-filled attention prefix.
    pub fn synthetic_qwen27b(context: usize, history: usize) -> Result<Self> {
        const H: usize = 5120;
        const F: usize = 17408;
        const VOCAB: usize = 248320;
        const KD: usize = 16 * 128;
        const VD: usize = 48 * 128;
        const QD: usize = 24 * 256;
        const KVD: usize = 4 * 256;
        const QKV: usize = 2 * KD + VD;

        let config = ModelConfig {
            hidden_size: H,
            intermediate_size: F,
            num_hidden_layers: 64,
            num_attention_heads: 24,
            num_key_value_heads: 4,
            head_dim: 256,
            vocab_size: VOCAB,
            linear_num_key_heads: 16,
            linear_num_value_heads: 48,
            linear_key_head_dim: 128,
            linear_value_head_dim: 128,
            linear_conv_kernel_dim: 4,
            full_attention_interval: 4,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000_000.0,
            partial_rotary_factor: 0.25,
            max_position_embeddings: 262_144,
            tie_word_embeddings: false,
            eos_token_ids: vec![248_044],
            layer_types: (0..64)
                .map(|i| {
                    if (i + 1) % 4 == 0 {
                        "full_attention".to_owned()
                    } else {
                        "linear_attention".to_owned()
                    }
                })
                .collect(),
        };
        config.validate()?;
        ensure!(
            context > 0 && context <= config.max_position_embeddings,
            "synthetic context must be in 1..={}",
            config.max_position_embeddings
        );
        ensure!(
            history < context,
            "synthetic history must be smaller than context"
        );
        // This is the dispatch grid's largest u32-indexed dimension.
        ensure!(
            (24u64) * (context as u64) * 32 <= u32::MAX as u64,
            "synthetic attention grid exceeds u32 indexing"
        );

        let gpu = Gpu::new()?;
        let unique_matrices = [
            (VOCAB, H), // embedding
            (VOCAB, H), // untied head
            (F, H),
            (F, H),
            (H, F), // shared MLP projections
            (QKV, H),
            (VD, H),
            (48, H),
            (48, H),
            (H, VD), // DeltaNet
            (QD * 2, H),
            (KVD, H),
            (KVD, H),
            (H, QD), // attention
        ];
        let weights_bytes: u64 = unique_matrices
            .into_iter()
            .map(|(rows, cols)| packed_bytes(rows, cols))
            .sum();
        let state_bytes = 48u64 * (QKV as u64 * 3 + 48 * 128 * 128) * 4;
        let cache_bytes = 16u64 * context as u64 * KVD as u64 * 2 * 4;
        // Includes ample room for scratch, small vectors, pipelines and Metal
        // bookkeeping. The actual allocation is checked again at the end.
        let required = weights_bytes + state_bytes + cache_bytes + 128 * 1024 * 1024;
        let budget = gpu.device.recommended_max_working_set_size();
        ensure!(
            gpu.device.current_allocated_size().saturating_add(required) < budget,
            "synthetic Qwen3.8-27B needs about {:.2} GiB plus Metal overhead; GPU recommends {:.2} GiB (reduce context)",
            required as f64 / 1073741824.0,
            budget as f64 / 1073741824.0
        );

        let scratch = Scratch {
            x: gpu.alloc_f32(H)?,
            normalized: gpu.alloc_f32(H)?,
            residual: gpu.alloc_f32(H)?,
            gate: gpu.alloc_f32(F)?,
            up: gpu.alloc_f32(F)?,
            activated: gpu.alloc_f32(F)?,
            qkv: gpu.alloc_f32(QKV)?,
            convolved: gpu.alloc_f32(QKV)?,
            z: gpu.alloc_f32(VD)?,
            a: gpu.alloc_f32(48)?,
            b: gpu.alloc_f32(48)?,
            mixed: gpu.alloc_f32(VD.max(QD))?,
            gated: gpu.alloc_f32(VD)?,
            qproj: gpu.alloc_f32(QD * 2)?,
            q: gpu.alloc_f32(QD)?,
            k: gpu.alloc_f32(KVD)?,
            v: gpu.alloc_f32(KVD)?,
            attention_gate: gpu.alloc_f32(QD)?,
            scores: gpu.alloc_f32(24 * context)?,
            logits: gpu.alloc_f32(VOCAB)?,
        };

        let embedding = packed_q4(&gpu, VOCAB, H)?;
        let head = Some(packed_q4(&gpu, VOCAB, H)?);
        let gate = packed_q4(&gpu, F, H)?;
        let up = packed_q4(&gpu, F, H)?;
        let down = packed_q4(&gpu, H, F)?;
        let delta_qkv = packed_q4(&gpu, QKV, H)?;
        let delta_z = packed_q4(&gpu, VD, H)?;
        let delta_a = packed_q4(&gpu, 48, H)?;
        let delta_b = packed_q4(&gpu, 48, H)?;
        let delta_out = packed_q4(&gpu, H, VD)?;
        let attn_q = packed_q4(&gpu, QD * 2, H)?;
        let attn_k = packed_q4(&gpu, KVD, H)?;
        let attn_v = packed_q4(&gpu, KVD, H)?;
        let attn_out = packed_q4(&gpu, H, QD)?;

        let norm_ones = gpu.upload_f32(&vec![1.; H])?;
        let delta_norm = gpu.upload_f32(&vec![1.; 128])?;
        let attn_norm = gpu.upload_f32(&vec![1.; 256])?;
        let conv = gpu.upload_f32(&vec![0.1; QKV * 4])?;
        let alog = gpu.upload_f32(&[0.; 48])?;
        let dt = gpu.upload_f32(&[0.; 48])?;

        let mut layers = Vec::with_capacity(64);
        for i in 0..64 {
            let mixer = if (i + 1) % 4 == 0 {
                Mixer::Attention(AttentionLayer {
                    q: share(&attn_q),
                    k: share(&attn_k),
                    v: share(&attn_v),
                    out: share(&attn_out),
                    qnorm: attn_norm.clone(),
                    knorm: attn_norm.clone(),
                    // alloc_f32 guarantees the historical prefix is all zeros.
                    kcache: gpu.alloc_f32(context * KVD)?,
                    vcache: gpu.alloc_f32(context * KVD)?,
                })
            } else {
                Mixer::Delta(DeltaLayer {
                    qkv: share(&delta_qkv),
                    z: share(&delta_z),
                    a: share(&delta_a),
                    b: share(&delta_b),
                    out: share(&delta_out),
                    conv: conv.clone(),
                    alog: alog.clone(),
                    dt: dt.clone(),
                    norm: delta_norm.clone(),
                    conv_state: gpu.alloc_f32(QKV * 3)?,
                    state: gpu.alloc_f32(48 * 128 * 128)?,
                })
            };
            layers.push(Layer {
                input_norm: norm_ones.clone(),
                post_norm: norm_ones.clone(),
                mixer,
                gate: share(&gate),
                up: share(&up),
                down: share(&down),
            });
        }
        ensure!(
            gpu.device.current_allocated_size() < budget,
            "synthetic graph exceeds Metal's recommended working set"
        );
        Ok(Self {
            gpu,
            config,
            embedding,
            head,
            norm: norm_ones,
            layers,
            scratch,
            position: history,
            context,
            block: None,
            prompt_block: None,
        })
    }
}
