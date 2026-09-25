//! Layer-major target verification with a transaction over recurrent state.
//!
//! Matrix projections consume all token columns in one dispatch. Stateful
//! operations retain causal token order; accepting a prefix restores its GPU
//! snapshots rather than evaluating the target a second time.
use anyhow::{Context, Result, ensure};
use metal::{Buffer, BufferRef};

use super::{Engine, Matrix, Mixer, Scratch};
use crate::gpu::DispatchEncoder;

const MAX_BLOCK: usize = 4;
const PREFIX_SLOTS: usize = MAX_BLOCK + 1;

/// Target outputs for a tentative block. `hidden` is after final target RMS.
#[derive(Debug)]
pub struct BlockOutput {
    pub base_position: usize,
    pub logits: Vec<Vec<f32>>,
    pub hidden: Vec<Vec<f32>>,
}

#[derive(Clone, Copy)]
pub(super) struct Pending {
    base_position: usize,
    tokens: usize,
}

struct Snapshots {
    conv: Buffer,
    state: Buffer,
    conv_elements: usize,
    state_elements: usize,
}

pub(super) struct BlockState {
    scratch: Scratch,
    snapshots: Vec<Option<Snapshots>>,
    pub(super) pending: Option<Pending>,
}

impl BlockState {
    fn new(engine: &Engine) -> Result<Self> {
        ensure!(
            engine.kernel_mode() != "reference",
            "block verification is unavailable with reference kernels; use sequential forward for reference comparisons"
        );
        // Sequential affine kernels also support other group sizes. Reject a
        // block-only limitation before encoding anything or resetting history.
        let check = |matrix: &Matrix| -> Result<()> {
            ensure!(
                matrix.scales.is_none()
                    || (matches!(matrix.bits, 4 | 8) && matches!(matrix.group, 32 | 64 | 128)),
                "block verification supports affine Q4/Q8 groups 32, 64 or 128; found Q{} group {}",
                matrix.bits,
                matrix.group
            );
            Ok(())
        };
        check(engine.head.as_ref().unwrap_or(&engine.embedding))?;
        for layer in &engine.layers {
            for matrix in [&layer.gate, &layer.up, &layer.down] {
                check(matrix)?;
            }
            match &layer.mixer {
                Mixer::Delta(d) => {
                    for matrix in [&d.qkv, &d.z, &d.a, &d.b, &d.out] {
                        check(matrix)?;
                    }
                }
                Mixer::Attention(a) => {
                    for matrix in [&a.q, &a.k, &a.v, &a.out] {
                        check(matrix)?;
                    }
                }
            }
        }
        let (g, original) = (&engine.gpu, &engine.scratch);
        let batched_buffers = [
            &original.x,
            &original.normalized,
            &original.residual,
            &original.gate,
            &original.up,
            &original.activated,
            &original.qkv,
            &original.convolved,
            &original.z,
            &original.a,
            &original.b,
            &original.mixed,
            &original.gated,
            &original.qproj,
            &original.q,
            &original.k,
            &original.v,
            &original.attention_gate,
            &original.logits,
        ];
        let mut allocation_bytes = 0u64;
        let mut add_allocation = |buffer: &Buffer, copies: usize| -> Result<()> {
            let elements = (buffer.length() / 4)
                .checked_mul(copies as u64)
                .context("block allocation size overflow")?;
            ensure!(
                elements <= u32::MAX as u64,
                "block buffer exceeds 32-bit indexing"
            );
            let bytes = elements
                .checked_mul(4)
                .context("block byte count overflow")?;
            ensure!(
                bytes <= g.device.max_buffer_length(),
                "block buffer exceeds Metal buffer limit"
            );
            allocation_bytes = allocation_bytes
                .checked_add(bytes)
                .context("block allocation total overflow")?;
            Ok(())
        };
        for buffer in batched_buffers {
            add_allocation(buffer, MAX_BLOCK)?;
        }
        for layer in &engine.layers {
            if let Mixer::Delta(d) = &layer.mixer {
                add_allocation(&d.conv_state, PREFIX_SLOTS)?;
                add_allocation(&d.state, PREFIX_SLOTS)?;
            }
        }
        ensure!(
            g.device
                .current_allocated_size()
                .checked_add(allocation_bytes)
                .is_some_and(|n| n < g.device.recommended_max_working_set_size()),
            "block scratch and recurrent snapshots need {:.1} MiB beyond the current model; exceeds Metal's recommended memory budget",
            allocation_bytes as f64 / 1_048_576.
        );
        let batch = |buffer: &Buffer| g.alloc_f32(buffer.length() as usize / 4 * MAX_BLOCK);
        let scratch = Scratch {
            x: batch(&original.x)?,
            normalized: batch(&original.normalized)?,
            residual: batch(&original.residual)?,
            gate: batch(&original.gate)?,
            up: batch(&original.up)?,
            activated: batch(&original.activated)?,
            qkv: batch(&original.qkv)?,
            convolved: batch(&original.convolved)?,
            z: batch(&original.z)?,
            a: batch(&original.a)?,
            b: batch(&original.b)?,
            mixed: batch(&original.mixed)?,
            gated: batch(&original.gated)?,
            qproj: batch(&original.qproj)?,
            q: batch(&original.q)?,
            k: batch(&original.k)?,
            v: batch(&original.v)?,
            attention_gate: batch(&original.attention_gate)?,
            // Each token completes attention before scores are reused.
            scores: original.scores.clone(),
            logits: batch(&original.logits)?,
        };
        let mut snapshots = Vec::with_capacity(engine.layers.len());
        for layer in &engine.layers {
            snapshots.push(if let Mixer::Delta(d) = &layer.mixer {
                let conv_elements = d.conv_state.length() as usize / 4;
                let state_elements = d.state.length() as usize / 4;
                Some(Snapshots {
                    conv: g.alloc_f32(conv_elements * PREFIX_SLOTS)?,
                    state: g.alloc_f32(state_elements * PREFIX_SLOTS)?,
                    conv_elements,
                    state_elements,
                })
            } else {
                None
            });
        }
        Ok(Self {
            scratch,
            snapshots,
            pending: None,
        })
    }
}

impl Matrix {
    /// Input and output are contiguous token-major vectors, not independent
    /// GEMV dispatches: the block kernel reuses loaded weights across columns.
    fn matmul_block(
        &self,
        e: &DispatchEncoder<'_>,
        x: &BufferRef,
        y: &BufferRef,
        batch: usize,
    ) -> Result<()> {
        ensure!(
            (1..=MAX_BLOCK).contains(&batch),
            "matrix block width must be 1..=4"
        );
        if let (Some(scales), Some(biases)) = (&self.scales, &self.biases) {
            e.encode(
                if self.metadata_bf16 {
                    "matmul_affine_bf16"
                } else {
                    "matmul_affine"
                },
                &[&self.weight, scales, biases, x, y],
                &[
                    self.rows as u32,
                    self.cols as u32,
                    self.bits,
                    self.group as u32,
                    batch as u32,
                ],
                self.rows * 32,
                128,
            )
        } else {
            e.encode(
                "matmul_f16",
                &[&self.weight, x, y],
                &[self.rows as u32, self.cols as u32, batch as u32],
                self.rows * 32,
                128,
            )
        }
    }

    fn embed_offset(
        &self,
        e: &DispatchEncoder<'_>,
        token: u32,
        y: &BufferRef,
        offset: usize,
    ) -> Result<()> {
        if let (Some(scales), Some(biases)) = (&self.scales, &self.biases) {
            e.encode_offsets(
                if self.metadata_bf16 {
                    "embed_affine_bf16"
                } else {
                    "embed_affine"
                },
                &[&self.weight, scales, biases, y],
                &[0, 0, 0, offset],
                &[token, self.cols as u32, self.bits, self.group as u32],
                self.cols,
                128,
            )
        } else {
            e.encode_offsets(
                "embed_f16",
                &[&self.weight, y],
                &[0, offset],
                &[token, self.cols as u32],
                self.cols,
                128,
            )
        }
    }
}

fn copy(
    e: &DispatchEncoder<'_>,
    source: &BufferRef,
    source_offset: usize,
    target: &BufferRef,
    target_offset: usize,
    elements: usize,
) -> Result<()> {
    e.encode_offsets(
        "copy_f32",
        &[source, target],
        &[source_offset, target_offset],
        &[elements as u32],
        elements,
        128,
    )
}

impl Engine {
    /// Evaluate one to four consecutive known/proposed inputs. The resulting
    /// state is tentative until `commit_block_prefix`, including full acceptance.
    pub fn verify_block(&mut self, tokens: &[u32]) -> Result<BlockOutput> {
        ensure!(
            self.block.as_ref().is_none_or(|b| b.pending.is_none()),
            "commit or reset the pending verification block before another block"
        );
        ensure!(
            (1..=MAX_BLOCK).contains(&tokens.len()),
            "verification block width must be 1..=4"
        );
        ensure!(
            tokens
                .iter()
                .all(|&token| (token as usize) < self.config.vocab_size),
            "verification block contains a token outside vocabulary"
        );
        ensure!(
            self.position
                .checked_add(tokens.len())
                .is_some_and(|end| end <= self.context),
            "verification block exceeds context capacity {}",
            self.context
        );
        if self.block.is_none() {
            self.block = Some(BlockState::new(self)?);
        }
        let result = objc::rc::autoreleasepool(|| self.verify_block_inner(tokens));
        match result {
            Ok(output) => {
                self.block.as_mut().unwrap().pending = Some(Pending {
                    base_position: self.position,
                    tokens: tokens.len(),
                });
                self.position += tokens.len();
                Ok(output)
            }
            Err(error) => {
                self.reset();
                Err(error)
            }
        }
    }

    /// Accept the first `consumed` inputs. Restores both convolution and DeltaNet
    /// state on the GPU. Attention suffix slots are masked by the new position.
    pub fn commit_block_prefix(&mut self, consumed: usize) -> Result<()> {
        let pending = self
            .block
            .as_ref()
            .and_then(|b| b.pending)
            .context("there is no pending verification block")?;
        ensure!(
            consumed <= pending.tokens,
            "accepted prefix exceeds pending block length"
        );
        let result = if consumed == pending.tokens {
            Ok(())
        } else {
            objc::rc::autoreleasepool(|| {
                let g = &self.gpu;
                let command = g.begin();
                let encoder = g.begin_encoding(command);
                for (layer, snapshots) in self
                    .layers
                    .iter()
                    .zip(&self.block.as_ref().unwrap().snapshots)
                {
                    if let (Mixer::Delta(d), Some(s)) = (&layer.mixer, snapshots) {
                        copy(
                            &encoder,
                            &s.conv,
                            consumed * s.conv_elements * 4,
                            &d.conv_state,
                            0,
                            s.conv_elements,
                        )?;
                        copy(
                            &encoder,
                            &s.state,
                            consumed * s.state_elements * 4,
                            &d.state,
                            0,
                            s.state_elements,
                        )?;
                    }
                }
                encoder.end_encoding()?;
                g.finish(command)
            })
        };
        if let Err(error) = result {
            self.reset();
            return Err(error);
        }
        self.position = pending.base_position + consumed;
        self.block.as_mut().unwrap().pending = None;
        Ok(())
    }

    fn verify_block_inner(&self, tokens: &[u32]) -> Result<BlockOutput> {
        let (g, c, block) = (&self.gpu, &self.config, self.block.as_ref().unwrap());
        let s = &block.scratch;
        let batch = tokens.len();
        let h = c.hidden_size;
        let hd = c.head_dim;
        let kh = c.linear_num_key_heads;
        let vh = c.linear_num_value_heads;
        let kd = c.linear_key_head_dim;
        let vd = c.linear_value_head_dim;
        let linear_width = vh * vd;
        let qkv = 2 * kh * kd + linear_width;
        let nh = c.num_attention_heads;
        let nk = c.num_key_value_heads;
        let q_width = nh * hd;
        let kv_width = nk * hd;
        let eps = c.rms_norm_eps.to_bits();
        let cmd = g.begin();
        let e = g.begin_encoding(cmd);
        let run = |name: &str,
                   buffers: &[&BufferRef],
                   offsets: &[usize],
                   p: &[u32],
                   threads: usize,
                   group: usize| {
            e.encode_offsets(name, buffers, offsets, p, threads, group)
        };
        for (i, &token) in tokens.iter().enumerate() {
            self.embedding.embed_offset(&e, token, &s.x, i * h * 4)?;
        }
        for (li, layer) in self.layers.iter().enumerate() {
            for i in 0..batch {
                run(
                    "rms_norm",
                    &[&s.x, &layer.input_norm, &s.normalized],
                    &[i * h * 4, 0, i * h * 4],
                    &[h as u32, eps],
                    32,
                    32,
                )?;
            }
            match &layer.mixer {
                Mixer::Delta(d) => {
                    d.qkv.matmul_block(&e, &s.normalized, &s.qkv, batch)?;
                    d.z.matmul_block(&e, &s.normalized, &s.z, batch)?;
                    d.a.matmul_block(&e, &s.normalized, &s.a, batch)?;
                    d.b.matmul_block(&e, &s.normalized, &s.b, batch)?;
                    let snapshots = block.snapshots[li].as_ref().unwrap();
                    copy(
                        &e,
                        &d.conv_state,
                        0,
                        &snapshots.conv,
                        0,
                        snapshots.conv_elements,
                    )?;
                    copy(
                        &e,
                        &d.state,
                        0,
                        &snapshots.state,
                        0,
                        snapshots.state_elements,
                    )?;
                    for i in 0..batch {
                        run(
                            "conv_silu",
                            &[&s.qkv, &d.conv, &d.conv_state, &s.convolved],
                            &[i * qkv * 4, 0, 0, i * qkv * 4],
                            &[qkv as u32, c.linear_conv_kernel_dim as u32],
                            qkv,
                            128,
                        )?;
                        run(
                            "delta_norm",
                            &[&s.convolved],
                            &[i * qkv * 4],
                            &[kh as u32, kd as u32, 1e-6f32.to_bits()],
                            kh * 32,
                            128,
                        )?;
                        run(
                            "delta_step",
                            &[&s.convolved, &s.a, &s.b, &d.alog, &d.dt, &d.state, &s.mixed],
                            &[
                                i * qkv * 4,
                                i * vh * 4,
                                i * vh * 4,
                                0,
                                0,
                                0,
                                i * linear_width * 4,
                            ],
                            &[kh as u32, vh as u32, kd as u32, vd as u32],
                            linear_width * 32,
                            128,
                        )?;
                        copy(
                            &e,
                            &d.conv_state,
                            0,
                            &snapshots.conv,
                            (i + 1) * snapshots.conv_elements * 4,
                            snapshots.conv_elements,
                        )?;
                        copy(
                            &e,
                            &d.state,
                            0,
                            &snapshots.state,
                            (i + 1) * snapshots.state_elements * 4,
                            snapshots.state_elements,
                        )?;
                        run(
                            "gated_rms",
                            &[&s.mixed, &s.z, &d.norm, &s.gated],
                            &[
                                i * linear_width * 4,
                                i * linear_width * 4,
                                0,
                                i * linear_width * 4,
                            ],
                            &[vh as u32, vd as u32, eps],
                            vh * 32,
                            128,
                        )?;
                    }
                    d.out.matmul_block(&e, &s.gated, &s.residual, batch)?;
                }
                Mixer::Attention(a) => {
                    a.q.matmul_block(&e, &s.normalized, &s.qproj, batch)?;
                    a.k.matmul_block(&e, &s.normalized, &s.k, batch)?;
                    a.v.matmul_block(&e, &s.normalized, &s.v, batch)?;
                    for i in 0..batch {
                        run(
                            "split_q_gate",
                            &[&s.qproj, &s.q, &s.attention_gate],
                            &[i * q_width * 2 * 4, i * q_width * 4, i * q_width * 4],
                            &[nh as u32, hd as u32],
                            q_width,
                            128,
                        )?;
                        run(
                            "head_rms",
                            &[&s.q, &a.qnorm],
                            &[i * q_width * 4, 0],
                            &[nh as u32, hd as u32, eps],
                            nh * 32,
                            128,
                        )?;
                        run(
                            "head_rms",
                            &[&s.k, &a.knorm],
                            &[i * kv_width * 4, 0],
                            &[nk as u32, hd as u32, eps],
                            nk * 32,
                            128,
                        )?;
                        let position = self.position + i;
                        for (buffer, heads, stride) in [(&s.q, nh, q_width), (&s.k, nk, kv_width)] {
                            run(
                                "rope",
                                &[buffer],
                                &[i * stride * 4],
                                &[
                                    heads as u32,
                                    hd as u32,
                                    c.rotary_dim() as u32,
                                    position as u32,
                                    c.rope_theta.to_bits(),
                                ],
                                heads * c.rotary_dim() / 2,
                                128,
                            )?;
                        }
                        run(
                            "kv_append",
                            &[&s.k, &s.v, &a.kcache, &a.vcache],
                            &[i * kv_width * 4, i * kv_width * 4, 0, 0],
                            &[nk as u32, hd as u32, position as u32],
                            kv_width,
                            128,
                        )?;
                        let length = position + 1;
                        let p = [nh as u32, nk as u32, hd as u32, length as u32];
                        run(
                            "attn_scores",
                            &[&s.q, &a.kcache, &s.scores],
                            &[i * q_width * 4, 0, 0],
                            &p,
                            nh * length * 32,
                            128,
                        )?;
                        run(
                            "softmax",
                            &[&s.scores],
                            &[0],
                            &[nh as u32, length as u32],
                            nh * 32,
                            128,
                        )?;
                        run(
                            "attn_values",
                            &[&s.scores, &a.vcache, &s.attention_gate, &s.mixed],
                            &[0, 0, i * q_width * 4, i * q_width * 4],
                            &p,
                            q_width,
                            128,
                        )?;
                    }
                    a.out.matmul_block(&e, &s.mixed, &s.residual, batch)?;
                }
            }
            e.encode(
                "add",
                &[&s.x, &s.residual, &s.x],
                &[(batch * h) as u32],
                batch * h,
                128,
            )?;
            for i in 0..batch {
                run(
                    "rms_norm",
                    &[&s.x, &layer.post_norm, &s.normalized],
                    &[i * h * 4, 0, i * h * 4],
                    &[h as u32, eps],
                    32,
                    32,
                )?;
            }
            layer.gate.matmul_block(&e, &s.normalized, &s.gate, batch)?;
            layer.up.matmul_block(&e, &s.normalized, &s.up, batch)?;
            e.encode(
                "swiglu",
                &[&s.gate, &s.up, &s.activated],
                &[(batch * c.intermediate_size) as u32],
                batch * c.intermediate_size,
                128,
            )?;
            layer
                .down
                .matmul_block(&e, &s.activated, &s.residual, batch)?;
            e.encode(
                "add",
                &[&s.x, &s.residual, &s.x],
                &[(batch * h) as u32],
                batch * h,
                128,
            )?;
        }
        for i in 0..batch {
            run(
                "rms_norm",
                &[&s.x, &self.norm, &s.normalized],
                &[i * h * 4, 0, i * h * 4],
                &[h as u32, eps],
                32,
                32,
            )?;
        }
        self.head.as_ref().unwrap_or(&self.embedding).matmul_block(
            &e,
            &s.normalized,
            &s.logits,
            batch,
        )?;
        e.end_encoding()?;
        g.finish(cmd)?;
        let logits = g.read_f32(&s.logits, batch * c.vocab_size)?;
        let hidden = g.read_f32(&s.normalized, batch * h)?;
        Ok(BlockOutput {
            base_position: self.position,
            logits: logits
                .chunks_exact(c.vocab_size)
                .map(<[f32]>::to_vec)
                .collect(),
            hidden: hidden.chunks_exact(h).map(<[f32]>::to_vec).collect(),
        })
    }
}
