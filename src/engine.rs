//! Own autoregressive execution graph. No third-party inference runtime.
mod synthetic;
use std::{path::Path, time::Instant};

use anyhow::{Context, Result, ensure};
use metal::{Buffer, BufferRef};

use crate::{
    chat::{ChatTokenizer, GenerationOutput, GenerationRequest, Sampler, TextGenerator},
    config::ModelConfig,
    gpu::{DispatchEncoder, FrameTiming, Gpu, ProfileRow},
    weights::{Checkpoint, MatrixData},
};

struct Matrix {
    weight: Buffer,
    scales: Option<Buffer>,
    biases: Option<Buffer>,
    rows: usize,
    cols: usize,
    bits: u32,
    group: usize,
    metadata_bf16: bool,
}

impl Matrix {
    fn load(cp: &Checkpoint, gpu: &Gpu, name: &str, rows: usize, cols: usize) -> Result<Self> {
        let data = cp.matrix(name).with_context(|| format!("loading {name}"))?;
        let compacted = match &data {
            MatrixData::Packed {
                bits,
                group_size,
                scales,
                biases,
                ..
            } => gpu.compact_metadata(*bits, *group_size, scales, biases),
            MatrixData::Dense { .. } => None,
        };
        let saved_bytes = compacted
            .as_ref()
            .map_or(0, |(s, b)| (s.len() + b.len()) as u64 * 2);
        let (r, c, bytes) = match &data {
            MatrixData::Dense { rows, cols, values } => (*rows, *cols, values.len() * 2),
            MatrixData::Packed {
                rows,
                cols,
                weights,
                scales,
                biases,
                ..
            } => (
                *rows,
                *cols,
                (weights.len() + scales.len() + biases.len()) * 4 - saved_bytes as usize,
            ),
        };
        ensure!(
            (r, c) == (rows, cols),
            "{name}: expected [{rows},{cols}], got [{r},{c}]"
        );
        let budget = gpu.device.recommended_max_working_set_size();
        ensure!(
            gpu.device
                .current_allocated_size()
                .saturating_add(bytes as u64)
                < budget,
            "model exceeds Metal's recommended memory budget ({:.1} GiB); use packed 4-bit weights or more RAM",
            budget as f64 / 1073741824.
        );
        match data {
            MatrixData::Dense { values, .. } => Ok(Self {
                weight: gpu.upload_bytes(bytemuck::cast_slice(&values))?,
                scales: None,
                biases: None,
                rows,
                cols,
                bits: 16,
                group: 0,
                metadata_bf16: false,
            }),
            MatrixData::Packed {
                weights,
                scales,
                biases,
                bits,
                group_size,
                ..
            } => {
                let metadata_bf16 = compacted.is_some();
                let (scales, biases) = if let Some((s, b)) = compacted {
                    (
                        gpu.upload_bytes(bytemuck::cast_slice(&s))?,
                        gpu.upload_bytes(bytemuck::cast_slice(&b))?,
                    )
                } else {
                    (gpu.upload_f32(&scales)?, gpu.upload_f32(&biases)?)
                };
                let matrix = Self {
                    weight: gpu.upload_bytes(bytemuck::cast_slice(&weights))?,
                    scales: Some(scales),
                    biases: Some(biases),
                    rows,
                    cols,
                    bits,
                    group: group_size,
                    metadata_bf16,
                };
                if metadata_bf16 {
                    gpu.record_compacted_metadata(saved_bytes);
                }
                Ok(matrix)
            }
        }
    }

    fn matvec(&self, e: &DispatchEncoder<'_>, x: &BufferRef, y: &BufferRef) -> Result<()> {
        if let (Some(s), Some(b)) = (&self.scales, &self.biases) {
            e.encode(
                if self.metadata_bf16 {
                    "matvec_affine_bf16"
                } else {
                    "matvec_affine"
                },
                &[&self.weight, s, b, x, y],
                &[
                    self.rows as u32,
                    self.cols as u32,
                    self.bits,
                    self.group as u32,
                ],
                self.rows * 32,
                128,
            )
        } else {
            e.encode(
                "matvec_f16",
                &[&self.weight, x, y],
                &[self.rows as u32, self.cols as u32],
                self.rows * 32,
                128,
            )
        }
    }

    fn embed(&self, e: &DispatchEncoder<'_>, token: u32, y: &BufferRef) -> Result<()> {
        if let (Some(s), Some(b)) = (&self.scales, &self.biases) {
            e.encode(
                if self.metadata_bf16 {
                    "embed_affine_bf16"
                } else {
                    "embed_affine"
                },
                &[&self.weight, s, b, y],
                &[token, self.cols as u32, self.bits, self.group as u32],
                self.cols,
                128,
            )
        } else {
            e.encode(
                "embed_f16",
                &[&self.weight, y],
                &[token, self.cols as u32],
                self.cols,
                128,
            )
        }
    }
}

fn vector(cp: &Checkpoint, g: &Gpu, name: &str, len: usize) -> Result<Buffer> {
    let v = cp.vector(name)?;
    ensure!(
        v.len() == len,
        "{name}: expected {len} elements, got {}",
        v.len()
    );
    ensure!(
        v.iter().all(|x| x.is_finite()),
        "{name}: non-finite weights"
    );
    g.upload_f32(&v)
}

struct DeltaLayer {
    qkv: Matrix,
    z: Matrix,
    a: Matrix,
    b: Matrix,
    out: Matrix,
    conv: Buffer,
    alog: Buffer,
    dt: Buffer,
    norm: Buffer,
    conv_state: Buffer,
    state: Buffer,
}
struct AttentionLayer {
    q: Matrix,
    k: Matrix,
    v: Matrix,
    out: Matrix,
    qnorm: Buffer,
    knorm: Buffer,
    kcache: Buffer,
    vcache: Buffer,
}
enum Mixer {
    Delta(DeltaLayer),
    Attention(AttentionLayer),
}
struct Layer {
    input_norm: Buffer,
    post_norm: Buffer,
    mixer: Mixer,
    gate: Matrix,
    up: Matrix,
    down: Matrix,
}
struct Scratch {
    x: Buffer,
    normalized: Buffer,
    residual: Buffer,
    gate: Buffer,
    up: Buffer,
    activated: Buffer,
    qkv: Buffer,
    convolved: Buffer,
    z: Buffer,
    a: Buffer,
    b: Buffer,
    mixed: Buffer,
    gated: Buffer,
    qproj: Buffer,
    q: Buffer,
    k: Buffer,
    v: Buffer,
    attention_gate: Buffer,
    scores: Buffer,
    logits: Buffer,
}

pub struct Engine {
    gpu: Gpu,
    config: ModelConfig,
    embedding: Matrix,
    head: Option<Matrix>,
    norm: Buffer,
    layers: Vec<Layer>,
    scratch: Scratch,
    position: usize,
    context: usize,
}

impl Engine {
    pub fn load(directory: &Path, context: usize) -> Result<Self> {
        let cp = Checkpoint::open(directory)?;
        let c = cp.config.clone();
        ensure!(
            context > 0 && context <= c.max_position_embeddings,
            "context must be in 1..={}",
            c.max_position_embeddings
        );
        ensure!(
            c.linear_num_value_heads % c.linear_num_key_heads == 0,
            "DeltaNet value heads must be a multiple of key heads"
        );
        // Metal shader addresses use u32; reject overflow before multiplying/allocating.
        for dims in [
            vec![c.vocab_size, c.hidden_size],
            vec![c.intermediate_size, c.hidden_size],
            vec![c.num_attention_heads, c.head_dim, 2, c.hidden_size],
            vec![
                c.linear_num_value_heads,
                c.linear_value_head_dim,
                c.linear_key_head_dim,
            ],
            vec![context, c.num_key_value_heads, c.head_dim],
            vec![c.num_attention_heads, context, 32],
        ] {
            let product = dims
                .into_iter()
                .try_fold(1usize, |p, v| p.checked_mul(v))
                .context("tensor size overflow")?;
            ensure!(
                product <= u32::MAX as usize,
                "tensor exceeds 32-bit GPU indexing"
            );
        }
        let g = Gpu::new()?;
        let h = c.hidden_size;
        let kd = c.linear_num_key_heads * c.linear_key_head_dim;
        let vd = c.linear_num_value_heads * c.linear_value_head_dim;
        let qd = c.num_attention_heads * c.head_dim;
        let kvd = c.num_key_value_heads * c.head_dim;
        let qkv = 2 * kd + vd;
        let cache_bytes = (context as u64)
            * (kvd as u64)
            * 8
            * (c.layer_types
                .iter()
                .filter(|s| *s == "full_attention")
                .count() as u64);
        ensure!(
            cache_bytes < g.device.recommended_max_working_set_size() / 2,
            "KV cache alone would use {:.1} GiB; lower --context",
            cache_bytes as f64 / 1073741824.
        );
        // Allocate caches/scratch before matrices so the matrix budget guard includes them.
        let scratch = Scratch {
            x: g.alloc_f32(h)?,
            normalized: g.alloc_f32(h)?,
            residual: g.alloc_f32(h)?,
            gate: g.alloc_f32(c.intermediate_size)?,
            up: g.alloc_f32(c.intermediate_size)?,
            activated: g.alloc_f32(c.intermediate_size)?,
            qkv: g.alloc_f32(qkv)?,
            convolved: g.alloc_f32(qkv)?,
            z: g.alloc_f32(vd)?,
            a: g.alloc_f32(c.linear_num_value_heads)?,
            b: g.alloc_f32(c.linear_num_value_heads)?,
            mixed: g.alloc_f32(vd.max(qd))?,
            gated: g.alloc_f32(vd)?,
            qproj: g.alloc_f32(qd * 2)?,
            q: g.alloc_f32(qd)?,
            k: g.alloc_f32(kvd)?,
            v: g.alloc_f32(kvd)?,
            attention_gate: g.alloc_f32(qd)?,
            scores: g.alloc_f32(c.num_attention_heads * context)?,
            logits: g.alloc_f32(c.vocab_size)?,
        };
        let embedding = Matrix::load(&cp, &g, "model.embed_tokens", c.vocab_size, h)?;
        let mut layers = Vec::with_capacity(c.num_hidden_layers);
        for li in 0..c.num_hidden_layers {
            let p = format!("model.layers.{li}");
            let load =
                |suffix: &str, r, col| Matrix::load(&cp, &g, &format!("{p}.{suffix}"), r, col);
            let mixer = if c.is_linear(li) {
                let conv = cp.conv(
                    &format!("{p}.linear_attn.conv1d"),
                    qkv,
                    c.linear_conv_kernel_dim,
                )?;
                Mixer::Delta(DeltaLayer {
                    qkv: load("linear_attn.in_proj_qkv", qkv, h)?,
                    z: load("linear_attn.in_proj_z", vd, h)?,
                    a: load("linear_attn.in_proj_a", c.linear_num_value_heads, h)?,
                    b: load("linear_attn.in_proj_b", c.linear_num_value_heads, h)?,
                    out: load("linear_attn.out_proj", h, vd)?,
                    conv: g.upload_f32(&conv)?,
                    alog: vector(
                        &cp,
                        &g,
                        &format!("{p}.linear_attn.A_log"),
                        c.linear_num_value_heads,
                    )?,
                    dt: vector(
                        &cp,
                        &g,
                        &format!("{p}.linear_attn.dt_bias"),
                        c.linear_num_value_heads,
                    )?,
                    norm: vector(
                        &cp,
                        &g,
                        &format!("{p}.linear_attn.norm"),
                        c.linear_value_head_dim,
                    )?,
                    conv_state: g.alloc_f32(qkv * (c.linear_conv_kernel_dim - 1))?,
                    state: g.alloc_f32(
                        c.linear_num_value_heads * c.linear_value_head_dim * c.linear_key_head_dim,
                    )?,
                })
            } else {
                Mixer::Attention(AttentionLayer {
                    q: load("self_attn.q_proj", qd * 2, h)?,
                    k: load("self_attn.k_proj", kvd, h)?,
                    v: load("self_attn.v_proj", kvd, h)?,
                    out: load("self_attn.o_proj", h, qd)?,
                    qnorm: vector(&cp, &g, &format!("{p}.self_attn.q_norm"), c.head_dim)?,
                    knorm: vector(&cp, &g, &format!("{p}.self_attn.k_norm"), c.head_dim)?,
                    kcache: g.alloc_f32(context * kvd)?,
                    vcache: g.alloc_f32(context * kvd)?,
                })
            };
            layers.push(Layer {
                input_norm: vector(&cp, &g, &format!("{p}.input_layernorm"), h)?,
                post_norm: vector(&cp, &g, &format!("{p}.post_attention_layernorm"), h)?,
                mixer,
                gate: load("mlp.gate_proj", c.intermediate_size, h)?,
                up: load("mlp.up_proj", c.intermediate_size, h)?,
                down: load("mlp.down_proj", h, c.intermediate_size)?,
            });
        }
        let norm = vector(&cp, &g, "model.norm", h)?;
        let head = if c.tie_word_embeddings {
            None
        } else {
            Some(Matrix::load(&cp, &g, "lm_head", c.vocab_size, h)?)
        };
        ensure!(
            g.device.current_allocated_size() < g.device.recommended_max_working_set_size(),
            "model and caches exceed recommended GPU memory budget"
        );
        let mut engine = Self {
            gpu: g,
            config: c,
            embedding,
            head,
            norm,
            layers,
            scratch,
            position: 0,
            context,
        };
        engine.reset();
        Ok(engine)
    }

    pub fn config(&self) -> &ModelConfig {
        &self.config
    }
    pub fn kernel_mode(&self) -> &str {
        self.gpu.kernel_mode()
    }
    pub fn norm_mode(&self) -> &str {
        self.gpu.norm_mode()
    }
    pub fn metadata_mode(&self) -> &str {
        self.gpu.metadata_mode()
    }
    pub fn metadata_stats(&self) -> (usize, u64) {
        self.gpu.metadata_stats()
    }
    pub fn enable_profiling(&mut self) -> Result<()> {
        self.gpu.enable_profiling()
    }
    pub fn enable_command_profiling(&mut self) -> Result<()> {
        self.gpu.enable_command_profiling()
    }
    pub fn profile_backend(&self) -> Option<&str> {
        self.gpu.profile_backend()
    }
    pub fn disable_profiling(&mut self) {
        self.gpu.disable_profiling();
    }
    pub fn set_parallel_norm(&mut self, enabled: bool) {
        self.gpu.set_parallel_norm(enabled);
    }
    pub fn profile_report(&self) -> Result<Vec<ProfileRow>> {
        self.gpu.profile_report()
    }
    pub fn last_frame_timing(&self) -> Option<FrameTiming> {
        self.gpu.last_frame_timing()
    }
    pub fn device_name(&self) -> &str {
        self.gpu.device.name()
    }
    pub fn allocated_bytes(&self) -> u64 {
        self.gpu.device.current_allocated_size()
    }
    pub fn position(&self) -> usize {
        self.position
    }
    pub fn context_capacity(&self) -> usize {
        self.context
    }

    /// Called only when the previous synchronous forward has completed.
    pub fn reset(&mut self) {
        for layer in &self.layers {
            if let Mixer::Delta(d) = &layer.mixer {
                for buffer in [&d.conv_state, &d.state] {
                    // Shared storage, no GPU command outstanding: CPU owns these bytes.
                    unsafe {
                        std::ptr::write_bytes(
                            buffer.contents().cast::<u8>(),
                            0,
                            buffer.length() as usize,
                        );
                    }
                }
            }
        }
        // Attention reads only positions <= position, overwritten on the next forward.
        self.position = 0;
    }

    pub fn forward(&mut self, token: u32) -> Result<Vec<f32>> {
        self.forward_mode(token, true)
    }

    pub fn prefill_token(&mut self, token: u32, final_token: bool) -> Result<Vec<f32>> {
        self.forward_mode(token, final_token)
    }

    fn forward_mode(&mut self, token: u32, logits: bool) -> Result<Vec<f32>> {
        ensure!(
            (token as usize) < self.config.vocab_size,
            "token {token} outside vocabulary"
        );
        ensure!(
            self.position < self.context,
            "context capacity {} exhausted",
            self.context
        );
        let result = objc::rc::autoreleasepool(|| self.forward_inner(token, logits));
        // A failed command may have updated recurrent state; never reuse it.
        if result.is_err() {
            self.reset();
        }
        result
    }

    fn forward_inner(&mut self, token: u32, logits: bool) -> Result<Vec<f32>> {
        let (g, c, s) = (&self.gpu, &self.config, &self.scratch);
        let cmd = g.begin();
        let e = g.begin_encoding(cmd);
        self.embedding.embed(&e, token, &s.x)?;
        let h = c.hidden_size;
        let hd = c.head_dim;
        let kh = c.linear_num_key_heads;
        let vh = c.linear_num_value_heads;
        let kd = c.linear_key_head_dim;
        let vd = c.linear_value_head_dim;
        let qkv = 2 * kh * kd + vh * vd;
        let nh = c.num_attention_heads;
        let nk = c.num_key_value_heads;
        let eps = c.rms_norm_eps.to_bits();
        let run = |name: &str, buffers: &[&BufferRef], p: &[u32], threads: usize, group: usize| {
            e.encode(name, buffers, p, threads, group)
        };
        for layer in &self.layers {
            run(
                "rms_norm",
                &[&s.x, &layer.input_norm, &s.normalized],
                &[h as u32, eps],
                32,
                32,
            )?;
            match &layer.mixer {
                Mixer::Delta(d) => {
                    d.qkv.matvec(&e, &s.normalized, &s.qkv)?;
                    d.z.matvec(&e, &s.normalized, &s.z)?;
                    d.a.matvec(&e, &s.normalized, &s.a)?;
                    d.b.matvec(&e, &s.normalized, &s.b)?;
                    run(
                        "conv_silu",
                        &[&s.qkv, &d.conv, &d.conv_state, &s.convolved],
                        &[qkv as u32, c.linear_conv_kernel_dim as u32],
                        qkv,
                        128,
                    )?;
                    run(
                        "delta_norm",
                        &[&s.convolved],
                        &[kh as u32, kd as u32, 1e-6f32.to_bits()],
                        kh * 32,
                        128,
                    )?;
                    run(
                        "delta_step",
                        &[&s.convolved, &s.a, &s.b, &d.alog, &d.dt, &d.state, &s.mixed],
                        &[kh as u32, vh as u32, kd as u32, vd as u32],
                        vh * vd * 32,
                        128,
                    )?;
                    run(
                        "gated_rms",
                        &[&s.mixed, &s.z, &d.norm, &s.gated],
                        &[vh as u32, vd as u32, eps],
                        vh * 32,
                        128,
                    )?;
                    d.out.matvec(&e, &s.gated, &s.residual)?;
                }
                Mixer::Attention(a) => {
                    a.q.matvec(&e, &s.normalized, &s.qproj)?;
                    a.k.matvec(&e, &s.normalized, &s.k)?;
                    a.v.matvec(&e, &s.normalized, &s.v)?;
                    run(
                        "split_q_gate",
                        &[&s.qproj, &s.q, &s.attention_gate],
                        &[nh as u32, hd as u32],
                        nh * hd,
                        128,
                    )?;
                    run(
                        "head_rms",
                        &[&s.q, &a.qnorm],
                        &[nh as u32, hd as u32, eps],
                        nh * 32,
                        128,
                    )?;
                    run(
                        "head_rms",
                        &[&s.k, &a.knorm],
                        &[nk as u32, hd as u32, eps],
                        nk * 32,
                        128,
                    )?;
                    for (buffer, heads) in [(&s.q, nh), (&s.k, nk)] {
                        run(
                            "rope",
                            &[buffer],
                            &[
                                heads as u32,
                                hd as u32,
                                c.rotary_dim() as u32,
                                self.position as u32,
                                c.rope_theta.to_bits(),
                            ],
                            heads * c.rotary_dim() / 2,
                            128,
                        )?;
                    }
                    run(
                        "kv_append",
                        &[&s.k, &s.v, &a.kcache, &a.vcache],
                        &[nk as u32, hd as u32, self.position as u32],
                        nk * hd,
                        128,
                    )?;
                    let length = self.position + 1;
                    let p = [nh as u32, nk as u32, hd as u32, length as u32];
                    run(
                        "attn_scores",
                        &[&s.q, &a.kcache, &s.scores],
                        &p,
                        nh * length * 32,
                        128,
                    )?;
                    run(
                        "softmax",
                        &[&s.scores],
                        &[nh as u32, length as u32],
                        nh * 32,
                        128,
                    )?;
                    run(
                        "attn_values",
                        &[&s.scores, &a.vcache, &s.attention_gate, &s.mixed],
                        &p,
                        nh * hd,
                        128,
                    )?;
                    a.out.matvec(&e, &s.mixed, &s.residual)?;
                }
            }
            run("add", &[&s.x, &s.residual, &s.x], &[h as u32], h, 128)?;
            run(
                "rms_norm",
                &[&s.x, &layer.post_norm, &s.normalized],
                &[h as u32, eps],
                32,
                32,
            )?;
            layer.gate.matvec(&e, &s.normalized, &s.gate)?;
            layer.up.matvec(&e, &s.normalized, &s.up)?;
            run(
                "swiglu",
                &[&s.gate, &s.up, &s.activated],
                &[c.intermediate_size as u32],
                c.intermediate_size,
                128,
            )?;
            layer.down.matvec(&e, &s.activated, &s.residual)?;
            run("add", &[&s.x, &s.residual, &s.x], &[h as u32], h, 128)?;
        }
        if logits {
            run(
                "rms_norm",
                &[&s.x, &self.norm, &s.normalized],
                &[h as u32, eps],
                32,
                32,
            )?;
            self.head
                .as_ref()
                .unwrap_or(&self.embedding)
                .matvec(&e, &s.normalized, &s.logits)?;
        }
        e.end_encoding()?;
        g.finish(cmd)?;
        self.position += 1;
        if logits {
            g.read_f32(&s.logits, c.vocab_size)
        } else {
            Ok(Vec::new())
        }
    }
}

pub struct ChatEngine {
    engine: Engine,
    tokenizer: ChatTokenizer,
    model_id: String,
    cached_tokens: Vec<u32>,
    cached_logits: Vec<f32>,
}
impl ChatEngine {
    pub fn load(path: &Path, context: usize) -> Result<Self> {
        let tokenizer = ChatTokenizer::load(path)?;
        let engine = Engine::load(path, context)?;
        let model_id = path
            .file_name()
            .context("model path needs a directory name")?
            .to_string_lossy()
            .into_owned();
        Ok(Self {
            engine,
            tokenizer,
            model_id,
            cached_tokens: Vec::new(),
            cached_logits: Vec::new(),
        })
    }
    pub fn engine(&self) -> &Engine {
        &self.engine
    }
    pub fn clear_cache(&mut self) {
        self.engine.reset();
        self.cached_tokens.clear();
        self.cached_logits.clear();
    }
    fn step(&mut self, token: u32) -> Result<()> {
        match self.engine.forward(token) {
            Ok(logits) => {
                self.cached_logits = logits;
                self.cached_tokens.push(token);
                Ok(())
            }
            Err(err) => {
                self.clear_cache();
                Err(err)
            }
        }
    }
}
impl TextGenerator for ChatEngine {
    fn model_id(&self) -> &str {
        &self.model_id
    }
    fn generate(
        &mut self,
        r: &GenerationRequest,
        on_text: &mut dyn FnMut(&str) -> bool,
    ) -> Result<GenerationOutput> {
        ensure!(r.max_tokens > 0, "max_tokens must be positive");
        ensure!(
            r.temperature.is_finite() && r.temperature >= 0.,
            "invalid temperature"
        );
        ensure!(
            r.top_p.is_finite() && r.top_p > 0. && r.top_p <= 1.,
            "invalid top_p"
        );
        let prompt = self
            .tokenizer
            .encode(&self.tokenizer.render(&r.messages, r.enable_thinking)?)?;
        ensure!(!prompt.is_empty(), "empty prompt after tokenization");
        ensure!(
            prompt
                .len()
                .checked_add(r.max_tokens)
                .is_some_and(|n| n <= self.engine.context_capacity()),
            "prompt ({} tokens) plus max_tokens ({}) exceeds context {}",
            prompt.len(),
            r.max_tokens,
            self.engine.context_capacity()
        );
        if !prompt.starts_with(&self.cached_tokens) || self.cached_tokens.is_empty() {
            self.clear_cache();
        }
        let prefill = Instant::now();
        let offset = self.cached_tokens.len();
        for (i, &id) in prompt.iter().enumerate().skip(offset) {
            if !on_text("") {
                self.clear_cache();
                anyhow::bail!("request cancelled during prefill");
            }
            match self.engine.prefill_token(id, i + 1 == prompt.len()) {
                Ok(logits) => {
                    self.cached_logits = logits;
                    self.cached_tokens.push(id);
                }
                Err(error) => {
                    self.clear_cache();
                    return Err(error);
                }
            }
        }
        let prefill_seconds = prefill.elapsed().as_secs_f64();
        let decode = Instant::now();
        let mut sampler = Sampler::new(r.seed);
        let mut tokens = Vec::with_capacity(r.max_tokens);
        let mut text = String::new();
        let mut count = 0;
        let mut finish = "length";
        for i in 0..r.max_tokens {
            // Check cancellation even when a tokenizer hasn't emitted a full UTF-8 sequence yet.
            if !on_text("") {
                finish = "cancelled";
                break;
            }
            let token = sampler.sample(&self.cached_logits, r.temperature, r.top_p, r.top_k)?;
            count += 1;
            if self.engine.config.eos_token_ids.contains(&token) {
                finish = "stop";
                break;
            }
            tokens.push(token);
            let decoded = if r.enable_thinking {
                self.tokenizer.decode_with_special_tokens(&tokens)?
            } else {
                self.tokenizer.decode(&tokens)?
            };
            if !decoded.ends_with('\u{fffd}') {
                ensure!(
                    decoded.starts_with(&text),
                    "tokenizer decoder changed previously emitted text"
                );
                let chunk = &decoded[text.len()..];
                if !chunk.is_empty() && !on_text(chunk) {
                    finish = "cancelled";
                    break;
                }
                text = decoded;
            }
            if i + 1 < r.max_tokens {
                self.step(token)?;
            }
        }
        // If output ended in a partial UTF-8 sequence, return the tokenizer's replacement,
        // ensuring non-stream and stream represent the same completed text.
        if finish != "cancelled" {
            let final_text = if r.enable_thinking {
                self.tokenizer.decode_with_special_tokens(&tokens)?
            } else {
                self.tokenizer.decode(&tokens)?
            };
            ensure!(
                final_text.starts_with(&text),
                "unstable final tokenizer output"
            );
            if final_text.len() > text.len() {
                on_text(&final_text[text.len()..]);
            }
            text = final_text;
        }
        Ok(GenerationOutput {
            text,
            prompt_tokens: prompt.len(),
            completion_tokens: count,
            finish_reason: finish.into(),
            prefill_seconds,
            decode_seconds: decode.elapsed().as_secs_f64(),
        })
    }
}
