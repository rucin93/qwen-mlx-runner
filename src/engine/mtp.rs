//! Native one-layer Qwen MTP drafter. Only immutable embedding/head storage is shared.
use super::{AttentionLayer, Engine, Matrix, vector};
use crate::{config::ModelConfig, gpu::Gpu, weights::Checkpoint};
use anyhow::{Result, ensure};
use metal::Buffer;
use std::path::Path;

pub struct MtpOutput {
    /// Hidden state after the MTP final RMS, used by the next draft step.
    pub hidden: Vec<f32>,
    pub logits: Vec<f32>,
}

struct Scratch {
    embedding: Buffer,
    hidden: Buffer,
    joined: Buffer,
    x: Buffer,
    normalized: Buffer,
    residual: Buffer,
    qproj: Buffer,
    q: Buffer,
    k: Buffer,
    v: Buffer,
    attention_gate: Buffer,
    scores: Buffer,
    mixed: Buffer,
    gate: Buffer,
    up: Buffer,
    activated: Buffer,
    logits: Buffer,
}

pub struct Mtp {
    gpu: Gpu,
    config: ModelConfig,
    embedding: Matrix,
    head: Matrix,
    pre_embedding: Buffer,
    pre_hidden: Buffer,
    fc: Matrix,
    input_norm: Buffer,
    post_norm: Buffer,
    norm: Buffer,
    attention: AttentionLayer,
    gate: Matrix,
    up: Matrix,
    down: Matrix,
    scratch: Scratch,
    position: usize,
    context: usize,
}

impl Mtp {
    pub fn load(target: &Engine, directory: &Path, context: usize) -> Result<Self> {
        let cp = Checkpoint::open_mtp(directory, &target.config)?;
        let c = target.config.clone();
        ensure!(
            context > 0 && context <= target.context,
            "MTP context exceeds target capacity"
        );
        let g = Gpu::new()?;
        ensure!(
            g.device.registry_id() == target.gpu.device.registry_id(),
            "MTP and target must use the same Metal device"
        );
        let h = c.hidden_size;
        let qd = c.num_attention_heads * c.head_dim;
        let kvd = c.num_key_value_heads * c.head_dim;
        for dims in [
            [h, 2 * h],
            [context, kvd],
            [c.num_attention_heads * 32, context],
        ] {
            ensure!(
                dims[0]
                    .checked_mul(dims[1])
                    .is_some_and(|n| n <= u32::MAX as usize),
                "MTP tensor exceeds GPU indexing capacity"
            );
        }
        // Allocate persistent KV and temporaries first, so matrix loading includes them in its budget.
        let s = Scratch {
            embedding: g.alloc_f32(h)?,
            hidden: g.alloc_f32(h)?,
            joined: g.alloc_f32(2 * h)?,
            x: g.alloc_f32(h)?,
            normalized: g.alloc_f32(h)?,
            residual: g.alloc_f32(h)?,
            qproj: g.alloc_f32(qd * 2)?,
            q: g.alloc_f32(qd)?,
            k: g.alloc_f32(kvd)?,
            v: g.alloc_f32(kvd)?,
            attention_gate: g.alloc_f32(qd)?,
            scores: g.alloc_f32(c.num_attention_heads * context)?,
            mixed: g.alloc_f32(qd)?,
            gate: g.alloc_f32(c.intermediate_size)?,
            up: g.alloc_f32(c.intermediate_size)?,
            activated: g.alloc_f32(c.intermediate_size)?,
            logits: g.alloc_f32(c.vocab_size)?,
        };
        let kcache = g.alloc_f32(context * kvd)?;
        let vcache = g.alloc_f32(context * kvd)?;
        let load = |name: &str, rows, cols| Matrix::load(&cp, &g, name, rows, cols);
        let vec = |name: &str, len| vector(&cp, &g, name, len);
        let pre_embedding = vec("pre_fc_norm_embedding", h)?;
        let pre_hidden = vec("pre_fc_norm_hidden", h)?;
        let input_norm = vec("layers.0.input_layernorm", h)?;
        let post_norm = vec("layers.0.post_attention_layernorm", h)?;
        let norm = vec("norm", h)?;
        let fc = load("fc", h, 2 * h)?;
        let attention = AttentionLayer {
            q: load("layers.0.self_attn.q_proj", qd * 2, h)?,
            k: load("layers.0.self_attn.k_proj", kvd, h)?,
            v: load("layers.0.self_attn.v_proj", kvd, h)?,
            out: load("layers.0.self_attn.o_proj", h, qd)?,
            qnorm: vec("layers.0.self_attn.q_norm", c.head_dim)?,
            knorm: vec("layers.0.self_attn.k_norm", c.head_dim)?,
            kcache,
            vcache,
        };
        let gate = load("layers.0.mlp.gate_proj", c.intermediate_size, h)?;
        let up = load("layers.0.mlp.up_proj", c.intermediate_size, h)?;
        let down = load("layers.0.mlp.down_proj", h, c.intermediate_size)?;
        ensure!(
            g.device.current_allocated_size() < g.device.recommended_max_working_set_size(),
            "MTP and target exceed recommended GPU memory budget"
        );
        Ok(Self {
            gpu: g,
            config: c,
            embedding: target.embedding.share(),
            head: target.head.as_ref().unwrap_or(&target.embedding).share(),
            pre_embedding,
            pre_hidden,
            fc,
            input_norm,
            post_norm,
            norm,
            attention,
            gate,
            up,
            down,
            scratch: s,
            position: 0,
            context,
        })
    }

    pub fn position(&self) -> usize {
        self.position
    }

    /// Mask the rejected suffix. Every later append overwrites its own KV slot.
    pub fn truncate(&mut self, position: usize) -> Result<()> {
        ensure!(
            position <= self.position,
            "MTP truncate cannot advance the cache"
        );
        self.position = position;
        Ok(())
    }

    pub fn reset(&mut self) {
        self.position = 0;
    }

    pub fn forward(
        &mut self,
        token: u32,
        hidden: &[f32],
        output_logits: bool,
    ) -> Result<MtpOutput> {
        ensure!(
            self.position < self.context,
            "MTP context capacity exhausted"
        );
        ensure!(
            (token as usize) < self.config.vocab_size,
            "MTP token exceeds vocabulary"
        );
        ensure!(
            hidden.len() == self.config.hidden_size && hidden.iter().all(|v| v.is_finite()),
            "MTP needs a finite normalized target/draft hidden vector"
        );
        let result = objc::rc::autoreleasepool(|| self.forward_inner(token, hidden, output_logits));
        if result.is_err() {
            self.reset();
        }
        result
    }

    fn forward_inner(
        &mut self,
        token: u32,
        hidden: &[f32],
        output_logits: bool,
    ) -> Result<MtpOutput> {
        let (g, c, s, a) = (&self.gpu, &self.config, &self.scratch, &self.attention);
        let (h, nh, nk, hd) = (
            c.hidden_size,
            c.num_attention_heads,
            c.num_key_value_heads,
            c.head_dim,
        );
        let eps = c.rms_norm_eps.to_bits();
        // Shared storage, with synchronous finish on every preceding use. No in-flight GPU reader.
        unsafe {
            std::ptr::copy_nonoverlapping(hidden.as_ptr(), s.hidden.contents().cast::<f32>(), h);
        }
        let cmd = g.begin();
        let e = g.begin_encoding(cmd);
        self.embedding.embed(&e, token, &s.embedding)?;
        e.encode(
            "rms_norm",
            &[&s.embedding, &self.pre_embedding, &s.joined],
            &[h as u32, eps],
            32,
            32,
        )?;
        e.encode_offsets(
            "rms_norm",
            &[&s.hidden, &self.pre_hidden, &s.joined],
            &[0, 0, h * 4],
            &[h as u32, eps],
            32,
            32,
        )?;
        self.fc.matvec(&e, &s.joined, &s.x)?;
        e.encode(
            "rms_norm",
            &[&s.x, &self.input_norm, &s.normalized],
            &[h as u32, eps],
            32,
            32,
        )?;
        a.q.matvec(&e, &s.normalized, &s.qproj)?;
        a.k.matvec(&e, &s.normalized, &s.k)?;
        a.v.matvec(&e, &s.normalized, &s.v)?;
        e.encode(
            "split_q_gate",
            &[&s.qproj, &s.q, &s.attention_gate],
            &[nh as u32, hd as u32],
            nh * hd,
            128,
        )?;
        for (x, w, heads) in [(&s.q, &a.qnorm, nh), (&s.k, &a.knorm, nk)] {
            e.encode(
                "head_rms",
                &[x, w],
                &[heads as u32, hd as u32, eps],
                heads * 32,
                128,
            )?;
            e.encode(
                "rope",
                &[x],
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
        e.encode(
            "kv_append",
            &[&s.k, &s.v, &a.kcache, &a.vcache],
            &[nk as u32, hd as u32, self.position as u32],
            nk * hd,
            128,
        )?;
        let length = self.position + 1;
        let p = [nh as u32, nk as u32, hd as u32, length as u32];
        e.encode(
            "attn_scores",
            &[&s.q, &a.kcache, &s.scores],
            &p,
            nh * length * 32,
            128,
        )?;
        e.encode(
            "softmax",
            &[&s.scores],
            &[nh as u32, length as u32],
            nh * 32,
            128,
        )?;
        e.encode(
            "attn_values",
            &[&s.scores, &a.vcache, &s.attention_gate, &s.mixed],
            &p,
            nh * hd,
            128,
        )?;
        a.out.matvec(&e, &s.mixed, &s.residual)?;
        e.encode("add", &[&s.x, &s.residual, &s.x], &[h as u32], h, 128)?;
        e.encode(
            "rms_norm",
            &[&s.x, &self.post_norm, &s.normalized],
            &[h as u32, eps],
            32,
            32,
        )?;
        self.gate.matvec(&e, &s.normalized, &s.gate)?;
        self.up.matvec(&e, &s.normalized, &s.up)?;
        e.encode(
            "swiglu",
            &[&s.gate, &s.up, &s.activated],
            &[c.intermediate_size as u32],
            c.intermediate_size,
            128,
        )?;
        self.down.matvec(&e, &s.activated, &s.residual)?;
        e.encode("add", &[&s.x, &s.residual, &s.x], &[h as u32], h, 128)?;
        e.encode(
            "rms_norm",
            &[&s.x, &self.norm, &s.normalized],
            &[h as u32, eps],
            32,
            32,
        )?;
        if output_logits {
            self.head.matvec(&e, &s.normalized, &s.logits)?;
        }
        e.end_encoding()?;
        g.finish(cmd)?;
        let hidden = g.read_f32(&s.normalized, h)?;
        let logits = if output_logits {
            g.read_f32(&s.logits, c.vocab_size)?
        } else {
            Vec::new()
        };
        ensure!(
            hidden.iter().chain(&logits).all(|v| v.is_finite()),
            "MTP produced non-finite output"
        );
        self.position += 1;
        Ok(MtpOutput { hidden, logits })
    }
}
