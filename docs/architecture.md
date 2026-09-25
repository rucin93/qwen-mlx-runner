# Independent Qwen Rust/Metal engine

Requested target: MacBook M5 Pro, 48 GB unified memory; maximize single-chat
generation tokens/second, no GUI, no llama.cpp, MTPLX, MLX, Candle or other
inference runtime. Rust owns inference and serving; our Metal shaders own GPU
math. Rust crates for Metal bindings, tokenization, file formats and HTTP are OK.

The current development host is M1 / 16 GB. Tests here cannot establish 27B
correctness or performance on the target. Report kernel/synthetic measurements
separately from real checkpoint generation and target performance.

## Implementation

Load a local Hugging Face directory with config.json, tokenizer.json, the
checkpoint chat template, and safetensors shards. Support dense Qwen3.5/3.8
hybrid text architecture, affine packed 4/8-bit MLX-format safetensors and
unquantized F16/BF16/F32. MLX is a weight format only, never a runtime dependency.
Reject unsupported architectures, quantization modes, malformed tensor shapes,
out-of-bounds data, and unsupported API options explicitly.

Use FP32 activations/state for an initial numerical baseline, GPU-resident packed
weights and preallocated scratch/cache. All layer operations for one token share
one command buffer with a synchronization only before CPU sampling. No full
dequantized matrix is created during inference. Full-attention KV grows within
an explicit context capacity; DeltaNet has recurrent state and causal conv state.
Prefill initially executes recurrent token steps; this is a known optimization
target, not a claimed high-throughput implementation. MTP requires a separate
validated future implementation; do not present ordinary decode as MTP.

## Boundaries

`config.rs`: ModelConfig with public fields matching HF text_config names:
hidden_size, intermediate_size, num_hidden_layers, num_attention_heads,
num_key_value_heads, head_dim, vocab_size, linear_num_key_heads,
linear_num_value_heads, linear_key_head_dim, linear_value_head_dim,
linear_conv_kernel_dim, full_attention_interval, rms_norm_eps,
rope_theta, partial_rotary_factor, max_position_embeddings,
tie_word_embeddings, eos_token_ids: Vec<u32>, layer_types: Vec<String>.
Methods `from_json(&str) -> Result<Self>`, `validate() -> Result<()>`,
`is_linear(usize) -> bool`, `rotary_dim() -> usize`.

`weights.rs`: Checkpoint::open(&Path) -> Result<Checkpoint>; public config.
matrix(&str) -> Result<MatrixData>, vector(&str) -> Result<Vec<f32>>,
conv(&str, channels: usize, width: usize) -> Result<Vec<f32>> (channel-major),
contains(&str) -> bool. Names requested by engine: `model.embed_tokens`,
`model.layers.N.input_layernorm`, `...post_attention_layernorm`,
`...linear_attn.in_proj_qkv/in_proj_z/in_proj_a/in_proj_b/out_proj`,
`...linear_attn.conv1d/A_log/dt_bias/norm`,
`...self_attn.q_proj/k_proj/v_proj/o_proj/q_norm/k_norm`,
`...mlp.gate_proj/up_proj/down_proj`, `model.norm`, `lm_head`.
Name resolution handles HF and MLX prefixes and `.weight` automatically;
A_log/dt_bias have no .weight. Normalize HF zero-centered RMS weights to
ordinary multiplicative weights on load, but never add one to gated RMS norm.
MatrixData variants: Dense { rows: usize, cols: usize, values: Vec<u16> } (F16)
and Packed { rows: usize, cols: usize, bits: u32, group_size: usize,
weights: Vec<u32>, scales: Vec<f32>, biases: Vec<f32> }.

`gpu.rs` and `kernels/*.metal`: independent GPU operations, parameters and API
agreed directly with engine implementer. Expose a generic command encoder path
so multiple operations do not each submit/wait. Tests compare numerical outputs
with independent scalar calculations; GPU tests must fail if explicitly invoked
without a Metal device, never silently count as passed.

`chat.rs`: message/template/tokenizer and sampling, no inference framework.
`server.rs`: loopback HTTP API, one loaded engine, serialized generation on a
dedicated worker, streaming, cancellation and bounded request queue.
Public shared interface (owned by chat.rs):
Message { role: String, content: String };
GenerationRequest { messages: Vec<Message>, max_tokens: usize,
temperature: f32, top_p: f32, top_k: usize, seed: u64,
enable_thinking: bool };
GenerationOutput { text: String, prompt_tokens: usize, completion_tokens: usize,
finish_reason: String, prefill_seconds: f64, decode_seconds: f64 };
trait TextGenerator: Send { fn generate(&mut self, request: &GenerationRequest,
on_text: &mut dyn FnMut(&str) -> bool) -> anyhow::Result<GenerationOutput>;
fn model_id(&self) -> &str; }
Server entry: async fn serve(engine: Box<dyn TextGenerator>, address:
std::net::SocketAddr) -> anyhow::Result<()>.

## Verification and performance contract

Test safetensor corruption, packed quantization order, norm offsets, GQA layout,
partial RoPE, causal conv, recurrent state across tokens, attention gating,
context exhaustion, session reset and streaming cancellation. Compile shaders
and run numerical kernel checks on a real Metal device. Exercise full forward
with a tiny deterministic hybrid model; independent reference logits are needed.
Do not claim validated Qwen3.8-27B chat until real checkpoint output agrees with
a reference. Do not claim faster until same-target same-workload measurements.

Primary references (read as architecture/format documentation, not runtime deps):
- https://huggingface.co/Qwen/Qwen3.8-27B/raw/main/config.json
- https://huggingface.co/mlx-community/Qwen3.8-27B-4bit/raw/main/config.json
- https://github.com/huggingface/transformers/tree/main/src/transformers/models/qwen3_5
- https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/qwen3_5.py
- https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/qwen3_next.py
