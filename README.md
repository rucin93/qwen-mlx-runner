# qwen-mlx-runner

An independent **Rust + Metal** inference engine for the dense Qwen3.5 / Qwen3.8
text architecture on Apple Silicon. The target machine is a MacBook with an
M5 Pro and 48 GB unified memory. It provides a headless local chat server.

Despite the repository name, **MLX is only a supported checkpoint format**.
Inference uses this project's Rust execution graph and original Metal shaders.
There is no llama.cpp, MTPLX, MLX, Candle, PyTorch, or other inference runtime in
the executable. Standard crates provide Metal bindings, tokenization, Jinja
templates, JSON, memory mapping, and HTTP.

## Status

This is a numerically tested implementation under active performance tuning.
The optimized path specializes packed Q4/Q8 matrix-vector products and keeps
the original path available for same-binary comparisons. Real Metal GPU tests
cover an independent FP64 matrix oracle and complete four-layer dense and Q4
synthetic hybrid models, including reset and context bounds.

**Trained Qwen3.8-27B output quality has not yet been independently validated.**
Target-machine throughput is documented through user-supplied M5 Pro results;
local GPU verification uses an M1. Synthetic tests and microbenchmarks do not
establish trained-model quality or target throughput.

The reported M5 Pro result improved from a measured run of 1.45 decode tokens/s
to a three-run median of **8.16 tokens/s** with the aligned kernels. The requested
10× improvement remains a target, **not a verified result**. See the
[first iteration](docs/performance-2026-09-25.md) and
[v0.2 profiling and optimization notes](docs/performance-v0.2.md).
The [v0.3 notes](docs/performance-v0.3.md) cover the M5 zero-counter fallback
and optional lossless BF16 metadata storage.
The [v0.4 follow-up](docs/performance-v0.4.md) records the successful M5 profile:
parallel RMS reduced one normal step from 83.18 to 72.08 ms. These single-step
measurements do not establish a new full-benchmark token rate.

Implemented:

- Safetensors loading, including indexed shards and bounds/shape validation.
- Affine packed 4-bit and 8-bit matrices in MLX format; F16/BF16/F32 source
  matrices are converted to F16 on load.
- Original fused dequantization/matrix-vector Metal kernels: weights remain
  packed during inference.
- Hybrid Gated DeltaNet recurrence, causal depthwise convolution, grouped-query
  attention, partial RoPE, gated norms, SwiGLU, and residual connections.
- GPU-resident weights, preallocated scratch space and explicit context capacity.
- One GPU command buffer per token; no CPU/GPU synchronization between layers.
- Checkpoint-provided tokenizer and chat template, deterministic seeded sampling.
- Single active generation, bounded request queue, streaming HTTP responses,
  cancellation, and exact-prefix reuse for the most recent conversation.

Current limits:

- Text input only; no images, audio, tools, MoE, or non-default RoPE scaling.
- Prefill is sequential. Intermediate prompt tokens skip the final vocabulary
  projection, but there is no batched matrix-matrix prefill yet.
- No MTP/speculative decoding, Flash Attention, GPU sampling, or M5-specific
  tensor acceleration yet. These are potential improvements, not hidden features.
- Activations, KV cache, and recurrent state use FP32. This is a numerical
  baseline and uses more cache memory than a mixed-precision implementation.
- GGUF, AWQ, GPTQ, NVFP4, MXFP4, and arbitrary quantization schemes are unsupported.
- HTTP implements a documented subset of the Chat Completions API. Unsupported
  request fields are rejected rather than silently ignored.

## Build

Use an Apple Silicon Mac, a current macOS installation, Rust 1.88 or newer
(edition 2024 and let chains), and Xcode Command Line Tools. The checked build
used Rust 1.96 and macOS 26.6.2. Metal shaders are embedded and compiled at
runtime; the separately downloaded offline Metal compiler is not needed.

```sh
cargo build --release --locked
```

Build on the target Mac. If a Homebrew Rust build reports a macOS deployment
target mismatch, use a Rust toolchain supporting your OS or set
`MACOSX_DEPLOYMENT_TARGET` to your installed macOS version. Do not select a newer
deployment target than your actual OS.

## Checkpoints

Supply an existing local Hugging Face checkpoint directory containing:

```text
config.json
tokenizer.json
chat_template.jinja                 # or a string chat_template in tokenizer_config.json
generation_config.json             # recommended for the model's EOS IDs
model.safetensors                  # or shards plus model.safetensors.index.json
```

The intended initial full-size checkpoint is
[`mlx-community/Qwen3.8-27B-4bit`](https://huggingface.co/mlx-community/Qwen3.8-27B-4bit).
Only its text weights are used. Download all checkpoint files into a directory,
for example `models/Qwen3.8-27B-4bit`. Loading that weight format does not install
or execute MLX. Model weights are not included in this repository.

Keep checkpoint files unchanged while the process is running: the loader maps
them read-only. GPU allocations are checked against Metal's recommended working
set; the engine does not modify system GPU memory limits.

```sh
./target/release/qwen-metal inspect --model models/Qwen3.8-27B-4bit
```

`inspect` checks configuration and file structure. It does not establish model
output correctness.

## Local chat server

```sh
./target/release/qwen-metal serve \
  --model models/Qwen3.8-27B-4bit \
  --context 8192 \
  --listen 127.0.0.1:8080
```

The served model ID is the final component of the model directory path. Confirm
it with `GET /v1/models`.

```sh
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen3.8-27B-4bit",
    "messages": [{"role": "user", "content": "Wyjaśnij krótko, czym jest Rust."}],
    "max_tokens": 256,
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "seed": 42,
    "enable_thinking": false,
    "stream": true
  }'
```

Routes: `GET /health`, `GET /v1/models`, `POST /v1/chat/completions`.
The server accepts text messages with `system`, `user`, and `assistant` roles.
Supported request fields are `model`, `messages`, `max_tokens` (or
`max_completion_tokens`), `temperature`, `top_p`, `top_k`, `seed`, `stream`, and
`enable_thinking`. The output cap is 4096 tokens per request. Prompt plus output
budget must fit `--context`. It binds only to loopback.

Thinking is off by default. When requested, reasoning is returned as visible
`<think>...</think>` text rather than a separate `reasoning_content` field.
Historical reasoning is not preserved as a separate chat-template field.

For a single terminal request:

```sh
./target/release/qwen-metal generate \
  --model models/Qwen3.8-27B-4bit \
  --prompt 'Napisz funkcję Rust obliczającą NWD.' \
  --max-tokens 256
```

## Reproducible measurements

```sh
./target/release/qwen-metal bench \
  --model models/Qwen3.8-27B-4bit \
  --context 8192 --prompt-tokens 512 --generate-tokens 128 --runs 3
```

This runs one warmup and three measured iterations, resetting state each time.
It uses a synthetic fixed input-token sequence, greedy autoregressive steps,
and deliberately ignores EOS. JSON reports prefill and decode separately,
device, memory allocation and median decode tokens/second. It is not a chat
quality benchmark. Record the exact model revision and source commit alongside
results, and keep context, quantization, power mode and workload unchanged when
comparing builds.

```sh
./target/release/qwen-metal kernel-bench --rows 17408 --cols 5120 --iterations 50
```

`kernel-bench` measures packed Q4/Q8 matrix-vector multiplication (`--bits 4|8`,
`--group 32|64|128`). `--reference` selects the original kernel and dispatch
path. Its effective
weight bandwidth can benefit from repeated cached reads; it is **not LLM
tokens/second**.

The default `aligned` mode uses a vectorized Q4 kernel for compatible shapes
and falls back to a tail-safe packed kernel. `QWEN_METAL_GEMV=packed4` selects
the alternative packed implementation; `QWEN_METAL_GEMV=stream` selects an
experimental lane-per-quantization-group Q4 implementation. Stream is opt-in:
it has no established target-machine speed advantage.
`QWEN_METAL_REFERENCE=1` restores the
original matrix and barrier-list path for `bench`, `generate`, `serve`, and
`synthetic-bench`. These switches leave model weights and context unchanged.
Final model benchmark JSON reports `kernel_mode`.

`QWEN_METAL_METADATA=bf16` keeps affine scales/biases in 16-bit BF16 storage
only when every value reconstructs the original FP32 bits exactly. Matrices
with inexact metadata or unsupported groups retain FP32 storage. This is not
weight requantization; arithmetic and Q4/Q8 values are unchanged. The default
is `f32`, and reference mode always uses FP32. Benchmark JSON reports the
actual `compacted_matrices` and `metadata_saved_bytes`. The `stream` mode uses
the packed BF16 kernel when metadata is compacted.

RMS normalization retains the original single-SIMD path by default.
`QWEN_METAL_NORM=parallel` enables the new 256-thread reduction for large
disjoint buffers. It remains opt-in until full target-model improvement is
measured. Final JSON also reports `norm_mode`.

`QWEN_METAL_ATTN_VALUES=parallel` enables an optional attention-value reduction
that splits the history across eight SIMD groups. It retains FP32 cache and
arithmetic and the original GQA mapping. It applies at history length 128 or
greater and head dimension 32 or greater, with a serial fallback for smaller
shapes or aliased output. The default and reference modes remain `serial`.
Benchmark/profile JSON reports the requested `attention_mode`; eligible
dispatches use the parallel kernel. This flag does not change attention-score
computation or softmax. Run the standalone, reused-buffer microbenchmark with:

```sh
cargo run --release --locked --example attention_bench
```

Its GPU and wall intervals overlap, and neither is a model token-rate result.

To locate costs on the actual model, run one diagnostic capture:

```sh
./target/release/qwen-metal profile \
  --model models/Qwen3.8-27B-4bit --context 8192 --history 512 \
  --compare-norm --profile-backend commands > profile.json
```

This prefills the fixed history once, then captures both serial and parallel
RMS. Each capture warms up a complete step, times a normal step, and profiles
the next step. Within each `captures` entry, `normal_timing` separates CPU encoding,
CPU commit, completion wait, and Metal's whole-command GPU interval. GPU time
overlaps the wait and must not be added to it. `commands` uses GPU start/end
timestamps from one synchronous command buffer per operation. This avoids
hardware counters that returned zero samples on the target M5. The default
`auto` backend tries calibrated stage counters and falls back when samples are
unavailable. Both change scheduling; profiled elapsed time is **not normal
inference throughput**. Missing timings produce explicit errors/null data;
normal-command timing is preserved. An actual execution failure stops comparison.
Omitting `--model` profiles the smaller synthetic reused-weight graph instead.

For a same-build comparison, save the final JSON from two otherwise identical
`bench` commands (progress is printed to stderr):

```sh
QWEN_METAL_REFERENCE=1 ./target/release/qwen-metal bench \
  --model models/Qwen3.8-27B-4bit \
  --context 8192 --prompt-tokens 512 --generate-tokens 128 --runs 3 > before.json
QWEN_METAL_REFERENCE=0 ./target/release/qwen-metal bench \
  --model models/Qwen3.8-27B-4bit \
  --context 8192 --prompt-tokens 512 --generate-tokens 128 --runs 3 > after.json
python3 scripts/compare_bench.py before.json after.json --require-speedup 10
```

The comparison rejects different workloads and synthetic/microbenchmark JSON.
It excludes warmup and exits unsuccessfully when the requested speedup is unmet.

## Tests

```sh
cargo test --locked
cargo test --locked -- --ignored
cargo fmt --all -- --check
```

The second command explicitly runs tests requiring a real Metal GPU. They fail
if no device is available; they do not silently succeed. The first command lists
them as ignored so file-format/API tests can also run in GPU-restricted sandboxes.

`tests/fixtures/tiny` and `tiny-q4` are small, untrained synthetic models,
not Qwen weights. Regenerate them and their independent scalar reference logits with:

```sh
python3 scripts/make_fixture.py
```

See [architecture](docs/architecture.md) and [implementation checklist](docs/implementation-plan.md).

## License and references

The original Rust and Metal implementation is MIT licensed. The official Qwen
chat template retained as a test fixture is Apache-2.0 licensed; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Downloaded models retain their
own licenses.

Architecture and format references:

- [Qwen3.8-27B configuration](https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/config.json)
- [Hugging Face Qwen3.5 implementation](https://github.com/huggingface/transformers/tree/main/src/transformers/models/qwen3_5)
- [MLX Qwen3.5 format and architecture reference](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/qwen3_5.py)
