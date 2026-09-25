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

The user's full-model 0.6.4 M5 Pro comparison measures **28.69 sustained tokens/s
with selective MLP R2, versus 27.81 with legacy MTP (+3.16%)** on five mixed
prompts. Each prompt has three measured runs; R2 improves every paired run and
matches ordinary target generation's token IDs, text and finish reason,
including warmups. **The mixed-use 32 tokens/s goal is not reached:** R2 prompt
medians are 26.18 (explanation), 33.95 (code), 29.29 (analysis), 29.36 (rewrite)
and 26.03 (planning) tokens/s; only code exceeds 32.
See the [audited full-model result and remaining budget](docs/m5-selective-r2.md).

Native MTP verifies one target token plus two proposals from the separate
239 MB adapter. The measured configuration uses aligned GEMV, parallel RMS,
exact BF16 metadata and serial attention values. The **0.6.5 default policy
selects selective R2 only for the exact Metal device name `Apple M5 Pro`**;
other devices retain legacy matrices, and batched DeltaNet remains the default.
R2 applies only to the two eligible B3 BF16 Q4/group64 MLP shapes; the vocabulary
head and other shapes/formats/block widths retain their existing dispatch.
Set `QWEN_METAL_BLOCK_MATMUL=legacy` for an explicit fallback.

Version 0.6.5 also includes a standalone [FP32 TensorOps experiment](docs/tensorops-probe.md).
The user's M5 Pro sweep rejects this implementation for B3 decode: all 288
paired samples are slower, with candidate medians 4.21–8.10 times the scalar
controls. Numerical checks pass, but the experiment stays outside inference.
The measured R2 configuration remains selected on M5 Pro.

Earlier evidence includes the [16.13 tokens/s ordinary-target measurement](docs/m5-pro.md),
the [26.77 tokens/s first MTP result](docs/m5-mtp-v0.6.md), and the
[controlled recovery from the 0.6.1 shared-matrix regression](docs/m5-kernel-comparison.md).
The latest same-session R2/legacy comparison isolates the R2 gain; differences
from historical runs include other changes and measurement conditions. See
[MTP setup and limitations](docs/native-mtp.md) before comparing chat workloads.

Earlier results and implementation evidence remain in the
[first iteration](docs/performance-2026-09-25.md),
[v0.2 profiling](docs/performance-v0.2.md),
[v0.3 BF16 metadata](docs/performance-v0.3.md),
[v0.4 attention follow-up](docs/performance-v0.4.md), and
[v0.5 sustained diagnostics](docs/performance-v0.5.md).

Implemented:

- Safetensors loading, including indexed shards and bounds/shape validation.
- Affine packed 4-bit and 8-bit matrices in MLX format; F16/BF16/F32 source
  matrices are converted to F16 on load.
- Original fused dequantization/matrix-vector Metal kernels: weights remain
  packed during inference.
- Hybrid Gated DeltaNet recurrence, causal depthwise convolution, grouped-query
  attention, partial RoPE, gated norms, SwiGLU, and residual connections.
- GPU-resident weights, preallocated scratch space and explicit context capacity.
- One GPU command buffer per target token or verification block; no CPU/GPU
  synchronization between target layers.
- Checkpoint-provided tokenizer and chat template, deterministic seeded sampling.
- Single active generation, bounded request queue, streaming HTTP responses,
  cancellation, and exact-prefix reuse for the most recent conversation.
- Optional native MTP generation with causal block verification, GPU state
  rollback, and greedy or corrected stochastic sampling. No external inference
  runtime is required; the adapter shares the target embedding and output head.
- OpenCode-compatible Chat Completions with function tools, tool-result history,
  streaming usage, stop sequences, reasoning fields, and sampling penalties.

Current limits:

- Text input and function tools only; no images, audio, MoE, or non-default RoPE scaling.
- The ordinary path uses sequential prefill and supports recent-prefix reuse.
  The experimental MTP path resets caches per request; see its separate timing
  and prefill results before choosing it for repeated long conversations.
- No Flash Attention, GPU sampling, or M5-specific tensor acceleration yet.
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

For the measured M5 Pro MTP configuration, download the companion adapter as
described in [MTP setup](docs/native-mtp.md), connect external power, disable
Low Power Mode, and use:

```sh
QWEN_METAL_REFERENCE=0 QWEN_METAL_GEMV=aligned \
QWEN_METAL_NORM=parallel QWEN_METAL_METADATA=bf16 QWEN_METAL_ATTN_VALUES=serial \
  ./target/release/qwen-metal serve \
  --model models/Qwen3.8-27B-4bit \
  --mtp models/Qwen3.8-27B-MTP-4bit --mtp-block-size 3 \
  --context 8192 \
  --listen 127.0.0.1:8080
```

The 28.69 tokens/s result measures greedy decode over five prompts, not HTTP
end-to-end latency. Chat speed also depends on prompt length and sampling.
MTP resets caches per request; omit `--mtp` and `--mtp-block-size` to use ordinary
generation with recent-prefix reuse. Version 0.6.5 chooses selective R2 on
`Apple M5 Pro`; for the same selection on 0.6.4, add
`QWEN_METAL_BLOCK_MATMUL=mlp-r2`. Model quality remains subject to the validation
limit above.

The served model ID is the final component of the model directory path. Confirm
it with `GET /v1/models`.

```sh
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen3.8-27B-4bit",
    "messages": [{"role": "user", "content": "Wyjaśnij krótko, czym jest Rust."}],
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "seed": 42,
    "enable_thinking": false,
    "stream": true
  }'
```

Routes: `GET /health`, `GET /v1/models`, `POST /v1/chat/completions`.
The server accepts text messages with `system`, `developer`, `user`, `assistant`,
and `tool` roles, including multipart text and assistant tool-call history.
Supported controls include `tools`, `tool_choice`, `parallel_tool_calls`,
`stream_options`, `stop`, `max_tokens` / `max_completion_tokens`, `temperature`,
`top_p`, `top_k`, `seed`, `frequency_penalty`, `presence_penalty`, `logit_bias`,
`reasoning_effort`, and `enable_thinking`. From version 0.7.1, the HTTP API has no
fixed output-token cap or 256-token default. Omit the output limit or set it to
null to use the context remaining after the complete prompt is tokenized.
An explicit positive limit is an upper bound: the effective budget is the
smaller of that limit and the remaining context. A prompt that leaves no room
for an output token returns HTTP 400 with `context_length_exceeded`, including
before streaming starts. EOS and stop sequences can end generation earlier.
The server binds only to loopback.

Thinking is off by default. HTTP responses separate reasoning into
`reasoning_content`, and the same field is supported in assistant history.
The CLI retains its existing text presentation. Function calls are returned to
the client for execution; the server does not execute tools. Tool constraints
use checkpoint prompting and output validation, not grammar-constrained decoding.
See the [complete compatibility contract](docs/openai-compatibility.md) for
accepted defaults, JSON object mode, streaming semantics and explicit limits.

### OpenCode

Start the server above and copy [examples/opencode.json](examples/opencode.json)
to `opencode.json` in the project you want OpenCode to work on, or merge its
provider/model settings into your existing config. Then run `opencode` in that
project. The config uses `@ai-sdk/openai-compatible`, the local `/v1` endpoint,
and this model for both main and small-model tasks. No API key is needed.

From 0.7.2 the example enables reasoning at `medium` effort and preserves it
through tool-call history with `interleaved.field: "reasoning_content"`.
Explicit variants `none`, `low`, `medium`, and `xhigh` let you change the effort.
In the OpenCode TUI, `/thinking` toggles the reasoning display; in the CLI use
`opencode run --thinking 'your prompt'`. This display flag does not itself enable
model reasoning. Copy or merge the updated example if you already have a config.

The example sets `limit.context`, `limit.input` and `limit.output` to 8192;
keep all three equal to the server's `--context` when changing it. The server
reduces each requested output budget to the space the actual prompt leaves.
`compaction.reserved:1024` gives OpenCode headroom for summarizing a growing
conversation; it does not cap an answer at 1024 tokens. The explicit input limit
lets OpenCode use that reserve instead of subtracting the entire output limit
from the context and immediately requesting compaction. See the
[verified compaction calculation](docs/openai-compatibility.md#configure-opencode).
Large repositories or long tool histories can still require compaction.
Integration tests exercise the actual adapter
version used by the inspected OpenCode release; trained-model tool selection
and argument quality still require evaluation with your checkpoint.

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

Add `--timing` to `bench` to record ordinary per-token CPU/GPU intervals during
the sustained workload. It uses the same one-encoder, one-command-buffer path,
without hardware counters or per-operation waits. It adds host clocks and
preallocated scalar samples; durations include this small bookkeeping overhead.
Each run's `timing` contains prefill/decode distributions and blocks of 32 tokens,
separating sampling, forward, CPU encoding, commit, completion wait and GPU time.
GPU and wait overlap. `unclassified_host` is the remainder of forward wall time,
including readback, autorelease and other host work; it is not isolated readback.
Missing command timestamps produce null frame-derived summaries with coverage
counts. A separate `warmup_run` is retained but excluded from the median.

Power/thermal conditions are read-only macOS snapshots outside the timed phases.
They do not report GPU frequency or temperature and do not change power settings.
The final prefill token is marked as computing logits, unlike preceding tokens.
Example diagnostic run (one warmup and one measured run):

```sh
QWEN_METAL_NORM=parallel QWEN_METAL_METADATA=bf16 \
  ./target/release/qwen-metal bench --timing \
  --model models/Qwen3.8-27B-4bit --context 8192 \
  --prompt-tokens 512 --generate-tokens 128 --runs 1 > timing.json
```

## Tests

```sh
cargo test --locked
cargo test --locked -- --ignored --test-threads=1 --skip opencode_sdk_tool_round_trip --skip opencode_cli_reasoning_and_tool_history
cargo fmt --all -- --check
```

The second command explicitly runs tests requiring a real Metal GPU. They fail
if no device is available; they do not silently succeed. The first command lists
them as ignored so file-format/API tests can also run in GPU-restricted sandboxes.
The separate ignored OpenCode tests require Node.js and either an isolated install
of the exact SDK version or an installed OpenCode CLI; see
[the test commands](docs/openai-compatibility.md).

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
