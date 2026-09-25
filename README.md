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

This is an initial, numerically tested implementation, **not a demonstrated
performance winner**. The six GPU kernel tests and a complete four-layer
synthetic hybrid model have been tested on a real Apple M1 GPU. The complete
model test compares every output logit over five tokens against an independent
scalar Python oracle and checks reset and context bounds.

**Real Qwen3.8-27B output and M5 Pro throughput have not yet been validated.**
Do not interpret a successful synthetic test, a matrix microbenchmark, or the
availability of the server as that evidence.

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

`kernel-bench` measures only packed Q4 matrix-vector multiplication. Its effective
weight bandwidth can benefit from repeated cached reads; it is **not LLM
tokens/second**.

## Tests

```sh
cargo test --locked
cargo test --locked -- --ignored
cargo fmt --all -- --check
```

The second command explicitly runs tests requiring a real Metal GPU. They fail
if no device is available; they do not silently succeed. The first command lists
them as ignored so file-format/API tests can also run in GPU-restricted sandboxes.

`tests/fixtures/tiny` is a small, untrained synthetic model, not Qwen weights.
Regenerate it and its independent scalar reference logits with:

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
