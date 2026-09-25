# Native MTP: candidate toward 32 tokens/s

The user-supplied M5 Pro 0.6.0 benchmark measures **26.77 sustained tokens/s with
MTP**, versus **16.63** for sequential generation on the same target. Complete
greedy traces agree across all five prompts. **The mixed-use 32 tokens/s goal
is not reached yet.** See the [audited result and time budget](m5-mtp-v0.6.md).
The [0.6.1 follow-up](performance-v0.6.1.md) regressed to 23.34 sustained tokens/s
on the same M5 workload despite isolated M1 improvements. The subsequent
[same-model comparison](m5-kernel-comparison.md) isolates the regression to
shared matrix unpacking. Version 0.6.3 defaults to legacy matrices and batched
DeltaNet: the combination measured at 28.53 sustained tokens/s over the two
diagnostic prompts. The full five-prompt result for that combination is pending.
Version 0.6.4 also offers the opt-in `QWEN_METAL_BLOCK_MATMUL=mlp-r2` path,
restricted to the two B3 BF16 MLP shapes supported by the M5 microbenchmark.
Use [`mtp-bench --compare-mlp-r2`](m5-selective-r2.md) to compare it with both
legacy MTP and ordinary target generation on one loaded model.

Every matrix pass in the measured target graph streams approximately 14.41 GB
of Q4 weights and BF16 metadata. The [Apple M5 Pro specification](https://www.apple.com/macbook-pro/specs/)
lists 307 GB/s memory bandwidth. About 21.3 tokens/s is consequently a
bandwidth-only ceiling for that single-token streaming workload. This is an
estimate that ignores other costs and cache effects. Sharing weight reads across
several verified positions offers a path beyond it.

## What runs

The original Rust execution graph and original Metal kernels execute both the
target and its native one-layer MTP adapter. No llama.cpp, MTPLX, MLX, PyTorch or
other inference runtime is installed or executed. `mlx-community` identifies
the checkpoint's storage format and publisher.

The default block has **one already selected target token plus two proposals**.
All target linear projections process the block with shared packed-weight reads.
DeltaNet, convolution and attention retain causal token order. GPU snapshots
allow any accepted prefix to be committed without rerunning target layers.
The adapter shares immutable embedding/head buffers with the target; it owns
only its small set of weights, scratch and one layer's KV cache.

Greedy verification selects target argmax tokens. Stochastic verification uses
the configured temperature/top-k/top-p distributions, acceptance `min(1,p/q)`,
and the positive `p-q` residual on rejection. This preserves the target sampling
law; it does not reproduce the ordinary sampler's random stream for a given
seed. Floating-point block/sequential differences can affect near-tied logits,
so the benchmark records complete greedy token traces for comparison.

## Build and download

Build the candidate checkout on the target Mac and download only the companion
adapter. The existing 27B checkpoint stays in place.

```sh
cargo build --release --locked
python3 scripts/download_mtp.py
```

The standard-library Python downloader pins
[`mlx-community/Qwen3.8-27B-MTP-4bit`](https://huggingface.co/mlx-community/Qwen3.8-27B-MTP-4bit/tree/b643c01b6d3b094e325edb6ebd832e16c486c575)
to revision `b643c01b6d3b094e325edb6ebd832e16c486c575`. It downloads config plus
**238,934,137 bytes** of safetensors and verifies SHA-256 before accepting either
file. Adapter weight SHA-256:
`76663c101e7e8ea9c0ae17bcb95183cd7f733ce424c912b8b264a7b1c48e4cc6`.
There is no second tokenizer download: token IDs, template, embedding and output
head all come from the target.

## Measure representative output

Connect external power, disable Low Power Mode, and keep other GPU generation
stopped. Confirm `low_power_mode: false` in the report: connection to a charger
alone is not a measurement of that setting. This command runs
five mixed Polish/English prompts, honors EOS, compares the ordinary target and
MTP sequentially on the same loaded weights, and excludes warmups:

```sh
QWEN_METAL_REFERENCE=0 QWEN_METAL_GEMV=aligned \
QWEN_METAL_NORM=parallel QWEN_METAL_METADATA=bf16 QWEN_METAL_ATTN_VALUES=serial \
  ./target/release/qwen-metal mtp-bench \
  --model models/Qwen3.8-27B-4bit \
  --mtp models/Qwen3.8-27B-MTP-4bit \
  --context 8192 --max-tokens 128 --runs 3 --block-size 3 \
  --temperature 0 --compare > mtp-mixed-greedy.json
```

For normal stochastic chat, repeat with `--temperature 0.7 --top-p 0.8 --top-k 20`
and a different output filename. Equal greedy token traces are a correctness
check; equal stochastic traces are not expected. Use `--prompts` with a JSON
array of `{ "name": "...", "prompt": "..." }` objects to supply your actual
workloads. The default suite is also available in `docs/mixed-prompts.json`.

The report measures actual verified completion IDs and wall time, not proposed
tokens, synthetic fixed steps, or aggregate multi-user throughput. It reports
both completion tokens/time and `(completion_tokens - 1)/decode_time`, since
the first token is obtained from prefill logits. The latter is the conservative
sustained decode metric to compare with the 32 tokens/s target. Prompt
processing and time to first token are reported separately. Acceptance rate,
target/draft/rollback times, output text, token IDs and per-prompt results make
regressions visible. Small early-EOS outputs are not sustained speed evidence.

`goal_32_tps_confirmed` is true only for the measured workload when every prompt
has a median of at least 32 sustained tokens/s over at least three measured runs,
the greedy token IDs/text/finish reason match the sequential target, and **every
measured MTP run contains at least 64 sustained decoded tokens**. This flag is not
a guarantee for other prompts. Stochastic reports expose throughput and
acceptance without using greedy trace equality as a goal gate.

## Start the headless server

```sh
QWEN_METAL_REFERENCE=0 QWEN_METAL_GEMV=aligned \
QWEN_METAL_NORM=parallel QWEN_METAL_METADATA=bf16 QWEN_METAL_ATTN_VALUES=serial \
  ./target/release/qwen-metal serve \
  --model models/Qwen3.8-27B-4bit \
  --mtp models/Qwen3.8-27B-MTP-4bit --mtp-block-size 3 \
  --context 8192 --listen 127.0.0.1:8080
```

The existing HTTP chat API and sampling fields work unchanged. `generate` also
accepts `--mtp` and `--mtp-block-size`, printing acceptance/timing statistics to
stderr. Omitting `--mtp` selects the ordinary engine.

## Limits and verification

- MTP acceptance depends on the prompt and sampling settings. Low acceptance
  can make this path slower. Compare actual outputs from your use cases.
- MTP currently resets both caches per request, including cancellation/error
  exits. Recent-prefix reuse remains available on the ordinary path.
- Block width is limited to 1–4; the native default is 3. Block dispatch requires
  the optimized kernel mode and supported affine group sizes 32/64/128 or dense
  F16. Unsupported combinations fail before changing target history.
- Recurrent prefix snapshots require additional GPU memory. All caches and
  snapshots are included in Metal's working-set checks; no OS limits are raised.
- Native MTP supports the separate sanitized one-layer adapter with shared
  embeddings. It does not reinterpret arbitrary model tensors as an adapter.
- The trained 27B checkpoint cannot fit the local M1 test machine's safe budget.
  Local evidence comprises independent FP64 fixtures, real Metal tests,
  synthetic hybrid models and matrix microbenchmarks. Target M5 throughput and
  trained-model quality require the explicit full-model benchmark above.

The mathematical and file-format references are recorded in
[the implementation plan](mtp-32-plan.md). They are references only, not runtime
dependencies or sources of performance claims.

The [recorded local matrix microbenchmarks](benchmarks/m1-block-matrix-b3.json)
compare B=3 against three ordinary GEMVs on identical inputs and weights:

| Matrix | Three GEMVs, GPU ms | Shared-weight block, GPU ms | Ratio |
| --- | ---: | ---: | ---: |
| MLP up, 17408 × 5120 | 4.591 | 2.270 | 2.02× |
| MLP down, 5120 × 17408 | 3.191 | 1.807 | 1.77× |
| Vocabulary, 248320 × 5120 | 75.805 | 31.974 | 2.37× |

The device was **M1**, weights were reused between iterations, and the three
outputs were numerically identical to the existing aligned kernels for these
inputs. These are GPU kernel intervals; wall time and host overhead are also
in the raw report. They establish actual weight-sharing speedups for these
primitives, not full-model or M5 chat speed.

## Release-candidate verification (local M1)

Version 0.6.0 was built with `cargo build --release --offline --locked`.
The complete all-target test run passed 108 top-level tests with real GPU tests
enabled; a subsequently added end-to-end `mtp-bench` CLI regression also passed
(109 distinct tests total). The changed sampling and chat-manager tests were
rerun after final iterator/validation cleanup. The chat-manager GPU suite also
passed with `QWEN_METAL_METADATA=bf16`.

```sh
cargo test --offline --locked --all-targets -- --include-ignored --test-threads=1
QWEN_METAL_METADATA=bf16 cargo test --offline --locked --lib mtp_chat \
  -- --include-ignored --test-threads=1
```

The compiled server was exercised over HTTP with the tiny target/adapter:
health, non-stream response, SSE response equality, `[DONE]`, and a fresh request
after disconnecting an SSE consumer all passed. The real 239 MB adapter was
also downloaded, checksum-verified, loaded and executed for three steps with
synthetic target embedding/head buffers; every 5120-element hidden vector and
248320-element logit vector was finite. That check confirms real adapter format
compatibility, not trained-target quality or acceptance.

Formatting, Python syntax and `git diff --check` passed. Clippy completed with
style warnings (including existing modulo/range conventions and new wide GPU
dispatch signatures); strict `-D warnings` is not a passing gate in this release.
The build also reports a future-compatibility warning in the existing `block`
0.1.6 dependency used by the Metal bindings.
