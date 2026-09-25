# v0.3: counter fallback and lossless metadata storage

The latest full-model target result remains the user-reported **8.16096 decode
tokens/s**. Neither this release nor local tests establish 15 tokens/s on M5.

## M5 profiler failure

The target returned `invalid metal timestamp pair for matvec_affine: start=0 end=0`
with v0.2. Its normal forward had completed, but invalid operation samples caused
the report to be discarded. Zero counter samples cannot be treated as measured
zero-duration operations.

`profile --profile-backend commands` bypasses counters. Each profiled operation
executes in its own command buffer and completes before the next operation;
GPUStartTime/GPUEndTime supply GPU elapsed seconds. The normal step still uses
one encoder and one command buffer. The default `auto` profiler attempts counters
and retries the next token with command timing if counter sampling/reporting is
unavailable. The JSON reports the backend, actual history, and any counter error.

If completed operations have unavailable timestamps, the graph continues and
the report contains `profile_error`, `operations: null`, and no fabricated total.
The previously measured normal timing remains available. Actual GPU execution
failure marks the capture and stops comparison to avoid reusing partial state.
For the command backend, `profiled_timing` is null: its outer command is empty
and must not be mistaken for the model's GPU execution interval.

```sh
./target/release/qwen-metal profile \
  --model models/Qwen3.8-27B-4bit --context 8192 --history 512 \
  --compare-norm --profile-backend commands > profile.json
```

This loads and prefills once. Both profiling backends change scheduling;
per-command timings include per-command GPU overhead and are diagnostic, not
normal token throughput. Do not subtract their sum from the normal step and
label the remainder dispatch overhead.

## Exact BF16 metadata

A ranged read of the first public `mlx-community/Qwen3.8-27B-4bit` safetensors
header found 276 scale/bias tensors with dtype BF16. No model weights were
downloaded. This observation describes that public shard, not a verified hash
of the user's local checkpoint.

The original loader expands these values to FP32. With
`QWEN_METAL_METADATA=bf16`, both metadata arrays of an eligible matrix are
compacted only if every value is finite and its low 16 FP32 bits are zero.
The shaders reconstruct exactly those FP32 bits with an integer shift, without
converting through F16. Signed zero and BF16 subnormals retain their storage
bit patterns. Subsequent arithmetic remains FP32 with the existing fast-math
policy, so numerical tests use tolerances rather than claim bitwise logits.

Compaction supports Q4/Q8 groups 32/64/128. Inexact arrays and unsupported groups
retain FP32. Reference mode forces FP32. Only compatible Q4/group64 matrices
use the aligned BF16 kernel; other supported cases use its packed counterpart.
`stream` also uses packed BF16 because no stream BF16 specialization is present.

For Q4/group64, payload falls from 0.625 to 0.5625 bytes per weight when eligible:
a **10% byte reduction**, with an ideal bandwidth-only ceiling of **1.111×**.
That is not a measured model speedup. Default storage remains FP32; opt in with:

```sh
QWEN_METAL_METADATA=bf16 ./target/release/qwen-metal bench \
  --model models/Qwen3.8-27B-4bit \
  --context 8192 --prompt-tokens 512 --generate-tokens 128 --runs 3
```

Inspect `metadata_mode`, `compacted_matrices`, `metadata_saved_bytes` and actual
allocation in the final JSON. A model with ineligible metadata can save nothing.
`kernel-bench --bf16-metadata` compares compact storage on synthetic matrices;
v0.3 uses BF16-exact scale 0.015625 and bias -0.0625 in both storage modes.
These differ from older microbenchmark constants, and microbenchmarks remain
separate from model throughput measurements.

A single exploratory local M1 pass gave mixed results: BF16 was slower for
17,408 × 5,120 and faster for 5,120 × 17,408 and 248,320 × 5,120. These are
repeated synthetic matrices, with one measurement per configuration, not a
robust speedup estimate. The raw records and release-binary hash are in
[`benchmarks/m1-v0.3-metadata-kernels.json`](benchmarks/m1-v0.3-metadata-kernels.json).
This evidence does not justify changing the default or predicting an M5 gain.

## Numerical verification

The BF16 matrix/embedding oracle covers Q4/Q8, all three group sizes, short/odd
row groups and aligned dimensions, across aligned/packed4/stream choices.
An independent four-layer BF16-metadata fixture contains 33 quantized matrices,
66 metadata tensors and 3,288 metadata values; compaction saves exactly 6,576
bytes. Its complete logits over five tokens and a reset match the scalar oracle.
The previous tiny fixtures remain unchanged. The command-timing profiler also
preserves complete-model fixture logits and rejects missing or partial timings.
