# Validation of the initial implementation

Date: 2026-09-25. Host: Apple M1, 16 GB unified memory, macOS 26.6.2;
Rust 1.96.0. This host is not the intended M5 Pro / 48 GB target.

Commands run on the final code before the initial push:

```text
cargo test --offline --locked -- --include-ignored
  library: 35 passed, 0 failed, 0 ignored
  integration: 2 passed, 0 failed, 0 ignored
cargo fmt --all -- --check
  passed
./target/debug/qwen-metal inspect --model tests/fixtures/tiny
  parsed 4-layer synthetic hybrid model (3 DeltaNet, 1 full attention)
```

The test run was given actual Metal device access. It includes six numerical
GPU fixtures and two complete-engine tests:

- Packed Q4/Q8 nibble order, affine scale/bias, F16 matrix-vector products,
  embeddings, odd dimensions and tail handling.
- RMS and gated norms, causal convolution, recurrent DeltaNet updates,
  partial rotate-half RoPE, GQA grouping, softmax, KV append and output gating.
- Every logit over five tokens from a four-layer synthetic model, compared with
  an independent scalar Python implementation. Tolerance: absolute error below
  0.0003. The sequence is repeated after reset; token and context bounds are tested.
- Multi-turn chat prefix reuse and recovery after cancelled generation compared
  with fresh generation using the same synthetic model.
- Checkpoint format/norm layout errors, official template rendering, sampling,
  HTTP response structure, streaming termination, validation and cancellation.

Independent static review found a checkpoint norm-layout bug and duplicate HTTP
completion IDs. Both were corrected before the final run.

Cargo reported a future-compatibility warning in the transitive dependency
`block` 0.1.6, brought in by Metal bindings. It did not prevent compilation or
the tests. The offline `metal` compiler component was not installed; Metal's
runtime source compiler successfully compiled and executed all kernels.

## What this evidence does not establish

- Real Qwen3.8-27B checkpoint accuracy or chat quality.
- M5 Pro throughput, sustained thermals, or full-size memory use.
- A speed advantage over any existing engine.
- MTP/speculative decoding, batched prefill, or M5-specific acceleration.

The synthetic fixture is deliberately small and untrained. No model
tokens/second result is claimed from it.

## Performance iteration validation (2026-09-25)

On the same real M1 GPU, after selecting the packed and aligned kernels:

```text
cargo build --release --offline --locked -j 2
  passed
cargo test --offline --locked -- --include-ignored
  library: 36 passed, 0 failed, 0 ignored
  inference integration: 3 passed, 0 failed, 0 ignored
  quantized matrix integration: 1 passed, 0 failed, 0 ignored
cargo fmt --all -- --check
  passed
```

The new matrix test checks 270 GPU dispatches against FP64 dot products; its
cases include Q4/Q8, groups 32/64/128, short and non-SIMD-aligned columns,
partial output row groups, and both optimized paths plus reference. A new
packed Q4 hybrid fixture compares five autoregressive steps and a state reset
against independent scalar Python logits. A host-only test covers unsafe
signed index products and dispatch fallback.

Independent read-only review found one benchmark-comparator integration bug
(the accepted mode names differed from the executable); it was fixed and
rechecked. No remaining actionable correctness findings were reported.
The reviewer also confirmed that the explicit barriers are ignored on this
engine's serial encoder, so the resource-list change must not be described as
a demonstrated GPU fencing speedup.

[Performance evidence](performance-2026-09-25.md) records the final same-binary
release timings and their limits. Full trained-checkpoint accuracy and the
requested M5 Pro 10× speedup still need target-machine validation.

## v0.2 profiling and normalization validation

The target user subsequently supplied a three-run M5 Pro median of 8.160957654
decode tokens/s; the complete reported record is retained in
`docs/benchmarks/m5-pro-aligned-user.json`. It is user-provided evidence, not
a local M5 execution. The absolute 15 tokens/s target is still unmet in the
available full-model measurements.

The final full test command for v0.2 was
`cargo test --offline --locked -- --include-ignored` with actual Metal access:
41 library tests, four inference tests, one normalization test and one
quantized-matrix test passed; zero failed or ignored (47 total). Matrix coverage
now includes 360 dispatch/oracle cases across aligned, packed4, stream, and
reference modes. New RMS coverage checks zero/nonzero inputs, large/odd lengths
and alias fallback. Profiling compares every output logit with independent
goldens over five steps and checks sample reset between tokens.

Independent review found a clock-unit error in the initial profiler. Raw GPU
counter ticks are now calibrated using paired CPU/GPU timestamps; tests cover
non-unit ratios, invalid samples/spans, large timestamps and overflow. Review
of the correction found no remaining actionable issue. Whole-command Metal
timestamps were already in seconds and did not need that conversion.

The optional stream path also passed all four inference tests in a separate
`QWEN_METAL_GEMV=stream` run. The comparator was checked for below/above absolute
token-rate targets, combined constraints, and invalid rates. These checks do
not establish trained-model quality or a v0.2 M5 throughput improvement.

## v0.3 counter fallback and exact metadata validation

The user reported zero stage-counter timestamps on the M5 Pro. The release
adds a counter-free command-buffer backend and keeps the normal-step timing
when diagnostic samples are unavailable. A failed GPU execution is handled
separately from unavailable timestamps and stops the comparison.

Final local validation used the same M1 host with actual Metal access:

```text
cargo test --offline --locked -- --include-ignored
  48 library tests and 8 integration tests passed (56 total)
  0 failed, 0 ignored
cargo build --release --offline --locked -j 2
  passed
cargo fmt --all -- --check
  passed
git diff --check
  passed
```

The integration tests include the independent five-token BF16 checkpoint
oracle, BF16 matrix/embedding cases, complete-engine command-profiling logits,
and the previous inference, normalization and quantized-matrix coverage.
A separate `QWEN_METAL_METADATA=bf16` run of `bf16_inference` and `inference`
passed all five tests, including exact storage-savings accounting.

The final release CLI was exercised with `--compare-norm --profile-backend
commands` on both the small BF16 checkpoint and the full 64-layer synthetic
graph. Both normalization captures completed without profile errors, with
75 and 1,155 timed dispatches respectively. The JSON contained positive normal
GPU times and operation durations, null counter ticks, and null outer-command
`profiled_timing`. The BF16 checkpoint reported 33 compacted matrices and
exactly 6,576 bytes saved. Synthetic weights are reused; these runs establish
profiling execution and output structure, not trained-model throughput.

Independent read-only review found no remaining actionable correctness issue
after the missing-timestamp and partial-state error paths were corrected.
The command backend still needs confirmation on the user's M5 driver. BF16
metadata remains opt-in, and no v0.3 full-model M5 speedup is claimed.
