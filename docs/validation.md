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

## v0.4 attention follow-up validation

The user subsequently provided a successful v0.3 M5 profile: both captures
completed all 1,155 operations with valid normal GPU times and no profile
errors. Its single normal steps support testing parallel RMS in the sustained
benchmark; they do not establish a new sustained rate. Raw target evidence and
analysis are in `docs/performance-v0.4.md`.

The optional parallel attention-value kernel was checked on the local M1:

```text
cargo test --offline --locked -- --include-ignored
  48 library tests and 10 integration tests passed (58 total)
  0 failed, 0 ignored
cargo build --release --offline --locked --bin qwen-metal --example attention_bench -j 2
  passed
```

The independent FP64 attention test runs 51 cases across serial, parallel and
reference modes (153 GPU dispatches). It covers GQA head mapping, short and
odd dimensions/history lengths, threshold boundaries, 8,192-token histories,
mixed signs, extreme gates, zero values, output sentinels and safe gate/output
aliasing. The whole-engine test compares all 64 logits over 130 fixed tokens,
crossing the parallel attention history threshold, then verifies reset/replay.
It uses an isolated temporary copy of the synthetic Q4 fixture, changing only
the allowed position count; committed fixture bytes remain unchanged.

The release microbenchmark compares the same production logical dispatch and
input buffers in both modes, with five unreported warmups per case. It measures
actual GPU and wall intervals separately and compares final outputs. Its local
results and limits are retained in `docs/benchmarks/m1-v0.4-attention.json`.
The release CLI also completed a synthetic 64-layer profile with both parallel
RMS and parallel attention selected: all 1,155 operations returned valid timings,
including 16 attention-value dispatches at history 515, with no profile errors.
The recorded mode fields matched the selected options. Final formatting and
`git diff --check` passed.
Read-only review found no actionable indexing, barrier, aliasing or mode-routing
issue. The default remains serial until full target-machine evidence supports
promotion. No v0.4 M5 speedup is claimed.

## v0.5 sustained timing validation

The next user-provided sustained M5 results were 7.44877 tokens/s for parallel
RMS/FP32 metadata and 8.03204 for parallel RMS/BF16 metadata. They did not establish
the intended 15 tokens/s. The user confirmed sequential execution without another
GPU generation workload. Both records and their provenance are retained in
`docs/performance-v0.5.md` and its linked JSON files.

The new `bench --timing` leaves the normal GPU execution graph unchanged. Local
validation with actual Metal access passed:

```text
cargo test --offline --locked -- --include-ignored
  55 library tests and 11 integration tests passed (66 total)
  0 failed, 0 ignored
cargo test --offline --locked --test bench_timing_cli -- --include-ignored
  passed after adding phase-boundary and untimed-output checks
cargo build --release --offline --locked -j 2
  passed
cargo fmt --all -- --check
  passed
git diff --check
  passed
```

Seven new CPU tests check known aggregate values and nearest-rank percentiles,
empty/partial phases, 1/31/32/33-token block boundaries, missing timestamps,
invalid/overflowing durations, relative floating-point rounding, and GPU/wait
overlap. A real CLI test checks the warmup and measured 33-token phases, exact
history ranges, sample coverage, phase wall-time containment, and exclusion of
warmup from the median. A separate untimed invocation retains the old output
shape without timing or warmup additions.

Independent review found no actionable issue in timestamp aggregation, execution
boundaries, warmup handling or the availability-guarded Objective-C system-state
reads. Those snapshots follow the BOOL and NSInteger ABIs verified in the local
SDK; unavailable properties yield null. They neither change power settings nor
measure GPU frequency or temperature.

The actual release greedy sampler was also measured independently on the M1
using 248,320 finite logits: 100 warmups and 500 measured calls gave a median of
1.059 ms. This bounds a local CPU cost; it is not a target M5 result. No v0.5
inference speedup is claimed: its purpose is to resolve the unexplained sustained
latency using the existing inference path.

The final v0.5 release CLI was also exercised with parallel RMS and BF16 metadata
on the independent tiny BF16 fixture. Warmup and measured phases each contained
33 valid GPU samples for prefill and decode, with correct block/history ranges
and phase containment. It reported 33 compacted matrices and 6,576 bytes saved.
The local system snapshots returned Low Power Mode false and thermal state
`fair`; these describe the M1 test host only, not the user's M5.

## Returned M5 v0.5 report: observed target threshold

The user supplied `docs/benchmarks/m5-pro-v0.5-timing-user.json` and clarified
that external power had been connected before this run. It reports one measured
128-token decode at 16.1308010524 tokens/s (7.935129792 s), after a separately
excluded warmup at 16.1329225936 tokens/s. This exceeds 15 tokens/s in the measured
run, and is 10.7539x the rounded original 1.5 tokens/s baseline, or 11.1062x the
original measured 1.4524158079 tokens/s observation.

Direct parsing and independent review verified rate identities, phase and block
totals, mean identities, exact contiguous history coverage, 512 prefill and 128
decode GPU samples without missing data, and forward/sampler containment in
wall time. `read_benchmark` from the existing comparator accepts the single
record and its reported median; a direct absolute-threshold check passes.
The two-report comparator correctly rejects comparison against the prior
three-run report because measured run counts differ. That guard was not relaxed.

No three-run uninstrumented median is claimed yet. The active GPU shader path
is unchanged from the prior BF16/parallel-RMS configuration, with parallel
attention disabled. External power is the newly reported condition and the
leading explanation of the roughly 2x change; it is not proof of a new v0.5
kernel speedup or an isolated same-build battery penalty. All six system
snapshots show nominal thermal state and Low Power Mode off in this run only.

The [M5 configuration guide](m5-pro.md) preserves the result and provides the
headless server command plus a three-run uninstrumented confirmation command.
This update changes documentation and retains the supplied JSON; it changes
no executable code or defaults. JSON validation and `git diff --check` passed.
