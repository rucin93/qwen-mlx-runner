# Performance iteration, 2026-09-25

The requested result is at least 10× the reported ~1.5 decode tokens/s on an
M5 Pro with 48 GB, using `mlx-community` packed Q4 weights. **That target is
not verified.** Only an M1 with 16 GB is available in this workspace, and it
cannot fit this trained 27B checkpoint under the engine's memory budget.

## Reported target baseline

The user ran:

```sh
./target/release/qwen-metal bench --model models/Qwen3.8-27B-4bit \
  --context 8192 --prompt-tokens 512 --generate-tokens 128 --runs 3
```

The supplied warmup had 1.550455707 decode tokens/s and 1.560779651 prefill
tokens/s. Measured run 1 had 1.452415808 decode tokens/s and 1.679659483 prefill
tokens/s. The remaining runs/final median were not supplied. These are a user
report, not a benchmark executed in this workspace. Use 15 decode tokens/s as
the practical acceptance threshold; do not substitute a matrix speedup for it.

## Changes

- Compile-time Q4/Q8 group specializations remove per-value dynamic division
  and repeated packed-word loads from the original shader.
- Four output rows share each input-vector load. The aligned Q4 path handles
  16 weights per lane using vector loads, signed bounds-proven offsets and
  explicit accumulators. Non-aligned matrices use the safe packed fallback.
- Pipeline selection uses static names. Barrier resource lists use stack
  storage and name actual shader writes. Metal ignores explicit barriers on
  this engine's serial encoder, so this is host-allocation cleanup, not a
  demonstrated reduction in GPU weight fences.
- The original path stays selectable with `QWEN_METAL_REFERENCE=1`; the final
  JSON identifies the selected mode. Within each local comparison, reference
  and candidate use identical weights, context, executed layers and output
  count. There is no requantization or external inference runtime.

## Final local release measurements

Raw records, workload dimensions and the executable SHA-256 are in
[2026-09-25-m1.json](benchmarks/2026-09-25-m1.json). Both modes use the same
release executable and were run sequentially on the M1. This was an active
16 GB desktop with memory pressure, not an isolated benchmarking machine;
repeat and compare on the target hardware before drawing throughput conclusions.

Q4 group 64 matrix-vector timings, including host dispatch and completion:

| Matrix rows × columns | Reference | Aligned | Speedup |
| --- | ---: | ---: | ---: |
| 17,408 × 5,120 | 17.355 ms | 1.905 ms | 9.11× |
| 5,120 × 17,408 | 18.945 ms | 1.852 ms | 10.23× |
| 248,320 × 5,120 | 274.244 ms | 27.961 ms | 9.81× |

The microbenchmarks use 60 measured repetitions for each MLP shape and eight
for the vocabulary head, after three warmups. They reuse weights and are not
model token throughput.

The full 64-layer **synthetic reused-weight, zero-history** diagnostic ran with
`--context 2048 --history 512 --steps 6`, after one warmup step:

| Mode | Mean milliseconds per step | Relative to reference |
| --- | ---: | ---: |
| Reference | 3991.342 | 1.00× |
| Packed4 | 698.176 | 5.72× |
| Aligned (default) | 626.852 | 6.37× |

Reference is one six-step execution; the optimized entries average two
six-step executions each. Earlier exploratory debug runs varied (including
~7.4× for packed4); the final release comparison above is the retained evidence.
The synthetic model shares layer weights and allocates only 2,322,677,760 bytes.
It does not perform a real prompt prefill, generate meaningful text, or load
the trained checkpoint. Its speedup must not be presented as M5 or 27B tok/s.
See [synthetic workload details](synthetic-benchmark.md).

## Correctness and target validation

The GPU matrix test compares 270 cases against scalar FP64 sums: Q4/Q8,
groups 32/64/128, one-group/three-group/5120-wide inputs, rows 1/3/7/19/20,
and both optimized modes plus reference. An additional host test exercises
the signed-index guard and fallback selection without allocating huge buffers.
The new packed Q4 four-layer fixture checks every logit over five tokens,
then repeats after reset, against an independent Python scalar implementation.
Reassociated FP32 arithmetic is tolerance-checked, not claimed bit-identical;
these synthetic tests do not establish trained-model quality.

On the M5, build and rerun the exact workload:

```sh
cargo build --release --locked
QWEN_METAL_REFERENCE=0 QWEN_METAL_GEMV=aligned \
  ./target/release/qwen-metal bench --model models/Qwen3.8-27B-4bit \
  --context 8192 --prompt-tokens 512 --generate-tokens 128 --runs 3 > after.json
```

For a fresh paired comparison, run the same command with
`QWEN_METAL_REFERENCE=1` into `before.json`, then:

```sh
python3 scripts/compare_bench.py before.json after.json --require-speedup 10
```

Keep the checkpoint files/revision, context, power mode and competing workloads
unchanged. The comparator checks matching reported model path/device/token
workload and excludes warmup, but cannot prove the files at that path stayed
unchanged. Read the generated text separately to evaluate quality. If default
`aligned` is slower on the target, measure `QWEN_METAL_GEMV=packed4` with the
same command; it is retained for that hardware-specific comparison.
