# M5 regression audit and independent kernel comparison

The user's 0.6.1 full-model M5 Pro report **regressed from 26.7659 to 23.3430
sustained MTP tokens/s (−12.79%)**. The paired sequential target stayed almost
unchanged: 16.6282 to 16.6204 tokens/s. Version 0.6.2 restores the earlier
matrix and recurrent execution schedules as defaults and keeps the two 0.6.1
variants independently selectable. **0.6.2 full-model M5 throughput is not
yet measured; neither recovery nor 32 tokens/s is claimed.**

## Evidence

The complete [0.6.0 report](benchmarks/m5-pro-v0.6-mtp-b3-user.json) and
[0.6.1 report](benchmarks/m5-pro-v0.6.1-mtp-b3-user.json) were supplied by the
user. The latter is preserved byte-for-byte, SHA-256
`e04e571d62e392d419755fdf4265065a206231c06b4c5bd735e582e9822736e3`.
Each has 40 captures: 10 excluded warmups and 15 measured generations per mode.
All matching captures across versions have identical token IDs, output text,
finish reason, accepted/proposed counts, verification rounds and block widths.
Workload, sampling, context capacity and block size also match.

| Measured MTP work | 0.6.0 | 0.6.1 |
| --- | ---: | ---: |
| Sustained decoded tokens | 1,905 | 1,905 |
| Total decode interval | 71.173 s | 81.609 s |
| Target execution | 61.764 s | 72.007 s |
| Draft work | 7.746 s | 7.861 s |
| CPU verification | 1.090 s | 1.159 s |
| Rollback | 0.544 s | 0.551 s |
| Verification rounds | 780 | 780 |
| Accepted / proposed drafts | 1,125 / 1,551 | 1,125 / 1,551 |

Target execution accounts for **98.15% of the extra decode duration**. This
timer includes CPU encoding, GPU completion and readback; the old reports do
not identify which of those increased. M1 microbenchmark improvements from
0.6.1 are therefore insufficient evidence for choosing M5 defaults.

Low Power Mode is false in every snapshot of both reports. The first 21
captures of 0.6.1 remain thermally `nominal`; capture 21 (zero-based, analysis,
MTP run 2) first changes to `fair`. The first two prompts, entirely nominal in
both versions, already fall from 27.7548 to 24.6702 MTP tokens/s (−11.11%), while
their sequential rate changes from 16.6603 to 16.6875. Later thermal changes
may contribute to the larger late-prompt slowdown but do not explain the early
regression. The readings are snapshots, not continuous power/clock telemetry.

Recompute the comparison from the archived raw captures:

```sh
python3 scripts/audit_mtp_reports.py \
  docs/benchmarks/m5-pro-v0.6-mtp-b3-user.json \
  docs/benchmarks/m5-pro-v0.6.1-mtp-b3-user.json > regression-audit.json
```

## Controlled comparison

The new `mtp-bench --compare-block-kernels` runs these configurations with one
loaded target and adapter:

| Matrix schedule | DeltaNet execution | Purpose |
| --- | --- | --- |
| `legacy` | `sequential` | Earlier execution schedules, now the default |
| `shared` | `sequential` | Isolate shared Q4 unpacking |
| `legacy` | `batched` | Isolate batched recurrence |
| `shared` | `batched` | 0.6.1 combination |

All modes retain 0.6.1's reduced snapshot allocation and omit unused final
snapshots. Thus the first row restores the execution schedules, but is not an
exact replay of the 0.6.0 binary. No weights or precision settings change.
Shared unpacking applies only to aligned Q4/group64 width-2/3 matrix blocks;
batched recurrence supports key dimensions up to 128. Other shapes keep their
validated fallback paths.

Every prompt/configuration has a separately excluded warmup. Measured order
is balanced over four runs, with the order recorded in the captures. Each
generation starts with reset caches. Throughput is summarized independently
for each configuration and prompt, using actual verified tokens and the full
decode interval. Exact token IDs, text and finish reasons are compared against
the legacy/sequential configuration. This diagnostic comparison alone does
not establish ordinary-target agreement or achievement of 32 tokens/s.

For a shorter first diagnosis, use the two original prompts that already
regressed while thermally nominal:

```sh
git pull --ff-only
cargo build --release --locked
QWEN_METAL_REFERENCE=0 QWEN_METAL_GEMV=aligned \
QWEN_METAL_NORM=parallel QWEN_METAL_METADATA=bf16 QWEN_METAL_ATTN_VALUES=serial \
./target/release/qwen-metal mtp-bench \
  --model models/Qwen3.8-27B-4bit \
  --mtp models/Qwen3.8-27B-MTP-4bit \
  --context 8192 --max-tokens 128 --runs 4 --block-size 3 \
  --temperature 0 --prompts docs/kernel-ab-prompts.json \
  --compare-block-kernels > kernel-ab.json
```

Keep external power connected and Low Power Mode off. Run this once, without
another inference workload. Omit `--prompts` later to compare all five mixed
prompts. Do not add `--compare`: that is the separate sequential-target
comparison and is mutually exclusive with this diagnostic mode.

`stats.target_decode_timing` summarizes existing command-buffer timestamps
without adding profiling dispatches. `observed_gpu_seconds` covers the whole
GPU command interval; it overlaps `observed_completion_wait_seconds`, so the
two must not be added. CPU encoding and commit have separate fields. Missing
GPU timestamps are counted explicitly, and all sums cover only available
samples. Prefill and rollback are outside this target-decode timing summary.

For ordinary generation or the server, explicit overrides are
`QWEN_METAL_BLOCK_MATMUL=legacy|shared` and
`QWEN_METAL_BLOCK_DELTA=sequential|batched`. Leave them unset for the restored
defaults until the M5 comparison identifies the better combination. The
factorial command selects its configurations explicitly and records each one.

## Acceptance still outstanding

Local verification on Apple M1: **120 unique all-target tests passed**, zero
failed or ignored with real Metal tests enabled, using
`cargo test --offline --locked --all-targets -- --include-ignored --test-threads=1`.
This includes both matrix schedules against independent FP64 references and
bitwise comparisons, all four combinations across rollback prefixes and causal
continuation, K160 fallback, normal and factorial CLI accounting, configuration
guards and missing/unknown timing-condition handling. The small hybrid fixture
does not have aligned group64 matrices; those paths are covered separately by
the direct matrix tests at columns 512/5120/17408 with F32/BF16 metadata.

The two archived reports also pass the independent raw-capture audit, and an
altered token trace is rejected. These checks establish correctness and report
accounting locally, not performance recovery on M5.

The 0.6.2 release build also passed a two-prompt/four-run factorial CLI smoke
on the tiny fixtures: 40 captures, 30 exact greedy comparisons, every variant
in each timing position, and no missing timing samples. Formatting and
`git diff --check` passed. The dependency `block` 0.1.6 continues to emit its
existing future-compatibility warning.

The goal remains sustained 32 tokens/s on the mixed workload, with matching
ordinary-target greedy traces, at least three measured runs per prompt and
sufficient decoded tokens. After selecting a configuration, repeat
`mtp-bench --compare` over all five prompts under the same power conditions.
The kernel diagnostic is a step toward that evidence, not a replacement for it.
