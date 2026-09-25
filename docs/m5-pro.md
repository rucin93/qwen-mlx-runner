# M5 Pro / 48 GB: observed configuration

The user supplied the complete v0.5 report in
[`m5-pro-v0.5-timing-user.json`](benchmarks/m5-pro-v0.5-timing-user.json), then
clarified that the MacBook had been connected to external power for this run.
This is user-reported target evidence, not a local M5 execution.

## Result and acceptance limits

| Measurement | Value |
| --- | ---: |
| Measured decode | **16.13080 tokens/s** |
| Generated steps / time | 128 / 7.93513 s |
| Warmup decode, separately excluded | 16.13292 tokens/s |
| Measured prefill | 512 tokens / 29.41935 s = 17.40351 tokens/s |
| Context capacity | 8192 |
| Reported GPU allocations | 16,374,677,504 bytes |
| Compacted matrices | 498 |
| Exact metadata bytes saved | 1,680,834,560 |

The original measured decode run was 1.4524158079 tokens/s; the new rate is
11.1062 times that observation, or 10.7539 times the original approximately
1.5 tokens/s target baseline. **The 15 tokens/s threshold is exceeded in this
full 128-token run.** A three-run uninstrumented median and longer-session
stability have not yet been established. This result also does not establish
trained-model quality, chat throughput, or superiority over another engine.

The JSON's `median_decode_tokens_per_second` equals its only measured run.
The warmup is not included. All 512 prefill and 128 decode GPU samples are
present; block counts, history ranges, duration sums and rate calculations
were independently checked against the report.

## Where the measured time goes

Measured decode averages per token:

| Interval | Time |
| --- | ---: |
| Entire decode loop | 61.9932 ms |
| Forward call | 61.5218 ms |
| GPU interval | 60.6605 ms |
| Completion wait (overlaps GPU) | 61.1694 ms |
| CPU encoding | 0.3078 ms |
| CPU commit | 0.0033 ms |
| Token selection | 0.4712 ms |
| Other host work within forward | 0.0412 ms |

GPU execution accounts for 97.85% of decode wall time. GPU and wait must not
be added. There is no unexplained 50–60 ms gap in this run: forward plus
sampling leaves only about 28 microseconds of total loop bookkeeping across
all 128 tokens.

The four 32-token blocks averaged 60.177, 60.671, 60.687 and 61.107 ms of GPU
time. Combined forward/sampling means were 61.502–62.472 ms, a gradual change
of about 1.6% across the decode window. The report does not show a progressive
large slowdown over those 128 tokens. Every phase-boundary snapshot reports
Low Power Mode off and thermal state `nominal`; these are snapshots, not a
continuous temperature or clock-frequency trace.

Warmup prefill contains a roughly 1.536-second forward outlier with substantial
completion-wait time; warmup GPU durations never exceed about 63.9 ms. The
aggregate data does not identify its cause or prove it is shader compilation.
It does not appear in measured prefill/decode and cannot explain the earlier
sustained roughly 8 tokens/s result.

## Interpreting the jump from 8 to 16 tokens/s

The preceding parallel RMS / BF16 report measured 8.03204 tokens/s. The new
run is 2.0083x that rate, with the same 16.37 GB allocation and 498 compacted
matrices. The user identified connecting external power as a changed condition.

The active aligned BF16 matvec and parallel RMS shaders did not change between
v0.3 and v0.5. The newer parallel attention kernel was disabled in this run.
v0.5 adds optional timing and read-only system-state queries, without changing
the command graph or power settings. The leading explanation is therefore
power conditions; **the 2x change is not established as a v0.5 code speedup**.
Earlier runs did not record power/thermal state. A same-build power-source
comparison would be needed to isolate that cause or quantify a battery penalty.
No claim about exact GPU clock or an OS power limit follows from these data.

## Reproduce the configuration

Connect external power and build the current checkout:

```sh
cargo build --release --locked
```

Use the measured configuration for the headless server:

```sh
QWEN_METAL_REFERENCE=0 QWEN_METAL_GEMV=aligned \
QWEN_METAL_NORM=parallel QWEN_METAL_METADATA=bf16 QWEN_METAL_ATTN_VALUES=serial \
  ./target/release/qwen-metal serve \
  --model models/Qwen3.8-27B-4bit --context 8192 --listen 127.0.0.1:8080
```

For the remaining three-run confirmation, use the same configuration without
diagnostic timing. Keep the power source and other workloads consistent:

```sh
QWEN_METAL_REFERENCE=0 QWEN_METAL_GEMV=aligned \
QWEN_METAL_NORM=parallel QWEN_METAL_METADATA=bf16 QWEN_METAL_ATTN_VALUES=serial \
  ./target/release/qwen-metal bench \
  --model models/Qwen3.8-27B-4bit --context 8192 \
  --prompt-tokens 512 --generate-tokens 128 --runs 3 > bench-ac.json

python3 scripts/compare_bench.py \
  docs/benchmarks/m5-pro-v0.3-parallel-bf16-user.json bench-ac.json --require-tps 15
```

The comparator validates workload/rate consistency and the absolute 15 tokens/s
threshold. Its speed ratio in this command uses the previous 8.03 observation,
not the original 1.45 observation. These measurements compare different runs
and versions; they do not isolate an individual optimization's causal effect.
