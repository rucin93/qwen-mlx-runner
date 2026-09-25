# M5 Pro: audited native MTP result, block size 3

This is a user-supplied 0.6.0 report from Apple M5 Pro / 48 GB, independently
recalculated from its individual captures. The original JSON is preserved in
[`m5-pro-v0.6-mtp-b3-user.json`](benchmarks/m5-pro-v0.6-mtp-b3-user.json).
It contains trained-target generation, not synthetic weights or a kernel test.

**MTP sustained decode is 26.76587 tokens/s, versus 16.62821 tokens/s for ordinary
generation on the same loaded target: 1.60967x (+60.97%). The 32 tokens/s mixed
workload goal is not yet reached.**

## Workload and checks

- Five mixed Polish/English prompts, three measured runs per prompt and mode.
  Forty captures contain ten separate warmups and thirty measured generations.
- Each output reaches the 128-token length cap. No early EOS shortens a run.
  Each mode has 1920 completion tokens and 1905 sustained decoded tokens.
  The first completion of each request is supplied by prefill logits.
- Actual prompt lengths are 43–65 tokens. The 8192 setting is cache capacity;
  this is not an 8192-token-history measurement.
- Temperature 0, block size 3, aligned kernels, parallel RMS, BF16 quantization
  metadata, serial attention values. Metal reports 17,475,239,936 allocated bytes
  including the target, MTP, caches and scratch.
- Raw token arrays, text and finish reasons agree for all twenty paired runs,
  including the fifteen measured pairs. These are five distinct deterministic
  128-token traces repeated across runs, not twenty distinct prompts.
- All eighty phase-boundary snapshots report Low Power Mode off and nominal
  thermal state. The report does not directly capture AC connection or clocks.
- Aggregate counts, rates, medians, per-prompt summaries and warmup exclusion
  were recalculated and agree with the reported fields.

## Per-prompt result

| Prompt | Median sustained MTP tok/s | Median sequential tok/s | Draft acceptance |
| --- | ---: | ---: | ---: |
| Polish explanation | 24.5864 | 16.7051 | 61.40% |
| English code | 31.8825 | 16.6432 | 94.32% |
| Polish analysis | 26.7417 | 16.6865 | 74.51% |
| English rewrite | 27.6216 | 16.7323 | 76.00% |
| Polish planning | 24.4000 | 16.7335 | 61.95% |

The code task's completion-count median is 32.1335 tokens/s; its sustained median
is **31.8825** after subtracting the prefill-derived token. It therefore does not
meet the strict sustained 32 threshold. All fifteen measured MTP runs are long
enough for the report's minimum-sample guard, but the throughput gate is false.

## Decode budget

MTP emits 1905 sustained tokens in 71.172721124 seconds, using 780 target rounds.
Accepted proposals plus rounds equal emitted tokens exactly: 1125 + 780 = 1905.
The mean yield is 2.44231 tokens/round; proposal acceptance is 1125/1551 = 72.53%.
There are 774 width-3 rounds, three width-2 rounds and three width-1 rounds.

| Component | Total seconds | Decode share | Mean ms/round |
| --- | ---: | ---: | ---: |
| Target model block execution/readback | 61.764455 | 86.78% | 79.1852 |
| MTP proposals, seeding and cache repair | 7.745531 | 10.88% | 9.9302 |
| CPU acceptance and sampling | 1.089786 | 1.53% | 1.3972 |
| Prefix commit/cache truncation | 0.544057 | 0.76% | 0.6975 |
| Other host/output work | 0.028892 | 0.04% | 0.0370 |

`verification_seconds` in this report means CPU acceptance, not model execution.
Model verification is `target_seconds`, which includes encoding, GPU completion
and readback. These are wall-time components; this report does not separate
target GPU activity from CPU encoding.

At the same acceptance pattern, deleting all MTP time would yield only 30.0344
tokens/s. Deleting every overhead outside target execution would yield 30.8430.
Thus draft-only or sampler-only tuning cannot reach the aggregate 32 target.

Keeping other costs fixed, aggregate 32 requires target time to fall by about
18.85% (79.19 to 64.26 ms/round). Reaching 32 on the weakest individual prompts
requires roughly 27% lower target time, or an increase in accepted tokens per
round. These are calculated requirements, not predicted optimization gains.

The next measurements compare block width 4, packed matrix-kernel work, and
recurrent-state snapshot traffic. Correctness remains gated by target trace
agreement; neither a microbenchmark nor a changed block width establishes full
model speed without another M5 measurement.

## Block-4 follow-up: different power conditions

The user then ran block size 4 on the same 0.6.0 build, with one measured run
per prompt. They pasted the complete report into the conversation; a
[transcribed summary](benchmarks/m5-pro-v0.6-mtp-b4-low-power-summary.json) preserves
the relevant values and explicitly identifies itself as a summary. Its result
was **11.24778 sustained MTP tokens/s versus 9.18317 sequentially**, with greedy
agreement still true, 635 sustained output tokens, 409 accepted proposals out
of 671, and 226 target rounds (2.80973 output tokens/round).

Every before/after snapshot in that report has **`low_power_mode: true`**;
block 3 above had false. Even the unchanged sequential path slowed from
16.63 to 9.18 tokens/s. Therefore 11.25 versus 26.77 is not a controlled estimate
of the effect of block width. Within its own conditions, block 4 improved over
its paired baseline by 1.22483x; comparing that ratio across power modes also
does not isolate block width, because power limits can affect kernels differently.

Block 3 stays the default. Peak-speed retests must record Low Power Mode off
as well as consistent external-power and thermal conditions. The benchmark now
prints a warning before weight loading if the OS reports Low Power Mode enabled.
It leaves system power settings untouched.
