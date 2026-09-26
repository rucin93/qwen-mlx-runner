# OpenCode prompt cost and live diagnostics

The user reported on 2026-09-26 that the M5 Pro takes about ten minutes to
answer `Hej` in OpenCode's interactive TUI, both with and without `--pure`,
while direct curl requests work. The plugin-isolation command therefore does
not establish a latency fix. This issue is separate from the noninteractive
OpenCode CLI event-drain problem.

## Measured request size

Isolated OpenCode 1.15.12 CLI and actual TUI sessions in an empty project sent
`Hej` through the repository's example provider configuration with external
plugins disabled.
The model endpoint was a deterministic loopback fixture: no full model
generation or M5 timing was involved. The real server request parser and
`ChatTokenizer` processed the captured request using the Qwen3.8-27B tokenizer
and template at model revision `10c35caafbb80f7dc6a7a432cdd11af10a6d4818`.

The [sanitized count report](benchmarks/opencode-v1.15.12-prompt-cost.json)
retains the per-message/tool counts, exact output-budget results and source
fingerprints. Raw client prompts and downloaded tokenizer files are not
included in the repository.

| Input | Rendered prompt tokens |
| --- | ---: |
| User-only `Hej`, same medium reasoning setting | 12 |
| Captured OpenCode messages with tools removed | 2,141 |
| CLI request, including ten tool definitions | 7,734 |
| Actual TUI request, including eleven tool definitions | **8,065** |

The system message alone contained 9,664 characters; the user message had
three. The CLI tools were `bash`, `edit`, `glob`, `grep`, `read`, `skill`,
`task`, `todowrite`, `webfetch` and `write`; the TUI also included `question`.
The TUI tool definitions added 5,924 tokens. Its complete rendered prompt was
34,736 bytes. With context 8,192 and requested output 8,192, the actual server
preparation left only **127 output tokens** for the TUI request (458 for the
CLI), shared by reasoning and the answer.

The TUI capture used the installed client in an isolated local PTY, not the
user's failing session or a full model on M5. Project instructions, skills, agents, versions,
history and plugins can change the actual request. It establishes why a
short visible message does not imply a short model prompt; it does not by
itself attribute all ten minutes in the user's session.

The earlier full-model M5 comparison used prompts of only 43–65 tokens.
Those samples had median prompt-processing rates of approximately 17.44
tokens/s for ordinary generation and 38.05 tokens/s pooled across the two MTP
variants. A linear extrapolation to 8,065 tokens gives about 462 / 212 seconds
before the answer, respectively. **These are estimates from short prompts,
not measured long-context performance or a prediction for 0.7.4.** The actual
long request requires phase timing on the target Mac.

Increasing the context window alone does not accelerate prompt processing.
Disabling reasoning does not remove the tools or system prompt. The current
ordinary path processes prompt tokens sequentially; MTP processes small blocks
and resets its caches for each request. These remain performance limitations
for large coding-agent prompts. [Current MTP behavior](native-mtp.md).

## User M5 log: 81,000 context and native MTP

The user then supplied a 0.7.4 server launch and lifecycle prefix with native
MTP block size 3, aligned GEMV, parallel norms, BF16 metadata, serial attention
values and **context 81,000**. The displayed device allocation was 24.99 GiB.
The [normalized log extract](benchmarks/m5-pro-v0.7.4-opencode-lifecycle-user.json)
records the exact supplied counts and times without prompt content.

A three-message request without tools processed 551 prompt tokens in
15.6827 seconds, then generated 118 tokens in 5.3054 seconds. It occupied the
worker for 20.9987 seconds. A following request with 11 tools, 24,453 message
content bytes and 22,371 serialized tool bytes waited **20.2123 seconds** in
the queue. Its CPU preparation took only 0.0110 seconds before entering
generation. The provided excerpt ends there; the main request's first text,
token count and final engine timings were not supplied.

The first request is consistent with an auxiliary task such as title
generation; counts alone cannot identify it. Explicitly disabling the title
agent in the effective OpenCode configuration is a useful isolation step.
The 20-second queue explains only part of the reported delay.

**The 127-token output remainder measured with context 8,192 does not apply
to this user's 81,000-token server.** The logged effective output maximum is
8,192. The relevant engine limitation is already the MTP small-block prompt
path, so optimizing the ordinary sequential path would not accelerate this
launch. Larger prompt batches must be evaluated separately from the
three-token speculative decode block.

## Capture the wait without waiting for completion

Build version 0.7.4 or newer and check `qwen-metal --version`.
Restart the server with **the same model, adapter,
context and kernel options as the slow run**, adding
`QWEN_METAL_LOG_REQUESTS=1` before its existing launch command. This flag is
read at server startup; setting it in the OpenCode terminal does not affect an
already-running model server.

Open a new OpenCode session with the explicit local model and send `Hej` once.
The server now writes metadata-only JSON records as the request progresses:

| Event in `kind: "request_lifecycle"` | Meaning |
| --- | --- |
| `admission` | The parsed request is about to enter the bounded queue. |
| `worker_start` | The worker takes the request; `queue_seconds` measures waiting. |
| `generation_start` | Preparation completed; includes `prepare_seconds` and effective `max_tokens`. |
| `first_model_text` | The first nonempty decoded model text is observed. |
| `first_reasoning`, `first_content`, `first_tool_call` | The corresponding first delta becomes ready for HTTP. |
| `terminal` | Completion, failure, rejection or client cancellation, including cancellation during a blocked generation. |

Correlate records by `id`. They include `engine_version`, message/tool counts,
message/tool byte sizes and elapsed times. They do not contain prompts, tool
arguments, reasoning, answer text or raw exception messages. Byte sizes are
not tokenizer counts. The existing final `request_timing` record and HTTP
`timings` extension remain available.

Lifecycle logging starts after HTTP parsing/schema/sampling validation. A
request rejected before admission still returns its existing HTTP error and
does not produce these lifecycle records.

`terminal` with `status: "completed"` and `stage: "generation"` means model
processing completed, before final HTTP delivery. It does not prove TUI
rendering. Disconnect and worker logging can race; an already-started readiness
write can appear after a cancellation line. Use the request ID and elapsed
times rather than assuming all cross-thread output is strictly ordered.

If generation is still waiting, the lifecycle records already emitted are
useful; there is no need to let another ten-minute attempt finish. A long gap
after `generation_start` but before first model text points to work before
visible output, including prompt processing. A first reasoning event followed
much later by first answer content identifies time spent before the answer.
The completed summary reports exact engine prefill and decode durations.

Version 0.7.4 adds diagnostic evidence. It does not claim to solve the
reported ten-minute latency, change reasoning defaults or speed up M5 kernels.

## Local validation of 0.7.4

Five new tests use the real router and worker to check logs before a blocked
generator completes, immediate client cancellation, preparation failure,
first tool readiness and output-processing failure. The first three failed by
assertion before the change and passed afterwards. The base test suite passed
142 tests with 53 environment-dependent tests ignored; the two ignored real
OpenCode/SDK contracts were then run explicitly and passed.

The release build passed. A real release server on the local M1 GPU emitted
the expected admission/worker/generation/first-text/first-content/completed
sequence for a tiny fixture and an error terminal for context overflow. The
test checked version 0.7.4, HTTP 200/400 and metadata-only logging. Independent
read-only review found no blocking correctness or privacy issue. Full-model
M5 latency remains to be measured with the new diagnostics.

## Opt-in prompt batching in 0.7.5

`serve` and `generate` now accept `--mtp-prefill-batch-size 8` or `16` with
`--mtp`. This groups known prompt positions independently of the speculative
decode width. Omitting it retains the original prompt path and width; the
measured M5 decode setting remains `--mtp-block-size 3`.

The candidate traverses more prompt positions per layer, removes recurrent
rollback snapshots that known inputs do not need, and retains the existing
matrix kernels in groups of at most three. Attention and DeltaNet remain
causal, and every target hidden output is passed to the adapter. It does not
change weights, quantization, context, reasoning or the per-request cache reset.

On a synthetic four-layer, 512-wide mixed-format graph on **M1**, processing
48 known inputs took median wall times of 35.095 ms for the existing path,
28.536 ms for prompt batch 8 and 25.683 ms for prompt batch 16. Matrix-only
controls at the real projection dimensions were approximately neutral.
These are local fixture results, not full-model M5 or OpenCode measurements.
The [numeric record](benchmarks/m1-v0.7.5-known-prefill.json) retains all five
matrix controls. Two true-wide matrix experiments were slower and were removed
from the runtime; their [negative measurements](benchmarks/m1-v0.7.5-rejected-wide-prefill.json)
are retained. No larger prompt batch is enabled by default.

### Short comparison on the target Mac

Build both the binary and the comparison example:

```sh
cargo build --release --locked --bin qwen-metal --example mtp_prefill_bench
./target/release/qwen-metal --version
```

Stop the running model server first. The supplied launch already allocates
24.99 GiB; loading another target and adapter concurrently on the 48 GiB Mac
would put the benchmark under avoidable memory pressure. Keep the same kernel
options, context, power source and power mode for the comparison:

```sh
QWEN_METAL_REFERENCE=0 QWEN_METAL_GEMV=aligned \
QWEN_METAL_NORM=parallel QWEN_METAL_METADATA=bf16 QWEN_METAL_ATTN_VALUES=serial \
  ./target/release/examples/mtp_prefill_bench \
  --model models/Qwen3.8-27B-4bit \
  --mtp models/Qwen3.8-27B-MTP-4bit \
  --context 81000 --prompt-tokens 512 --max-tokens 16 --runs 3 \
  --prefill-batch-size 16 > prefill-512.json
```

The example loads the target and adapter once, runs one excluded warmup pair,
then three measured pairs in alternating AB/BA order. All runs use greedy
generation, the same synthetic text and the checkpoint's original template;
the report records the actual rendered token count. Speculative decode stays
at three in both variants. Token IDs, text, finish reason and counts must agree
in every pair, including warmup. A mismatch produces a nonzero exit and withholds
all comparison ratios. This is a numerical/performance probe, not a coding
quality test or a replay of the user's OpenCode request.

Inspect `correct_comparison`, the `baseline` and `candidate` summaries, and
`baseline_over_candidate_median_prefill_ratio`. A ratio above one means lower
candidate latency. The report separates target and adapter prompt time, records
time to first token and captures device/power/thermal conditions. Check these
conditions before attributing a small difference to the candidate. A short
probe is the first gate; it does not establish the latency of the much larger
OpenCode request. If it is correct and beneficial, repeat with a representative
longer prompt before deciding whether to add `--mtp-prefill-batch-size 16` to
the server command.

### Local validation of 0.7.5

`cargo test --all-targets` passed 150 tests with 61 environment-dependent tests
ignored. The relevant ignored tests were then run on the real M1 GPU:

```sh
QWEN_METAL_METADATA=bf16 cargo test --test known_prefill --test known_delta \
  -- --ignored --nocapture --test-threads=1
cargo test --test grouped_prefill_matrix -- --ignored --nocapture --test-threads=1
cargo test --test mtp_prefill --test block_inference --test delta_snapshot \
  --test batch_matrix -- --ignored --test-threads=1
```

These passed 7, 1 and 20 tests respectively. Coverage includes widths 1–16 and
all tails, dense/Q4/BF16 fixtures, direct scalar/FP64 DeltaNet comparison,
hidden/logit history and future decoding, cancellation/reset, invalid inputs,
pending verification, the larger-key scalar fallback and snapshot-free dispatch
profiles. Existing speculative verification and rollback tests remain green.
The CLI opt-in test was observed failing before implementation and passing
afterwards. An independent read-only source review found no actionable
correctness or safety issue.

The release binary and example built successfully and reported version 0.7.5.
The real GPU release example ran a 54-token synthetic fixture with one warmup
pair and two measured pairs: all outputs matched, exactly two measurements per
variant entered summaries, and the target/adapter were loaded once. A release
server with prompt batch 16 returned HTTP 200 and a completed lifecycle record;
the release `generate` command reported prompt batch 16 and decode width 3.
The tiny fixture supports context at most 128, which was used for these release
checks. Formatting and whitespace checks passed. These checks establish local
integration and numerical consistency, not resolution of the M5 delay.
