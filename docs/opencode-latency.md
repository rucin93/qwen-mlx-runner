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

## Capture the wait without waiting for completion

Build the version containing this diagnostic change and verify that the binary
reports `qwen-metal 0.7.4`. Restart the server with **the same model, adapter,
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

This release adds diagnostic evidence. It does not claim to solve the
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
