# OpenCode and Chat Completions compatibility

The local Rust/Metal server implements the text and function-tool subset of
`POST /v1/chat/completions` described below. It also exposes `GET /v1/models`
and `GET /health`. This is a local API contract, not a claim of complete OpenAI
service parity or trained-model tool-calling quality. The implementation is in
[request parsing](../src/server/request.rs), [response processing](../src/server/output.rs),
[HTTP serving](../src/server.rs) and [tool parsing](../src/chat/tools.rs).

## Configure OpenCode

Use the [example configuration](../examples/opencode.json) as your project's
`opencode.json`, or merge its provider and model entries into an existing config.
It selects `qwen-metal/Qwen3.8-27B-4bit` for both normal work and lightweight
auxiliary tasks. From 0.7.3 the example disables the automatic title agent to
avoid a competing request on the same local 27B model. The main model's reasoning
remains enabled. The adapter is **`@ai-sdk/openai-compatible`**,
and its base URL is **`http://127.0.0.1:8080/v1`**. This adapter uses Chat
Completions; the server does not implement `/v1/responses`.
[OpenCode custom-provider instructions](https://opencode.ai/docs/providers/#custom-provider),
[OpenCode model configuration](https://opencode.ai/docs/config/#models).

Start the server from the repository after building and downloading the target
and companion adapter as described in [native MTP setup](native-mtp.md):

```sh
QWEN_METAL_REFERENCE=0 QWEN_METAL_GEMV=aligned \
QWEN_METAL_NORM=parallel QWEN_METAL_METADATA=bf16 QWEN_METAL_ATTN_VALUES=serial \
  ./target/release/qwen-metal serve \
  --model models/Qwen3.8-27B-4bit \
  --mtp models/Qwen3.8-27B-MTP-4bit --mtp-block-size 3 \
  --context 8192 --listen 127.0.0.1:8080
```

The ordinary target path is also supported: omit `--mtp` and `--mtp-block-size`.
Check the exact model ID with `curl http://127.0.0.1:8080/v1/models`; it is the
target directory's final component. Update the OpenCode model key if yours
differs. This loopback server does not require an API key.

Keep `limit.context`, `limit.input` and `limit.output` equal to the server's actual
`--context`; the example sets all three to 8192. They describe the configured
window. The server computes the actual output budget from the space left after
tokenizing the prompt, including its template, tool definitions, instructions
and history. Increasing client limits does not enlarge the server's caches.

The example also sets top-level `compaction.reserved` to 1024. This is headroom
for OpenCode's context management, **not a per-answer output cap**. With an
explicit input limit, the inspected OpenCode implementation computes its usable
threshold as `max(0, limit.input - reserved)`: 7168 for this example. Without an
input limit it instead subtracts its output budget from `limit.context`, which
would give zero with context/output both 8192. Setting the reserve alone does
not change that second branch. Keep both `limit.input` and the reserve.
[Pinned OpenCode compaction calculation](https://github.com/anomalyco/opencode/blob/fe3f3a41f79ad292cc3c7c629567385a20ec5130/packages/opencode/src/session/overflow.ts#L10-L33).

OpenCode's output calculation falls back to 32,000 when its resolved output
limit is zero or absent. Its configuration schema requires `output` whenever
a `limit` object is supplied, so deleting only that field is not a supported
way to remove a cap. Declaring the real window size and letting this server
reduce the requested budget avoids the previous 2048-token example limit.
[Pinned output calculation](https://github.com/anomalyco/opencode/blob/fe3f3a41f79ad292cc3c7c629567385a20ec5130/packages/opencode/src/provider/transform.ts#L1468-L1470),
[pinned limit schema](https://github.com/anomalyco/opencode/blob/fe3f3a41f79ad292cc3c7c629567385a20ec5130/packages/core/src/v1/config/provider.ts#L47-L52),
[pinned reserve schema](https://github.com/anomalyco/opencode/blob/fe3f3a41f79ad292cc3c7c629567385a20ec5130/packages/core/src/v1/config/config.ts#L149-L166).

## Accepted requests

Optional `null` values are generally treated as omitted. `model` and a nonempty
`messages` array remain required. Unknown top-level fields are rejected. The
HTTP request-body limit is 1 MiB.

| Field | Implemented behavior |
| --- | --- |
| `model` | Must match the served model ID. |
| `messages` | Text messages with roles `system`, `developer`, `user`, `assistant`, or `tool`; at least one user message is required. |
| `max_tokens`, `max_completion_tokens` | Alternative positive-integer upper bounds, with no fixed server cap. Omission/null uses the remaining context. An explicit value is reduced to `min(requested, remaining context)`. Supply at most one non-null limit. |
| `temperature` | Number 0–2; default 1. Zero selects greedy sampling. |
| `top_p` | Number greater than zero and at most 1; default 1. |
| `top_k` | Local extension: nonnegative integer; default 0 disables top-k filtering. |
| `seed` | Integer; default 0. Negative integers are mapped to the internal unsigned seed. |
| `frequency_penalty`, `presence_penalty` | Numbers from −2 to 2; default 0. Count only tokens generated in the current completion. |
| `logit_bias` | Object mapping vocabulary token-ID strings to biases from −100 to 100; out-of-vocabulary IDs are rejected. |
| `stop` | One string or up to four nonempty strings, each at most 4096 bytes. Matched sequences are omitted, including when split across streamed chunks. |
| `stream` | Boolean, default false; true selects SSE. |
| `stream_options` | Requires `stream:true`; supports `include_usage` and `include_obfuscation`. |
| `tools`, `tool_choice`, `parallel_tool_calls` | Function tools and tool-choice behavior described below. |
| `response_format` | Exactly `{"type":"text"}` or `{"type":"json_object"}`; defaults to text. |
| `enable_thinking` | Local boolean extension. Defaults to false unless a non-`none` reasoning effort is supplied. |
| `reasoning_effort` | `none`, `minimal`, `low`, `medium`, `high`, or `xhigh`; maps to checkpoint template controls. `minimal` maps to `low`, `high` to `xhigh`. Conflicting thinking settings are rejected. |
| `n`, `store`, `logprobs`, `top_logprobs`, `modalities` | Only `1`, `false`, `false`, `0`, and `["text"]`, respectively. These do not enable multiple completions, storage, log probabilities, or multimodal inference. |
| `user`, `safety_identifier`, `prompt_cache_key` | Optional strings accepted as passive request metadata. They do not activate persistence, moderation, or a keyed prompt-cache service. |
| `metadata` | Passive object of at most 16 string entries; keys up to 64 characters, values up to 512 characters. |
| `service_tier` | `auto` or `default`; when supplied, the response reports `default`. It does not alter scheduling. |

Penalties use completion-token counts, not prompt/history counts. A repeated
token's adjusted logit subtracts `frequency_penalty × count` and
`presence_penalty`; negative penalties reward repetition. Logit bias applies
before sampling, including the first completion token. These controls are
also applied in MTP's verified sampling path.

The HTTP budget behavior above starts in **0.7.1**. For example, if a fully
rendered prompt occupies 1200 tokens of an 8192-token window, omission/null or
`max_tokens:32000` permits up to 6992 generated tokens; `max_tokens:1000` permits
up to 1000. Reasoning and tool-call syntax count toward that generated-token
budget. EOS and configured stop sequences can finish earlier. A prompt that
fills or exceeds the window is rejected because no output token would fit.
The CLI and benchmark commands retain their existing explicit token controls.

Message content may be a string or an array of `{"type":"text","text":"..."}`
parts, concatenated in order. Images, audio, files and other content-part types
are rejected. Initial system/developer messages must precede conversation
messages; they are adapted to the checkpoint's initial system prompt.
Optional message `name` must contain 1–64 ASCII letters, digits, `_` or `-`.
`reasoning_content` is accepted only on assistant history and is passed through
the checkpoint's history adapter.

## Reasoning in OpenCode

The example configuration enables reasoning by default from **0.7.2**. Its
model entry advertises `reasoning:true`, sets `options.reasoningEffort:"medium"`,
and uses `interleaved:{"field":"reasoning_content"}` to preserve reasoning in
assistant history, including tool continuations. The compatible adapter sends
the request option as `reasoning_effort` and converts `delta.reasoning_content`
into separate reasoning events. Ordinary answer text remains `delta.content`.

The model entry defines explicit `none`, `low`, `medium`, and `xhigh` variants,
because the inspected OpenCode versions do not automatically build reasoning
variants for Qwen names. For example:

```sh
opencode run --thinking 'Explain this function.'
opencode run --variant low --thinking 'Explain this function.'
opencode run --variant none 'Explain this function.'
```

`--thinking` controls display, not inference: `reasoningEffort` enables reasoning.
The TUI `/thinking` command toggles expanded/collapsed reasoning. Display
preferences remain under client control. The server's HTTP default is still
thinking off for requests that supply neither effort nor `enable_thinking`.
Reasoning consumes generation time and the shared output/context budget.

For same-model tool turns, OpenCode moves reasoning into assistant message
metadata and the adapter serializes it as `reasoning_content`; the server
passes it to the checkpoint template. Switching to another provider/model has
different history semantics in OpenCode and is outside this same-model test.

Primary references for installed OpenCode 1.15.12:
[model schema](https://github.com/anomalyco/opencode/blob/58a27b95c155d3f7d9b9f25b30eb1233bfb0eae5/packages/opencode/src/config/provider.ts#L5-L70),
[reasoning history normalization](https://github.com/anomalyco/opencode/blob/58a27b95c155d3f7d9b9f25b30eb1233bfb0eae5/packages/opencode/src/provider/transform.ts#L308-L339),
[Qwen variant handling](https://github.com/anomalyco/opencode/blob/58a27b95c155d3f7d9b9f25b30eb1233bfb0eae5/packages/opencode/src/provider/transform.ts#L632-L650),
[CLI reasoning events](https://github.com/anomalyco/opencode/blob/58a27b95c155d3f7d9b9f25b30eb1233bfb0eae5/packages/opencode/src/cli/cmd/run.ts#L618-L710).

## Latency diagnostics

OpenCode and a plain terminal `generate` request usually submit different work.
The OpenCode example enables medium reasoning and sends system/project
instructions, tool schemas and conversation history; `generate` defaults to
thinking off and a small user prompt. When `--thinking` is enabled, the terminal
prints an opening `<think>` marker before inference starts, so that marker is
not a measurement of the first generated token.

The example's `agent.title.disable:true` removes a separate title-generation
request that OpenCode otherwise launches alongside the first main turn using
the same `small_model`. Both would share the server's single generation worker.
The main chat retains reasoning and its existing output/context behavior.
[OpenCode title launch](https://github.com/anomalyco/opencode/blob/58a27b95c155d3f7d9b9f25b30eb1233bfb0eae5/packages/opencode/src/session/prompt.ts#L1293-L1300).

For a measured diagnosis, add `QWEN_METAL_LOG_REQUESTS=1` to the existing server
launch. Completed generations write a `kind:"request_timing"` JSON record to
stderr with prompt/completion token counts, message/tool counts, actual output
budget, reasoning settings and timings. These records contain no prompt,
argument, reasoning or answer text. Timing fields are also returned in the
response's existing `timings` extension, including the final SSE choice chunk.

| Timing field | What it measures |
| --- | --- |
| `queue_seconds` | Time from enqueue to the model worker accepting the request. |
| `prepare_seconds` | Worker-side request preparation, including template/tokenizer validation. |
| `prefill_seconds` | Engine prompt-processing time before generation. |
| `first_model_text_seconds` | Enqueue to the first nonempty decoded model text callback; this may be structural text and is not an exact first-token measurement. |
| `first_reasoning_seconds` | Enqueue to the first non-whitespace reasoning delta ready for HTTP. |
| `first_content_seconds` | Enqueue to the first non-whitespace answer-content delta ready for HTTP. |
| `first_tool_call_seconds` | Enqueue to the first complete validated tool call ready for HTTP. |
| `decode_seconds` | Engine generation interval, including reasoning and tool-call text. |
| `total_seconds` | Enqueue to worker completion, before final HTTP delivery/rendering. |

Missing delta types are `null`, not zero. Queue/first-delta/total clocks overlap;
do not add them. JSON-object mode buffers content until validation, so its
first-content time reflects that buffering. These are server-side readiness
measurements, not client network or UI presentation times.

A long queue interval points to competing requests. A long prefill interval
points to prompt size and prompt processing. A long gap between first reasoning
and first content means the model is generating reasoning before answering.
The `low` variant keeps reasoning enabled with a lower requested effort; `none`
can provide a controlled comparison with the terminal's default. Expanding
thinking in the UI changes visibility only.

From 0.7.3, MTP prompt initialization skips unused vocabulary projections in
nonfinal target blocks and uses a cache-only K/V append for the draft model.
The final prompt block and decode verification retain their existing numeric
paths. MTP still starts fresh for each request; this does not introduce prompt
prefix reuse or establish a particular M5 time-to-first-token improvement.

## Function calls and client execution

**OpenCode executes tools.** The server supplies tool definitions to the model,
parses its generated call, and returns a function name and JSON arguments. It
does not read files, launch commands, or execute a function named in a response.
The client decides how to execute the call and submits its result as history.

Supply up to 128 `type:"function"` definitions with unique names, optional
descriptions and object-shaped `parameters`. Omitted parameters become an empty
object schema. `tool_choice` accepts `auto`, `none`, `required`, or
`{"type":"function","function":{"name":"read_file"}}`. The default is `auto`
when definitions exist, otherwise `none`. A named choice must refer to a
supplied function. `parallel_tool_calls` defaults to true; false permits at
most one generated call in the response.

The output parser recognizes the checkpoint's Qwen XML tool frames and JSON
tool frames. It withholds each frame until complete, then checks its function
name, object arguments, required properties, top-level property types and
`additionalProperties:false`. Arguments are returned as a JSON-encoded string.
Malformed/incomplete calls, forbidden functions, missing required calls and
multiple calls with parallel calls disabled cause a generation error.

These are **post-generation checks**, not grammar-constrained decoding or a
complete JSON Schema validator. Nested constraints, references, enums and other
schema keywords are not guaranteed. `function.strict:true` is rejected;
false, null or omission is accepted. `required` and named choices add model
instructions and validate the result; they cannot guarantee the model complies.

For the next turn, retain the assistant's `tool_calls` and append a `role:"tool"`
message for each call with the matching `tool_call_id` and textual result.
Assistant tool-call content may be empty or null. Historical arguments must be
a JSON object encoded as a string. IDs must be unique within the submitted
history, every pending call must receive exactly one result before another
conversation message, and a request cannot end with unanswered calls.

## Streaming, JSON output and errors

SSE begins with an assistant-role chunk. Text uses `delta.content`; a preopened
thinking phase is separated into `delta.reasoning_content`. Validated calls are
emitted as complete `delta.tool_calls` entries with stable IDs, indices, names
and JSON-string arguments. A normally finished call response uses
`finish_reason:"tool_calls"`; ordinary completion uses `stop`, and token-limit
completion uses `length`. Streaming ends with `[DONE]`.

`stream_options.include_usage:true` adds null usage to ordinary chunks and a
separate final usage chunk with `choices:[]`. OpenCode enables this option by
default. `include_obfuscation` defaults to true and adds random padding in an
`obfuscation` field on choice chunks; false disables it. Non-streaming responses
include usage, and completed responses also expose local timing fields. Usage
counts model-generated tokens, including structural text; it is not inferred
from visible characters after stop/tool filtering.

JSON-object mode adds an instruction asking for one JSON object, buffers output
and validates the final content before releasing it. Even with `stream:true`,
JSON content is sent only after validation. Invalid JSON or a non-object result
causes a generation error. This is **prompting plus post-validation**, not
constrained decoding. JSON-object mode cannot be combined with enabled tool
calls; `response_format:{"type":"json_schema",...}` is unsupported.

Unsupported request fields/values normally return HTTP 400; an unknown model
returns 404. Errors use `{"error":{"message":...,"type":...,"param":...,"code":...}}`.
Context preparation and tokenization happen before successful SSE headers. A
prompt with no room for output returns HTTP 400 with
`code:"context_length_exceeded"`; an oversized positive output upper bound is
reduced to the remaining space. Generation
failures return HTTP 500 for non-streaming requests or an SSE error followed by
`[DONE]` after streaming has started. A full/stopped worker queue returns 503.
Request metadata is not stored in a database or echoed as a stored completion;
`store:true` is unsupported.

Current OpenCode's [API-error classifier](https://github.com/anomalyco/opencode/blob/dev/packages/opencode/src/provider/error.ts)
recognizes `error.code:"context_length_exceeded"` from the JSON response body;
it does not rely solely on matching the human-readable message.

The public [OpenAI Chat Completions reference](https://developers.openai.com/api/reference/resources/chat)
defines the broader protocol. The table and limits above describe this engine's
implemented subset.

## Reproduce the SDK contract test

The [Node integration script](../tests/opencode_sdk_contract.cjs) loads the real,
exact **`@ai-sdk/openai-compatible@2.0.41`** adapter. Its package version uses the
LanguageModelV3 interface. The Rust test starts the actual HTTP router on an
ephemeral loopback port with a deterministic generator, then invokes Node.

From the repository root, with Node and npm available:

```sh
mkdir -p results/opencode-sdk
npm install --prefix "$PWD/results/opencode-sdk" --save-exact --ignore-scripts \
  @ai-sdk/openai-compatible@2.0.41
OPENCODE_SDK_DIR="$PWD/results/opencode-sdk" \
  cargo test --locked --lib server::tests::opencode_sdk_tool_round_trip \
  -- --ignored --exact --nocapture --test-threads=1
```

If Node is not on `PATH`, set `NODE_BINARY` to its absolute executable path on
the Cargo command. The Rust test supplies `BASE_URL` and `MODEL_ID=test-model`
to the script; no trained model or GPU execution is needed for this HTTP fixture.

The assertions cover two system messages, multipart user text, actual SDK
request serialization, streamed tool-call lifecycle and usage, matching call-ID
history, and non-streaming follow-ups. Five HTTP requests cover ordinary output,
reasoning before a tool call, and preserved reasoning in both native SDK parts
and explicit OpenCode-shaped message metadata. This SDK test does not itself
execute OpenCode's history transform. The fixture emits a `read_file` call for
`README.md`, receives an in-memory result, and answers `Done.`. No file tool is
executed. Fixed token counts of 12 input and 7 output tokens verify usage
transport; they are not model measurements.

This fixture does not establish the trained Qwen3.8-27B model's tool selection, argument
quality or coding-agent reliability. Those require an additional run using the
real checkpoint and its unchanged chat template. The pinned adapter's
[message converter](https://github.com/vercel/ai/blob/%40ai-sdk%2Fopenai-compatible%402.0.41/packages/openai-compatible/src/chat/convert-to-openai-compatible-chat-messages.ts)
and [stream parser](https://github.com/vercel/ai/blob/%40ai-sdk%2Fopenai-compatible%402.0.41/packages/openai-compatible/src/chat/openai-compatible-chat-language-model.ts)
are the client-side contract references.

## Reproduce the installed OpenCode reasoning test

With Node.js and OpenCode on `PATH`, run:

```sh
cargo test --locked --lib server::tests::opencode_cli_reasoning_and_tool_history \
  -- --ignored --exact --nocapture --test-threads=1
```

`NODE_BINARY` and `OPENCODE_BINARY` can supply absolute executable paths.
The test starts the real HTTP router with deterministic generated text. The
[CLI harness](../tests/opencode_reasoning_contract.cjs) uses the example config,
an isolated temporary project and XDG data directories. It uses the example's
disabled title agent and checks that only the two main tool-round requests are
made. It disables external plugins and automatic updates, restricts provider
selection to the loopback fixture, and allows only a read of its project files.
OpenCode reads the fixture README, sends its result and prior reasoning back,
and receives a second reasoning part plus `Done: 42.` as separate answer text.
The Rust fixture verifies the incoming effort and exact reasoning history.

The harness checks live reasoning/tool events and independently exports the
saved session to verify both completed reasoning parts, tool output, final
answer, and completion status. Installed OpenCode **1.15.12** exhibited an
NDJSON event-drain race: it exited before printing the final reasoning/text
events even though its session contained the complete result. The report
records `final_cli_events_complete:false` for that case. This is not a claim
that every final CLI event was displayed, or a visual TUI test. No artificial
generation delays are used to conceal it. The inspected CLI starts its event
consumer without awaiting it before returning and disposing the instance:
[CLI lifecycle](https://github.com/anomalyco/opencode/blob/58a27b95c155d3f7d9b9f25b30eb1233bfb0eae5/packages/opencode/src/cli/cmd/run.ts#L768-L803),
[instance disposal](https://github.com/anomalyco/opencode/blob/58a27b95c155d3f7d9b9f25b30eb1233bfb0eae5/packages/opencode/src/cli/effect-cmd.ts#L84-L92).

This test verifies actual client reasoning reception, same-model history
preservation, tool execution and persistence. It uses deterministic fixture
output, so it does not measure the trained 27B model's reasoning quality.

## Recorded validation: 0.7.0

- `cargo test --locked --all-targets`: **135 passed**, zero failed, 45 ignored
  (121 library, 12 binary and two CLI tests passed).
- The serial ignored Metal suite, excluding the separate SDK fixture:
  **44 distinct GPU tests passed** on M1. A nested subprocess reruns one test;
  it is not counted twice.
- The exact SDK HTTP round-trip test above: **one passed**, zero failed,
  rerun against version 0.7.0.
- `cargo build --release --locked`, formatting and diff checks passed.

The GPU evidence uses synthetic model/matrix fixtures on M1. The SDK evidence
uses the real adapter and HTTP server with deterministic generated text. Neither
is a trained-27B agent evaluation or a new M5 throughput result.

## Recorded validation: 0.7.1

- **136 non-GPU tests** and **45 distinct Metal tests** passed.
- The real-engine HTTP regression uses a four-token prompt in a 520-token
  synthetic context. Both ordinary and MTP generation produce 516 tokens for
  omitted/null limits, 8192, and the largest unsigned 64-bit integer. An explicit
  limit of 8 produces 8 tokens. A full prompt returns HTTP 400 before SSE starts.
- The exact SDK round trip passed with `maxOutputTokens:8192`, exercising a
  request above the old 4096-token cap.
- Installed OpenCode **1.15.12** accepted the example through
  `opencode debug config --pure` with isolated XDG configuration/state paths.
  The resolved context/input/output limits were all 8192 and the compaction
  reserve was 1024. This is configuration validation, not a trained-model run.

## Recorded validation: 0.7.2

- **136 non-GPU tests** passed; the two external-client tests run separately.
- The exact compatible SDK passed all five HTTP requests, including streamed
  reasoning, non-streaming reasoning, and both history representations.
- Installed **OpenCode 1.15.12** completed the isolated read-tool conversation.
  Its saved session contained `I should read README.md.` and
  `The file is available.` as reasoning, and `Done: 42.` as separate answer text.
  The server verified the prior reasoning arrived in the tool-result turn.
- The CLI emitted the first reasoning/tool events but omitted the final NDJSON
  tail; the report recorded `final_cli_events_complete:false`. Complete session
  assertions used OpenCode's export, as described above.

The engine's existing reasoning wire format required no change. This update
corrects client configuration and adds integration coverage; it does not
establish trained-model reasoning quality or a new performance result.
