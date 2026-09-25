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
tasks such as title generation. The adapter is **`@ai-sdk/openai-compatible`**,
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

Keep `limit.context` equal to the server's actual `--context`. The example sets
`limit.output` to 2048, below the server's per-request cap of 4096. Prompt,
template, tool definitions, history and requested output must fit the context.
OpenCode's output-limit fallback can otherwise request 32,000 tokens, so do not
omit the output limit. A larger value in client configuration does not enlarge
the server's caches. [OpenCode output-limit calculation](https://github.com/anomalyco/opencode/blob/dev/packages/opencode/src/provider/transform.ts).

## Accepted requests

Optional `null` values are generally treated as omitted. `model` and a nonempty
`messages` array remain required. Unknown top-level fields are rejected. The
HTTP request-body limit is 1 MiB.

| Field | Implemented behavior |
| --- | --- |
| `model` | Must match the served model ID. |
| `messages` | Text messages with roles `system`, `developer`, `user`, `assistant`, or `tool`; at least one user message is required. |
| `max_tokens`, `max_completion_tokens` | Alternative output limits, integer 1–4096; default 256. Supply at most one non-null limit. |
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

Message content may be a string or an array of `{"type":"text","text":"..."}`
parts, concatenated in order. Images, audio, files and other content-part types
are rejected. Initial system/developer messages must precede conversation
messages; they are adapted to the checkpoint's initial system prompt.
Optional message `name` must contain 1–64 ASCII letters, digits, `_` or `-`.
`reasoning_content` is accepted only on assistant history and is passed through
the checkpoint's history adapter.

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
Context validation and tokenization happen before successful SSE headers, so
context overflow is HTTP 400 with `code:"context_length_exceeded"`. Generation
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
history, and a non-streaming follow-up. The fixture emits a `read_file` call for
`README.md`, receives an in-memory result, and answers `Done.`. No file tool is
executed. Fixed token counts of 12 input and 7 output tokens verify usage
transport; they are not model measurements.

This fixture does not establish the trained Qwen3.8-27B model's tool selection, argument
quality or coding-agent reliability. Those require an additional run using the
real checkpoint and its unchanged chat template. The pinned adapter's
[message converter](https://github.com/vercel/ai/blob/%40ai-sdk%2Fopenai-compatible%402.0.41/packages/openai-compatible/src/chat/convert-to-openai-compatible-chat-messages.ts)
and [stream parser](https://github.com/vercel/ai/blob/%40ai-sdk%2Fopenai-compatible%402.0.41/packages/openai-compatible/src/chat/openai-compatible-chat-language-model.ts)
are the client-side contract references.

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
