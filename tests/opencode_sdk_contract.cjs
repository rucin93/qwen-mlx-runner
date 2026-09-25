#!/usr/bin/env node
'use strict';

// Real HTTP contract test for OpenCode's pinned adapter, not a mock SDK.
// Root supplies a running deterministic Rust fixture server and installs the SDK:
// OPENCODE_SDK_DIR=/absolute/sdk/project BASE_URL=http://127.0.0.1:PORT/v1 \
// MODEL_ID=test-model node tests/opencode_sdk_contract.cjs
// The fixture emits a read_file(path="README.md") call, then "Done." after a
// matching tool-result message; each response reports 12 input / 7 output tokens.
// No tools are executed: the tool result below is an in-memory fixture string.
// Package 2.0.41 implements LanguageModelV3, despite its package major version.

const assert = require('node:assert/strict');
const path = require('node:path');
const { createRequire } = require('node:module');

const SDK_NAME = '@ai-sdk/openai-compatible';
const SDK_VERSION = '2.0.41';
const XML_MARKER = /<\/?(?:tool_call|function|parameter)\b/i;

function requiredEnvironment(name) {
  const value = process.env[name];
  assert.ok(value, `${name} must be set`);
  return value;
}

function assertUsage(usage, label) {
  assert.equal(usage?.inputTokens?.total, 12, `${label}: input-token usage`);
  assert.equal(usage?.outputTokens?.total, 7, `${label}: output-token usage`);
  assert.equal(usage?.raw?.prompt_tokens, 12, `${label}: raw prompt usage`);
  assert.equal(usage?.raw?.completion_tokens, 7, `${label}: raw completion usage`);
}

async function main() {
  const sdkDirectory = requiredEnvironment('OPENCODE_SDK_DIR');
  assert.ok(path.isAbsolute(sdkDirectory), 'OPENCODE_SDK_DIR must be absolute');
  const sdkRequire = createRequire(path.join(sdkDirectory, 'package.json'));
  const sdkPackage = sdkRequire(`${SDK_NAME}/package.json`);
  assert.equal(sdkPackage.version, SDK_VERSION, 'contract requires the exact pinned SDK');
  const { createOpenAICompatible } = sdkRequire(SDK_NAME);

  const endpoint = new URL(requiredEnvironment('BASE_URL'));
  assert.equal(endpoint.protocol, 'http:', 'fixture URL must use loopback HTTP');
  assert.ok(['127.0.0.1', 'localhost', '[::1]'].includes(endpoint.hostname),
    'fixture URL must point to loopback');
  assert.equal(endpoint.pathname.replace(/\/$/, ''), '/v1', 'BASE_URL must end in /v1');
  assert.equal(endpoint.search, '', 'fixture URL must have no query');
  assert.equal(endpoint.hash, '', 'fixture URL must have no fragment');
  assert.equal(endpoint.username + endpoint.password, '', 'fixture URL must contain no credentials');
  const baseURL = endpoint.href.replace(/\/$/, '');
  const modelId = process.env.MODEL_ID || 'Qwen3.8-27B-4bit';

  assert.equal(typeof globalThis.fetch, 'function', 'Node with built-in fetch is required');
  const fetchHTTP = globalThis.fetch.bind(globalThis);
  const requests = [];
  const responseBodies = [];
  const provider = createOpenAICompatible({
    name: 'qwen-metal',
    baseURL,
    includeUsage: true,
    // Observe the actual adapter serialization while retaining real HTTP I/O.
    fetch: async (input, init) => {
      const url = new URL(typeof input === 'string' ? input : input.url || input.href);
      assert.equal(url.origin, endpoint.origin, 'adapter changed fixture origin');
      assert.equal(url.pathname, '/v1/chat/completions', 'adapter endpoint');
      assert.equal(init?.method, 'POST', 'adapter HTTP method');
      assert.equal(typeof init?.body, 'string', 'adapter must serialize JSON');
      requests.push(JSON.parse(init.body));
      const response = await fetchHTTP(input, init);
      const responseBody = response.clone().text();
      // A later assertion may abort the request before the wire body is awaited.
      responseBody.catch(() => {});
      responseBodies.push(responseBody);
      return response;
    },
  });
  const model = provider.chatModel(modelId);
  assert.equal(model.specificationVersion, 'v3', 'pinned adapter specification version');

  const inputSchema = {
    type: 'object',
    properties: { path: { type: 'string' } },
    required: ['path'],
    additionalProperties: false,
  };
  const tools = [{
    type: 'function',
    name: 'read_file',
    description: 'Read one file from the current project.',
    inputSchema,
  }];
  const prompt = [
    { role: 'system', content: 'You are a local coding assistant.' },
    { role: 'system', content: 'Use the supplied tools when file contents are needed.' },
    {
      role: 'user',
      content: [
        { type: 'text', text: 'Read README.md. ' },
        { type: 'text', text: 'After receiving its contents, answer Done.' },
      ],
    },
  ];
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(new Error('contract test timed out after 60s')), 60_000);
  try {
    const options = {
      prompt,
      tools,
      toolChoice: { type: 'auto' },
      maxOutputTokens: 8192,
      abortSignal: controller.signal,
    };
    const streamed = await model.doStream(options);
    const reader = streamed.stream.getReader();
    const events = [];
    try {
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        assert.notEqual(value.type, 'error', `SDK stream error: ${String(value.error?.stack || value.error)}`);
        events.push(value);
      }
    } finally {
      reader.releaseLock();
    }

    assert.equal(requests.length, 1, 'first operation should issue one HTTP request');
    const firstRequest = requests[0];
    assert.equal(firstRequest.model, modelId);
    assert.equal(firstRequest.max_tokens, 8192);
    assert.equal(firstRequest.stream, true);
    assert.deepEqual(firstRequest.stream_options, { include_usage: true });
    assert.equal(firstRequest.tool_choice, 'auto');
    assert.deepEqual(firstRequest.tools, [{
      type: 'function',
      function: { name: 'read_file', description: tools[0].description, parameters: inputSchema },
    }]);
    assert.deepEqual(firstRequest.messages, [
      prompt[0], prompt[1], { role: 'user', content: prompt[2].content },
    ], 'adapter must preserve both system messages and multipart text');

    const text = events.filter(event => event.type === 'text-delta').map(event => event.delta).join('');
    assert.doesNotMatch(text, XML_MARKER, 'raw model tool markup leaked into text');
    assert.equal(text.trim(), '', 'tool-only fixture should produce no assistant prose');
    const calls = events.filter(event => event.type === 'tool-call');
    assert.equal(calls.length, 1, 'expected exactly one completed tool call');
    const call = calls[0];
    assert.equal(call.toolName, 'read_file');
    assert.equal(typeof call.toolCallId, 'string');
    assert.ok(call.toolCallId.length > 0, 'tool call needs a stable nonempty ID');
    assert.equal(typeof call.input, 'string', 'SDK tool input must be JSON text');
    const argumentsObject = JSON.parse(call.input);
    assert.deepEqual(argumentsObject, { path: 'README.md' });

    const starts = events.filter(event => event.type === 'tool-input-start');
    const ends = events.filter(event => event.type === 'tool-input-end');
    assert.equal(starts.length, 1, 'expected one tool-input-start');
    assert.equal(ends.length, 1, 'expected one tool-input-end');
    assert.equal(starts[0].toolName, call.toolName);
    for (const event of events.filter(event => event.type.startsWith('tool-input-'))) {
      assert.equal(event.id, call.toolCallId, `unstable call ID in ${event.type}`);
    }
    const streamedInput = events.filter(event => event.type === 'tool-input-delta').map(event => event.delta).join('');
    assert.equal(streamedInput, call.input, 'argument fragments must reconstruct the completed call');
    const finishes = events.filter(event => event.type === 'finish');
    assert.equal(finishes.length, 1, 'expected one terminal SDK finish event');
    assert.deepEqual(finishes[0].finishReason, { unified: 'tool-calls', raw: 'tool_calls' });
    assertUsage(finishes[0].usage, 'stream');

    const fakeToolResult = '# Fixture README\nThis text is supplied by the test; no file was read.';
    const followup = await model.doGenerate({
      ...options,
      prompt: [
        ...prompt,
        {
          role: 'assistant',
          content: [{ type: 'tool-call', toolCallId: call.toolCallId, toolName: call.toolName, input: argumentsObject }],
        },
        {
          role: 'tool',
          content: [{
            type: 'tool-result',
            toolCallId: call.toolCallId,
            toolName: call.toolName,
            output: { type: 'text', value: fakeToolResult },
          }],
        },
      ],
    });
    assert.equal(requests.length, 2, 'follow-up should issue exactly one more HTTP request');
    const secondRequest = requests[1];
    assert.equal(secondRequest.model, modelId);
    assert.equal(secondRequest.max_tokens, 8192);
    assert.equal(secondRequest.stream, undefined, 'doGenerate must use non-streaming HTTP');
    assert.equal(secondRequest.stream_options, undefined);
    assert.equal(secondRequest.messages.length, 5);
    assert.deepEqual(secondRequest.messages.slice(0, 3), firstRequest.messages);
    const assistant = secondRequest.messages[3];
    assert.equal(assistant.role, 'assistant');
    assert.equal(assistant.content, '', 'tool-only history uses empty assistant content');
    assert.equal(assistant.tool_calls.length, 1);
    assert.deepEqual(assistant.tool_calls[0], {
      id: call.toolCallId,
      type: 'function',
      function: { name: 'read_file', arguments: JSON.stringify(argumentsObject) },
    });
    assert.deepEqual(secondRequest.messages[4], {
      role: 'tool', tool_call_id: call.toolCallId, content: fakeToolResult,
    }, 'tool-result history must match the emitted call ID');
    assert.deepEqual(followup.content, [{ type: 'text', text: 'Done.' }]);
    assert.deepEqual(followup.finishReason, { unified: 'stop', raw: 'stop' });
    assertUsage(followup.usage, 'follow-up');

    const wireBodies = await Promise.all(responseBodies);
    assert.equal(wireBodies.length, 2);
    for (const [index, body] of wireBodies.entries()) {
      assert.doesNotMatch(body, XML_MARKER, `raw model tool markup leaked into HTTP response ${index + 1}`);
    }
    console.log(JSON.stringify({
      kind: 'opencode_sdk_http_contract',
      passed: true,
      sdk: SDK_NAME,
      sdk_version: sdkPackage.version,
      specification_version: model.specificationVersion,
      model: modelId,
      requests: requests.length,
      stream_event_types: events.map(event => event.type),
      tool_call_id: call.toolCallId,
      tool_name: call.toolName,
      tool_arguments: argumentsObject,
      stream_finish: finishes[0].finishReason,
      followup_finish: followup.finishReason,
      stream_usage: finishes[0].usage,
      followup_usage: followup.usage,
      followup_text: 'Done.',
      raw_tool_markup_leaked: false,
      tools_executed: 0,
    }, null, 2));
  } finally {
    clearTimeout(timeout);
    controller.abort();
  }
}

main().catch(error => {
  console.error(error?.stack || String(error));
  process.exitCode = 1;
});
