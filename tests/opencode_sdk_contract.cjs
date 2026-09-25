#!/usr/bin/env node
'use strict';

// Real HTTP contract test for OpenCode's pinned adapter, not a mock SDK.
// Root supplies a running deterministic Rust fixture server and installs the SDK:
// OPENCODE_SDK_DIR=/absolute/sdk/project BASE_URL=http://127.0.0.1:PORT/v1 \
// MODEL_ID=test-model node tests/opencode_sdk_contract.cjs
// The fixture emits a read_file(path="README.md") call, then "Done." after a
// matching tool-result message; each response reports 12 input / 7 output tokens.
// With reasoningEffort="medium", it also emits the two reasoning strings below.
// No tools are executed: the tool result below is an in-memory fixture string.
// Package 2.0.41 implements LanguageModelV3, despite its package major version.

const assert = require('node:assert/strict');
const path = require('node:path');
const { createRequire } = require('node:module');

const SDK_NAME = '@ai-sdk/openai-compatible';
const SDK_VERSION = '2.0.41';
const XML_MARKER = /<\/?(?:tool_call|function|parameter)\b/i;
const THINK_MARKER = /<\/?think\s*>/i;
const TOOL_REASONING = 'I should read README.md.';
const FOLLOWUP_REASONING = 'The file is available.';

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

async function collectStream(stream) {
  const reader = stream.getReader();
  const events = [];
  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) return events;
      assert.notEqual(value.type, 'error', `SDK stream error: ${String(value.error?.stack || value.error)}`);
      events.push(value);
    }
  } finally {
    reader.releaseLock();
  }
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
    const events = await collectStream(streamed.stream);

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

    // The request namespace follows the provider name. The SDK maps the known
    // camelCase option to the server's reasoning_effort wire field.
    const thinkingOptions = {
      ...options,
      providerOptions: { 'qwen-metal': { reasoningEffort: 'medium' } },
    };
    const thinkingStream = await model.doStream(thinkingOptions);
    const thinkingEvents = await collectStream(thinkingStream.stream);
    assert.equal(requests.length, 3, 'thinking stream should issue one HTTP request');
    assert.deepEqual(requests[2], { ...firstRequest, reasoning_effort: 'medium' },
      'provider reasoningEffort must become reasoning_effort without other request changes');
    const reasoningEvents = thinkingEvents.filter(event => event.type.startsWith('reasoning-'));
    assert.equal(reasoningEvents[0]?.type, 'reasoning-start');
    assert.equal(reasoningEvents.at(-1)?.type, 'reasoning-end');
    assert.equal(reasoningEvents.filter(event => event.type === 'reasoning-start').length, 1);
    assert.equal(reasoningEvents.filter(event => event.type === 'reasoning-end').length, 1);
    for (const event of reasoningEvents) {
      assert.equal(event.id, reasoningEvents[0].id, 'reasoning stream ID must remain stable');
    }
    const reasoningText = reasoningEvents.filter(event => event.type === 'reasoning-delta')
      .map(event => event.delta).join('');
    assert.equal(reasoningText, TOOL_REASONING, 'reasoning must remain separate from tool-call content');
    assert.equal(thinkingEvents.filter(event => event.type === 'text-delta')
      .map(event => event.delta).join(''), '', 'thinking tool call must not leak reasoning into assistant text');
    const reasoningEnd = thinkingEvents.findIndex(event => event.type === 'reasoning-end');
    const thinkingToolStart = thinkingEvents.findIndex(event => event.type === 'tool-input-start');
    assert.ok(reasoningEnd >= 0 && reasoningEnd < thinkingToolStart,
      'SDK must close reasoning before opening tool input');
    const thinkingCalls = thinkingEvents.filter(event => event.type === 'tool-call');
    assert.equal(thinkingCalls.length, 1, 'thinking stream must finish one tool call');
    const thinkingCall = thinkingCalls[0];
    assert.equal(thinkingCall.toolName, 'read_file');
    assert.equal(typeof thinkingCall.toolCallId, 'string');
    assert.ok(thinkingCall.toolCallId.length > 0);
    assert.equal(typeof thinkingCall.input, 'string');
    const thinkingArguments = JSON.parse(thinkingCall.input);
    assert.deepEqual(thinkingArguments, { path: 'README.md' });
    assert.equal(thinkingEvents.filter(event => event.type === 'tool-input-start').length, 1);
    assert.equal(thinkingEvents.filter(event => event.type === 'tool-input-end').length, 1);
    for (const event of thinkingEvents.filter(event => event.type.startsWith('tool-input-'))) {
      assert.equal(event.id, thinkingCall.toolCallId, `unstable thinking call ID in ${event.type}`);
    }
    assert.equal(thinkingEvents.filter(event => event.type === 'tool-input-delta')
      .map(event => event.delta).join(''), thinkingCall.input);
    const thinkingFinishes = thinkingEvents.filter(event => event.type === 'finish');
    assert.equal(thinkingFinishes.length, 1);
    assert.deepEqual(thinkingFinishes[0].finishReason, { unified: 'tool-calls', raw: 'tool_calls' });
    assertUsage(thinkingFinishes[0].usage, 'thinking stream');

    const thinkingToolPart = {
      type: 'tool-call',
      toolCallId: thinkingCall.toolCallId,
      toolName: thinkingCall.toolName,
      input: thinkingArguments,
    };
    const thinkingToolResult = {
      role: 'tool',
      content: [{
        type: 'tool-result',
        toolCallId: thinkingCall.toolCallId,
        toolName: thinkingCall.toolName,
        output: { type: 'text', value: fakeToolResult },
      }],
    };
    const historyCases = [
      {
        name: 'native-sdk-reasoning-part',
        message: { role: 'assistant', content: [{ type: 'reasoning', text: reasoningText }, thinkingToolPart] },
      },
      {
        name: 'opencode-shaped-message-metadata',
        // This explicitly supplies the shape produced by OpenCode's interleaved
        // reasoning_content transform. The SDK contract does not run OpenCode.
        // Message metadata uses openaiCompatible, not the request namespace.
        message: {
          role: 'assistant',
          content: [thinkingToolPart],
          providerOptions: { openaiCompatible: { reasoning_content: reasoningText } },
        },
      },
    ];
    const reasoningFollowups = [];
    for (const historyCase of historyCases) {
      const result = await model.doGenerate({
        ...thinkingOptions,
        prompt: [...prompt, historyCase.message, thinkingToolResult],
      });
      const request = requests.at(-1);
      assert.deepEqual(request, {
        ...secondRequest,
        reasoning_effort: 'medium',
        messages: [
          ...firstRequest.messages,
          {
            role: 'assistant',
            content: '',
            reasoning_content: TOOL_REASONING,
            tool_calls: [{
              id: thinkingCall.toolCallId,
              type: 'function',
              function: { name: 'read_file', arguments: JSON.stringify(thinkingArguments) },
            }],
          },
          { role: 'tool', tool_call_id: thinkingCall.toolCallId, content: fakeToolResult },
        ],
      }, `${historyCase.name}: reasoning and matching tool history must survive SDK serialization`);
      // doGenerate places text before reasoning; assert types separately so the
      // contract checks semantic separation without imposing streaming order.
      assert.equal(result.content.length, 2, `${historyCase.name}: unexpected generated content`);
      assert.deepEqual(result.content.filter(part => part.type === 'text'), [{ type: 'text', text: 'Done.' }]);
      assert.deepEqual(result.content.filter(part => part.type === 'reasoning'),
        [{ type: 'reasoning', text: FOLLOWUP_REASONING }]);
      assert.deepEqual(result.finishReason, { unified: 'stop', raw: 'stop' });
      assertUsage(result.usage, historyCase.name);
      reasoningFollowups.push({ history_shape: historyCase.name, content: result.content,
        finish: result.finishReason, usage: result.usage });
    }
    assert.equal(requests.length, 5, 'all scenarios should issue exactly five HTTP requests');

    const wireBodies = await Promise.all(responseBodies);
    assert.equal(wireBodies.length, 5);
    for (const [index, body] of wireBodies.entries()) {
      assert.doesNotMatch(body, XML_MARKER, `raw model tool markup leaked into HTTP response ${index + 1}`);
      assert.doesNotMatch(body, THINK_MARKER, `raw thinking delimiter leaked into HTTP response ${index + 1}`);
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
      reasoning_stream_event_types: thinkingEvents.map(event => event.type),
      reasoning_stream_text: reasoningText,
      reasoning_stream_finish: thinkingFinishes[0].finishReason,
      reasoning_stream_usage: thinkingFinishes[0].usage,
      reasoning_history_followups: reasoningFollowups,
      opencode_transform_executed: false,
      raw_tool_markup_leaked: false,
      raw_thinking_markup_leaked: false,
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
