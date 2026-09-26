#!/usr/bin/env node
'use strict';

// Runs the installed OpenCode CLI against a deterministic loopback Rust server.
// This exercises client configuration, reasoning events, and a real read tool
// in an isolated temporary project. It does not evaluate trained model quality.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { spawnSync } = require('node:child_process');

const project = process.env.OPENCODE_TEST_PROJECT;
assert.ok(project && path.isAbsolute(project), 'absolute temporary project required');
const endpoint = new URL(process.env.BASE_URL);
assert.equal(endpoint.protocol, 'http:');
assert.equal(endpoint.hostname, '127.0.0.1');
const binary = process.env.OPENCODE_BINARY || 'opencode';
const config = JSON.parse(fs.readFileSync(path.join(__dirname, '../examples/opencode.json'), 'utf8'));
config.provider['qwen-metal'].options.baseURL = endpoint.href.replace(/\/$/, '');
// The fixture requests only a read of this test's own file.
config.permission = { '*': 'deny', read: 'allow' };
config.enabled_providers = ['qwen-metal'];
config.autoupdate = false;
fs.writeFileSync(path.join(project, 'opencode.json'), JSON.stringify(config));
fs.writeFileSync(path.join(project, 'README.md'), 'Fixture document: 42.\n');
const env = {
  ...process.env,
  XDG_CONFIG_HOME: path.join(project, 'config'),
  XDG_DATA_HOME: path.join(project, 'data'),
  XDG_CACHE_HOME: path.join(project, 'cache'),
  XDG_STATE_HOME: path.join(project, 'state'),
  OPENCODE_CONFIG: path.join(project, 'opencode.json'),
};
const version = spawnSync(binary, ['--version'], { env, encoding: 'utf8', timeout: 10000 });
assert.equal(version.status, 0, version.stderr || String(version.error));
const result = spawnSync(binary, [
  'run', '--pure', '--format', 'json', '--thinking',
  '--dir', project,
  '--model', config.model,
  `Read ${path.join(project, 'README.md')} and report its number.`,
], { cwd: project, env, encoding: 'utf8', timeout: 60000, maxBuffer: 8 * 1024 * 1024 });
assert.equal(result.status, 0, `${result.error || ''}\n${result.stderr}\n${result.stdout}`);
const events = result.stdout.split(/\r?\n/).filter(line => line.trim()).map(line => JSON.parse(line));
assert.equal(events.some(event => event.type === 'error'), false, result.stdout);
const sessionID = events[0]?.sessionID;
assert.ok(sessionID, result.stdout);
const exported = spawnSync(binary, ['export', sessionID, '--pure'], {
  cwd: project, env, encoding: 'utf8', timeout: 15000, maxBuffer: 8 * 1024 * 1024,
});
assert.equal(exported.status, 0, exported.stderr || String(exported.error));
const session = JSON.parse(exported.stdout);
assert.equal(fs.realpathSync(session.info.directory), project);
const assistant = session.messages.filter(message => message.info.role === 'assistant');
const parts = assistant.flatMap(message => message.parts);
const reasoning = parts.filter(part => part.type === 'reasoning').map(part => part.text);
assert.deepEqual(reasoning, ['I should read README.md.', 'The file is available.']);
assert.ok(parts.filter(part => part.type === 'reasoning').every(part => part.time.end >= part.time.start));
const text = parts.filter(part => part.type === 'text').map(part => part.text).join('');
assert.equal(text, 'Done: 42.');
assert.doesNotMatch(text, /<\/?think>|I should read|The file is available/);
assert.equal(assistant.at(-1).info.finish, 'stop');
assert.ok(assistant.at(-1).info.time.completed);
// OpenCode 1.15.12 can return before its unawaited NDJSON consumer drains the
// final events. Verify complete persisted state independently; do not hide the
// missing live tail by adding sleeps to the fixture or claiming it was emitted.
const liveReasoning = events.filter(event => event.type === 'reasoning').map(event => event.part.text);
assert.ok(liveReasoning.length >= 1, result.stdout);
assert.deepEqual(liveReasoning, reasoning.slice(0, liveReasoning.length));
const liveText = events.filter(event => event.type === 'text').map(event => event.part.text).join('');
assert.ok(text.startsWith(liveText));
const finalStepVisible = events.some(event => event.type === 'step_finish'
  && event.part.reason === 'stop' && event.part.messageID === assistant.at(-1).info.id);
const tools = events.filter(event => event.type === 'tool_use');
assert.equal(tools.length, 1, result.stdout);
assert.equal(tools[0].part.tool, 'read');
assert.equal(tools[0].part.state.status, 'completed', result.stdout);
const savedTools = parts.filter(part => part.type === 'tool');
assert.equal(savedTools.length, 1);
assert.equal(savedTools[0].state.status, 'completed');
assert.equal(savedTools[0].state.input.filePath, path.join(project, 'README.md'));
assert.match(savedTools[0].state.output, /Fixture document: 42\./);
console.log(JSON.stringify({
  kind: 'opencode_cli_reasoning_contract', passed: true,
  opencode_version: version.stdout.trim(), event_types: events.map(event => event.type),
  reasoning, text, completed_tool: 'read', tool_scope: 'temporary fixture README.md',
  live_reasoning: liveReasoning, live_text: liveText,
  final_cli_events_complete: liveReasoning.length === reasoning.length && liveText === text && finalStepVisible,
  complete_session_verified_by: 'OpenCode export from isolated session database',
}, null, 2));
