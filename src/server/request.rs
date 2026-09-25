use std::collections::{BTreeMap, BTreeSet};

use serde_json::{Map, Value};

use crate::chat::{
    GenerationRequest, Message,
    tools::{ToolCall, ToolConfig},
};

#[derive(Clone, Copy, Debug)]
pub(super) struct StreamOptions {
    pub include_usage: bool,
    pub include_obfuscation: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ResponseFormat {
    Text,
    JsonObject,
}

pub(super) struct ParsedRequest {
    pub generation: GenerationRequest,
    pub stream: bool,
    pub stream_options: StreamOptions,
    pub stop: Vec<String>,
    pub response_format: ResponseFormat,
    pub service_tier: bool,
}

#[derive(Debug)]
pub(super) struct RequestError {
    pub message: String,
    pub param: String,
    pub code: &'static str,
}

impl RequestError {
    fn invalid(param: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            param: param.into(),
            code: "invalid_value",
        }
    }
    fn unsupported(param: impl Into<String>) -> Self {
        let param = param.into();
        Self {
            message: format!("unsupported parameter or value: {param}"),
            param,
            code: "unsupported_parameter",
        }
    }
}

type Result<T> = std::result::Result<T, RequestError>;
fn supplied<'a>(object: &'a Map<String, Value>, key: &str) -> Option<&'a Value> {
    object.get(key).filter(|v| !v.is_null())
}
fn boolean(object: &Map<String, Value>, key: &str, default: bool) -> Result<bool> {
    supplied(object, key).map_or(Ok(default), |v| {
        v.as_bool()
            .ok_or_else(|| RequestError::invalid(key, format!("{key} must be a boolean")))
    })
}
fn number(object: &Map<String, Value>, key: &str, default: f64, min: f64, max: f64) -> Result<f32> {
    let value = supplied(object, key)
        .map_or(Some(default), Value::as_f64)
        .ok_or_else(|| RequestError::invalid(key, format!("{key} must be a number")))?;
    if !value.is_finite() || value < min || value > max {
        return Err(RequestError::invalid(
            key,
            format!("{key} must be in [{min}, {max}]"),
        ));
    }
    Ok(value as f32)
}
fn string_option(object: &Map<String, Value>, key: &str) -> Result<Option<String>> {
    supplied(object, key)
        .map(|v| {
            v.as_str()
                .map(str::to_owned)
                .ok_or_else(|| RequestError::invalid(key, format!("{key} must be a string")))
        })
        .transpose()
}
fn text_content(value: &Value, path: &str) -> Result<String> {
    if let Some(text) = value.as_str() {
        return Ok(text.to_owned());
    }
    let items = value.as_array().ok_or_else(|| {
        RequestError::invalid(path, "content must be a string or an array of text parts")
    })?;
    let mut text = String::new();
    for (i, item) in items.iter().enumerate() {
        let p = format!("{path}[{i}]");
        let part = item
            .as_object()
            .ok_or_else(|| RequestError::invalid(&p, "content part must be an object"))?;
        if part.get("type").and_then(Value::as_str) != Some("text") {
            return Err(RequestError::unsupported(format!("{p}.type")));
        }
        if part.keys().any(|k| !matches!(k.as_str(), "type" | "text")) {
            return Err(RequestError::unsupported(&p));
        }
        text.push_str(
            part.get("text")
                .and_then(Value::as_str)
                .ok_or_else(|| RequestError::invalid(&p, "text part requires a string text"))?,
        );
    }
    Ok(text)
}
fn valid_name(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= 64
        && name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
}

fn messages(value: Option<&Value>) -> Result<Vec<Message>> {
    let raw = value
        .and_then(Value::as_array)
        .filter(|v| !v.is_empty())
        .ok_or_else(|| RequestError::invalid("messages", "messages must be a nonempty array"))?;
    let mut result = Vec::with_capacity(raw.len());
    let mut pending = BTreeMap::new();
    let mut used_ids = BTreeSet::new();
    let mut conversation_started = false;
    for (i, item) in raw.iter().enumerate() {
        let path = format!("messages[{i}]");
        let object = item
            .as_object()
            .ok_or_else(|| RequestError::invalid(&path, "message must be an object"))?;
        for key in object.keys() {
            if !matches!(
                key.as_str(),
                "role" | "content" | "name" | "tool_calls" | "tool_call_id" | "reasoning_content"
            ) {
                return Err(RequestError::unsupported(format!("{path}.{key}")));
            }
        }
        let role = object.get("role").and_then(Value::as_str).ok_or_else(|| {
            RequestError::invalid(format!("{path}.role"), "message role must be a string")
        })?;
        if !matches!(role, "system" | "developer" | "user" | "assistant" | "tool") {
            return Err(RequestError::unsupported(format!("{path}.role")));
        }
        if matches!(role, "system" | "developer") {
            if conversation_started {
                return Err(RequestError::invalid(
                    &path,
                    "system/developer messages must precede the conversation",
                ));
            }
        } else {
            conversation_started = true;
        }
        if role != "tool" && !pending.is_empty() {
            return Err(RequestError::invalid(
                &path,
                "tool results must answer every preceding tool_call before another message",
            ));
        }
        let name = string_option(object, "name")?;
        if name.as_ref().is_some_and(|v| !valid_name(v)) {
            return Err(RequestError::invalid(
                format!("{path}.name"),
                "name must contain 1..64 letters, digits, underscores or hyphens",
            ));
        }
        let reasoning_content = string_option(object, "reasoning_content")?;
        if reasoning_content.is_some() && role != "assistant" {
            return Err(RequestError::invalid(
                &path,
                "reasoning_content requires assistant role",
            ));
        }
        let mut calls = Vec::new();
        if let Some(value) = supplied(object, "tool_calls") {
            if role != "assistant" {
                return Err(RequestError::invalid(
                    &path,
                    "tool_calls requires assistant role",
                ));
            }
            let array = value
                .as_array()
                .filter(|a| !a.is_empty() && a.len() <= 128)
                .ok_or_else(|| {
                    RequestError::invalid(&path, "tool_calls must contain 1..128 calls")
                })?;
            for raw_call in array {
                let call: ToolCall = serde_json::from_value(raw_call.clone())
                    .map_err(|_| RequestError::invalid(&path, "invalid function tool_call"))?;
                if call.kind != "function" || call.id.is_empty() || !valid_name(&call.function.name)
                {
                    return Err(RequestError::invalid(
                        &path,
                        "tool_call requires id, type=function and a valid function name",
                    ));
                }
                if !used_ids.insert(call.id.clone()) {
                    return Err(RequestError::invalid(&path, "duplicate tool_call id"));
                }
                let args: Value = serde_json::from_str(&call.function.arguments).map_err(|_| {
                    RequestError::invalid(
                        &path,
                        "tool_call arguments must be a JSON object encoded as a string",
                    )
                })?;
                if !args.is_object() {
                    return Err(RequestError::invalid(
                        &path,
                        "tool_call arguments must be a JSON object",
                    ));
                }
                pending.insert(call.id.clone(), call.function.name.clone());
                calls.push(call);
            }
        }
        let tool_call_id = string_option(object, "tool_call_id")?;
        if role == "tool" {
            let id = tool_call_id.as_ref().ok_or_else(|| {
                RequestError::invalid(&path, "tool message requires tool_call_id")
            })?;
            if pending.remove(id).is_none() {
                return Err(RequestError::invalid(
                    &path,
                    "tool_call_id does not match a pending assistant call",
                ));
            }
        } else if tool_call_id.is_some() {
            return Err(RequestError::invalid(
                &path,
                "tool_call_id requires tool role",
            ));
        }
        let content = match supplied(object, "content") {
            Some(value) => text_content(value, &format!("{path}.content"))?,
            None if role == "assistant" && !calls.is_empty() => String::new(),
            None => {
                return Err(RequestError::invalid(
                    format!("{path}.content"),
                    "content is required except for assistant tool calls",
                ));
            }
        };
        result.push(Message {
            role: role.into(),
            content,
            name,
            tool_calls: calls,
            tool_call_id,
            reasoning_content,
        });
    }
    if !pending.is_empty() {
        return Err(RequestError::invalid(
            "messages",
            "missing tool results for pending assistant calls",
        ));
    }
    if !result.iter().any(|m| m.role == "user") {
        return Err(RequestError::invalid(
            "messages",
            "at least one user message is required",
        ));
    }
    Ok(result)
}

pub(super) fn parse(value: Value, model_id: &str) -> Result<ParsedRequest> {
    let object = value
        .as_object()
        .ok_or_else(|| RequestError::invalid("body", "request must be a JSON object"))?;
    for key in object.keys() {
        if !matches!(
            key.as_str(),
            "model"
                | "messages"
                | "max_tokens"
                | "max_completion_tokens"
                | "temperature"
                | "top_p"
                | "top_k"
                | "seed"
                | "stream"
                | "enable_thinking"
                | "reasoning_effort"
                | "tools"
                | "tool_choice"
                | "parallel_tool_calls"
                | "stream_options"
                | "stop"
                | "frequency_penalty"
                | "presence_penalty"
                | "logit_bias"
                | "n"
                | "store"
                | "logprobs"
                | "top_logprobs"
                | "response_format"
                | "user"
                | "metadata"
                | "service_tier"
                | "modalities"
                | "safety_identifier"
                | "prompt_cache_key"
        ) {
            return Err(RequestError::unsupported(key));
        }
    }
    let model = object
        .get("model")
        .and_then(Value::as_str)
        .ok_or_else(|| RequestError::invalid("model", "model must be a string"))?;
    if model != model_id {
        return Err(RequestError {
            message: format!("unknown model: {model}"),
            param: "model".into(),
            code: "model_not_found",
        });
    }
    let messages = messages(object.get("messages"))?;
    if supplied(object, "max_tokens").is_some()
        && supplied(object, "max_completion_tokens").is_some()
    {
        return Err(RequestError::invalid(
            "max_completion_tokens",
            "set only one token limit",
        ));
    }
    let limit_key = if supplied(object, "max_completion_tokens").is_some() {
        "max_completion_tokens"
    } else {
        "max_tokens"
    };
    let max_tokens = supplied(object, limit_key)
        .map_or(Some(usize::MAX as u64), Value::as_u64)
        .and_then(|n| usize::try_from(n).ok())
        .filter(|&n| n > 0)
        .ok_or_else(|| {
            RequestError::invalid(limit_key, format!("{limit_key} must be a positive integer"))
        })?;
    let temperature = number(object, "temperature", 1.0, 0.0, 2.0)?;
    let top_p = number(object, "top_p", 1.0, 0.0, 1.0)?;
    if top_p == 0.0 {
        return Err(RequestError::invalid(
            "top_p",
            "top_p must be greater than zero",
        ));
    }
    let top_k = supplied(object, "top_k")
        .map_or(Some(0), Value::as_u64)
        .and_then(|n| usize::try_from(n).ok())
        .ok_or_else(|| RequestError::invalid("top_k", "top_k must be a nonnegative integer"))?;
    let seed = match supplied(object, "seed") {
        None => 0,
        Some(v) => v
            .as_u64()
            .or_else(|| v.as_i64().map(|s| s as u64))
            .ok_or_else(|| RequestError::invalid("seed", "seed must be an integer"))?,
    };
    let stream = boolean(object, "stream", false)?;
    let options = supplied(object, "stream_options");
    if options.is_some() && !stream {
        return Err(RequestError::invalid(
            "stream_options",
            "stream_options requires stream=true",
        ));
    }
    let mut stream_options = StreamOptions {
        include_usage: false,
        include_obfuscation: true,
    };
    if let Some(options) = options {
        let options = options.as_object().ok_or_else(|| {
            RequestError::invalid("stream_options", "stream_options must be an object")
        })?;
        for k in options.keys() {
            if !matches!(k.as_str(), "include_usage" | "include_obfuscation") {
                return Err(RequestError::unsupported(format!("stream_options.{k}")));
            }
        }
        stream_options.include_usage = boolean(options, "include_usage", false)?;
        stream_options.include_obfuscation = boolean(options, "include_obfuscation", true)?;
    }
    let tools = ToolConfig::parse(
        supplied(object, "tools"),
        supplied(object, "tool_choice"),
        supplied(object, "parallel_tool_calls"),
    )
    .map_err(|message| RequestError::invalid("tools", message))?;
    let stop = match supplied(object, "stop") {
        None => Vec::new(),
        Some(Value::String(s)) => vec![s.clone()],
        Some(Value::Array(a)) => a
            .iter()
            .map(|v| {
                v.as_str()
                    .map(str::to_owned)
                    .ok_or_else(|| RequestError::invalid("stop", "stop entries must be strings"))
            })
            .collect::<Result<Vec<_>>>()?,
        _ => {
            return Err(RequestError::invalid(
                "stop",
                "stop must be a string or array of strings",
            ));
        }
    };
    if stop.len() > 4 || stop.iter().any(|s| s.is_empty() || s.len() > 4096) {
        return Err(RequestError::invalid(
            "stop",
            "provide up to four nonempty stop strings, each at most 4096 bytes",
        ));
    }
    let mut sampling = crate::chat::sampling::SamplingOptions {
        frequency_penalty: number(object, "frequency_penalty", 0.0, -2.0, 2.0)?,
        presence_penalty: number(object, "presence_penalty", 0.0, -2.0, 2.0)?,
        ..Default::default()
    };
    if let Some(bias) = supplied(object, "logit_bias") {
        let bias = bias.as_object().ok_or_else(|| {
            RequestError::invalid("logit_bias", "logit_bias must map token IDs to numbers")
        })?;
        for (id, value) in bias {
            let token = id.parse::<u32>().map_err(|_| {
                RequestError::invalid("logit_bias", "token IDs must be unsigned integers")
            })?;
            let value = value
                .as_f64()
                .filter(|v| v.is_finite() && (-100.0..=100.0).contains(v))
                .ok_or_else(|| {
                    RequestError::invalid("logit_bias", "bias values must be in [-100, 100]")
                })? as f32;
            if sampling.logit_bias.insert(token, value).is_some() {
                return Err(RequestError::invalid(
                    "logit_bias",
                    "duplicate numeric token ID",
                ));
            }
        }
    }
    for key in ["n", "top_logprobs"] {
        let expected = if key == "n" { 1 } else { 0 };
        if supplied(object, key).is_some_and(|v| v.as_u64() != Some(expected)) {
            return Err(RequestError::unsupported(key));
        }
    }
    for key in ["store", "logprobs"] {
        if boolean(object, key, false)? {
            return Err(RequestError::unsupported(key));
        }
    }
    if supplied(object, "modalities").is_some_and(|v| v != &serde_json::json!(["text"])) {
        return Err(RequestError::unsupported("modalities"));
    }
    for key in ["user", "safety_identifier", "prompt_cache_key"] {
        let _ = string_option(object, key)?;
    }
    if let Some(meta) = supplied(object, "metadata") {
        let m = meta.as_object().filter(|m| m.len() <= 16).ok_or_else(|| {
            RequestError::invalid(
                "metadata",
                "metadata must be an object with at most 16 entries",
            )
        })?;
        if m.iter().any(|(k, v)| {
            k.chars().count() > 64 || v.as_str().is_none_or(|s| s.chars().count() > 512)
        }) {
            return Err(RequestError::invalid(
                "metadata",
                "metadata keys must be at most 64 characters and string values at most 512",
            ));
        }
    }
    let service_tier = string_option(object, "service_tier")?;
    if service_tier
        .as_deref()
        .is_some_and(|s| !matches!(s, "auto" | "default"))
    {
        return Err(RequestError::unsupported("service_tier"));
    }
    let response_format = match supplied(object, "response_format") {
        None => ResponseFormat::Text,
        Some(v) if v == &serde_json::json!({"type":"text"}) => ResponseFormat::Text,
        Some(v) if v == &serde_json::json!({"type":"json_object"}) => ResponseFormat::JsonObject,
        _ => return Err(RequestError::unsupported("response_format")),
    };
    if response_format == ResponseFormat::JsonObject && tools.enabled() {
        return Err(RequestError::invalid(
            "response_format",
            "json_object cannot be combined with enabled tool calls",
        ));
    }
    let effort = string_option(object, "reasoning_effort")?;
    let reasoning_effort = match effort.as_deref() {
        None | Some("none") => None,
        Some("minimal" | "low") => Some("low".into()),
        Some("medium") => Some("medium".into()),
        Some("high" | "xhigh") => Some("xhigh".into()),
        _ => return Err(RequestError::unsupported("reasoning_effort")),
    };
    let enable_thinking = boolean(object, "enable_thinking", reasoning_effort.is_some())?;
    if (effort.as_deref() == Some("none") && enable_thinking)
        || (reasoning_effort.is_some() && !enable_thinking)
    {
        return Err(RequestError::invalid(
            "reasoning_effort",
            "reasoning_effort conflicts with enable_thinking",
        ));
    }
    Ok(ParsedRequest {
        generation: GenerationRequest {
            messages,
            max_tokens,
            temperature,
            top_p,
            top_k,
            seed,
            enable_thinking,
            tools,
            sampling,
            reasoning_effort,
        },
        stream,
        stream_options,
        stop,
        response_format,
        service_tier: service_tier.is_some(),
    })
}
