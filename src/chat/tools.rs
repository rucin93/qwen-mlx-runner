use std::collections::HashSet;

use anyhow::{Result, anyhow, bail, ensure};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum ToolChoice {
    #[default]
    None,
    Auto,
    Required,
    Named(String),
}

#[derive(Clone, Debug)]
pub struct ToolConfig {
    pub definitions: Vec<Value>,
    pub choice: ToolChoice,
    pub parallel: bool,
}

impl Default for ToolConfig {
    fn default() -> Self {
        Self {
            definitions: Vec::new(),
            choice: ToolChoice::None,
            parallel: true,
        }
    }
}

impl ToolConfig {
    pub fn parse(
        tools: Option<&Value>,
        choice: Option<&Value>,
        parallel: Option<&Value>,
    ) -> std::result::Result<Self, String> {
        let mut definitions = match tools {
            None | Some(Value::Null) => Vec::new(),
            Some(Value::Array(values)) if values.len() <= 128 => values.clone(),
            Some(Value::Array(_)) => {
                return Err("tools accepts at most 128 function definitions".into());
            }
            _ => return Err("tools must be an array of function definitions".into()),
        };
        let mut names = HashSet::new();
        for (index, definition) in definitions.iter_mut().enumerate() {
            if definition.get("type").and_then(Value::as_str) != Some("function") {
                return Err(format!("tools[{index}].type must be function"));
            }
            let function = definition
                .get_mut("function")
                .and_then(Value::as_object_mut)
                .ok_or_else(|| format!("tools[{index}].function must be an object"))?;
            let name = function
                .get("name")
                .and_then(Value::as_str)
                .ok_or_else(|| format!("tools[{index}].function.name must be a string"))?;
            if !valid_name(name) {
                return Err(format!(
                    "tools[{index}].function.name must contain 1..64 ASCII letters, digits, underscores or hyphens"
                ));
            }
            if !names.insert(name.to_owned()) {
                return Err(format!("duplicate tool name: {name}"));
            }
            match function.get("strict") {
                None | Some(Value::Null) | Some(Value::Bool(false)) => {}
                Some(Value::Bool(true)) => return Err("strict: true is unsupported: this engine does not guarantee grammar-constrained tool arguments".into()),
                _ => return Err(format!("tools[{index}].function.strict must be a boolean or null")),
            }
            if function
                .get("description")
                .is_some_and(|value| !value.is_string())
            {
                return Err(format!(
                    "tools[{index}].function.description must be a string"
                ));
            }
            let parameters = function
                .entry("parameters")
                .or_insert_with(|| serde_json::json!({"type":"object","properties":{}}));
            let schema = parameters
                .as_object()
                .ok_or_else(|| format!("tools[{index}].function.parameters must be an object"))?;
            if schema
                .get("type")
                .is_some_and(|value| value.as_str() != Some("object"))
            {
                return Err(format!(
                    "tools[{index}].function.parameters.type must be object"
                ));
            }
            if schema
                .get("properties")
                .is_some_and(|value| !value.is_object())
            {
                return Err(format!(
                    "tools[{index}].function.parameters.properties must be an object"
                ));
            }
            if schema.get("required").is_some_and(|value| {
                value
                    .as_array()
                    .is_none_or(|items| items.iter().any(|item| !item.is_string()))
            }) {
                return Err(format!(
                    "tools[{index}].function.parameters.required must be an array of strings"
                ));
            }
        }
        let choice = match choice {
            None | Some(Value::Null) => {
                if definitions.is_empty() {
                    ToolChoice::None
                } else {
                    ToolChoice::Auto
                }
            }
            Some(Value::String(value)) => match value.as_str() {
                "none" => ToolChoice::None,
                "auto" => ToolChoice::Auto,
                "required" => ToolChoice::Required,
                _ => {
                    return Err(
                        "tool_choice must be none, auto, required, or a named function".into(),
                    );
                }
            },
            Some(Value::Object(value))
                if value.get("type").and_then(Value::as_str) == Some("function") =>
            {
                let name = value
                    .get("function")
                    .and_then(|value| value.get("name"))
                    .and_then(Value::as_str)
                    .ok_or_else(|| "tool_choice.function.name must be a string".to_owned())?;
                if !names.contains(name) {
                    return Err(format!("tool_choice names an undefined tool: {name}"));
                }
                ToolChoice::Named(name.to_owned())
            }
            _ => {
                return Err(
                    "unsupported tool_choice; use none, auto, required, or a named function".into(),
                );
            }
        };
        if definitions.is_empty() && choice != ToolChoice::None {
            return Err("tool_choice requires at least one function definition".into());
        }
        let parallel = match parallel {
            None | Some(Value::Null) => true,
            Some(Value::Bool(value)) => *value,
            _ => return Err("parallel_tool_calls must be a boolean or null".into()),
        };
        Ok(Self {
            definitions,
            choice,
            parallel,
        })
    }
    pub fn enabled(&self) -> bool {
        self.choice != ToolChoice::None && !self.definitions.is_empty()
    }
    pub fn effective_definitions(&self) -> Vec<Value> {
        match &self.choice {
            ToolChoice::None => Vec::new(),
            ToolChoice::Named(name) => self
                .definitions
                .iter()
                .filter(|definition| definition["function"]["name"].as_str() == Some(name))
                .cloned()
                .collect(),
            _ => self.definitions.clone(),
        }
    }
    pub fn instruction(&self) -> Option<String> {
        if !self.enabled() {
            return None;
        }
        let mut instructions = Vec::new();
        match &self.choice {
            ToolChoice::Required => instructions.push("You must call at least one of the provided functions before ending this response.".to_owned()),
            ToolChoice::Named(name) => instructions.push(format!("You must call the function {name} before ending this response. Do not call any other function.")),
            _ => {}
        }
        if !self.parallel {
            instructions.push("Return at most one function call in this response.".into());
        }
        (!instructions.is_empty()).then(|| instructions.join("\n"))
    }

    fn schema_for_call(&self, name: &str) -> Result<&Value> {
        ensure!(
            self.enabled(),
            "model emitted a tool call when tool_choice is none"
        );
        if let ToolChoice::Named(expected) = &self.choice {
            ensure!(
                name == expected,
                "model called {name}, but tool_choice requires {expected}"
            );
        }
        self.definitions
            .iter()
            .find_map(|definition| {
                let function = definition.get("function")?;
                (function.get("name")?.as_str()? == name)
                    .then(|| function.get("parameters"))
                    .flatten()
            })
            .ok_or_else(|| anyhow!("model called an undefined function: {name}"))
    }
}

fn valid_name(name: &str) -> bool {
    (1..=64).contains(&name.len())
        && name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
}

#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
pub struct ToolFunction {
    pub name: String,
    pub arguments: String,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: String,
    pub function: ToolFunction,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ToolEvent {
    Text(String),
    Call(ToolCall),
}

/// Tool frames are withheld until complete and validated. Ordinary text,
/// including reasoning tags, is returned unchanged for the caller's text path.
pub struct ToolOutputParser {
    config: ToolConfig,
    id_prefix: String,
    pending: String,
    thinking: bool,
    in_call: bool,
    calls: usize,
    finished: bool,
    failed: bool,
}

impl ToolOutputParser {
    pub fn new(config: ToolConfig, id_prefix: String, thinking: bool) -> Self {
        Self {
            config,
            id_prefix,
            pending: String::new(),
            thinking,
            in_call: false,
            calls: 0,
            finished: false,
            failed: false,
        }
    }

    pub fn push(&mut self, text: &str) -> Result<Vec<ToolEvent>> {
        ensure!(
            !self.finished && !self.failed,
            "tool output parser is already finished or failed"
        );
        self.pending.push_str(text);
        let result = self.drain(false);
        if result.is_err() {
            self.failed = true;
        }
        result
    }

    pub fn finish(&mut self) -> Result<Vec<ToolEvent>> {
        ensure!(!self.failed, "tool output parser has failed");
        if self.finished {
            return Ok(Vec::new());
        }
        self.finished = true;
        let result = self.drain(true).and_then(|events| {
            if matches!(
                self.config.choice,
                ToolChoice::Required | ToolChoice::Named(_)
            ) && self.calls == 0
            {
                bail!("model did not produce the tool call required by tool_choice");
            }
            Ok(events)
        });
        if result.is_err() {
            self.failed = true;
        }
        result
    }

    fn drain(&mut self, finishing: bool) -> Result<Vec<ToolEvent>> {
        const OPEN: &str = "<tool_call>";
        const CLOSE: &str = "</tool_call>";
        const THINK: &str = "<think>";
        const THINK_END: &str = "</think>";
        let mut events = Vec::new();
        loop {
            if self.in_call {
                let Some(end) = call_end(&self.pending) else {
                    ensure!(!finishing, "incomplete tool call at end of generation");
                    break;
                };
                ensure!(
                    self.config.parallel || self.calls == 0,
                    "model emitted multiple calls with parallel_tool_calls=false"
                );
                let body = self.pending[..end].to_owned();
                let (name, arguments) = self.parse_body(&body)?;
                let call = ToolCall {
                    id: format!("{}_{}", self.id_prefix, self.calls),
                    kind: "function".into(),
                    function: ToolFunction {
                        name,
                        arguments: serde_json::to_string(&arguments)?,
                    },
                };
                self.calls += 1;
                self.pending.drain(..end + CLOSE.len());
                self.in_call = false;
                events.push(ToolEvent::Call(call));
                continue;
            }
            if self.pending.is_empty() {
                break;
            }
            if self.thinking {
                if let Some(end) = self.pending.find(THINK_END) {
                    let text: String = self.pending.drain(..end + THINK_END.len()).collect();
                    emit_text(&mut events, text);
                    self.thinking = false;
                    continue;
                }
                let keep = if finishing {
                    0
                } else {
                    marker_suffix(&self.pending, &[THINK_END])
                };
                let end = self.pending.len() - keep;
                emit_text(&mut events, self.pending.drain(..end).collect());
                break;
            }
            let next = [(OPEN, false), (THINK, true)]
                .into_iter()
                .filter_map(|(marker, think)| {
                    self.pending
                        .find(marker)
                        .map(|index| (index, marker, think))
                })
                .min_by_key(|(index, _, _)| *index);
            if let Some((index, marker, think)) = next {
                emit_text(&mut events, self.pending.drain(..index).collect());
                self.pending.drain(..marker.len());
                if think {
                    emit_text(&mut events, marker.to_owned());
                    self.thinking = true;
                } else {
                    self.in_call = true;
                }
                continue;
            }
            let keep = marker_suffix(&self.pending, &[OPEN, THINK]);
            let end = self.pending.len() - keep;
            emit_text(&mut events, self.pending.drain(..end).collect());
            if finishing {
                // A lone '<' can be ordinary text. A recognized but truncated
                // tool opener cannot be silently exposed as a successful answer.
                ensure!(
                    !self.pending.starts_with("<tool_"),
                    "incomplete tool call marker at end of generation"
                );
                emit_text(&mut events, std::mem::take(&mut self.pending));
            }
            break;
        }
        Ok(events)
    }

    fn parse_body(&self, body: &str) -> Result<(String, Value)> {
        let body = body.trim();
        let (name, arguments) = if let Some(rest) = body.strip_prefix("<function=") {
            let end = rest
                .find('>')
                .ok_or_else(|| anyhow!("malformed function opening tag"))?;
            let name = &rest[..end];
            let schema = self.config.schema_for_call(name)?;
            let mut rest = &rest[end + 1..];
            let mut arguments = Map::new();
            loop {
                rest = rest.trim_start();
                if let Some(suffix) = rest.strip_prefix("</function>") {
                    ensure!(
                        suffix.trim().is_empty(),
                        "unexpected content after function closing tag"
                    );
                    break;
                }
                let parameter = rest
                    .strip_prefix("<parameter=")
                    .ok_or_else(|| anyhow!("expected parameter or function closing tag"))?;
                let end = parameter
                    .find('>')
                    .ok_or_else(|| anyhow!("malformed parameter opening tag"))?;
                let key = &parameter[..end];
                ensure!(
                    !key.is_empty() && !key.contains(['<', '\n', '\r']),
                    "invalid parameter name"
                );
                ensure!(
                    !arguments.contains_key(key),
                    "duplicate tool parameter: {key}"
                );
                let value = &parameter[end + 1..];
                let end = value
                    .find("</parameter>")
                    .ok_or_else(|| anyhow!("missing parameter closing tag"))?;
                let property = schema
                    .get("properties")
                    .and_then(|properties| properties.get(key));
                arguments.insert(
                    key.to_owned(),
                    xml_value(strip_framing_newlines(&value[..end]), property)?,
                );
                rest = &value[end + "</parameter>".len()..];
            }
            (name.to_owned(), Value::Object(arguments))
        } else {
            let value: Value = serde_json::from_str(body)
                .map_err(|error| anyhow!("malformed JSON tool call: {error}"))?;
            ensure!(value.is_object(), "tool call must be a JSON object");
            if let Some(kind) = value.get("type") {
                ensure!(
                    kind.as_str() == Some("function"),
                    "tool call type must be function"
                );
            }
            let function = value.get("function").unwrap_or(&value);
            let name = function
                .get("name")
                .and_then(Value::as_str)
                .ok_or_else(|| anyhow!("tool call requires a function name"))?;
            let arguments = match function.get("arguments") {
                None => Value::Object(Map::new()),
                Some(Value::String(encoded)) => serde_json::from_str(encoded)
                    .map_err(|error| anyhow!("tool arguments are not valid JSON: {error}"))?,
                Some(value) => value.clone(),
            };
            (name.to_owned(), arguments)
        };
        let schema = self.config.schema_for_call(&name)?;
        validate_arguments(&arguments, schema)?;
        Ok((name, arguments))
    }
}

fn emit_text(events: &mut Vec<ToolEvent>, text: String) {
    if text.is_empty() {
        return;
    }
    if let Some(ToolEvent::Text(previous)) = events.last_mut() {
        previous.push_str(&text);
    } else {
        events.push(ToolEvent::Text(text));
    }
}

fn marker_suffix(text: &str, markers: &[&str]) -> usize {
    markers
        .iter()
        .flat_map(|marker| (1..marker.len()).map(move |length| &marker[..length]))
        .filter(|prefix| text.ends_with(*prefix))
        .map(str::len)
        .max()
        .unwrap_or(0)
}

/// Closing markers inside a JSON string are argument data, not frame boundaries.
fn call_end(text: &str) -> Option<usize> {
    if text.trim_start().starts_with("<function=") {
        return xml_call_end(text);
    }
    if !text.trim_start().starts_with('{') {
        return text.find("</tool_call>");
    }
    let mut quoted = false;
    let mut escaped = false;
    for (index, byte) in text.bytes().enumerate() {
        if quoted {
            if escaped {
                escaped = false;
            } else if byte == b'\\' {
                escaped = true;
            } else if byte == b'"' {
                quoted = false;
            }
        } else if byte == b'"' {
            quoted = true;
        } else if byte == b'<' && text[index..].starts_with("</tool_call>") {
            return Some(index);
        }
    }
    None
}

/// Qwen XML parameter bodies are raw strings, not XML-escaped content. Treat
/// embedded tool/function tags as data until the parameter's own closing tag.
/// A literal </parameter> remains ambiguous in this checkpoint's protocol.
fn xml_call_end(text: &str) -> Option<usize> {
    const PARAMETER: &str = "<parameter=";
    const PARAMETER_END: &str = "</parameter>";
    const FUNCTION_END: &str = "</function>";
    const CALL_END: &str = "</tool_call>";
    let mut position = 0;
    loop {
        let remaining = &text[position..];
        let parameter = remaining.find(PARAMETER);
        let function_end = remaining.find(FUNCTION_END);
        if let Some(end) = function_end
            && parameter.is_none_or(|parameter| end < parameter)
        {
            position += end + FUNCTION_END.len();
            return text[position..].find(CALL_END).map(|end| position + end);
        }
        let parameter = parameter?;
        position += parameter;
        let opening_end = text[position..].find('>')?;
        position += opening_end + 1;
        let closing_end = text[position..].find(PARAMETER_END)?;
        position += closing_end + PARAMETER_END.len();
    }
}

fn strip_framing_newlines(value: &str) -> &str {
    let value = value
        .strip_prefix("\r\n")
        .or_else(|| value.strip_prefix('\n'))
        .unwrap_or(value);
    value
        .strip_suffix("\r\n")
        .or_else(|| value.strip_suffix('\n'))
        .unwrap_or(value)
}

fn schema_types(schema: Option<&Value>) -> Vec<&str> {
    match schema.and_then(|schema| schema.get("type")) {
        Some(Value::String(kind)) => vec![kind.as_str()],
        Some(Value::Array(kinds)) => kinds.iter().filter_map(Value::as_str).collect(),
        _ => Vec::new(),
    }
}

fn matches_type(value: &Value, kind: &str) -> bool {
    match kind {
        "string" => value.is_string(),
        "number" => value.is_number(),
        "integer" => value
            .as_f64()
            .is_some_and(|number| number.is_finite() && number.fract() == 0.0),
        "boolean" => value.is_boolean(),
        "object" => value.is_object(),
        "array" => value.is_array(),
        "null" => value.is_null(),
        _ => true,
    }
}

fn xml_value(raw: &str, schema: Option<&Value>) -> Result<Value> {
    let types = schema_types(schema);
    if types.contains(&"string") {
        if types.contains(&"null") && raw.trim() == "null" {
            return Ok(Value::Null);
        }
        return Ok(Value::String(raw.to_owned()));
    }
    let parsed = serde_json::from_str::<Value>(raw.trim());
    if types.is_empty() {
        return Ok(parsed.unwrap_or_else(|_| Value::String(raw.to_owned())));
    }
    let value =
        parsed.map_err(|error| anyhow!("tool parameter is not valid {:?}: {error}", types))?;
    ensure!(
        types.iter().any(|kind| matches_type(&value, kind)),
        "tool parameter does not match declared type {:?}",
        types
    );
    Ok(value)
}

/// Post-generation shape checks, not a complete JSON Schema implementation or
/// constrained-decoding guarantee. Strict function schemas are rejected above.
fn validate_arguments(arguments: &Value, schema: &Value) -> Result<()> {
    let arguments = arguments
        .as_object()
        .ok_or_else(|| anyhow!("tool arguments must be a JSON object"))?;
    if let Some(required) = schema.get("required").and_then(Value::as_array) {
        for key in required.iter().filter_map(Value::as_str) {
            ensure!(
                arguments.contains_key(key),
                "missing required tool parameter: {key}"
            );
        }
    }
    let properties = schema.get("properties").and_then(Value::as_object);
    for (key, value) in arguments {
        let property = properties.and_then(|properties| properties.get(key));
        if schema.get("additionalProperties") == Some(&Value::Bool(false)) {
            ensure!(property.is_some(), "undeclared tool parameter: {key}");
        }
        let types = schema_types(property);
        ensure!(
            types.is_empty() || types.iter().any(|kind| matches_type(value, kind)),
            "tool parameter {key} does not match declared type {:?}",
            types
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn definitions() -> Value {
        json!([
            {"type":"function","function":{"name":"lookup","parameters":{
                "type":"object", "properties": {
                    "code":{"type":"string"}, "count":{"type":"integer"},
                    "ratio":{"type":"number"}, "flag":{"type":"boolean"},
                    "options":{"type":"object"}, "items":{"type":"array"}
                }, "required":["code"]
            }}},
            {"type":"function","function":{"name":"other","parameters":{"type":"object"}}}
        ])
    }

    fn config() -> ToolConfig {
        ToolConfig::parse(Some(&definitions()), None, None).unwrap()
    }
    fn parser() -> ToolOutputParser {
        ToolOutputParser::new(config(), "call_test".into(), false)
    }
    fn xml() -> &'static str {
        "<tool_call>\n<function=lookup>\n<parameter=code>\n00123\n</parameter>\n<parameter=count>2</parameter><parameter=ratio>0.25</parameter><parameter=flag>true</parameter><parameter=options>{\"city\":\"Łódź\"}</parameter><parameter=items>[1,\"x\"]</parameter></function>\n</tool_call>"
    }
    fn collect(events: &[ToolEvent]) -> (String, Vec<ToolCall>) {
        let mut text = String::new();
        let mut calls = Vec::new();
        for event in events {
            match event {
                ToolEvent::Text(value) => text.push_str(value),
                ToolEvent::Call(call) => calls.push(call.clone()),
            }
        }
        (text, calls)
    }
    fn parse_all(text: &str) -> Result<(String, Vec<ToolCall>)> {
        let mut p = parser();
        let mut events = p.push(text)?;
        events.extend(p.finish()?);
        Ok(collect(&events))
    }

    #[test]
    fn configuration_defaults_and_explicit_none() {
        let empty = ToolConfig::parse(None, None, None).unwrap();
        assert!(!empty.enabled());
        assert!(empty.parallel);
        assert_eq!(config().choice, ToolChoice::Auto);
        let none = ToolConfig::parse(Some(&definitions()), Some(&json!("none")), None).unwrap();
        assert!(!none.enabled());
        assert!(none.effective_definitions().is_empty());
    }

    #[test]
    fn omitted_parameters_mean_an_empty_object_schema() {
        let defs = json!([{"type":"function","function":{"name":"ping"}}]);
        let cfg = ToolConfig::parse(Some(&defs), None, None).unwrap();
        let mut p = ToolOutputParser::new(cfg, "call".into(), false);
        let events = p
            .push("<tool_call>{\"name\":\"ping\",\"arguments\":{}}</tool_call>")
            .unwrap();
        assert_eq!(collect(&events).1[0].function.arguments, "{}");
        assert!(p.finish().is_ok());
    }

    #[test]
    fn validates_tool_names_duplicates_shapes_and_limit() {
        for name in ["", "has space", "ą", &"a".repeat(65)] {
            let value = json!([{"type":"function","function":{"name":name,"parameters":{}}}]);
            assert!(
                ToolConfig::parse(Some(&value), None, None).is_err(),
                "{name}"
            );
        }
        for value in [
            json!({}),
            json!([{"type":"custom","name":"x"}]),
            json!([{"type":"function","function":{"name":"x","parameters":[]}}]),
        ] {
            assert!(ToolConfig::parse(Some(&value), None, None).is_err());
        }
        let duplicate = json!([definitions()[0].clone(), definitions()[0].clone()]);
        assert!(ToolConfig::parse(Some(&duplicate), None, None).is_err());
        let too_many = Value::Array((0..129).map(|n| json!({"type":"function","function":{"name":format!("f{n}"),"parameters":{}}})).collect());
        assert!(ToolConfig::parse(Some(&too_many), None, None).is_err());
    }

    #[test]
    fn rejects_strict_guarantees_but_accepts_false_or_null() {
        for strict in [json!(false), Value::Null] {
            let mut defs = definitions();
            defs[0]["function"]["strict"] = strict;
            assert!(ToolConfig::parse(Some(&defs), None, None).is_ok());
        }
        let mut defs = definitions();
        defs[0]["function"]["strict"] = json!(true);
        assert!(
            ToolConfig::parse(Some(&defs), None, None)
                .unwrap_err()
                .contains("strict")
        );
    }

    #[test]
    fn validates_choice_and_parallel_and_emits_constraints() {
        for choice in [
            json!("required"),
            json!({"type":"function","function":{"name":"lookup"}}),
        ] {
            assert!(ToolConfig::parse(None, Some(&choice), None).is_err());
            let cfg = ToolConfig::parse(Some(&definitions()), Some(&choice), Some(&json!(false)))
                .unwrap();
            assert!(cfg.instruction().is_some());
            assert!(!cfg.parallel);
        }
        assert!(
            ToolConfig::parse(
                Some(&definitions()),
                Some(&json!({"type":"function","function":{"name":"absent"}})),
                None
            )
            .is_err()
        );
        assert!(ToolConfig::parse(Some(&definitions()), Some(&json!("unexpected")), None).is_err());
        assert!(ToolConfig::parse(None, None, Some(&json!("false"))).is_err());
    }

    #[test]
    fn xml_schema_preserves_numeric_strings_and_converts_values() {
        let (_, calls) = parse_all(xml()).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].kind, "function");
        assert_eq!(calls[0].function.name, "lookup");
        assert_eq!(
            serde_json::from_str::<Value>(&calls[0].function.arguments).unwrap(),
            json!({"code":"00123","count":2,"ratio":0.25,"flag":true,"options":{"city":"Łódź"},"items":[1,"x"]})
        );
    }

    #[test]
    fn all_char_chunks_match_whole_input_with_utf8_and_stable_ids() {
        let input = format!("Cześć 🦀\n{}\n{}done", xml(), xml());
        let expected = parse_all(&input).unwrap();
        let mut p = parser();
        let mut events = Vec::new();
        for ch in input.chars() {
            events.extend(p.push(&ch.to_string()).unwrap());
        }
        events.extend(p.finish().unwrap());
        assert_eq!(collect(&events), expected);
        assert_eq!(expected.0, "Cześć 🦀\n\ndone");
        assert_ne!(expected.1[0].id, expected.1[1].id);
    }

    #[test]
    fn every_marker_split_is_incremental_and_never_exposes_partial_call() {
        let input = xml();
        for split in input.char_indices().map(|(i, _)| i) {
            let mut p = parser();
            let first = p.push(&input[..split]).unwrap();
            assert!(first.is_empty(), "split {split}: {first:?}");
            let mut events = p.push(&input[split..]).unwrap();
            events.extend(p.finish().unwrap());
            assert_eq!(collect(&events).1.len(), 1);
        }
        let mut p = parser();
        assert_eq!(
            p.push("hello <tool_").unwrap(),
            vec![ToolEvent::Text("hello ".into())]
        );
        assert!(p.push("call>").unwrap().is_empty());
    }

    #[test]
    fn ordinary_similar_markers_and_utf8_are_preserved() {
        let input = "a <tool_box> żółw </think> < x";
        assert_eq!(parse_all(input).unwrap(), (input.into(), vec![]));
    }

    #[test]
    fn old_json_calls_accept_object_or_encoded_object() {
        for arguments in [json!({"code":"008"}), json!("{\"code\":\"008\"}")] {
            let call = json!({"name":"lookup","arguments":arguments});
            let (_, calls) = parse_all(&format!("<tool_call>{call}</tool_call>")).unwrap();
            assert_eq!(
                serde_json::from_str::<Value>(&calls[0].function.arguments).unwrap(),
                json!({"code":"008"})
            );
        }
    }

    #[test]
    fn json_string_markers_do_not_end_the_call_early() {
        let body = json!({"name":"lookup","arguments":{"code":"literal </tool_call> inside text"}});
        let (_, calls) = parse_all(&format!("<tool_call>{body}</tool_call>")).unwrap();
        assert_eq!(
            serde_json::from_str::<Value>(&calls[0].function.arguments).unwrap()["code"],
            "literal </tool_call> inside text"
        );
    }

    #[test]
    fn xml_strings_keep_spaces_and_embedded_newlines() {
        let input = "<tool_call><function=lookup><parameter=code>\n  Łódź\n\n</parameter></function></tool_call>";
        let (_, calls) = parse_all(input).unwrap();
        assert_eq!(
            serde_json::from_str::<Value>(&calls[0].function.arguments).unwrap()["code"],
            "  Łódź\n"
        );
    }

    #[test]
    fn xml_parameter_closing_tag_literals_survive_unicode_chunk_boundaries() {
        let code = "def przykład():\n    return '</tool_call> </function> 🦀'";
        let input = format!(
            "<tool_call><function=lookup><parameter=code>\n{code}\n</parameter></function></tool_call>"
        );
        let (_, whole_calls) = parse_all(&input).unwrap();
        assert_eq!(whole_calls.len(), 1);
        assert_eq!(
            serde_json::from_str::<Value>(&whole_calls[0].function.arguments).unwrap()["code"],
            code
        );

        let mut p = parser();
        let mut events = Vec::new();
        for ch in input.chars() {
            events.extend(p.push(&ch.to_string()).unwrap());
        }
        events.extend(p.finish().unwrap());
        let (text, calls) = collect(&events);
        assert!(text.is_empty());
        assert_eq!(calls, whole_calls);
    }

    #[test]
    fn rejects_unknown_name_nonobject_arguments_and_duplicate_parameters() {
        for body in [
            "{\"name\":\"absent\",\"arguments\":{}}",
            "{\"name\":\"lookup\",\"arguments\":[]}",
            "{\"name\":\"lookup\",\"arguments\":{}}",
            "<function=lookup><parameter=code>a</parameter><parameter=code>b</parameter></function>",
            "<function=lookup><parameter=count>not-a-number</parameter></function>",
        ] {
            assert!(
                parse_all(&format!("<tool_call>{body}</tool_call>")).is_err(),
                "{body}"
            );
        }
    }

    #[test]
    fn required_named_none_and_parallel_constraints_are_enforced() {
        let defs = definitions();
        let mut required = ToolOutputParser::new(
            ToolConfig::parse(Some(&defs), Some(&json!("required")), None).unwrap(),
            "id".into(),
            false,
        );
        required.push("answer without call").unwrap();
        assert!(required.finish().is_err());
        let mut named = ToolOutputParser::new(
            ToolConfig::parse(
                Some(&defs),
                Some(&json!({"type":"function","function":{"name":"other"}})),
                None,
            )
            .unwrap(),
            "id".into(),
            false,
        );
        assert!(named.push(xml()).is_err());
        let mut none = ToolOutputParser::new(ToolConfig::default(), "id".into(), false);
        assert!(none.push(xml()).is_err());
        let mut sequential = ToolOutputParser::new(
            ToolConfig::parse(Some(&defs), None, Some(&json!(false))).unwrap(),
            "id".into(),
            false,
        );
        sequential.push(xml()).unwrap();
        assert!(sequential.push(xml()).is_err());
    }

    #[test]
    fn malformed_and_truncated_calls_fail_explicitly() {
        for input in [
            "<tool_call>",
            "<tool_call>{",
            "<tool_call><function=lookup></tool_call>",
            "<tool_call>{no}</tool_call>",
            "<tool_call><function=lookup></function>junk</tool_call>",
            "<tool_call><function=lookup><parameter=code>x</function></tool_call>",
            "<tool_",
        ] {
            assert!(parse_all(input).is_err(), "{input}");
        }
    }

    #[test]
    fn failed_or_finished_parsers_cannot_resume_generation() {
        let mut p = parser();
        assert!(p.push("<tool_call>{bad}</tool_call>").is_err());
        assert!(p.push(xml()).is_err());
        assert!(p.finish().is_err());
        let mut p = parser();
        p.finish().unwrap();
        assert!(p.push("new text").is_err());
    }

    #[test]
    fn tool_like_reasoning_stays_text_for_explicit_and_preopened_think() {
        let reasoning = format!("<think>consider {} first</think>", xml());
        let input = format!("{reasoning}{}", xml());
        let (text, calls) = parse_all(&input).unwrap();
        assert_eq!(text, reasoning);
        assert_eq!(calls.len(), 1);
        let mut p = ToolOutputParser::new(config(), "call_test".into(), true);
        let input = format!("consider {} first</think>{}", xml(), xml());
        let mut events = Vec::new();
        for ch in input.chars() {
            events.extend(p.push(&ch.to_string()).unwrap());
        }
        events.extend(p.finish().unwrap());
        let (text, calls) = collect(&events);
        assert_eq!(text, format!("consider {} first</think>", xml()));
        assert_eq!(calls.len(), 1);
    }
}
