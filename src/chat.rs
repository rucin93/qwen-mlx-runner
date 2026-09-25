use std::fs;
use std::path::Path;

use anyhow::{Context, Result, anyhow, bail, ensure};
use minijinja::{Environment, context};
use serde::{Deserialize, Serialize};
use tokenizers::Tokenizer;

pub mod sampling;
pub mod tools;
pub use sampling::SamplingOptions;
pub use tools::{ToolCall, ToolChoice, ToolConfig, ToolEvent, ToolFunction, ToolOutputParser};

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct Message {
    pub role: String,
    pub content: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCall>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
}

#[derive(Clone, Debug, Default)]
pub struct GenerationRequest {
    pub messages: Vec<Message>,
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub seed: u64,
    pub enable_thinking: bool,
    pub tools: ToolConfig,
    pub sampling: SamplingOptions,
    pub reasoning_effort: Option<String>,
}

impl GenerationRequest {
    pub fn preserve_special_tokens(&self) -> bool {
        self.enable_thinking || self.tools.enabled()
    }
}

#[derive(Clone, Debug)]
pub struct GenerationOutput {
    pub text: String,
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub finish_reason: String,
    pub prefill_seconds: f64,
    pub decode_seconds: f64,
}

pub trait TextGenerator: Send {
    fn generate(
        &mut self,
        request: &GenerationRequest,
        on_text: &mut dyn FnMut(&str) -> bool,
    ) -> Result<GenerationOutput>;
    fn model_id(&self) -> &str;
    /// Validate an HTTP request and bound its output budget to available context
    /// before generation can allocate buffers. `usize::MAX` means no client cap.
    fn prepare_request(&self, request: &mut GenerationRequest) -> Result<()>;
    /// Allows HTTP validation before queuing an otherwise expensive generation.
    fn vocab_size(&self) -> Option<usize> {
        None
    }
}

#[derive(Debug)]
pub struct ContextLengthExceeded {
    pub prompt_tokens: usize,
    pub capacity: usize,
}
impl std::fmt::Display for ContextLengthExceeded {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "prompt ({} tokens) must be shorter than context {} to leave room for a completion",
            self.prompt_tokens, self.capacity
        )
    }
}
impl std::error::Error for ContextLengthExceeded {}

pub struct ChatTokenizer {
    tokenizer: Tokenizer,
    template: String,
}

impl ChatTokenizer {
    pub fn prepare_request(&self, request: &mut GenerationRequest, capacity: usize) -> Result<()> {
        anyhow::ensure!(request.max_tokens > 0, "max_tokens must be positive");
        let tokens = self.encode(&self.render_request(request)?)?;
        anyhow::ensure!(!tokens.is_empty(), "empty prompt after tokenization");
        let remaining = capacity.saturating_sub(tokens.len());
        if remaining == 0 {
            return Err(ContextLengthExceeded {
                prompt_tokens: tokens.len(),
                capacity,
            }
            .into());
        }
        request.max_tokens = request.max_tokens.min(remaining);
        Ok(())
    }
    pub fn load(directory: &Path) -> Result<Self> {
        let tokenizer = Tokenizer::from_file(directory.join("tokenizer.json"))
            .map_err(|e| anyhow!("cannot load tokenizer.json: {e}"))?;
        let template_path = directory.join("chat_template.jinja");
        let template = if template_path.is_file() {
            fs::read_to_string(&template_path)
                .with_context(|| format!("cannot read {}", template_path.display()))?
        } else {
            let config_path = directory.join("tokenizer_config.json");
            let config: serde_json::Value =
                serde_json::from_slice(&fs::read(&config_path).with_context(|| {
                    format!(
                        "missing chat template: {} and {}",
                        template_path.display(),
                        config_path.display()
                    )
                })?)
                .context("invalid tokenizer_config.json")?;
            config["chat_template"]
                .as_str()
                .ok_or_else(|| anyhow!("tokenizer_config.json has no string chat_template"))?
                .to_owned()
        };
        if template.is_empty() {
            bail!("chat template is empty");
        }
        let mut env = Self::environment();
        env.add_template("chat", &template)
            .context("invalid checkpoint chat template")?;
        Ok(Self {
            tokenizer,
            template,
        })
    }

    fn environment<'a>() -> Environment<'a> {
        let mut env = Environment::new();
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        env.add_function(
            "raise_exception",
            |message: String| -> std::result::Result<String, minijinja::Error> {
                Err(minijinja::Error::new(
                    minijinja::ErrorKind::InvalidOperation,
                    message,
                ))
            },
        );
        env
    }

    pub fn render(&self, messages: &[Message], enable_thinking: bool) -> Result<String> {
        let mut env = Self::environment();
        env.add_template("chat", &self.template)
            .context("invalid checkpoint chat template")?;
        let template = env.get_template("chat")?;
        template
            .render(context! {
                messages => messages,
                add_generation_prompt => true,
                enable_thinking => enable_thinking,
                preserve_thinking => false,
                tools => Vec::<serde_json::Value>::new(),
                add_vision_id => false,
            })
            .context("checkpoint chat template rejected messages")
    }

    /// Adapt API history to the checkpoint's unchanged template. OpenAI stores
    /// historical function arguments as JSON strings; Qwen's template iterates
    /// their object items. Keep the public messages intact and normalize a copy.
    pub fn render_request(&self, request: &GenerationRequest) -> Result<String> {
        if let Some(effort) = request.reasoning_effort.as_deref() {
            ensure!(
                matches!(effort, "none" | "low" | "medium" | "xhigh"),
                "unsupported checkpoint reasoning effort {effort}"
            );
        }
        let messages = Self::request_messages(request)?;
        let preserve_thinking = request.messages.iter().any(|message| {
            message
                .reasoning_content
                .as_ref()
                .is_some_and(|reasoning| !reasoning.is_empty())
        });
        let mut values = serde_json::json!({
            "messages": messages,
            "add_generation_prompt": true,
            "enable_thinking": request.enable_thinking
                && request.reasoning_effort.as_deref() != Some("none"),
            "preserve_thinking": preserve_thinking,
            "tools": request.tools.effective_definitions(),
            "add_vision_id": false,
        });
        // An absent effort must be undefined, not JSON null: the checkpoint's
        // Jinja default filter supplies xhigh only for an undefined variable.
        if let Some(effort) = &request.reasoning_effort {
            values["reasoning_effort"] = serde_json::json!(effort);
        }
        let mut env = Self::environment();
        env.add_template("chat", &self.template)
            .context("invalid checkpoint chat template")?;
        env.get_template("chat")?
            .render(values)
            .context("checkpoint chat template rejected messages")
    }

    fn request_messages(request: &GenerationRequest) -> Result<Vec<serde_json::Value>> {
        let prefix = request
            .messages
            .iter()
            .take_while(|message| matches!(message.role.as_str(), "system" | "developer"))
            .count();
        ensure!(
            request.messages[prefix..]
                .iter()
                .all(|message| !matches!(message.role.as_str(), "system" | "developer")),
            "system and developer instructions must precede conversation messages"
        );
        let mut rendered = Vec::with_capacity(request.messages.len() + 1);
        let mut functions = std::collections::HashMap::new();
        for message in &request.messages {
            let mut value = serde_json::to_value(message)?;
            let mut annotations = Vec::new();
            let inferred_name = message
                .tool_call_id
                .as_ref()
                .and_then(|id| functions.get(id));
            if let Some(name) = message.name.as_ref().or(inferred_name) {
                annotations.push(format!("[name: {name}]"));
            }
            if let Some(id) = &message.tool_call_id {
                annotations.push(format!("[tool_call_id: {id}]"));
            }
            for (index, call) in message.tool_calls.iter().enumerate() {
                let arguments: serde_json::Value = serde_json::from_str(&call.function.arguments)
                    .with_context(|| {
                    format!("tool call {} has invalid JSON arguments", call.id)
                })?;
                ensure!(
                    arguments.is_object(),
                    "tool call {} arguments must be a JSON object",
                    call.id
                );
                value["tool_calls"][index]["function"]["arguments"] = arguments;
                // Stock Qwen templates ignore call IDs. Keep ordered bindings
                // in text so parallel replies can retain their association,
                // including two invocations of the same function.
                annotations.push(format!(
                    "[tool_call_id: {}; function: {}]",
                    call.id, call.function.name
                ));
                functions.insert(call.id.clone(), call.function.name.clone());
            }
            if !annotations.is_empty() {
                annotations.push(message.content.clone());
                value["content"] = serde_json::json!(annotations.join("\n"));
            }
            rendered.push(value);
        }

        // Qwen permits one initial system message. Keep system instructions
        // before developer instructions, with stable order within each role.
        // A single ordinary system message remains byte-for-byte unchanged.
        if prefix > 0 {
            let mut parts = rendered[..prefix]
                .iter()
                .filter(|message| message["role"] == "system")
                .map(|message| message["content"].as_str().unwrap_or_default().to_owned())
                .collect::<Vec<_>>();
            let developer = rendered[..prefix]
                .iter()
                .filter(|message| message["role"] == "developer")
                .map(|message| message["content"].as_str().unwrap_or_default())
                .collect::<Vec<_>>()
                .join("\n\n");
            if !developer.is_empty() {
                parts.push(format!(
                    "[Developer instructions; follow system instructions if they conflict]\n{developer}"
                ));
            }
            if prefix != 1 || request.messages[0].role != "system" {
                rendered.splice(
                    ..prefix,
                    [serde_json::json!({"role":"system", "content":parts.join("\n\n")})],
                );
            }
        }
        if let Some(instruction) = request.tools.instruction() {
            if let Some(first) = rendered
                .first_mut()
                .filter(|message| message["role"] == "system")
            {
                let content = first["content"].as_str().unwrap_or_default();
                first["content"] = serde_json::json!(if content.is_empty() {
                    instruction
                } else {
                    format!("{content}\n\n{instruction}")
                });
            } else {
                rendered.insert(
                    0,
                    serde_json::json!({"role":"system", "content":instruction}),
                );
            }
        }
        Ok(rendered)
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        Ok(self
            .tokenizer
            .encode(text, false)
            .map_err(|e| anyhow!("tokenization failed: {e}"))?
            .get_ids()
            .to_vec())
    }

    /// Decodes ordinary visible text, excluding the tokenizer's registered special tokens.
    pub fn decode(&self, tokens: &[u32]) -> Result<String> {
        self.tokenizer
            .decode(tokens, true)
            .map_err(|e| anyhow!("decoding failed: {e}"))
    }

    /// Keep registered special tokens such as `</think>` visible when serving reasoning.
    pub fn decode_with_special_tokens(&self, tokens: &[u32]) -> Result<String> {
        self.tokenizer
            .decode(tokens, false)
            .map_err(|e| anyhow!("decoding failed: {e}"))
    }
}

/// Small deterministic PRNG so identical seeds give identical samples on every host.
pub struct Sampler {
    state: u64,
}

impl Sampler {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    pub(crate) fn uniform(&mut self) -> f64 {
        self.state = self.state.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        ((z ^ (z >> 31)) >> 11) as f64 * (1.0 / ((1u64 << 53) as f64))
    }

    pub fn sample(
        &mut self,
        logits: &[f32],
        temperature: f32,
        top_p: f32,
        top_k: usize,
    ) -> Result<u32> {
        if logits.is_empty() {
            bail!("cannot sample empty logits");
        }
        if logits.len() > u32::MAX as usize {
            bail!("vocabulary exceeds u32 token IDs");
        }
        if !temperature.is_finite() || temperature < 0.0 {
            bail!("temperature must be finite and nonnegative");
        }
        if !top_p.is_finite() || !(0.0 < top_p && top_p <= 1.0) {
            bail!("top_p must be in (0, 1]");
        }
        if logits.iter().any(|v| !v.is_finite()) {
            bail!("logits contain non-finite values");
        }
        if temperature == 0.0 {
            let best = logits
                .iter()
                .enumerate()
                .max_by(|(a, av), (b, bv)| av.total_cmp(bv).then_with(|| b.cmp(a)))
                .expect("nonempty logits")
                .0;
            return Ok(best as u32);
        }
        let mut order: Vec<usize> = (0..logits.len()).collect();
        let compare = |&a: &usize, &b: &usize| logits[b].total_cmp(&logits[a]).then(a.cmp(&b));
        let count = if top_k == 0 {
            order.len()
        } else {
            top_k.min(order.len())
        };
        if count < order.len() {
            order.select_nth_unstable_by(count - 1, compare);
        }
        order.truncate(count);
        order.sort_unstable_by(compare);
        let max = logits[order[0]] as f64;
        let weights: Vec<f64> = order
            .iter()
            .map(|&i| (((logits[i] as f64) - max) / temperature as f64).exp())
            .collect();
        let sum: f64 = weights.iter().sum();
        if !sum.is_finite() || sum <= 0.0 {
            bail!("sampling distribution is invalid");
        }
        let mut nucleus = order.len();
        let mut cumulative = 0.0;
        for (i, &weight) in weights.iter().enumerate() {
            cumulative += weight / sum;
            if cumulative >= top_p as f64 {
                nucleus = i + 1;
                break;
            }
        }
        let total: f64 = weights[..nucleus].iter().sum();
        let target = self.uniform() * total;
        let mut running = 0.0;
        for (i, &weight) in weights[..nucleus].iter().enumerate() {
            running += weight;
            if target < running {
                return Ok(order[i] as u32);
            }
        }
        Ok(order[nucleus - 1] as u32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokenizers::models::wordlevel::WordLevel;

    fn fixture(template: &str) -> ChatTokenizer {
        let directory = tempfile::tempdir().unwrap();
        let model = WordLevel::builder()
            .vocab([("[UNK]".to_owned(), 0), ("hello".to_owned(), 1)].into())
            .unk_token("[UNK]".to_owned())
            .build()
            .unwrap();
        Tokenizer::new(model)
            .save(directory.path().join("tokenizer.json"), false)
            .unwrap();
        fs::write(directory.path().join("chat_template.jinja"), template).unwrap();
        ChatTokenizer::load(directory.path()).unwrap()
    }

    #[test]
    fn request_preflight_accounts_for_rendered_prompt_and_output_budget() {
        let t = fixture("{{ messages[0].content }}");
        let mut request = GenerationRequest {
            messages: vec![Message {
                role: "user".into(),
                content: "hello".into(),
                ..Default::default()
            }],
            max_tokens: 4,
            ..Default::default()
        };
        t.prepare_request(&mut request, 5).unwrap();
        assert_eq!(request.max_tokens, 4);
        t.prepare_request(&mut request, 4).unwrap();
        assert_eq!(request.max_tokens, 3);
        let error = t.prepare_request(&mut request, 1).unwrap_err();
        let detail = error.downcast_ref::<ContextLengthExceeded>().unwrap();
        assert_eq!(detail.prompt_tokens, 1);
        assert_eq!(detail.capacity, 1);
        request.max_tokens = usize::MAX;
        t.prepare_request(&mut request, usize::MAX).unwrap();
        assert_eq!(request.max_tokens, usize::MAX - 1);
        request.max_tokens = 0;
        assert!(t.prepare_request(&mut request, 5).is_err());
    }

    #[test]
    fn renders_checkpoint_template_without_rewriting_it() {
        let t = fixture(
            "{{ messages[0].role }}:{{ messages[0].content }}:{{ enable_thinking }}:{{ preserve_thinking }}:{{ add_generation_prompt }}",
        );
        let result = t
            .render(
                &[Message {
                    role: "user".into(),
                    content: "hello".into(),
                    ..Default::default()
                }],
                false,
            )
            .unwrap();
        assert_eq!(result, "user:hello:False:False:True");
        assert_eq!(t.encode("hello").unwrap(), vec![1]);
        assert_eq!(t.decode(&[1]).unwrap(), "hello");
    }

    #[test]
    fn refuses_missing_template() {
        let directory = tempfile::tempdir().unwrap();
        let model = WordLevel::builder()
            .vocab([("[UNK]".to_owned(), 0)].into())
            .unk_token("[UNK]".to_owned())
            .build()
            .unwrap();
        Tokenizer::new(model)
            .save(directory.path().join("tokenizer.json"), false)
            .unwrap();
        assert!(ChatTokenizer::load(directory.path()).is_err());
    }

    #[test]
    fn loads_only_explicit_config_template_fallback() {
        let directory = tempfile::tempdir().unwrap();
        let model = WordLevel::builder()
            .vocab([("[UNK]".to_owned(), 0)].into())
            .unk_token("[UNK]".to_owned())
            .build()
            .unwrap();
        Tokenizer::new(model)
            .save(directory.path().join("tokenizer.json"), false)
            .unwrap();
        fs::write(
            directory.path().join("tokenizer_config.json"),
            r#"{"chat_template":"{{ messages[0].content }}"}"#,
        )
        .unwrap();
        let chat = ChatTokenizer::load(directory.path()).unwrap();
        assert_eq!(
            chat.render(
                &[Message {
                    role: "user".into(),
                    content: "hello".into(),
                    ..Default::default()
                }],
                false
            )
            .unwrap(),
            "hello"
        );
    }

    #[test]
    fn renders_official_qwen38_text_template() {
        // Fixture copied from Qwen/Qwen3.8-27B's chat_template.jinja on Hugging Face.
        // The template is used unchanged; this catches Python-Jinja compatibility regressions.
        let chat = fixture(include_str!("../tests/fixtures/qwen38-template.jinja"));
        let messages = [Message {
            role: "user".into(),
            content: "Hello".into(),
            ..Default::default()
        }];
        assert_eq!(
            chat.render(&messages, false).unwrap(),
            "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
        );
        let with_thinking = chat.render(&messages, true).unwrap();
        assert!(with_thinking.contains("Reasoning effort is set to xhigh."));
        assert!(with_thinking.ends_with("<|im_start|>assistant\n<think>\n"));
        let history = [
            Message {
                role: "user".into(),
                content: "First".into(),
                ..Default::default()
            },
            Message {
                role: "assistant".into(),
                content: "Answer".into(),
                ..Default::default()
            },
            Message {
                role: "user".into(),
                content: "Next".into(),
                ..Default::default()
            },
        ];
        let rendered = chat.render(&history, false).unwrap();
        assert!(rendered.contains("<|im_start|>assistant\nAnswer<|im_end|>"));
    }

    #[test]
    fn request_render_keeps_plain_checkpoint_prompts_unchanged() {
        let chat = fixture(include_str!("../tests/fixtures/qwen38-template.jinja"));
        for thinking in [false, true] {
            let messages: Vec<Message> = serde_json::from_value(serde_json::json!([
                {"role":"system","content":"Be concise."},
                {"role":"user","content":"Hello"},
                {"role":"assistant","content":"Hi"},
                {"role":"user","content":"Next"}
            ]))
            .unwrap();
            let old = chat.render(&messages, thinking).unwrap();
            let request = GenerationRequest {
                messages,
                enable_thinking: thinking,
                ..Default::default()
            };
            assert_eq!(chat.render_request(&request).unwrap(), old);
            assert_eq!(request.preserve_special_tokens(), thinking);
        }
    }

    #[test]
    fn request_render_normalizes_instruction_prefix_and_preserves_names_and_reasoning() {
        let chat = fixture(include_str!("../tests/fixtures/qwen38-template.jinja"));
        let messages = serde_json::from_value(serde_json::json!([
            {"role":"developer","content":"Use JSON."},
            {"role":"system","content":"Protect private data."},
            {"role":"developer","content":"Keep fields short."},
            {"role":"user","content":"First","name":"alice"},
            {"role":"assistant","content":"Answer","reasoning_content":"Check the units."},
            {"role":"user","content":"Next","name":"bob"}
        ]))
        .unwrap();
        let request = GenerationRequest {
            messages,
            ..Default::default()
        };
        let rendered = chat.render_request(&request).unwrap();
        assert_eq!(rendered.matches("<|im_start|>system\n").count(), 1);
        assert!(
            rendered.find("Protect private data.").unwrap() < rendered.find("Use JSON.").unwrap()
        );
        assert!(rendered.contains("Keep fields short."));
        assert!(rendered.contains("[name: alice]\nFirst"));
        assert!(rendered.contains("[name: bob]\nNext"));
        assert!(rendered.contains("<think>\nCheck the units.\n</think>\n\nAnswer"));
        for role in ["system", "developer"] {
            let messages = serde_json::from_value(serde_json::json!([
                {"role":"user","content":"Hello"}, {"role":role,"content":"Late instruction"}
            ]))
            .unwrap();
            assert!(
                chat.render_request(&GenerationRequest {
                    messages,
                    ..Default::default()
                })
                .is_err()
            );
        }
    }

    #[test]
    fn request_render_supports_checkpoint_reasoning_efforts_and_rejects_unknown_values() {
        let chat = fixture(include_str!("../tests/fixtures/qwen38-template.jinja"));
        for effort in ["none", "low", "medium", "xhigh"] {
            let request = GenerationRequest {
                messages: serde_json::from_value(
                    serde_json::json!([{"role":"user","content":"Hello"}]),
                )
                .unwrap(),
                enable_thinking: effort != "none",
                reasoning_effort: Some(effort.into()),
                ..Default::default()
            };
            let rendered = chat.render_request(&request).unwrap();
            if effort == "none" {
                assert!(rendered.ends_with("<think>\n\n</think>\n\n"));
            } else {
                assert!(rendered.ends_with("<think>\n"));
                if effort != "medium" {
                    assert!(rendered.contains(&format!("Reasoning effort is set to {effort}.")));
                }
            }
        }
        let request = GenerationRequest {
            messages: serde_json::from_value(
                serde_json::json!([{"role":"user","content":"Hello"}]),
            )
            .unwrap(),
            enable_thinking: true,
            reasoning_effort: Some("unsupported".into()),
            ..Default::default()
        };
        assert!(chat.render_request(&request).is_err());
    }

    #[test]
    fn request_render_round_trips_tool_arguments_and_tool_results() {
        let chat = fixture(include_str!("../tests/fixtures/qwen38-template.jinja"));
        let messages = serde_json::from_value(serde_json::json!([
            {"role":"user","content":"Check Warsaw weather"},
            {"role":"assistant","content":"","reasoning_content":"Use the weather function.","tool_calls":[
                {"id":"call_weather","type":"function","function":{"name":"weather","arguments":"{\"city\":\"Warsaw\",\"days\":2}"}}
            ]},
            {"role":"tool","content":"Sunny","name":"weather","tool_call_id":"call_weather"}
        ])).unwrap();
        let rendered = chat
            .render_request(&GenerationRequest {
                messages,
                ..Default::default()
            })
            .unwrap();
        assert!(rendered.contains("<function=weather>\n"));
        assert!(rendered.contains("<parameter=city>\nWarsaw\n</parameter>"));
        assert!(rendered.contains("<parameter=days>\n2\n</parameter>"));
        assert!(rendered.contains("Use the weather function."));
        assert!(rendered.contains("[tool_call_id: call_weather; function: weather]"));
        assert!(rendered.contains("[tool_call_id: call_weather]"));
        assert!(rendered.contains("[name: weather]"));
        assert!(rendered.contains("<tool_response>\n"));
        assert!(rendered.contains("Sunny"));
        for arguments in ["not json", "[]", "null", "42"] {
            let messages = serde_json::from_value(serde_json::json!([
                {"role":"user","content":"Run it"},
                {"role":"assistant","content":"","tool_calls":[
                    {"id":"call_bad","type":"function","function":{"name":"weather","arguments":arguments}}
                ]}
            ])).unwrap();
            assert!(
                chat.render_request(&GenerationRequest {
                    messages,
                    ..Default::default()
                })
                .is_err()
            );
        }
    }

    #[test]
    fn request_render_passes_effective_tools_and_forced_policy_to_checkpoint() {
        let chat = fixture(include_str!("../tests/fixtures/qwen38-template.jinja"));
        let definitions = serde_json::json!([
            {"type":"function","function":{"name":"weather","parameters":{"type":"object","properties":{"city":{"type":"string"}}}}},
            {"type":"function","function":{"name":"clock","parameters":{"type":"object","properties":{}}}}
        ]);
        let messages = || {
            serde_json::from_value(serde_json::json!([{"role":"user","content":"Hello"}])).unwrap()
        };
        let plain = chat
            .render_request(&GenerationRequest {
                messages: messages(),
                ..Default::default()
            })
            .unwrap();
        let none =
            tools::ToolConfig::parse(Some(&definitions), Some(&serde_json::json!("none")), None)
                .unwrap();
        let disabled = GenerationRequest {
            messages: messages(),
            tools: none,
            ..Default::default()
        };
        assert_eq!(chat.render_request(&disabled).unwrap(), plain);
        assert!(!disabled.preserve_special_tokens());
        let force = serde_json::json!({"type":"function","function":{"name":"weather"}});
        let forced = tools::ToolConfig::parse(Some(&definitions), Some(&force), None).unwrap();
        let instruction = forced.instruction().unwrap();
        let request = GenerationRequest {
            messages: messages(),
            tools: forced,
            ..Default::default()
        };
        let rendered = chat.render_request(&request).unwrap();
        assert!(rendered.contains("<tools>"));
        assert!(rendered.contains("weather"));
        assert!(!rendered.contains("\"clock\""));
        assert!(rendered.contains(&instruction));
        assert!(request.preserve_special_tokens());
    }

    #[test]
    fn seeded_sampling_and_validation() {
        let logits = [0.0, 2.0, 1.0];
        assert_eq!(Sampler::new(1).sample(&logits, 0.0, 1.0, 0).unwrap(), 1);
        let a = Sampler::new(42).sample(&logits, 1.0, 1.0, 0).unwrap();
        let b = Sampler::new(42).sample(&logits, 1.0, 1.0, 0).unwrap();
        assert_eq!(a, b);
        assert!(Sampler::new(1).sample(&[f32::NAN], 1.0, 1.0, 0).is_err());
        assert!(Sampler::new(1).sample(&logits, -1.0, 1.0, 0).is_err());
    }
}
