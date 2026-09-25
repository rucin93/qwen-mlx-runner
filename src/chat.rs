use std::fs;
use std::path::Path;

use anyhow::{Context, Result, anyhow, bail};
use minijinja::{Environment, context};
use serde::{Deserialize, Serialize};
use tokenizers::Tokenizer;

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Message {
    pub role: String,
    pub content: String,
}

#[derive(Clone, Debug)]
pub struct GenerationRequest {
    pub messages: Vec<Message>,
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub seed: u64,
    pub enable_thinking: bool,
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
}

pub struct ChatTokenizer {
    tokenizer: Tokenizer,
    template: String,
}

impl ChatTokenizer {
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

    fn uniform(&mut self) -> f64 {
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
    fn renders_checkpoint_template_without_rewriting_it() {
        let t = fixture(
            "{{ messages[0].role }}:{{ messages[0].content }}:{{ enable_thinking }}:{{ preserve_thinking }}:{{ add_generation_prompt }}",
        );
        let result = t
            .render(
                &[Message {
                    role: "user".into(),
                    content: "hello".into(),
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
                    content: "hello".into()
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
            },
            Message {
                role: "assistant".into(),
                content: "Answer".into(),
            },
            Message {
                role: "user".into(),
                content: "Next".into(),
            },
        ];
        let rendered = chat.render(&history, false).unwrap();
        assert!(rendered.contains("<|im_start|>assistant\nAnswer<|im_end|>"));
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
