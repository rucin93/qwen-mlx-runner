use anyhow::{Context, Result, ensure};
use serde_json::Value;

#[derive(Clone, Debug)]
pub struct ModelConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub linear_num_key_heads: usize,
    pub linear_num_value_heads: usize,
    pub linear_key_head_dim: usize,
    pub linear_value_head_dim: usize,
    pub linear_conv_kernel_dim: usize,
    pub full_attention_interval: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub partial_rotary_factor: f32,
    pub max_position_embeddings: usize,
    pub tie_word_embeddings: bool,
    pub eos_token_ids: Vec<u32>,
    pub layer_types: Vec<String>,
}

impl ModelConfig {
    pub fn from_json(json: &str) -> Result<Self> {
        let root: Value = serde_json::from_str(json).context("invalid config.json")?;
        let text = root.get("text_config").unwrap_or(&root);
        let model_type = root.get("model_type").and_then(Value::as_str).unwrap_or("");
        let text_type = text.get("model_type").and_then(Value::as_str).unwrap_or("");
        ensure!(
            matches!(model_type, "qwen3_5" | "qwen3_5_text")
                && matches!(text_type, "qwen3_5_text" | ""),
            "unsupported model architecture: {model_type}/{text_type}"
        );
        if let Some(architectures) = root.get("architectures").and_then(Value::as_array) {
            ensure!(
                architectures.iter().all(|a| matches!(
                    a.as_str(),
                    Some("Qwen3_5ForConditionalGeneration" | "Qwen3_5ForCausalLM")
                )),
                "unsupported architecture"
            );
        }
        for field in [
            "num_experts",
            "num_local_experts",
            "moe_intermediate_size",
            "shared_expert_intermediate_size",
        ] {
            ensure!(
                text.get(field).is_none(),
                "MoE field {field} is unsupported"
            );
        }
        if let Some(rope) = text.get("rope_scaling") {
            ensure!(rope.is_null(), "RoPE scaling is unsupported");
        }
        if let Some(attention_bias) = text.get("attention_bias") {
            ensure!(
                attention_bias.as_bool() == Some(false),
                "text_config.attention_bias must be false"
            );
        }
        if let Some(hidden_act) = text.get("hidden_act") {
            ensure!(
                hidden_act.as_str() == Some("silu"),
                "text_config.hidden_act must be silu"
            );
        }
        let rope = text.get("rope_parameters");
        if let Some(rope) = rope {
            ensure!(
                rope.get("rope_type")
                    .and_then(Value::as_str)
                    .unwrap_or("default")
                    == "default",
                "unsupported RoPE type"
            );
            ensure!(rope.get("factor").is_none(), "RoPE scaling is unsupported");
        }
        let get_usize = |key: &str| -> Result<usize> {
            let n = text
                .get(key)
                .and_then(Value::as_u64)
                .with_context(|| format!("missing or invalid text_config.{key}"))?;
            usize::try_from(n).with_context(|| format!("text_config.{key} overflows usize"))
        };
        let get_f32 = |key: &str| -> Result<f32> {
            let n = text
                .get(key)
                .and_then(Value::as_f64)
                .with_context(|| format!("missing or invalid text_config.{key}"))?;
            ensure!(
                n.is_finite() && n.abs() <= f32::MAX as f64,
                "invalid text_config.{key}"
            );
            Ok(n as f32)
        };
        let rope_theta = rope
            .and_then(|r| r.get("rope_theta"))
            .or_else(|| text.get("rope_theta"))
            .and_then(Value::as_f64)
            .context("missing RoPE theta")? as f32;
        let partial_rotary_factor = rope
            .and_then(|r| r.get("partial_rotary_factor"))
            .or_else(|| text.get("partial_rotary_factor"))
            .and_then(Value::as_f64)
            .context("missing partial rotary factor")? as f32;
        let layer_types = text
            .get("layer_types")
            .and_then(Value::as_array)
            .context("missing text_config.layer_types")?
            .iter()
            .map(|v| v.as_str().map(str::to_owned).context("invalid layer type"))
            .collect::<Result<Vec<_>>>()?;
        let eos = root
            .get("eos_token_id")
            .or_else(|| {
                root.get("generation_config")
                    .and_then(|v| v.get("eos_token_id"))
            })
            .or_else(|| text.get("eos_token_id"));
        let config = Self {
            hidden_size: get_usize("hidden_size")?,
            intermediate_size: get_usize("intermediate_size")?,
            num_hidden_layers: get_usize("num_hidden_layers")?,
            num_attention_heads: get_usize("num_attention_heads")?,
            num_key_value_heads: get_usize("num_key_value_heads")?,
            head_dim: get_usize("head_dim")?,
            vocab_size: get_usize("vocab_size")?,
            linear_num_key_heads: get_usize("linear_num_key_heads")?,
            linear_num_value_heads: get_usize("linear_num_value_heads")?,
            linear_key_head_dim: get_usize("linear_key_head_dim")?,
            linear_value_head_dim: get_usize("linear_value_head_dim")?,
            linear_conv_kernel_dim: get_usize("linear_conv_kernel_dim")?,
            full_attention_interval: get_usize("full_attention_interval")?,
            rms_norm_eps: get_f32("rms_norm_eps")?,
            rope_theta,
            partial_rotary_factor,
            max_position_embeddings: get_usize("max_position_embeddings")?,
            tie_word_embeddings: text
                .get("tie_word_embeddings")
                .or_else(|| root.get("tie_word_embeddings"))
                .and_then(Value::as_bool)
                .unwrap_or(false),
            eos_token_ids: parse_eos(eos)?,
            layer_types,
        };
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<()> {
        for (name, n) in [
            ("hidden_size", self.hidden_size),
            ("intermediate_size", self.intermediate_size),
            ("num_hidden_layers", self.num_hidden_layers),
            ("num_attention_heads", self.num_attention_heads),
            ("num_key_value_heads", self.num_key_value_heads),
            ("head_dim", self.head_dim),
            ("vocab_size", self.vocab_size),
            ("linear_num_key_heads", self.linear_num_key_heads),
            ("linear_num_value_heads", self.linear_num_value_heads),
            ("linear_key_head_dim", self.linear_key_head_dim),
            ("linear_value_head_dim", self.linear_value_head_dim),
            ("linear_conv_kernel_dim", self.linear_conv_kernel_dim),
            ("full_attention_interval", self.full_attention_interval),
            ("max_position_embeddings", self.max_position_embeddings),
        ] {
            ensure!(n > 0, "{name} must be positive");
        }
        ensure!(
            self.num_attention_heads % self.num_key_value_heads == 0,
            "attention heads must divide into KV heads"
        );
        ensure!(
            self.num_hidden_layers == self.layer_types.len(),
            "layer count mismatch"
        );
        for (i, kind) in self.layer_types.iter().enumerate() {
            let expected = if (i + 1) % self.full_attention_interval == 0 {
                "full_attention"
            } else {
                "linear_attention"
            };
            ensure!(kind == expected, "layer {i} must be {expected}, got {kind}");
        }
        ensure!(
            self.rms_norm_eps.is_finite() && self.rms_norm_eps > 0.0,
            "invalid RMS epsilon"
        );
        ensure!(
            self.rope_theta.is_finite() && self.rope_theta > 0.0,
            "invalid RoPE theta"
        );
        ensure!(
            self.partial_rotary_factor.is_finite()
                && self.partial_rotary_factor > 0.0
                && self.partial_rotary_factor <= 1.0,
            "invalid partial rotary factor"
        );
        ensure!(
            self.rotary_dim() > 0
                && self.rotary_dim() % 2 == 0
                && self.rotary_dim() <= self.head_dim,
            "partial rotary dimension must be even and fit head"
        );
        for (name, a, b) in [
            (
                "attention projection",
                self.num_attention_heads,
                self.head_dim,
            ),
            ("KV projection", self.num_key_value_heads, self.head_dim),
            (
                "linear key projection",
                self.linear_num_key_heads,
                self.linear_key_head_dim,
            ),
            (
                "linear value projection",
                self.linear_num_value_heads,
                self.linear_value_head_dim,
            ),
        ] {
            a.checked_mul(b)
                .with_context(|| format!("{name} dimension overflow"))?;
        }
        for &id in &self.eos_token_ids {
            ensure!(
                (id as usize) < self.vocab_size,
                "EOS token outside vocabulary"
            );
        }
        Ok(())
    }

    pub fn is_linear(&self, layer: usize) -> bool {
        self.layer_types
            .get(layer)
            .is_some_and(|kind| kind == "linear_attention")
    }

    pub fn rotary_dim(&self) -> usize {
        ((self.head_dim as f64) * (self.partial_rotary_factor as f64)) as usize
    }

    pub(crate) fn set_eos_from_generation_json(&mut self, json: &str) -> Result<()> {
        let generation: Value =
            serde_json::from_str(json).context("invalid generation_config.json")?;
        self.eos_token_ids = parse_eos(generation.get("eos_token_id"))?;
        self.validate()?;
        Ok(())
    }
}

fn parse_eos(value: Option<&Value>) -> Result<Vec<u32>> {
    let Some(value) = value else {
        return Ok(Vec::new());
    };
    let values: Vec<&Value> = if let Some(array) = value.as_array() {
        array.iter().collect()
    } else {
        vec![value]
    };
    let mut ids = Vec::with_capacity(values.len());
    for value in values {
        let raw = value.as_u64().context("invalid EOS token id")?;
        let id = u32::try_from(raw).context("EOS token id exceeds u32")?;
        if !ids.contains(&id) {
            ids.push(id);
        }
    }
    Ok(ids)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base() -> String {
        serde_json::json!({
          "model_type": "qwen3_5", "eos_token_id": [8,9],
          "text_config": {
            "model_type": "qwen3_5_text", "hidden_size": 8,
            "intermediate_size": 16, "num_hidden_layers": 2,
            "num_attention_heads": 2, "num_key_value_heads": 1,
            "head_dim": 4, "vocab_size": 32,
            "linear_num_key_heads": 2, "linear_num_value_heads": 2,
            "linear_key_head_dim": 4, "linear_value_head_dim": 4,
            "linear_conv_kernel_dim": 4, "full_attention_interval": 2,
            "rms_norm_eps": 0.000001, "rope_theta": 10000,
            "partial_rotary_factor": 0.5, "max_position_embeddings": 128,
            "tie_word_embeddings": false,
            "layer_types": ["linear_attention", "full_attention"],
            "eos_token_id": 7,
            "rope_parameters": {"rope_type": "default", "rope_theta": 10000,
                                "partial_rotary_factor": 0.5}
          }
        })
        .to_string()
    }

    #[test]
    fn root_eos_list_takes_precedence_over_text_scalar() {
        let c = ModelConfig::from_json(&base()).unwrap();
        assert_eq!(c.eos_token_ids, vec![8, 9]);
        assert!(c.is_linear(0));
        assert!(!c.is_linear(1));
        assert_eq!(c.rotary_dim(), 2);
    }

    #[test]
    fn unsupported_rope_and_moe_are_rejected() {
        let mut v: serde_json::Value = serde_json::from_str(&base()).unwrap();
        v["text_config"]["rope_parameters"]["rope_type"] = "yarn".into();
        assert!(ModelConfig::from_json(&v.to_string()).is_err());
        v["text_config"]["rope_parameters"]["rope_type"] = "default".into();
        v["text_config"]["num_experts"] = 8.into();
        assert!(ModelConfig::from_json(&v.to_string()).is_err());
    }

    #[test]
    fn malformed_layer_pattern_is_rejected() {
        let mut v: serde_json::Value = serde_json::from_str(&base()).unwrap();
        v["text_config"]["layer_types"][1] = "linear_attention".into();
        assert!(ModelConfig::from_json(&v.to_string()).is_err());
    }

    #[test]
    fn unsupported_attention_bias_and_activation_are_rejected() {
        let mut v: serde_json::Value = serde_json::from_str(&base()).unwrap();
        v["text_config"]["attention_bias"] = true.into();
        assert!(ModelConfig::from_json(&v.to_string()).is_err());
        v["text_config"]["attention_bias"] = false.into();
        v["text_config"]["hidden_act"] = "gelu".into();
        assert!(ModelConfig::from_json(&v.to_string()).is_err());
        v["text_config"]["hidden_act"] = "silu".into();
        ModelConfig::from_json(&v.to_string()).unwrap();
    }

    #[test]
    fn generation_config_supplies_eos_when_model_config_omits_it() {
        let mut v: serde_json::Value = serde_json::from_str(&base()).unwrap();
        v.as_object_mut().unwrap().remove("eos_token_id");
        v["text_config"]
            .as_object_mut()
            .unwrap()
            .remove("eos_token_id");
        let mut c = ModelConfig::from_json(&v.to_string()).unwrap();
        c.set_eos_from_generation_json(r#"{"eos_token_id":[9,8,9]}"#)
            .unwrap();
        assert_eq!(c.eos_token_ids, vec![9, 8]);
    }
}
