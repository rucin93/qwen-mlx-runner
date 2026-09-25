use std::{
    collections::{HashMap, HashSet},
    fs::{self, File},
    path::{Component, Path},
};

use crate::config::ModelConfig;
use anyhow::{Context, Result, bail, ensure};
use half::{bf16, f16};
use memmap2::Mmap;
use serde::de::{self, Deserialize, Deserializer, MapAccess, SeqAccess, Visitor};
use serde_json::Value;

#[derive(Debug)]
pub enum MatrixData {
    Dense {
        rows: usize,
        cols: usize,
        values: Vec<u16>,
    },
    Packed {
        rows: usize,
        cols: usize,
        bits: u32,
        group_size: usize,
        weights: Vec<u32>,
        scales: Vec<f32>,
        biases: Vec<f32>,
    },
}

struct Tensor {
    file: usize,
    offset: usize,
    length: usize,
    shape: Vec<usize>,
    dtype: String,
    raw_hf: bool,
}

#[derive(Clone, Copy)]
struct QuantSpec {
    bits: u32,
    group_size: usize,
}

pub struct Checkpoint {
    pub config: ModelConfig,
    files: Vec<Mmap>,
    tensors: HashMap<String, Tensor>,
    quant_default: Option<QuantSpec>,
    quant_overrides: HashMap<String, Option<QuantSpec>>,
}

impl Checkpoint {
    pub fn open(path: &Path) -> Result<Self> {
        ensure!(path.is_dir(), "checkpoint path must be a directory");
        let config_json =
            fs::read_to_string(path.join("config.json")).context("read checkpoint config.json")?;
        let config_value: Value = serde_json::from_str(&config_json)?;
        let mut config = ModelConfig::from_json(&config_json)?;
        if config_value.get("eos_token_id").is_none()
            && config_value
                .get("generation_config")
                .and_then(|v| v.get("eos_token_id"))
                .is_none()
            && path.join("generation_config.json").exists()
        {
            config.set_eos_from_generation_json(&fs::read_to_string(
                path.join("generation_config.json"),
            )?)?;
        }
        Self::from_config(path, config_value, config)
    }

    /// Load the separate, already-sanitized native MLX MTP adapter. It is not
    /// a standalone causal model and must match its target's feature geometry.
    pub(crate) fn open_mtp(path: &Path, target: &ModelConfig) -> Result<Self> {
        ensure!(path.is_dir(), "MTP checkpoint path must be a directory");
        let mut value: Value = serde_json::from_slice(&fs::read(path.join("config.json"))?)?;
        ensure!(
            value["model_type"] == "qwen3_5_mtp",
            "Expected native qwen3_5_mtp adapter"
        );
        ensure!(
            value["text_config"]["mtp_num_hidden_layers"] == 1,
            "Only one MTP layer is supported"
        );
        ensure!(
            value["text_config"]["mtp_use_dedicated_embeddings"] == false,
            "MTP requires shared target embeddings"
        );
        ensure!(
            value["text_config"]
                .get("attn_output_gate")
                .is_none_or(|v| v == true),
            "MTP requires gated attention"
        );
        ensure!(
            value["block_size"]
                .as_u64()
                .is_some_and(|n| (2..=4).contains(&n)),
            "MTP block_size must be 2..=4"
        );
        // Reuse text-architecture validation, preserving quantization metadata.
        value["model_type"] = Value::String("qwen3_5".into());
        let declared = ModelConfig::from_json(&serde_json::to_string(&value)?)?;
        for (name, actual, expected) in [
            ("hidden_size", declared.hidden_size, target.hidden_size),
            (
                "intermediate_size",
                declared.intermediate_size,
                target.intermediate_size,
            ),
            (
                "num_attention_heads",
                declared.num_attention_heads,
                target.num_attention_heads,
            ),
            (
                "num_key_value_heads",
                declared.num_key_value_heads,
                target.num_key_value_heads,
            ),
            ("head_dim", declared.head_dim, target.head_dim),
            ("vocab_size", declared.vocab_size, target.vocab_size),
            (
                "num_hidden_layers",
                declared.num_hidden_layers,
                target.num_hidden_layers,
            ),
            ("rotary_dim", declared.rotary_dim(), target.rotary_dim()),
        ] {
            ensure!(
                actual == expected,
                "MTP {name}={actual} differs from target {expected}"
            );
        }
        ensure!(
            declared.rope_theta == target.rope_theta
                && declared.rms_norm_eps == target.rms_norm_eps,
            "MTP RoPE or RMS parameters differ from target"
        );
        ensure!(
            declared.layer_types == target.layer_types
                && declared.tie_word_embeddings == target.tie_word_embeddings,
            "MTP declares a different target architecture"
        );
        let cp = Self::from_config(path, value, declared)?;
        ensure!(
            cp.tensors.keys().all(|key| key == "fc.weight"
                || key == "fc.scales"
                || key == "fc.biases"
                || key.starts_with("layers.0.")
                || matches!(
                    key.as_str(),
                    "norm.weight" | "pre_fc_norm_embedding.weight" | "pre_fc_norm_hidden.weight"
                )),
            "Unsupported MTP tensor namespace: use the separate sanitized adapter"
        );
        ensure!(
            cp.tensors.values().all(|tensor| !tensor.raw_hf),
            "MTP adapter norms must already be sanitized"
        );
        Ok(cp)
    }

    fn from_config(path: &Path, config_value: Value, config: ModelConfig) -> Result<Self> {
        let quant_value = config_value
            .get("quantization")
            .or_else(|| config_value.get("quantization_config"));
        let (quant_default, quant_overrides) = parse_quantization(quant_value)?;
        let index_path = path.join("model.safetensors.index.json");
        let (filenames, weight_map) = if index_path.exists() {
            let index: Value = serde_json::from_slice(&fs::read(&index_path)?)
                .context("invalid safetensors index")?;
            let map = index
                .get("weight_map")
                .and_then(Value::as_object)
                .context("missing safetensors weight_map")?;
            ensure!(!map.is_empty(), "empty safetensors index");
            let mut filenames = HashSet::new();
            let mut weight_map = HashMap::new();
            for (name, filename) in map {
                let filename = filename.as_str().context("invalid shard filename")?;
                validate_shard_name(filename)?;
                filenames.insert(filename.to_owned());
                weight_map.insert(name.clone(), filename.to_owned());
            }
            let mut filenames: Vec<_> = filenames.into_iter().collect();
            filenames.sort();
            (filenames, Some(weight_map))
        } else {
            let mut files = Vec::new();
            for entry in fs::read_dir(path)? {
                let entry = entry?;
                let name = entry.file_name();
                if let Some(name) = name.to_str()
                    && name.ends_with(".safetensors")
                {
                    files.push(name.to_owned());
                }
            }
            files.sort();
            ensure!(
                files.len() == 1,
                "expected one safetensors file or an index"
            );
            (files, None)
        };
        let canonical_root = path.canonicalize()?;
        let mut cp = Self {
            config,
            files: Vec::new(),
            tensors: HashMap::new(),
            quant_default,
            quant_overrides,
        };
        let mut seen_index_names = HashSet::new();
        let mut conv_layout = None;
        for filename in &filenames {
            validate_shard_name(filename)?;
            let file_path = path.join(filename);
            ensure!(
                file_path.canonicalize()?.starts_with(&canonical_root),
                "safetensors shard escapes checkpoint directory"
            );
            let file = File::open(&file_path).with_context(|| format!("open {filename}"))?;
            // SAFETY: mapping is read-only; all slices are bounds checked before use.
            let mmap = unsafe { Mmap::map(&file) }.with_context(|| format!("map {filename}"))?;
            let file_id = cp.files.len();
            let entries = parse_safetensors(&mmap).with_context(|| format!("parse {filename}"))?;
            for (name, mut tensor) in entries {
                if let Some(weight_map) = &weight_map {
                    ensure!(
                        weight_map.get(&name).is_some_and(|v| v == filename),
                        "tensor {name} missing or mapped to wrong shard"
                    );
                    seen_index_names.insert(name.clone());
                }
                tensor.file = file_id;
                tensor.raw_hf = name.starts_with("model.language_model.")
                    || (name.starts_with("model.") && cp.quant_default.is_none());
                let canonical = canonical_name(&name);
                if canonical.ends_with(".linear_attn.conv1d.weight") {
                    let shape = &tensor.shape;
                    let width = cp.config.linear_conv_kernel_dim;
                    ensure!(
                        shape.len() == 3 && shape[0] > 0,
                        "conv {canonical} has invalid shape {shape:?}"
                    );
                    let layout = match (
                        shape[1] == 1 && shape[2] == width,
                        shape[1] == width && shape[2] == 1,
                    ) {
                        (true, false) => true,  // HF [channels, 1, kernel]
                        (false, true) => false, // MLX [channels, kernel, 1]
                        _ => bail!("conv {canonical} has ambiguous or invalid shape {shape:?}"),
                    };
                    if let Some(previous) = conv_layout {
                        ensure!(
                            previous == layout,
                            "checkpoint mixes HF and MLX conv layouts"
                        );
                    }
                    conv_layout = Some(layout);
                }
                ensure!(
                    cp.tensors.insert(canonical.clone(), tensor).is_none(),
                    "duplicate canonical tensor {canonical}"
                );
            }
            cp.files.push(mmap);
        }
        if let Some(weight_map) = &weight_map {
            ensure!(
                seen_index_names.len() == weight_map.len(),
                "index names missing from shards"
            );
        }
        // Convolution orientation is the same signal used by the MLX sanitizer
        // to distinguish zero-centered HF norm weights from converted weights.
        // Tiny incomplete fixtures without a conv retain their namespace fallback.
        if let Some(raw_hf) = conv_layout {
            for tensor in cp.tensors.values_mut() {
                tensor.raw_hf = raw_hf;
            }
        }
        Ok(cp)
    }

    pub fn matrix(&self, name: &str) -> Result<MatrixData> {
        let key = weight_key(name);
        let tensor = self.tensor(&key)?;
        ensure!(tensor.shape.len() == 2, "matrix {key} must be rank 2");
        let rows = tensor.shape[0];
        if tensor.dtype == "U32" {
            let module = key.strip_suffix(".weight").context("packed matrix name")?;
            let quant = match self.quant_overrides.get(module) {
                Some(spec) => *spec,
                None => self.quant_default,
            }
            .context("packed matrix lacks quantization metadata")?;
            ensure!(matches!(quant.bits, 4 | 8), "unsupported packed bit width");
            ensure!(
                quant.group_size > 0 && quant.group_size % (32 / quant.bits as usize) == 0,
                "invalid quantization group size"
            );
            let words_per_group = quant.group_size / (32 / quant.bits as usize);
            ensure!(
                tensor.shape[1] % words_per_group == 0,
                "packed width not divisible by group width"
            );
            let groups = tensor.shape[1] / words_per_group;
            let cols = groups
                .checked_mul(quant.group_size)
                .context("packed matrix width overflow")?;
            let scale_key = format!("{module}.scales");
            let bias_key = format!("{module}.biases");
            let scales = self.quant_values(&scale_key, rows, groups)?;
            let biases = self.quant_values(&bias_key, rows, groups)?;
            let weights = self
                .bytes(tensor)
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
                .collect();
            Ok(MatrixData::Packed {
                rows,
                cols,
                bits: quant.bits,
                group_size: quant.group_size,
                weights,
                scales,
                biases,
            })
        } else {
            let cols = tensor.shape[1];
            let values = self
                .float_values(tensor)?
                .into_iter()
                .map(|v| {
                    ensure!(
                        v.is_finite() && v.abs() <= 65504.0,
                        "matrix {key} exceeds F16 range"
                    );
                    Ok(f16::from_f32(v).to_bits())
                })
                .collect::<Result<Vec<_>>>()?;
            Ok(MatrixData::Dense { rows, cols, values })
        }
    }

    pub fn vector(&self, name: &str) -> Result<Vec<f32>> {
        let key = weight_key(name);
        let tensor = self.tensor(&key)?;
        ensure!(tensor.shape.len() == 1, "vector {key} must be rank 1");
        let mut values = self.float_values(tensor)?;
        if tensor.raw_hf && is_zero_centered_norm(&key) {
            for value in &mut values {
                *value += 1.0;
            }
        }
        ensure!(
            values.iter().all(|v| v.is_finite()),
            "vector {key} has non-finite values"
        );
        Ok(values)
    }

    pub fn conv(&self, name: &str, channels: usize, width: usize) -> Result<Vec<f32>> {
        let key = weight_key(name);
        let tensor = self.tensor(&key)?;
        let shape = &tensor.shape;
        ensure!(channels > 0 && width > 0, "invalid conv dimensions");
        ensure!(
            shape.len() == 3 && shape[0] == channels,
            "conv {key} shape mismatch"
        );
        ensure!(
            (shape[1] == 1 && shape[2] == width) || (shape[1] == width && shape[2] == 1),
            "conv {key} kernel shape mismatch"
        );
        let values = self.float_values(tensor)?;
        // Both [channels, 1, width] (HF) and [channels, width, 1] (MLX)
        // flatten to channel-major width; no transpose is required in memory.
        Ok(values)
    }

    pub fn contains(&self, name: &str) -> bool {
        let key = weight_key(name);
        self.tensors.contains_key(&key)
            || (self.config.tie_word_embeddings
                && key == "lm_head.weight"
                && self.tensors.contains_key("model.embed_tokens.weight"))
    }

    fn tensor(&self, key: &str) -> Result<&Tensor> {
        if self.config.tie_word_embeddings
            && key == "lm_head.weight"
            && !self.tensors.contains_key(key)
        {
            return self
                .tensors
                .get("model.embed_tokens.weight")
                .context("tied output embedding is missing");
        }
        self.tensors
            .get(key)
            .with_context(|| format!("missing tensor {key}"))
    }

    fn bytes(&self, tensor: &Tensor) -> &[u8] {
        &self.files[tensor.file][tensor.offset..tensor.offset + tensor.length]
    }

    fn quant_values(&self, key: &str, rows: usize, groups: usize) -> Result<Vec<f32>> {
        let tensor = self.tensor(key)?;
        ensure!(tensor.shape == [rows, groups], "{key} shape mismatch");
        let values = self.float_values(tensor)?;
        ensure!(
            values.iter().all(|v| v.is_finite()),
            "{key} contains non-finite values"
        );
        Ok(values)
    }

    fn float_values(&self, tensor: &Tensor) -> Result<Vec<f32>> {
        let bytes = self.bytes(tensor);
        match tensor.dtype.as_str() {
            "F32" => Ok(bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect()),
            "F16" => Ok(bytes
                .chunks_exact(2)
                .map(|b| f16::from_bits(u16::from_le_bytes(b.try_into().unwrap())).to_f32())
                .collect()),
            "BF16" => Ok(bytes
                .chunks_exact(2)
                .map(|b| bf16::from_bits(u16::from_le_bytes(b.try_into().unwrap())).to_f32())
                .collect()),
            _ => bail!("tensor dtype {} is not floating point", tensor.dtype),
        }
    }
}

fn validate_shard_name(filename: &str) -> Result<()> {
    let path = Path::new(filename);
    ensure!(
        path.components().count() == 1
            && matches!(path.components().next(), Some(Component::Normal(_)))
            && filename.ends_with(".safetensors"),
        "unsafe shard filename {filename}"
    );
    Ok(())
}

fn canonical_name(name: &str) -> String {
    if let Some(rest) = name.strip_prefix("model.language_model.") {
        format!("model.{rest}")
    } else if let Some(rest) = name.strip_prefix("language_model.model.") {
        format!("model.{rest}")
    } else if let Some(rest) = name.strip_prefix("language_model.") {
        if rest.starts_with("lm_head") {
            rest.to_owned()
        } else {
            format!("model.{rest}")
        }
    } else {
        name.to_owned()
    }
}

fn weight_key(name: &str) -> String {
    let key = canonical_name(name);
    if key.ends_with(".weight")
        || key.ends_with(".scales")
        || key.ends_with(".biases")
        || key.ends_with(".A_log")
        || key.ends_with(".dt_bias")
    {
        key
    } else {
        format!("{key}.weight")
    }
}

fn is_zero_centered_norm(key: &str) -> bool {
    key == "model.norm.weight"
        || key.ends_with(".input_layernorm.weight")
        || key.ends_with(".post_attention_layernorm.weight")
        || key.ends_with(".q_norm.weight")
        || key.ends_with(".k_norm.weight")
}

// serde_json::Value retains the last duplicate key, while safetensors forbids duplicates.
struct StrictJson(Value);

impl<'de> Deserialize<'de> for StrictJson {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        struct StrictVisitor;

        impl<'de> Visitor<'de> for StrictVisitor {
            type Value = StrictJson;

            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("JSON without duplicate object keys")
            }
            fn visit_bool<E: de::Error>(self, v: bool) -> std::result::Result<Self::Value, E> {
                Ok(StrictJson(Value::Bool(v)))
            }
            fn visit_i64<E: de::Error>(self, v: i64) -> std::result::Result<Self::Value, E> {
                Ok(StrictJson(Value::Number(v.into())))
            }
            fn visit_u64<E: de::Error>(self, v: u64) -> std::result::Result<Self::Value, E> {
                Ok(StrictJson(Value::Number(v.into())))
            }
            fn visit_f64<E: de::Error>(self, v: f64) -> std::result::Result<Self::Value, E> {
                let number = serde_json::Number::from_f64(v)
                    .ok_or_else(|| E::custom("non-finite JSON number"))?;
                Ok(StrictJson(Value::Number(number)))
            }
            fn visit_str<E: de::Error>(self, v: &str) -> std::result::Result<Self::Value, E> {
                Ok(StrictJson(Value::String(v.to_owned())))
            }
            fn visit_string<E: de::Error>(self, v: String) -> std::result::Result<Self::Value, E> {
                Ok(StrictJson(Value::String(v)))
            }
            fn visit_unit<E: de::Error>(self) -> std::result::Result<Self::Value, E> {
                Ok(StrictJson(Value::Null))
            }
            fn visit_none<E: de::Error>(self) -> std::result::Result<Self::Value, E> {
                Ok(StrictJson(Value::Null))
            }
            fn visit_some<D: Deserializer<'de>>(
                self,
                deserializer: D,
            ) -> std::result::Result<Self::Value, D::Error> {
                StrictJson::deserialize(deserializer)
            }
            fn visit_seq<A: SeqAccess<'de>>(
                self,
                mut seq: A,
            ) -> std::result::Result<Self::Value, A::Error> {
                let mut values = Vec::new();
                while let Some(value) = seq.next_element::<StrictJson>()? {
                    values.push(value.0);
                }
                Ok(StrictJson(Value::Array(values)))
            }
            fn visit_map<A: MapAccess<'de>>(
                self,
                mut map: A,
            ) -> std::result::Result<Self::Value, A::Error> {
                let mut values = serde_json::Map::new();
                while let Some((key, value)) = map.next_entry::<String, StrictJson>()? {
                    if values.insert(key, value.0).is_some() {
                        return Err(de::Error::custom("duplicate JSON key"));
                    }
                }
                Ok(StrictJson(Value::Object(values)))
            }
        }

        deserializer.deserialize_any(StrictVisitor)
    }
}

fn parse_safetensors(mmap: &[u8]) -> Result<HashMap<String, Tensor>> {
    ensure!(mmap.len() >= 8, "safetensors header truncated");
    let header_len = usize::try_from(u64::from_le_bytes(mmap[..8].try_into().unwrap()))
        .context("safetensors header length overflow")?;
    ensure!(
        header_len <= 100_000_000,
        "safetensors header exceeds 100 MB"
    );
    let data_start = 8usize
        .checked_add(header_len)
        .context("safetensors header overflow")?;
    ensure!(data_start <= mmap.len(), "safetensors header truncated");
    let header = serde_json::from_slice::<StrictJson>(&mmap[8..data_start])
        .context("invalid safetensors JSON header")?
        .0;
    let object = header
        .as_object()
        .context("safetensors header is not an object")?;
    if let Some(metadata) = object.get("__metadata__") {
        let metadata = metadata
            .as_object()
            .context("invalid safetensors metadata")?;
        ensure!(
            metadata.values().all(Value::is_string),
            "invalid safetensors metadata value"
        );
    }
    let mut tensors = HashMap::new();
    let mut ranges = Vec::new();
    for (name, value) in object {
        if name == "__metadata__" {
            continue;
        }
        let dtype = value
            .get("dtype")
            .and_then(Value::as_str)
            .with_context(|| format!("{name}: missing dtype"))?;
        let item_size = match dtype {
            "F16" | "BF16" => 2usize,
            "F32" | "U32" | "I32" => 4,
            "F64" | "I64" | "U64" => 8,
            "U8" | "I8" | "BOOL" => 1,
            _ => bail!("{name}: unsupported dtype {dtype}"),
        };
        let shape = value
            .get("shape")
            .and_then(Value::as_array)
            .with_context(|| format!("{name}: missing shape"))?
            .iter()
            .map(|v| {
                let n = v.as_u64().context("invalid tensor dimension")?;
                let n = usize::try_from(n).context("tensor dimension overflow")?;
                ensure!(n > 0, "zero tensor dimension");
                Ok(n)
            })
            .collect::<Result<Vec<_>>>()?;
        let count = shape
            .iter()
            .try_fold(1usize, |a, &b| a.checked_mul(b))
            .with_context(|| format!("{name}: tensor element count overflow"))?;
        let length = count
            .checked_mul(item_size)
            .with_context(|| format!("{name}: tensor byte length overflow"))?;
        let offsets = value
            .get("data_offsets")
            .and_then(Value::as_array)
            .with_context(|| format!("{name}: missing data offsets"))?;
        ensure!(offsets.len() == 2, "{name}: invalid data offsets");
        let start = usize::try_from(offsets[0].as_u64().context("invalid start offset")?)?;
        let end = usize::try_from(offsets[1].as_u64().context("invalid end offset")?)?;
        ensure!(
            end >= start && end - start == length,
            "{name}: byte length mismatch"
        );
        let absolute_start = data_start
            .checked_add(start)
            .context("tensor offset overflow")?;
        let absolute_end = data_start
            .checked_add(end)
            .context("tensor offset overflow")?;
        ensure!(absolute_end <= mmap.len(), "{name}: tensor exceeds file");
        ranges.push((start, end));
        tensors.insert(
            name.clone(),
            Tensor {
                file: 0,
                offset: absolute_start,
                length,
                shape,
                dtype: dtype.to_owned(),
                raw_hf: false,
            },
        );
    }
    ranges.sort_unstable();
    let mut indexed_until = 0;
    for (start, end) in ranges {
        ensure!(
            start == indexed_until,
            "safetensors data ranges overlap or leave holes"
        );
        indexed_until = end;
    }
    ensure!(
        indexed_until == mmap.len() - data_start,
        "safetensors data has unindexed trailing bytes"
    );
    Ok(tensors)
}

fn parse_quantization(
    value: Option<&Value>,
) -> Result<(Option<QuantSpec>, HashMap<String, Option<QuantSpec>>)> {
    let Some(value) = value else {
        return Ok((None, HashMap::new()));
    };
    let object = value.as_object().context("invalid quantization metadata")?;
    let default = if object.contains_key("bits") || object.contains_key("group_size") {
        Some(parse_quant_spec(value)?)
    } else {
        None
    };
    let mut overrides = HashMap::new();
    for (key, value) in object {
        if matches!(key.as_str(), "bits" | "group_size" | "mode") {
            continue;
        }
        let spec = match value {
            Value::Bool(false) => None,
            Value::Bool(true) => default,
            Value::Object(_) => Some(parse_quant_spec(value)?),
            _ => bail!("invalid quantization entry {key}"),
        };
        let canonical = canonical_name(key);
        ensure!(
            overrides.insert(canonical.clone(), spec).is_none(),
            "duplicate quantization entry {canonical}"
        );
    }
    Ok((default, overrides))
}

fn parse_quant_spec(value: &Value) -> Result<QuantSpec> {
    let mode = value
        .get("mode")
        .and_then(Value::as_str)
        .unwrap_or("affine");
    ensure!(mode == "affine", "unsupported quantization mode {mode}");
    let bits = u32::try_from(
        value
            .get("bits")
            .and_then(Value::as_u64)
            .context("missing quantization bits")?,
    )?;
    ensure!(
        matches!(bits, 4 | 8),
        "unsupported quantization bit width {bits}"
    );
    let group_size = usize::try_from(
        value
            .get("group_size")
            .and_then(Value::as_u64)
            .context("missing quantization group size")?,
    )?;
    ensure!(
        group_size > 0 && group_size % (32 / bits as usize) == 0,
        "invalid quantization group size"
    );
    Ok(QuantSpec { bits, group_size })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{Value, json};
    use std::fs;
    use tempfile::tempdir;

    fn config() -> Value {
        json!({"model_type":"qwen3_5_text", "hidden_size":8,
          "intermediate_size":16,"num_hidden_layers":2,"num_attention_heads":2,
          "num_key_value_heads":1,"head_dim":4,"vocab_size":32,
          "linear_num_key_heads":2,"linear_num_value_heads":2,
          "linear_key_head_dim":4,"linear_value_head_dim":4,
          "linear_conv_kernel_dim":4,"full_attention_interval":2,
          "rms_norm_eps":0.000001,"rope_theta":10000,
          "partial_rotary_factor":0.5,"max_position_embeddings":128,
          "layer_types":["linear_attention","full_attention"]})
    }

    fn write_tensor_file(path: &Path, tensors: &[(&str, &str, Vec<usize>, Vec<u8>)]) {
        let mut header = serde_json::Map::new();
        let mut bytes = Vec::new();
        for (name, dtype, shape, data) in tensors {
            let start = bytes.len();
            bytes.extend_from_slice(data);
            header.insert(
                (*name).into(),
                json!({"dtype":dtype, "shape":shape,
                "data_offsets":[start, bytes.len()]}),
            );
        }
        let json = serde_json::to_vec(&header).unwrap();
        let mut file = (json.len() as u64).to_le_bytes().to_vec();
        file.extend(json);
        file.extend(bytes);
        fs::write(path, file).unwrap();
    }

    #[test]
    fn raw_norms_shift_but_gated_norm_does_not() {
        let dir = tempdir().unwrap();
        fs::write(dir.path().join("config.json"), config().to_string()).unwrap();
        let values = vec![0u16, 0x3c00u16]; // 0 and 1
        let data = values
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>();
        write_tensor_file(
            &dir.path().join("model.safetensors"),
            &[
                (
                    "model.layers.0.input_layernorm.weight",
                    "F16",
                    vec![2],
                    data.clone(),
                ),
                (
                    "model.layers.0.linear_attn.norm.weight",
                    "F16",
                    vec![2],
                    data,
                ),
            ],
        );
        let cp = Checkpoint::open(dir.path()).unwrap();
        assert_eq!(
            cp.vector("model.layers.0.input_layernorm").unwrap(),
            vec![1.0, 2.0]
        );
        assert_eq!(
            cp.vector("model.layers.0.linear_attn.norm").unwrap(),
            vec![0.0, 1.0]
        );
    }

    #[test]
    fn packed_nibbles_keep_low_to_high_order_and_per_layer_quant() {
        let dir = tempdir().unwrap();
        let mut cfg = config();
        cfg["quantization"] = json!({"bits":8,"group_size":8,"mode":"affine",
            "language_model.model.layers.0.mlp.gate_proj":{"bits":4,"group_size":8}});
        fs::write(dir.path().join("config.json"), cfg.to_string()).unwrap();
        write_tensor_file(
            &dir.path().join("model.safetensors"),
            &[
                (
                    "language_model.model.layers.0.mlp.gate_proj.weight",
                    "U32",
                    vec![1, 1],
                    0x76543210u32.to_le_bytes().to_vec(),
                ),
                (
                    "language_model.model.layers.0.mlp.gate_proj.scales",
                    "F16",
                    vec![1, 1],
                    0x3800u16.to_le_bytes().to_vec(),
                ),
                (
                    "language_model.model.layers.0.mlp.gate_proj.biases",
                    "F16",
                    vec![1, 1],
                    0xbc00u16.to_le_bytes().to_vec(),
                ),
            ],
        );
        let cp = Checkpoint::open(dir.path()).unwrap();
        match cp.matrix("model.layers.0.mlp.gate_proj").unwrap() {
            MatrixData::Packed {
                cols,
                bits,
                group_size,
                weights,
                scales,
                biases,
                ..
            } => {
                assert_eq!((cols, bits, group_size), (8, 4, 8));
                assert_eq!(weights[0], 0x76543210);
                assert_eq!(scales, vec![0.5]);
                assert_eq!(biases, vec![-1.0]);
                let decoded = (0..8)
                    .map(|i| {
                        let nibble = (weights[0] >> (i * 4)) & 15;
                        nibble as f32 * scales[0] + biases[0]
                    })
                    .collect::<Vec<_>>();
                assert_eq!(decoded, vec![-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5]);
            }
            _ => panic!("expected packed matrix"),
        }
    }

    #[test]
    fn truncated_tensor_and_index_traversal_are_rejected() {
        let dir = tempdir().unwrap();
        fs::write(dir.path().join("config.json"), config().to_string()).unwrap();
        write_tensor_file(
            &dir.path().join("model.safetensors"),
            &[(
                "model.norm.weight",
                "F32",
                vec![2],
                1f32.to_le_bytes().to_vec(),
            )],
        );
        assert!(Checkpoint::open(dir.path()).is_err());
        fs::write(
            dir.path().join("model.safetensors.index.json"),
            json!({"weight_map":{"model.norm.weight":"../other.safetensors"}}).to_string(),
        )
        .unwrap();
        assert!(Checkpoint::open(dir.path()).is_err());
    }

    #[test]
    fn mlx_norm_is_already_multiplicative_and_conv_is_channel_major() {
        let dir = tempdir().unwrap();
        fs::write(dir.path().join("config.json"), config().to_string()).unwrap();
        let norm = [0x3c00u16, 0x4000u16]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>();
        let conv = [1f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>();
        write_tensor_file(
            &dir.path().join("model.safetensors"),
            &[
                (
                    "language_model.model.layers.0.input_layernorm.weight",
                    "F16",
                    vec![2],
                    norm,
                ),
                (
                    "language_model.model.layers.0.linear_attn.conv1d.weight",
                    "F32",
                    vec![2, 4, 1],
                    conv,
                ),
            ],
        );
        let cp = Checkpoint::open(dir.path()).unwrap();
        assert_eq!(
            cp.vector("model.layers.0.input_layernorm").unwrap(),
            vec![1.0, 2.0]
        );
        assert_eq!(
            cp.conv("model.layers.0.linear_attn.conv1d", 2, 4).unwrap(),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        );
    }

    #[test]
    fn canonical_mlx_layout_without_global_quant_does_not_shift_norm() {
        let dir = tempdir().unwrap();
        let mut cfg = config();
        cfg["quantization"] = json!({
            "model.layers.0.mlp.gate_proj": {"bits":4,"group_size":8}
        });
        fs::write(dir.path().join("config.json"), cfg.to_string()).unwrap();
        write_tensor_file(
            &dir.path().join("model.safetensors"),
            &[
                (
                    "model.layers.0.input_layernorm.weight",
                    "F32",
                    vec![1],
                    1.25f32.to_le_bytes().to_vec(),
                ),
                (
                    "model.layers.0.linear_attn.conv1d.weight",
                    "F32",
                    vec![2, 4, 1],
                    vec![0; 2 * 4 * 4],
                ),
            ],
        );
        let cp = Checkpoint::open(dir.path()).unwrap();
        assert_eq!(
            cp.vector("model.layers.0.input_layernorm").unwrap(),
            vec![1.25]
        );
    }

    #[test]
    fn canonical_raw_layout_shifts_norm_even_with_global_quant() {
        let dir = tempdir().unwrap();
        let mut cfg = config();
        cfg["quantization"] = json!({"bits":4,"group_size":8});
        fs::write(dir.path().join("config.json"), cfg.to_string()).unwrap();
        write_tensor_file(
            &dir.path().join("model.safetensors"),
            &[
                (
                    "model.layers.0.input_layernorm.weight",
                    "F32",
                    vec![1],
                    0.25f32.to_le_bytes().to_vec(),
                ),
                (
                    "model.layers.0.linear_attn.conv1d.weight",
                    "F32",
                    vec![2, 1, 4],
                    vec![0; 2 * 4 * 4],
                ),
            ],
        );
        let cp = Checkpoint::open(dir.path()).unwrap();
        assert_eq!(
            cp.vector("model.layers.0.input_layernorm").unwrap(),
            vec![1.25]
        );
    }

    #[test]
    fn mixed_conv_layouts_are_rejected() {
        let dir = tempdir().unwrap();
        fs::write(dir.path().join("config.json"), config().to_string()).unwrap();
        write_tensor_file(
            &dir.path().join("model.safetensors"),
            &[
                (
                    "model.layers.0.linear_attn.conv1d.weight",
                    "F32",
                    vec![2, 1, 4],
                    vec![0; 2 * 4 * 4],
                ),
                (
                    "model.layers.1.linear_attn.conv1d.weight",
                    "F32",
                    vec![2, 4, 1],
                    vec![0; 2 * 4 * 4],
                ),
            ],
        );
        assert!(Checkpoint::open(dir.path()).is_err());
    }

    #[test]
    fn indexed_shards_must_match_named_file() {
        let dir = tempdir().unwrap();
        fs::write(dir.path().join("config.json"), config().to_string()).unwrap();
        write_tensor_file(
            &dir.path().join("a.safetensors"),
            &[(
                "model.norm.weight",
                "F32",
                vec![1],
                0f32.to_le_bytes().to_vec(),
            )],
        );
        let index = dir.path().join("model.safetensors.index.json");
        fs::write(
            &index,
            json!({"weight_map":{"model.norm.weight":"a.safetensors"}}).to_string(),
        )
        .unwrap();
        let cp = Checkpoint::open(dir.path()).unwrap();
        assert_eq!(cp.vector("model.norm").unwrap(), vec![1.0]);
        fs::write(
            &index,
            json!({"weight_map":{"model.norm.weight":"b.safetensors"}}).to_string(),
        )
        .unwrap();
        assert!(Checkpoint::open(dir.path()).is_err());
    }

    #[test]
    fn external_generation_eos_replaces_text_fallback() {
        let dir = tempdir().unwrap();
        let mut cfg = json!({"model_type":"qwen3_5", "text_config": config()});
        cfg["text_config"]["eos_token_id"] = 7.into();
        fs::write(dir.path().join("config.json"), cfg.to_string()).unwrap();
        fs::write(
            dir.path().join("generation_config.json"),
            json!({"eos_token_id":[8,9]}).to_string(),
        )
        .unwrap();
        write_tensor_file(
            &dir.path().join("model.safetensors"),
            &[(
                "model.norm.weight",
                "F32",
                vec![1],
                0f32.to_le_bytes().to_vec(),
            )],
        );
        let cp = Checkpoint::open(dir.path()).unwrap();
        assert_eq!(cp.config.eos_token_ids, vec![8, 9]);
    }

    #[test]
    fn safetensors_rejects_unindexed_data_bytes() {
        let dir = tempdir().unwrap();
        fs::write(dir.path().join("config.json"), config().to_string()).unwrap();
        let path = dir.path().join("model.safetensors");
        write_tensor_file(
            &path,
            &[(
                "model.norm.weight",
                "F32",
                vec![1],
                0f32.to_le_bytes().to_vec(),
            )],
        );
        let mut bytes = fs::read(&path).unwrap();
        bytes.push(0);
        fs::write(&path, bytes).unwrap();
        assert!(Checkpoint::open(dir.path()).is_err());
    }

    #[test]
    fn safetensors_rejects_non_string_metadata() {
        let dir = tempdir().unwrap();
        fs::write(dir.path().join("config.json"), config().to_string()).unwrap();
        let header = serde_json::to_vec(&json!({"__metadata__":{"producer":42}})).unwrap();
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend(header);
        fs::write(dir.path().join("model.safetensors"), bytes).unwrap();
        assert!(Checkpoint::open(dir.path()).is_err());
    }

    #[test]
    fn safetensors_rejects_duplicate_json_keys() {
        let dir = tempdir().unwrap();
        fs::write(dir.path().join("config.json"), config().to_string()).unwrap();
        let header = br#"{"model.norm.weight":{"dtype":"F32","shape":[1],"data_offsets":[0,4],"shape":[1]}}"#;
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header);
        bytes.extend_from_slice(&0f32.to_le_bytes());
        fs::write(dir.path().join("model.safetensors"), bytes).unwrap();
        assert!(Checkpoint::open(dir.path()).is_err());
    }
}
