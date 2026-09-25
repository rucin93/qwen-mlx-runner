#!/usr/bin/env python3
"""Generate a synthetic native-Qwen MTP adapter and independent FP64 oracle.

Uses only Python's standard library. Reads the existing tiny-q4 target without
modifying it and evaluates its actual packed embedding / LM-head values. This
is mathematical test data, not a trained model or an acceptance benchmark.
"""

import hashlib
import json
import math
import pathlib
import struct


REPO = pathlib.Path(__file__).resolve().parents[1]
TARGET = REPO / "tests/fixtures/tiny-q4"
OUTPUT = REPO / "tests/fixtures/tiny-mtp"


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def exact_bf16_bits(value):
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    assert math.isfinite(value) and bits & 0xFFFF == 0
    return bits >> 16


def read_safetensors(path):
    data = path.read_bytes()
    header_size = struct.unpack("<Q", data[:8])[0]
    header = json.loads(data[8:8 + header_size])
    body = data[8 + header_size:]
    result = {}
    for name, entry in header.items():
        if name == "__metadata__":
            continue
        start, end = entry["data_offsets"]
        assert 0 <= start <= end <= len(body)
        count = math.prod(entry["shape"])
        raw = body[start:end]
        dtype = entry["dtype"]
        if dtype == "BF16":
            words = struct.unpack("<" + "H" * count, raw)
            values = [struct.unpack("<f", struct.pack("<I", w << 16))[0]
                      for w in words]
        else:
            format_char = {"F16": "e", "F32": "f", "U32": "I"}[dtype]
            values = list(struct.unpack("<" + format_char * count, raw))
        result[name] = (entry["shape"], values)
    return result


def unpack_matrix(tensors, name, rows, cols, group_size):
    shape, words = tensors[name + ".weight"]
    assert shape == [rows, cols // 8]
    meta_shape, scales = tensors[name + ".scales"]
    bias_shape, biases = tensors[name + ".biases"]
    assert meta_shape == bias_shape == [rows, cols // group_size]
    matrix = []
    for row in range(rows):
        values = []
        for col in range(cols):
            packed = words[row * (cols // 8) + col // 8]
            q = (packed >> (4 * (col % 8))) & 15
            group = row * (cols // group_size) + col // group_size
            values.append(q * scales[group] + biases[group])
        matrix.append(values)
    return matrix


def matvec(matrix, x):
    assert all(len(row) == len(x) for row in matrix)
    return [sum(weight * value for weight, value in zip(row, x)) for row in matrix]


def rms(x, weight, epsilon):
    assert len(x) == len(weight)
    inv = 1.0 / math.sqrt(sum(v * v for v in x) / len(x) + epsilon)
    return [v * inv * w for v, w in zip(x, weight)]


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def main():
    source_files = {name: (TARGET / name).read_bytes()
                    for name in ("config.json", "model.safetensors")}
    target_config = json.loads(source_files["config.json"])
    text = target_config["text_config"]
    h, ff, vocab = (text[k] for k in ("hidden_size", "intermediate_size", "vocab_size"))
    heads, kv_heads, d = (text[k] for k in
                           ("num_attention_heads", "num_key_value_heads", "head_dim"))
    epsilon = text["rms_norm_eps"]
    rope = text["rope_parameters"]
    rotary_dim = int(d * rope["partial_rotary_factor"])
    group_size = 32
    target_tensors = read_safetensors(TARGET / "model.safetensors")
    prefix = "language_model."
    target_group_size = target_config["quantization"]["group_size"]
    embedding = unpack_matrix(target_tensors, prefix + "model.embed_tokens",
                              vocab, h, target_group_size)
    lm_head = unpack_matrix(target_tensors, prefix + "lm_head",
                            vocab, h, target_group_size)

    tensors = {}

    def store(name, shape, dtype, values):
        assert len(values) == math.prod(shape)
        tensors[name] = (shape, dtype, values)

    def quantized_matrix(name, rows, cols):
        assert cols % group_size == 0
        seed = sum((i + 1) * byte for i, byte in enumerate(name.encode("utf-8")))
        packed, scales, biases = [], [], []
        for row in range(rows):
            for group in range(cols // group_size):
                scale = (1 + (row + group + seed) % 3) / 256.0
                bias = -7 * scale
                exact_bf16_bits(scale)
                exact_bf16_bits(bias)
                scales.append(scale)
                biases.append(bias)
                values = []
                for col in range(group * group_size, (group + 1) * group_size):
                    # Independent integer hash varies rows and columns without
                    # introducing platform-dependent floating point weights.
                    mixed = ((row + 1) * 0x45D9F3B + (col + 1) * 0x119DE1F3 + seed)
                    mixed = ((mixed ^ (mixed >> 16)) * 0x45D9F3B) & 0xFFFFFFFF
                    values.append((mixed ^ (mixed >> 16)) & 15)
                for start in range(0, group_size, 8):
                    packed.append(sum(q << (4 * i)
                                      for i, q in enumerate(values[start:start + 8])))
        store(name + ".weight", [rows, cols // 8], "U32", packed)
        store(name + ".scales", [rows, cols // group_size], "BF16", scales)
        store(name + ".biases", [rows, cols // group_size], "BF16", biases)

    def norm(name, width):
        seed = sum(name.encode("utf-8"))
        values = [1.0 + ((i + seed) % 9 - 4) / 128.0 for i in range(width)]
        for value in values:
            exact_bf16_bits(value)
        store(name + ".weight", [width], "BF16", values)

    quantized_matrix("fc", h, 2 * h)
    for name, rows, cols in [
        ("q_proj", 2 * heads * d, h), ("k_proj", kv_heads * d, h),
        ("v_proj", kv_heads * d, h), ("o_proj", h, heads * d),
    ]:
        quantized_matrix("layers.0.self_attn." + name, rows, cols)
    for name, rows, cols in [("gate_proj", ff, h), ("up_proj", ff, h),
                             ("down_proj", h, ff)]:
        quantized_matrix("layers.0.mlp." + name, rows, cols)
    for name, width in [
        ("pre_fc_norm_embedding", h), ("pre_fc_norm_hidden", h), ("norm", h),
        ("layers.0.input_layernorm", h), ("layers.0.post_attention_layernorm", h),
        ("layers.0.self_attn.q_norm", d), ("layers.0.self_attn.k_norm", d),
    ]:
        norm(name, width)
    assert len(tensors) == 31

    header, body = {"__metadata__": {"format": "mlx"}}, bytearray()
    for name in sorted(tensors):
        shape, dtype, values = tensors[name]
        if dtype == "BF16":
            raw = struct.pack("<" + "H" * len(values),
                              *(exact_bf16_bits(v) for v in values))
        else:
            raw = struct.pack("<" + "I" * len(values), *values)
        header[name] = {"dtype": dtype, "shape": shape,
                        "data_offsets": [len(body), len(body) + len(raw)]}
        body.extend(raw)
    raw_header = json.dumps(header, separators=(",", ":")).encode("utf-8")
    raw_header += b" " * (-len(raw_header) % 8)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    model_bytes = struct.pack("<Q", len(raw_header)) + raw_header + body
    (OUTPUT / "model.safetensors").write_bytes(model_bytes)

    # Evaluate the serialized values through an independent file reader.
    actual = read_safetensors(OUTPUT / "model.safetensors")
    matrices = {}
    for name, (shape, _) in actual.items():
        if name.endswith(".weight") and len(shape) == 2:
            module = name.removesuffix(".weight")
            matrices[module] = unpack_matrix(actual, module, shape[0], shape[1] * 8,
                                              group_size)

    def normalize(x, name):
        return rms(x, actual[name + ".weight"][1], epsilon)

    def rotate(vector, position):
        result = list(vector)
        half = rotary_dim // 2
        for i in range(half):
            angle = position / (rope["rope_theta"] ** (2 * i / rotary_dim))
            u, v = vector[i], vector[i + half]
            result[i] = u * math.cos(angle) - v * math.sin(angle)
            result[i + half] = u * math.sin(angle) + v * math.cos(angle)
        return result

    def evaluate(tokens, positions, input_hidden):
        keys, values, hidden_out, logits_out = [], [], [], []
        for token, position, given_hidden in zip(tokens, positions, input_hidden):
            combined = (normalize(embedding[token], "pre_fc_norm_embedding")
                        + normalize(given_hidden, "pre_fc_norm_hidden"))
            x = matvec(matrices["fc"], combined)
            n = normalize(x, "layers.0.input_layernorm")
            a = "layers.0.self_attn."
            q_gate = matvec(matrices[a + "q_proj"], n)
            q = [rotate(normalize(q_gate[head * 2 * d:head * 2 * d + d],
                                  a + "q_norm"), position) for head in range(heads)]
            gate = [q_gate[head * 2 * d + d:(head + 1) * 2 * d]
                    for head in range(heads)]
            k_projected = matvec(matrices[a + "k_proj"], n)
            keys.append([rotate(normalize(k_projected[head * d:(head + 1) * d],
                                           a + "k_norm"), position)
                         for head in range(kv_heads)])
            v_projected = matvec(matrices[a + "v_proj"], n)
            values.append([v_projected[head * d:(head + 1) * d]
                           for head in range(kv_heads)])
            attended = []
            for head in range(heads):
                kv_head = head // (heads // kv_heads)
                scores = [sum(u * v for u, v in zip(q[head], entry[kv_head]))
                          / math.sqrt(d) for entry in keys]
                maximum = max(scores)
                exp_scores = [math.exp(score - maximum) for score in scores]
                total = sum(exp_scores)
                probabilities = [score / total for score in exp_scores]
                attended.extend(sum(probability * value[kv_head][column]
                                    for probability, value in zip(probabilities, values))
                                * sigmoid(gate[head][column]) for column in range(d))
            attention_out = matvec(matrices[a + "o_proj"], attended)
            x = [u + v for u, v in zip(x, attention_out)]
            n = normalize(x, "layers.0.post_attention_layernorm")
            gate_mlp = matvec(matrices["layers.0.mlp.gate_proj"], n)
            up = matvec(matrices["layers.0.mlp.up_proj"], n)
            activated = [g * sigmoid(g) * u for g, u in zip(gate_mlp, up)]
            down = matvec(matrices["layers.0.mlp.down_proj"], activated)
            x = normalize([u + v for u, v in zip(x, down)], "norm")
            hidden_out.append(x)
            logits_out.append(matvec(lm_head, x))
        return hidden_out, logits_out

    tokens = [3, 8, 4, 9, 6]
    positions = list(range(len(tokens)))
    target_norm = target_tensors[prefix + "model.norm.weight"][1]
    input_hidden = []
    for step in positions:
        raw = [((i * 7 + step * 11) % 31 - 15) / 16.0
               + ((i + step) % 3) / 32.0 for i in range(h)]
        input_hidden.append([f32(v) for v in rms(raw, target_norm, epsilon)])
    hidden, logits = evaluate(tokens, positions, input_hidden)
    assert (hidden, logits) == evaluate(tokens, positions, input_hidden)
    # Nonzero RoPE origin exercises explicit positions independently of KV length.
    shifted_positions = [p + 7 for p in positions]
    shifted_hidden, shifted_logits = evaluate(tokens, shifted_positions, input_hidden)
    text_config = dict(text)
    text_config.update(mtp_num_hidden_layers=1, mtp_use_dedicated_embeddings=False)
    config = {"model_type": "qwen3_5_mtp", "block_size": 3,
              "text_config": text_config, "tie_word_embeddings": False,
              "quantization": {"bits": 4, "group_size": group_size, "mode": "affine"}}
    golden = {
        "description": "Independent scalar Python FP64 MTP oracle; synthetic untrained adapter. Input hidden vectors are explicit FP32-rounded post-target-RMS inputs, not outputs from a trained model.",
        "target_fixture": "tests/fixtures/tiny-q4",
        "tokens": tokens, "positions": positions, "input_hidden": input_hidden,
        "hidden": hidden, "logits": logits,
        "shifted_position_case": {"positions": shifted_positions,
                                  "hidden": shifted_hidden, "logits": shifted_logits},
    }
    for name, value in (("config.json", config), ("golden.json", golden)):
        (OUTPUT / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    manifest = {
        "kind": "synthetic_mtp_oracle_fixture", "generator": "scripts/make_mtp_fixture.py",
        "generator_sha256": sha256(pathlib.Path(__file__).read_bytes()),
        "oracle": "Python standard-library scalar float64; evaluates serialized affine Q4 values; uses existing target embedding and LM head.",
        "training": "None: deterministic synthetic test weights, no performance or quality claim.",
        "normalization": "Native adapter norms are already multiplicative (HF weight+1 convention sanitized).",
        "tensor_count": len(tensors), "quantized_matrices": 8, "bf16_norm_vectors": 7,
        "body_bytes": len(body), "reset_replay_equal": True,
        "target_files_sha256": {"tests/fixtures/tiny-q4/" + name: sha256(data)
                                for name, data in source_files.items()},
        "files_sha256": {name: sha256((OUTPUT / name).read_bytes())
                         for name in ("config.json", "model.safetensors", "golden.json")},
        "architecture_reference": "https://github.com/Blaizzy/mlx-vlm/blob/ad4a3ccd6483aa2db4d84b70108aca103c6a9b15/mlx_vlm/speculative/drafters/qwen3_5_mtp/qwen3_5_mtp.py",
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for name, original in source_files.items():
        assert (TARGET / name).read_bytes() == original, "Target fixture was modified"
    print(json.dumps({"output": str(OUTPUT), "tensor_count": len(tensors),
                      "body_bytes": len(body), "tokens": len(tokens),
                      "hidden_width": h, "vocab": vocab, "reset_replay_equal": True}))


if __name__ == "__main__":
    main()
