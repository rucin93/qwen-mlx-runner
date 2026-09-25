#!/usr/bin/env python3
"""Independent scalar, double-precision oracle for a tiny hybrid Qwen model.

No inference framework or numerical library. Not a trained language model.
Equations follow the Qwen3.5 architecture (also used by Qwen3.8-27B).
Weights are rounded to F16 before both serialization and reference evaluation.
"""
import json
import math
import pathlib
import struct

ROOT = pathlib.Path(__file__).resolve().parents[1] / "tests/fixtures/tiny"
ROOT.mkdir(parents=True, exist_ok=True)
H, FF, VOCAB, HEADS, KVH, D, KH, VH, K, V = 32, 64, 64, 2, 1, 32, 1, 2, 32, 16
EPS = 1e-6
weights = {}
shapes = {}

def store(name, shape, values):
    values = [struct.unpack("<e", struct.pack("<e", x))[0] for x in values]
    weights[name] = values
    shapes[name] = shape
    return values

def matrix(name, rows, cols):
    offset = sum(name.encode())
    return store(name + ".weight", [rows, cols],
                 [0.11 * math.sin((i + 1) * 0.173 + offset * 0.31) for i in range(rows * cols)])

def norm_weight(name, n, gated=False):
    return store(name + ".weight", [n],
                 [(1.0 if gated else 0.0) + .025 * math.cos(i * .73) for i in range(n)])

matrix("model.embed_tokens", VOCAB, H)
for layer in range(4):
    p = f"model.layers.{layer}"
    norm_weight(p + ".input_layernorm", H)
    norm_weight(p + ".post_attention_layernorm", H)
    for name, r, c in [("gate_proj",FF,H),("up_proj",FF,H),("down_proj",H,FF)]:
        matrix(p + ".mlp." + name, r, c)
    if layer < 3:
        p += ".linear_attn"
        for name, r, c in [("in_proj_qkv",2*KH*K+VH*V,H),("in_proj_z",VH*V,H),
                           ("in_proj_a",VH,H),("in_proj_b",VH,H),("out_proj",H,VH*V)]:
            matrix(p + "." + name,r,c)
        store(p + ".conv1d.weight", [2*KH*K+VH*V,1,4],
              [.15 + .07*math.cos(i*.27) for i in range((2*KH*K+VH*V)*4)])
        store(p + ".A_log", [VH], [-.8,-.3])
        store(p + ".dt_bias", [VH], [.1,.2])
        norm_weight(p + ".norm",V,True)
    else:
        p += ".self_attn"
        for name,r,c in [("q_proj",2*HEADS*D,H),("k_proj",KVH*D,H),
                         ("v_proj",KVH*D,H),("o_proj",H,HEADS*D)]:
            matrix(p + "." + name,r,c)
        norm_weight(p + ".q_norm",D)
        norm_weight(p + ".k_norm",D)
norm_weight("model.norm",H)
matrix("lm_head",VOCAB,H)

def mv(name,x):
    w = weights[name+".weight"]
    return [sum(a*b for a,b in zip(w[r:r+len(x)],x)) for r in range(0,len(w),len(x))]

def rms(x,name,zero_centered=True):
    inv = 1/math.sqrt(sum(v*v for v in x)/len(x)+EPS)
    return [v*inv*(w+int(zero_centered)) for v,w in zip(x,weights[name+".weight"])]

def silu(v): return v/(1+math.exp(-v))
def sigmoid(v): return 1/(1+math.exp(-v))

conv = [[[0.]*3 for _ in range(2*KH*K+VH*V)] for _ in range(3)]
state = [[[[0.]*K for _ in range(V)] for _ in range(VH)] for _ in range(3)]
keys, values = [], []

def forward(token,pos):
    x = weights["model.embed_tokens.weight"][token*H:(token+1)*H]
    for li in range(4):
        p = f"model.layers.{li}"
        n = rms(x,p+".input_layernorm")
        if li < 3:
            a = p+".linear_attn"
            qkv = mv(a+".in_proj_qkv",n)
            z = mv(a+".in_proj_z",n)
            av = mv(a+".in_proj_a",n)
            bv = mv(a+".in_proj_b",n)
            for c in range(len(qkv)):
                seq = conv[li][c]+[qkv[c]]
                qkv[c] = silu(sum(s*w for s,w in zip(seq,weights[a+".conv1d.weight"][c*4:c*4+4])))
                conv[li][c] = seq[1:]
            q, k, val = qkv[:K], qkv[K:2*K], qkv[2*K:]
            qnorm,knorm = math.sqrt(sum(v*v for v in q)+EPS)*math.sqrt(K), math.sqrt(sum(v*v for v in k)+EPS)
            q,k = [v/qnorm for v in q],[v/knorm for v in k]
            out=[]
            for head in range(VH):
                decay = math.exp(-math.exp(weights[a+".A_log"][head])*math.log1p(math.exp(av[head]+weights[a+".dt_bias"][head])))
                beta = sigmoid(bv[head])
                yy=[]
                for vi in range(V):
                    row=[v*decay for v in state[li][head][vi]]
                    delta=beta*(val[head*V+vi]-sum(s*t for s,t in zip(row,k)))
                    row=[s+t*delta for s,t in zip(row,k)]
                    state[li][head][vi]=row
                    yy.append(sum(s*t for s,t in zip(row,q)))
                yy=rms(yy,a+".norm",False)
                out.extend(v*silu(z[head*V+i]) for i,v in enumerate(yy))
            out=mv(a+".out_proj",out)
        else:
            a=p+".self_attn"
            proj=mv(a+".q_proj",n)
            qs=[rms(proj[h*2*D:h*2*D+D],a+".q_norm") for h in range(HEADS)]
            gate=[proj[h*2*D+D:h*2*D+2*D] for h in range(HEADS)]
            kk=rms(mv(a+".k_proj",n),a+".k_norm")
            vv=mv(a+".v_proj",n)
            for t in qs+[kk]:
                rd=D//2
                for i in range(rd//2):
                    angle=pos/(10000.**(2*i/rd))
                    u,v=t[i],t[i+rd//2]
                    t[i]=u*math.cos(angle)-v*math.sin(angle)
                    t[i+rd//2]=u*math.sin(angle)+v*math.cos(angle)
            keys.append(kk)
            values.append(vv)
            out=[]
            for h in range(HEADS):
                scores=[sum(q*k for q,k in zip(qs[h],kk))/math.sqrt(D) for kk in keys]
                ee=[math.exp(s-max(scores)) for s in scores]
                probs=[e/sum(ee) for e in ee]
                out.extend(sum(pp*vv[d] for pp,vv in zip(probs,values))*sigmoid(gate[h][d]) for d in range(D))
            out=mv(a+".o_proj",out)
        x=[u+v for u,v in zip(x,out)]
        n=rms(x,p+".post_attention_layernorm")
        gate,up=mv(p+".mlp.gate_proj",n),mv(p+".mlp.up_proj",n)
        out=mv(p+".mlp.down_proj",[silu(g)*u for g,u in zip(gate,up)])
        x=[u+v for u,v in zip(x,out)]
    return mv("lm_head",rms(x,"model.norm"))

ids=[3,8,4,9,6]
golden={"tokens":ids,"logits":[forward(t,i) for i,t in enumerate(ids)],
        "description":"Scalar Python oracle, four-layer synthetic hybrid, not trained Qwen output"}
(ROOT/"golden.json").write_text(json.dumps(golden,indent=2)+"\n")
header={}
body=bytearray()
for name,values_ in weights.items():
    data=struct.pack("<"+"e"*len(values_),*values_)
    header[name]={"dtype":"F16","shape":shapes[name],"data_offsets":[len(body),len(body)+len(data)]}
    body.extend(data)
hdr=json.dumps(header,separators=(",",":")).encode()
hdr+=b" "*((-len(hdr))%8)
(ROOT/"model.safetensors").write_bytes(struct.pack("<Q",len(hdr))+hdr+body)
config={"model_type":"qwen3_5","eos_token_id":[1],"text_config":{
    "model_type":"qwen3_5_text","hidden_size":H,"intermediate_size":FF,"num_hidden_layers":4,
    "num_attention_heads":HEADS,"num_key_value_heads":KVH,"head_dim":D,"vocab_size":VOCAB,
    "linear_num_key_heads":KH,"linear_num_value_heads":VH,"linear_key_head_dim":K,
    "linear_value_head_dim":V,"linear_conv_kernel_dim":4,"full_attention_interval":4,
    "layer_types":["linear_attention"]*3+["full_attention"],"rms_norm_eps":EPS,
    "max_position_embeddings":128,"tie_word_embeddings":False,
    "rope_parameters":{"rope_type":"default","rope_theta":10000.,"partial_rotary_factor":.5}}}
(ROOT/"config.json").write_text(json.dumps(config,indent=2)+"\n")
tokens={"[UNK]":0,"<|im_end|>":1,"<|im_start|>":2,"user":3,"assistant":4,"system":5}
tokens.update({f"w{i}":i for i in range(6,VOCAB)})
tokenizer={"version":"1.0","truncation":None,"padding":None,"added_tokens":[],
    "normalizer":None,"pre_tokenizer":{"type":"WhitespaceSplit"},"post_processor":None,
    "decoder":None,"model":{"type":"WordLevel","vocab":tokens,"unk_token":"[UNK]"}}
(ROOT/"tokenizer.json").write_text(json.dumps(tokenizer)+"\n")
(ROOT/"chat_template.jinja").write_text("{% for m in messages %}{{ m.role }} {{ m.content }} {% endfor %}assistant")
print(f"Wrote {len(body)} bytes of synthetic F16 weights and {len(ids)} reference logit vectors to {ROOT}")

# A second checkpoint exercises the packed affine-Q4 path and the converted
# MLX tensor convention. The scalar oracle uses exactly the dequantized values
# represented by the packed words and rounded F16 affine parameters.
Q4_ROOT = ROOT.parent / "tiny-q4"
Q4_ROOT.mkdir(parents=True, exist_ok=True)
q4_header = {}
q4_body = bytearray()

def q4_tensor(name, shape, dtype, data):
    q4_header[name] = {"dtype":dtype,"shape":shape,
                       "data_offsets":[len(q4_body),len(q4_body)+len(data)]}
    q4_body.extend(data)

def half(value):
    return struct.unpack("<e", struct.pack("<e", value))[0]

def mlx_name(name):
    return "language_model." + name

for name, original in weights.items():
    shape = shapes[name]
    converted_name = mlx_name(name)
    if len(shape) == 2:
        rows, cols = shape
        assert cols % 32 == 0
        packed, scales, biases, dequantized = [], [], [], []
        for row in range(rows):
            for group_start in range(0, cols, 32):
                group = original[row*cols+group_start:row*cols+group_start+32]
                scale = half((max(group)-min(group))/15)
                bias = half(min(group))
                assert scale > 0
                scales.append(scale)
                biases.append(bias)
                quantized = [min(15,max(0,round((v-bias)/scale))) for v in group]
                dequantized.extend(q*scale+bias for q in quantized)
                packed.extend(sum(q << (4*i) for i,q in enumerate(quantized[j:j+8]))
                              for j in range(0,32,8))
        weights[name] = dequantized
        module = converted_name.removesuffix(".weight")
        groups = cols//32
        q4_tensor(converted_name,[rows,cols//8],"U32",
                  struct.pack("<"+"I"*len(packed),*packed))
        q4_tensor(module+".scales",[rows,groups],"F16",
                  struct.pack("<"+"e"*len(scales),*scales))
        q4_tensor(module+".biases",[rows,groups],"F16",
                  struct.pack("<"+"e"*len(biases),*biases))
    elif name.endswith(".conv1d.weight"):
        # MLX [channels, kernel, 1] has the same flattened channel-major data.
        q4_tensor(converted_name,[shape[0],shape[2],1],"F16",
                  struct.pack("<"+"e"*len(original),*original))
    elif name == "model.norm.weight" or name.endswith((
            ".input_layernorm.weight",".post_attention_layernorm.weight",
            ".q_norm.weight",".k_norm.weight")):
        # HF zero-centered norms become multiplicative in converted MLX files.
        # F32 preserves the original rounded-F16 value plus one exactly.
        q4_tensor(converted_name,shape,"F32",
                  struct.pack("<"+"f"*len(original),*(v+1 for v in original)))
    else:
        q4_tensor(converted_name,shape,"F16",
                  struct.pack("<"+"e"*len(original),*original))

q4_hdr = json.dumps(q4_header,separators=(",",":")).encode()
q4_hdr += b" "*((-len(q4_hdr))%8)
(Q4_ROOT/"model.safetensors").write_bytes(struct.pack("<Q",len(q4_hdr))+q4_hdr+q4_body)
q4_config = dict(config)
q4_config["quantization"] = {"bits":4,"group_size":32,"mode":"affine"}
(Q4_ROOT/"config.json").write_text(json.dumps(q4_config,indent=2)+"\n")
for filename in ("tokenizer.json","chat_template.jinja"):
    (Q4_ROOT/filename).write_bytes((ROOT/filename).read_bytes())

# The F16 fixture above has already been serialized; only matrix values in
# this oracle now change. Recurrent, convolution and attention caches restart.
conv = [[[0.]*3 for _ in range(2*KH*K+VH*V)] for _ in range(3)]
state = [[[[0.]*K for _ in range(V)] for _ in range(VH)] for _ in range(3)]
keys, values = [], []
q4_golden = {"tokens":ids,"logits":[forward(t,i) for i,t in enumerate(ids)],
             "description":"Scalar Python oracle for packed affine Q4, four-layer synthetic untrained hybrid"}
(Q4_ROOT/"golden.json").write_text(json.dumps(q4_golden,indent=2)+"\n")
print(f"Wrote {len(q4_body)} bytes of synthetic packed Q4 weights and {len(ids)} reference logit vectors to {Q4_ROOT}")
