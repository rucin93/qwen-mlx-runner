// Original Qwen dense hybrid decode kernels, FP32 activations and recurrent state.
// ABI: all parameter words are uint at buffer(15); floating parameters use bit casts.
// Buffers below are positional. SIMD kernels require a 32-thread SIMD width and
// a threadgroup size divisible by 32. Dispatch is padded, every kernel bounds checks.
// matvec_f16: w half[row,col],x,y; p rows,cols; grid rows*32.
// matvec_affine: w uint[row,col/(32/bits)],scale,bias,x,y;
//   p rows,cols,bits,group_size; grid rows*32. q is little-endian within uint.
// embed_f16: w,y; p row,cols; grid cols.
// embed_affine: w,scale,bias,y; p row,cols,bits,group_size; grid cols.
// rms_norm: x,w,y; p n,eps; grid 32.
// add: x,residual,y; p n; grid n. swiglu: gate,up,y; p n; grid n.
// conv_silu: x,w,state,y; p channels,width; grid channels;
//   w [channels,width], state [channels,width-1], oldest sample first.
// delta_norm: qkv in place; p key_heads,K,eps; grid key_heads*32;
//   qkv contiguous [key_heads*K,key_heads*K,value_heads*V].
// delta_step: qkv,a,b,Alog,dt,state,y; p KH,VH,K,V; grid VH*V*32;
//   state [VH,V,K]. q has already been scaled by 1/sqrt(K).
// gated_rms: x,z,w,y; p VH,V,eps; grid VH*32; w [V].
// split_q_gate: projection,q,gate; p H,D; grid H*D; input [H,2,D].
// head_rms: x in place,w[D]; p H,D,eps; grid H*32.
// rope: x in place; p H,D,rotary_dim,position,theta; grid H*rotary_dim/2;
//   partial, noninterleaved rotate-half RoPE, frequency exponent 2*i/rotary_dim.
// kv_append: k,v,kcache,vcache; p KVH,D,position; grid KVH*D;
//   both caches [token,KVH,D].
// attn_scores: q,kcache,scores; p H,KVH,D,length; grid H*length*32;
//   scores [H,length], GQA contiguous groups, scale 1/sqrt(D).
// softmax: scores in place; p H,length; grid H*32.
// attn_values: scores,vcache,gate,y; p H,KVH,D,length; grid H*D;
//   multiplies attention output by sigmoid(gate).
#include <metal_stdlib>
using namespace metal;
inline float sigmoid_stable(float x) { return 1.0f / (1.0f + exp(-x)); }
inline float silu(float x) { return x * sigmoid_stable(x); }
inline float softplus_stable(float x) {
    // Metal has no log1p. Compensate rounding of 1+t, including tiny t.
    float t=exp(-abs(x)), u=1.0f+t;
    float log_one_plus_t=(u==1.0f) ? t : log(u)*(t/(u-1.0f));
    return max(x,0.0f)+log_one_plus_t;
}
inline float affine(device const uint *w, device const float *s, device const float *b,
                    uint row, uint col, uint cols, uint bits, uint group) {
    uint pack = 32 / bits;
    uint word = w[row * (cols / pack) + col / pack];
    uint q = (word >> ((col % pack) * bits)) & ((1u << bits) - 1u);
    uint gi = row * (cols / group) + col / group;
    return float(q) * s[gi] + b[gi];
}
kernel void matvec_f16(device const half *w [[buffer(0)]], device const float *x [[buffer(1)]],
                       device float *y [[buffer(2)]], constant uint *p [[buffer(15)]],
                       uint gid [[thread_position_in_grid]], ushort lane [[thread_index_in_simdgroup]]) {
    uint row = gid / 32;
    if (row >= p[0]) return;
    float v = 0;
    for (uint c=lane;c<p[1];c+=32) v += float(w[row*p[1]+c])*x[c];
    v = simd_sum(v);
    if (lane==0) y[row]=v;
}
kernel void matvec_affine(device const uint *w [[buffer(0)]], device const float *s [[buffer(1)]],
                          device const float *b [[buffer(2)]], device const float *x [[buffer(3)]],
                          device float *y [[buffer(4)]], constant uint *p [[buffer(15)]],
                          uint gid [[thread_position_in_grid]], ushort lane [[thread_index_in_simdgroup]]) {
    uint row=gid/32;
    if(row>=p[0]) return;
    float v=0;
    for(uint c=lane;c<p[1];c+=32) v += affine(w,s,b,row,c,p[1],p[2],p[3])*x[c];
    v=simd_sum(v);
    if(lane==0) y[row]=v;
}
kernel void embed_f16(device const half *w [[buffer(0)]], device float *y [[buffer(1)]],
                      constant uint *p [[buffer(15)]], uint i [[thread_position_in_grid]]) {
    if(i<p[1]) y[i]=float(w[p[0]*p[1]+i]);
}
kernel void embed_affine(device const uint *w [[buffer(0)]], device const float *s [[buffer(1)]],
                         device const float *b [[buffer(2)]], device float *y [[buffer(3)]],
                         constant uint *p [[buffer(15)]], uint i [[thread_position_in_grid]]) {
    if(i<p[1]) y[i]=affine(w,s,b,p[0],i,p[1],p[2],p[3]);
}
kernel void rms_norm(device const float *x [[buffer(0)]], device const float *w [[buffer(1)]],
                     device float *y [[buffer(2)]], constant uint *p [[buffer(15)]],
                     uint gid [[thread_position_in_grid]], ushort lane [[thread_index_in_simdgroup]]) {
    if(gid>=32) return;
    float sum=0;
    for(uint i=lane;i<p[0];i+=32) sum += x[i]*x[i];
    float inv=rsqrt(simd_sum(sum)/float(p[0])+as_type<float>(p[1]));
    for(uint i=lane;i<p[0];i+=32) y[i]=x[i]*inv*w[i];
}
kernel void add(device const float *x [[buffer(0)]], device const float *r [[buffer(1)]],
                device float *y [[buffer(2)]], constant uint *p [[buffer(15)]], uint i [[thread_position_in_grid]]) {
    if(i<p[0]) y[i]=x[i]+r[i];
}
kernel void swiglu(device const float *g [[buffer(0)]], device const float *u [[buffer(1)]],
                   device float *y [[buffer(2)]], constant uint *p [[buffer(15)]], uint i [[thread_position_in_grid]]) {
    if(i<p[0]) y[i]=silu(g[i])*u[i];
}
kernel void conv_silu(device const float *x [[buffer(0)]], device const float *w [[buffer(1)]],
                      device float *state [[buffer(2)]], device float *y [[buffer(3)]],
                      constant uint *p [[buffer(15)]], uint c [[thread_position_in_grid]]) {
    if(c>=p[0]) return;
    uint width=p[1], base=c*(width-1);
    float sum=x[c]*w[c*width+width-1];
    for(uint k=0;k+1<width;++k) sum += state[base+k]*w[c*width+k];
    for(uint k=0;k+2<width;++k) state[base+k]=state[base+k+1];
    if(width>1) state[base+width-2]=x[c];
    y[c]=silu(sum);
}
kernel void delta_norm(device float *qkv [[buffer(0)]], constant uint *p [[buffer(15)]],
                       uint gid [[thread_position_in_grid]], ushort lane [[thread_index_in_simdgroup]]) {
    uint h=gid/32, K=p[1];
    if(h>=p[0]) return;
    uint qo=h*K, ko=p[0]*K+qo;
    float qs=0,ks=0;
    for(uint i=lane;i<K;i+=32) { qs+=qkv[qo+i]*qkv[qo+i];ks+=qkv[ko+i]*qkv[ko+i]; }
    float qi=rsqrt(simd_sum(qs)+as_type<float>(p[2]))*rsqrt(float(K));
    float ki=rsqrt(simd_sum(ks)+as_type<float>(p[2]));
    for(uint i=lane;i<K;i+=32) { qkv[qo+i]*=qi; qkv[ko+i]*=ki; }
}
kernel void delta_step(device const float *qkv [[buffer(0)]], device const float *a [[buffer(1)]],
                       device const float *b [[buffer(2)]], device const float *Alog [[buffer(3)]],
                       device const float *dt [[buffer(4)]], device float *state [[buffer(5)]],
                       device float *y [[buffer(6)]], constant uint *p [[buffer(15)]],
                       uint gid [[thread_position_in_grid]], ushort lane [[thread_index_in_simdgroup]]) {
    uint row=gid/32, KH=p[0],VH=p[1],K=p[2],V=p[3];
    if(row>=VH*V) return;
    uint h=row/V,kh=h/(VH/KH), qo=kh*K,ko=KH*K+qo,vo=2*KH*K+row;
    float decay=exp(-exp(Alog[h])*softplus_stable(a[h]+dt[h]));
    float prediction=0;
    for(uint i=lane;i<K;i+=32) prediction += state[row*K+i]*decay*qkv[ko+i];
    float delta=(qkv[vo]-simd_sum(prediction))*sigmoid_stable(b[h]);
    float output=0;
    for(uint i=lane;i<K;i+=32) {
        float next=state[row*K+i]*decay+qkv[ko+i]*delta;
        state[row*K+i]=next;
        output+=next*qkv[qo+i];
    }
    output=simd_sum(output);
    if(lane==0) y[row]=output;
}
// B=1..4 causal updates with K<=128. Each lane retains its four state
// elements across tokens. Prefix i is the state before input i; the final
// prefix B remains in live state and therefore needs no snapshot copy.
kernel void delta_step_block(device const float *qkv [[buffer(0)]], device const float *a [[buffer(1)]],
                             device const float *b [[buffer(2)]], device const float *Alog [[buffer(3)]],
                             device const float *dt [[buffer(4)]], device float *state [[buffer(5)]],
                             device float *y [[buffer(6)]], device float *snapshots [[buffer(7)]],
                             constant uint *p [[buffer(15)]], uint gid [[thread_position_in_grid]],
                             ushort lane [[thread_index_in_simdgroup]]) {
    uint row=gid/32, KH=p[0],VH=p[1],K=p[2],V=p[3],B=p[4],rows=VH*V;
    if(row>=rows) return;
    uint h=row/V,kh=h/(VH/KH),width=2*KH*K+rows;
    float current[4];
    for(uint j=0;j<4;++j) {
        uint i=uint(lane)+j*32;
        current[j]=i<K ? state[row*K+i] : 0.0f;
    }
    for(uint token=0;token<B;++token) {
        uint qo=token*width+kh*K,ko=qo+KH*K,vo=token*width+2*KH*K+row;
        float decay=exp(-exp(Alog[h])*softplus_stable(a[token*VH+h]+dt[h]));
        float prediction=0;
        for(uint j=0;j<4;++j) {
            uint i=uint(lane)+j*32;
            if(i<K) {
                snapshots[token*rows*K+row*K+i]=current[j];
                prediction+=current[j]*decay*qkv[ko+i];
            }
        }
        float delta=(qkv[vo]-simd_sum(prediction))*sigmoid_stable(b[token*VH+h]);
        float output=0;
        for(uint j=0;j<4;++j) {
            uint i=uint(lane)+j*32;
            if(i<K) {
                float next=current[j]*decay+qkv[ko+i]*delta;
                current[j]=next;
                output+=next*qkv[qo+i];
            }
        }
        output=simd_sum(output);
        if(lane==0) y[token*rows+row]=output;
    }
    for(uint j=0;j<4;++j) {
        uint i=uint(lane)+j*32;
        if(i<K) state[row*K+i]=current[j];
    }
}
// Known prompt batches need only live recurrent state, with no rollback writes.
kernel void delta_step_prefill(device const float *qkv [[buffer(0)]], device const float *a [[buffer(1)]],
                             device const float *b [[buffer(2)]], device const float *Alog [[buffer(3)]],
                             device const float *dt [[buffer(4)]], device float *state [[buffer(5)]],
                             device float *y [[buffer(6)]],
                             constant uint *p [[buffer(15)]], uint gid [[thread_position_in_grid]],
                             ushort lane [[thread_index_in_simdgroup]]) {
    uint row=gid/32, KH=p[0],VH=p[1],K=p[2],V=p[3],B=p[4],rows=VH*V;
    if(row>=rows) return;
    uint h=row/V,kh=h/(VH/KH),width=2*KH*K+rows;
    float current[4];
    for(uint j=0;j<4;++j) {
        uint i=uint(lane)+j*32;
        current[j]=i<K ? state[row*K+i] : 0.0f;
    }
    for(uint token=0;token<B;++token) {
        uint qo=token*width+kh*K,ko=qo+KH*K,vo=token*width+2*KH*K+row;
        float decay=exp(-exp(Alog[h])*softplus_stable(a[token*VH+h]+dt[h]));
        float prediction=0;
        for(uint j=0;j<4;++j) {
            uint i=uint(lane)+j*32;
            if(i<K) {
                prediction+=current[j]*decay*qkv[ko+i];
            }
        }
        float delta=(qkv[vo]-simd_sum(prediction))*sigmoid_stable(b[token*VH+h]);
        float output=0;
        for(uint j=0;j<4;++j) {
            uint i=uint(lane)+j*32;
            if(i<K) {
                float next=current[j]*decay+qkv[ko+i]*delta;
                current[j]=next;
                output+=next*qkv[qo+i];
            }
        }
        output=simd_sum(output);
        if(lane==0) y[token*rows+row]=output;
    }
    for(uint j=0;j<4;++j) {
        uint i=uint(lane)+j*32;
        if(i<K) state[row*K+i]=current[j];
    }
}
kernel void gated_rms(device const float *x [[buffer(0)]], device const float *z [[buffer(1)]],
                      device const float *w [[buffer(2)]], device float *y [[buffer(3)]],
                      constant uint *p [[buffer(15)]], uint gid [[thread_position_in_grid]],
                      ushort lane [[thread_index_in_simdgroup]]) {
    uint h=gid/32,V=p[1];
    if(h>=p[0]) return;
    float sum=0;
    for(uint i=lane;i<V;i+=32) sum+=x[h*V+i]*x[h*V+i];
    float inv=rsqrt(simd_sum(sum)/float(V)+as_type<float>(p[2]));
    for(uint i=lane;i<V;i+=32) y[h*V+i]=x[h*V+i]*inv*w[i]*silu(z[h*V+i]);
}
kernel void split_q_gate(device const float *x [[buffer(0)]],device float *q [[buffer(1)]],
                         device float *gate [[buffer(2)]],constant uint *p [[buffer(15)]],
                         uint i [[thread_position_in_grid]]) {
    if(i>=p[0]*p[1]) return;
    uint h=i/p[1],j=i%p[1];
    q[i]=x[h*2*p[1]+j];gate[i]=x[h*2*p[1]+p[1]+j];
}
kernel void head_rms(device float *x [[buffer(0)]], device const float *w [[buffer(1)]],
                     constant uint *p [[buffer(15)]], uint gid [[thread_position_in_grid]],
                     ushort lane [[thread_index_in_simdgroup]]) {
    uint h=gid/32,D=p[1];
    if(h>=p[0]) return;
    float sum=0;
    for(uint i=lane;i<D;i+=32) sum+=x[h*D+i]*x[h*D+i];
    float inv=rsqrt(simd_sum(sum)/float(D)+as_type<float>(p[2]));
    for(uint i=lane;i<D;i+=32) x[h*D+i]*=inv*w[i];
}
kernel void rope(device float *x [[buffer(0)]],constant uint *p [[buffer(15)]],uint i [[thread_position_in_grid]]) {
    uint halfdim=p[2]/2;
    if(i>=p[0]*halfdim) return;
    uint h=i/halfdim,j=i%halfdim,base=h*p[1];
    float angle=float(p[3])*pow(as_type<float>(p[4]),-float(2*j)/float(p[2]));
    float a=x[base+j],b=x[base+j+halfdim],cs=cos(angle),sn=sin(angle);
    x[base+j]=a*cs-b*sn;x[base+j+halfdim]=a*sn+b*cs;
}
kernel void kv_append(device const float *k [[buffer(0)]], device const float *v [[buffer(1)]],
                      device float *kc [[buffer(2)]],device float *vc [[buffer(3)]],
                      constant uint *p [[buffer(15)]],uint i [[thread_position_in_grid]]) {
    uint n=p[0]*p[1];
    if(i<n) { kc[p[2]*n+i]=k[i];vc[p[2]*n+i]=v[i]; }
}
kernel void attn_scores(device const float *q [[buffer(0)]],device const float *kc [[buffer(1)]],
                        device float *scores [[buffer(2)]],constant uint *p [[buffer(15)]],
                        uint gid [[thread_position_in_grid]],ushort lane [[thread_index_in_simdgroup]]) {
    uint row=gid/32,H=p[0],KVH=p[1],D=p[2],L=p[3];
    if(row>=H*L) return;
    uint h=row/L,t=row%L,kh=h/(H/KVH);
    float score=0;
    for(uint i=lane;i<D;i+=32) score+=q[h*D+i]*kc[(t*KVH+kh)*D+i];
    score=simd_sum(score)*rsqrt(float(D));
    if(lane==0) scores[row]=score;
}
kernel void softmax(device float *scores [[buffer(0)]],constant uint *p [[buffer(15)]],
                    uint gid [[thread_position_in_grid]],ushort lane [[thread_index_in_simdgroup]]) {
    uint h=gid/32,L=p[1];
    if(h>=p[0]) return;
    float m=-INFINITY;
    for(uint i=lane;i<L;i+=32) m=max(m,scores[h*L+i]);
    m=simd_max(m);
    float sum=0;
    for(uint i=lane;i<L;i+=32) sum+=exp(scores[h*L+i]-m);
    sum=simd_sum(sum);
    for(uint i=lane;i<L;i+=32) scores[h*L+i]=exp(scores[h*L+i]-m)/sum;
}
kernel void attn_values(device const float *scores [[buffer(0)]],device const float *vc [[buffer(1)]],
                        device const float *gate [[buffer(2)]],device float *y [[buffer(3)]],
                        constant uint *p [[buffer(15)]],uint i [[thread_position_in_grid]]) {
    uint H=p[0],KVH=p[1],D=p[2],L=p[3];
    if(i>=H*D) return;
    uint h=i/D,d=i%D,kh=h/(H/KVH);
    float v=0;
    for(uint t=0;t<L;++t) v+=scores[h*L+t]*vc[(t*KVH+kh)*D+d];
    y[i]=v*sigmoid_stable(gate[i]);
}
