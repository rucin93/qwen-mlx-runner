// Original aligned Q4 affine GEMV. Four rows/SIMD, 64 threads/threadgroup.
// ABI buffers0..4 w,scale,bias,x,y; uint parameters at15 rows,cols,bits,group.
// HOST PRECONDITIONS: rows%4==0, cols%512==0, matching GROUP specialization,
// and signed-int-safe dimensions/index products. Grid (rows/4)*32 threads.
// Padded SIMD groups return uniformly. No row/column tails in the inner loop.
// Configure pipeline maxTotalThreadsPerThreadgroup=64.
#include <metal_stdlib>
using namespace metal;

inline float aligned_u16_dot4(ushort q, float4 x) {
    return dot(float4(ushort4(q & ushort(0x000f), q & ushort(0x00f0),
                             q & ushort(0x0f00), q & ushort(0xf000))), x);
}

inline float aligned_u16_dot16(ushort4 q, float4 xa, float4 xb, float4 xc, float4 xd) {
    return aligned_u16_dot4(q.x,xa)+aligned_u16_dot4(q.y,xb)
         + aligned_u16_dot4(q.z,xc)+aligned_u16_dot4(q.w,xd);
}

template<int GROUP>
inline void affine_q4_aligned(device const ushort *w,
                             device const float *scales,
                             device const float *bias,
                             device const float *x,
                             device float *y,
                             constant uint *p,
                             uint gid, ushort lane) {
    const int row=(int(gid)/32)*4;
    if (row>=int(p[0])) return;
    const int cols=int(p[1]);
    const int packed_stride=cols/4;
    const int group_stride=cols/GROUP;
    const int tiles=cols/512;
    const int weight_base=row*packed_stride;
    const int scale_base=row*group_stride;
    float sum0=0.0f, sum1=0.0f, sum2=0.0f, sum3=0.0f;
    for (int tile=0; tile<tiles; ++tile) {
        const int col=tile*512+int(lane)*16;
        const float4 a=*reinterpret_cast<device const float4 *>(x+col);
        const float4 b=*reinterpret_cast<device const float4 *>(x+col+4);
        const float4 c=*reinterpret_cast<device const float4 *>(x+col+8);
        const float4 d=*reinterpret_cast<device const float4 *>(x+col+12);
        const float4 combined=a+b+c+d;
        const float xsum=combined.x+combined.y+combined.z+combined.w;
        const float4 factors=float4(1.0f,0x1p-4f,0x1p-8f,0x1p-12f);
        const float4 xa=a*factors, xb=b*factors, xc=c*factors, xd=d*factors;
        const int wi=weight_base+col/4;
        const int si=scale_base+col/GROUP;
        const ushort4 q0=*reinterpret_cast<device const ushort4 *>(w+wi);
        sum0+=fma(scales[si],aligned_u16_dot16(q0,xa,xb,xc,xd),bias[si]*xsum);
        const ushort4 q1=*reinterpret_cast<device const ushort4 *>(w+wi+packed_stride);
        sum1+=fma(scales[si+group_stride],aligned_u16_dot16(q1,xa,xb,xc,xd),
                  bias[si+group_stride]*xsum);
        const ushort4 q2=*reinterpret_cast<device const ushort4 *>(w+wi+2*packed_stride);
        sum2+=fma(scales[si+2*group_stride],aligned_u16_dot16(q2,xa,xb,xc,xd),
                  bias[si+2*group_stride]*xsum);
        const ushort4 q3=*reinterpret_cast<device const ushort4 *>(w+wi+3*packed_stride);
        sum3+=fma(scales[si+3*group_stride],aligned_u16_dot16(q3,xa,xb,xc,xd),
                  bias[si+3*group_stride]*xsum);
    }
    sum0=simd_sum(sum0); sum1=simd_sum(sum1);
    sum2=simd_sum(sum2); sum3=simd_sum(sum3);
    if (lane==0) {
        y[row]=sum0; y[row+1]=sum1; y[row+2]=sum2; y[row+3]=sum3;
    }
}

#define AFFINE_ALIGNED_ENTRY(NAME, GROUP) \
kernel void NAME(device const ushort *w [[buffer(0)]], \
                 device const float *scales [[buffer(1)]], \
                 device const float *bias [[buffer(2)]], \
                 device const float *x [[buffer(3)]], \
                 device float *y [[buffer(4)]], \
                 constant uint *p [[buffer(15)]], \
                 uint gid [[thread_position_in_grid]], \
                 ushort lane [[thread_index_in_simdgroup]]) { \
    affine_q4_aligned<GROUP>(w,scales,bias,x,y,p,gid,lane); \
}

AFFINE_ALIGNED_ENTRY(matvec_q4_g32_aligned,32)
AFFINE_ALIGNED_ENTRY(matvec_q4_g64_aligned,64)
AFFINE_ALIGNED_ENTRY(matvec_q4_g128_aligned,128)

#undef AFFINE_ALIGNED_ENTRY
