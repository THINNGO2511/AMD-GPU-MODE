#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

__device__ __constant__ float c_lut[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

__device__ __forceinline__ float e8m0f(uint8_t e) {
    uint32_t bits = ((uint32_t)e) << 23;
    float r; __builtin_memcpy(&r, &bits, sizeof(float)); return r;
}

__device__ __forceinline__ uint8_t compute_scale(float amax) {
    if (amax == 0.0f) return 0u;
    uint32_t bits; __builtin_memcpy(&bits, &amax, sizeof(uint32_t));
    int be = (int)((bits >> 23) & 0xFF);
    uint32_t m = bits & 0x7FFFFF;
    int se = be - ((m > 0x400000) ? 1 : 2);
    return (uint8_t)(se < 0 ? 0 : (se > 255 ? 255 : se));
}

__device__ __forceinline__ uint8_t qfp4(float v) {
    uint8_t s = (v < 0.0f) ? 8u : 0u;
    float a = fabsf(v);
    uint8_t m;
    if      (a < 0.25f) m = 0; else if (a < 0.75f) m = 1;
    else if (a < 1.25f) m = 2; else if (a < 1.75f) m = 3;
    else if (a < 2.5f)  m = 4; else if (a < 3.5f)  m = 5;
    else if (a < 5.0f)  m = 6; else m = 7;
    return s | m;
}

template<int RPB>
__global__ __attribute__((amdgpu_flat_work_group_size(256, 256)))
void mxfp4_gemm_main(
    const __hip_bfloat16* __restrict__ A, const uint8_t* __restrict__ B_qt,
    const uint8_t* __restrict__ B_st, __hip_bfloat16* __restrict__ C,
    const int M, const int N, const int K) {
    const int n = blockIdx.x * 256 + threadIdx.x;
    const int m_base = blockIdx.y * RPB;
    if (n >= N) return;
    const int ng = K >> 5;
    float acc[RPB];
    #pragma unroll
    for (int r = 0; r < RPB; r++) acc[r] = 0.0f;
    for (int g = 0; g < ng; g++) {
        const uint8_t* bq = B_qt + ((int64_t)g * N + n) * 16;
        float bsc = e8m0f(B_st[g * N + n]);
        const uint32_t* p = reinterpret_cast<const uint32_t*>(bq);
        uint32_t bp0=p[0], bp1=p[1], bp2=p[2], bp3=p[3];
        float bv[32];
        #pragma unroll
        for (int j=0;j<8;j++) {
            bv[j]=c_lut[(bp0>>(j*4))&0xF]; bv[8+j]=c_lut[(bp1>>(j*4))&0xF];
            bv[16+j]=c_lut[(bp2>>(j*4))&0xF]; bv[24+j]=c_lut[(bp3>>(j*4))&0xF];
        }
        #pragma unroll
        for (int r=0; r<RPB; r++) {
            int m = m_base + r; if (m >= M) break;
            const __hip_bfloat16* ap = A + (int64_t)m*K + (g<<5);
            const uint32_t* a32 = reinterpret_cast<const uint32_t*>(ap);
            float av[32]; float amax = 0.0f;
            #pragma unroll
            for (int i=0;i<16;i++) {
                uint32_t pk=a32[i]; __hip_bfloat16 lo,hi;
                __builtin_memcpy(&lo, &pk, 2);
                uint16_t hb=(uint16_t)(pk>>16); __builtin_memcpy(&hi, &hb, 2);
                av[i*2]=__bfloat162float(lo); av[i*2+1]=__bfloat162float(hi);
                amax=fmaxf(amax, fmaxf(fabsf(av[i*2]), fabsf(av[i*2+1])));
            }
            uint8_t ae=compute_scale(amax); float asc=e8m0f(ae);
            float inv=(asc>0.0f)?(1.0f/asc):0.0f;
            float dot=0.0f;
            #pragma unroll
            for (int i=0;i<32;i++) dot += c_lut[qfp4(av[i]*inv)] * bv[i];
            acc[r] += dot * asc * bsc;
        }
    }
    #pragma unroll
    for (int r=0;r<RPB;r++) { int m=m_base+r; if(m<M) C[(int64_t)m*N+n]=__float2bfloat16(acc[r]); }
}

torch::Tensor mxfp4_gemm_transposed(
    torch::Tensor A, torch::Tensor B_qt, torch::Tensor B_st,
    int64_t N_out, int64_t K_val) {
    const int M=A.size(0), K=(int)K_val, N=(int)N_out;
    auto C=torch::empty({M,N}, A.options());
    auto stream=at::cuda::getCurrentCUDAStream();
    int rpb = (M<=4)?4:((M<=64)?8:16);
    dim3 grid((N+255)/256, (M+rpb-1)/rpb), block(256);
    if (rpb==4) hipLaunchKernelGGL((mxfp4_gemm_main<4>),grid,block,0,stream,
        reinterpret_cast<const __hip_bfloat16*>(A.data_ptr()),
        B_qt.data_ptr<uint8_t>(),B_st.data_ptr<uint8_t>(),
        reinterpret_cast<__hip_bfloat16*>(C.data_ptr()),M,N,K);
    else if (rpb==8) hipLaunchKernelGGL((mxfp4_gemm_main<8>),grid,block,0,stream,
        reinterpret_cast<const __hip_bfloat16*>(A.data_ptr()),
        B_qt.data_ptr<uint8_t>(),B_st.data_ptr<uint8_t>(),
        reinterpret_cast<__hip_bfloat16*>(C.data_ptr()),M,N,K);
    else hipLaunchKernelGGL((mxfp4_gemm_main<16>),grid,block,0,stream,
        reinterpret_cast<const __hip_bfloat16*>(A.data_ptr()),
        B_qt.data_ptr<uint8_t>(),B_st.data_ptr<uint8_t>(),
        reinterpret_cast<__hip_bfloat16*>(C.data_ptr()),M,N,K);
    return C;
}

torch::Tensor mxfp4_gemm_hip(
    torch::Tensor A, torch::Tensor B_q, torch::Tensor B_scale, int64_t N_out) {
    const int M=A.size(0), K=A.size(1), N=(int)N_out;
    auto C=torch::empty({M,N}, A.options());
    auto stream=at::cuda::getCurrentCUDAStream();
    int rpb=(M<=4)?4:((M<=64)?8:16);
    dim3 grid((N+255)/256,(M+rpb-1)/rpb), block(256);
    if (rpb==4) hipLaunchKernelGGL((mxfp4_gemm_main<4>),grid,block,0,stream,
        reinterpret_cast<const __hip_bfloat16*>(A.data_ptr()),
        B_q.data_ptr<uint8_t>(),B_scale.data_ptr<uint8_t>(),
        reinterpret_cast<__hip_bfloat16*>(C.data_ptr()),M,N,K);
    else if (rpb==8) hipLaunchKernelGGL((mxfp4_gemm_main<8>),grid,block,0,stream,
        reinterpret_cast<const __hip_bfloat16*>(A.data_ptr()),
        B_q.data_ptr<uint8_t>(),B_scale.data_ptr<uint8_t>(),
        reinterpret_cast<__hip_bfloat16*>(C.data_ptr()),M,N,K);
    else hipLaunchKernelGGL((mxfp4_gemm_main<16>),grid,block,0,stream,
        reinterpret_cast<const __hip_bfloat16*>(A.data_ptr()),
        B_q.data_ptr<uint8_t>(),B_scale.data_ptr<uint8_t>(),
        reinterpret_cast<__hip_bfloat16*>(C.data_ptr()),M,N,K);
    return C;
}
