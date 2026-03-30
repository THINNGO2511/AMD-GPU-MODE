import torch
import os

os.environ["PYTORCH_ROCM_ARCH"] = "gfx950"

from torch.utils.cpp_extension import load_inline

cpp_sources = """
torch::Tensor launch_gemm(torch::Tensor A, torch::Tensor B_shuffled, torch::Tensor scales_A, torch::Tensor scales_B, int M, int N, int K);
"""

cuda_sources = r"""
#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>

typedef int __attribute__((ext_vector_type(4))) int4v;
typedef float __attribute__((ext_vector_type(4))) float4v;

// MFMA 16x16x128 FP4 intrinsic — compiler builtin, no extern declaration needed
// 9 args: a(int4v), b(int4v), c(float4v), cbsz, blgp, cbsz_sel, scale_a, op_sel, scale_b

#define BM 64
#define BN 64
#define BK 128
#define WARP_SZ 64
#define N_WAVES 4
#define N_THREADS (N_WAVES * WARP_SZ)
#define LDS_A_BYTES (BM * BK / 2)
#define LDS_B_BYTES (BN / 16 * 1024)
#define LDS_SLICE   (LDS_A_BYTES + LDS_B_BYTES)
#define SCALE_BLK   32

__global__ __attribute__((amdgpu_flat_work_group_size(256, 256)))
void fp4_gemm_16x16x128_kernel(
    const unsigned char* __restrict__ A,
    const unsigned char* __restrict__ B_shuf,
    const unsigned char* __restrict__ scl_A,
    const unsigned char* __restrict__ scl_B,
    hip_bfloat16* __restrict__ C,
    const int M, const int N, const int K)
{
    const int bm = blockIdx.x;
    const int bn = blockIdx.y;
    const int tid = threadIdx.x;
    const int wid = tid / WARP_SZ;
    const int lane = tid % WARP_SZ;
    const int wm = wid >> 1;
    const int wn = wid & 1;
    const int lr = lane & 15;
    const int lg = lane >> 4;

    const int row_base = bm * BM;
    const int col_base = bn * BN;
    const int Khalf = K / 2;
    const int scl_stride = K / SCALE_BLK;

    extern __shared__ unsigned char smem[];
    unsigned char* lds_buf[2];
    lds_buf[0] = smem;
    lds_buf[1] = smem + LDS_SLICE;

    float4v acc[2][2];
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            acc[i][j] = {0.f, 0.f, 0.f, 0.f};

    const int K_tiles = K / BK;

    // Load tile into LDS buffer
    auto loadTile = [&](int buf, int koff) {
        // A: BM rows x BK/2 bytes = 4096 bytes, 256 threads x 16B each
        {
            int off = tid * 16;
            int r = off / (BK / 2);
            int c = off % (BK / 2);
            int gr = row_base + r;
            unsigned char* dst = lds_buf[buf] + off;
            if (gr < M) {
                *(uint4*)dst = *(const uint4*)(A + (size_t)gr * Khalf + koff / 2 + c);
            } else {
                *(uint4*)dst = make_uint4(0,0,0,0);
            }
        }
        // B: 4 n_blocks x 1024 bytes = 4096 bytes
        {
            int off = tid * 16;
            int nb = off / 1024;
            int rem = off % 1024;
            int gnb = col_base / 16 + nb;
            unsigned char* dst = lds_buf[buf] + LDS_A_BYTES + off;
            if (gnb * 16 < N) {
                *(uint4*)dst = *(const uint4*)(B_shuf + (size_t)gnb * Khalf * 16 + (size_t)(koff / 2) * 16 + rem);
            } else {
                *(uint4*)dst = make_uint4(0,0,0,0);
            }
        }
    };

    loadTile(0, 0);
    __syncthreads();

    for (int kt = 0; kt < K_tiles; kt++) {
        int buf = kt & 1;

        if (kt + 1 < K_tiles) {
            loadTile(1 - buf, (kt + 1) * BK);
        }

        unsigned char* curA = lds_buf[buf];
        unsigned char* curB = lds_buf[buf] + LDS_A_BYTES;

        int4v a_op[2];
        for (int tm = 0; tm < 2; tm++) {
            int ar = wm * 32 + tm * 16 + lr;
            int aoff = ar * (BK / 2) + lg * 16;
            uint4 v = *(const uint4*)(curA + aoff);
            a_op[tm] = *(int4v*)&v;
        }

        int4v b_op[2];
        for (int tn = 0; tn < 2; tn++) {
            int nb = wn * 2 + tn;
            int boff = nb * 1024 + lg * 256 + lr * 16;
            uint4 v = *(const uint4*)(curB + boff);
            b_op[tn] = *(int4v*)&v;
        }

        int koff = kt * BK;
        int ksi = koff / SCALE_BLK;

        unsigned int sa[2], sb[2];
        for (int tm = 0; tm < 2; tm++) {
            int r = row_base + wm * 32 + tm * 16 + lr;
            if (r < M) {
                const unsigned char* p = scl_A + (size_t)r * scl_stride + ksi;
                sa[tm] = (unsigned int)p[0] | ((unsigned int)p[1] << 8) |
                         ((unsigned int)p[2] << 16) | ((unsigned int)p[3] << 24);
            } else {
                sa[tm] = 0x7f7f7f7fu;
            }
        }
        for (int tn = 0; tn < 2; tn++) {
            int c = col_base + wn * 32 + tn * 16 + lr;
            if (c < N) {
                const unsigned char* p = scl_B + (size_t)c * scl_stride + ksi;
                sb[tn] = (unsigned int)p[0] | ((unsigned int)p[1] << 8) |
                         ((unsigned int)p[2] << 16) | ((unsigned int)p[3] << 24);
            } else {
                sb[tn] = 0x7f7f7f7fu;
            }
        }

        for (int tm = 0; tm < 2; tm++) {
            for (int tn = 0; tn < 2; tn++) {
                acc[tm][tn] = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
                    a_op[tm], b_op[tn], acc[tm][tn],
                    4, 4, 0,
                    sa[tm], 0, sb[tn]);
            }
        }

        if (kt + 1 < K_tiles)
            __syncthreads();
    }

    for (int tm = 0; tm < 2; tm++) {
        for (int tn = 0; tn < 2; tn++) {
            for (int i = 0; i < 4; i++) {
                int r = row_base + wm * 32 + tm * 16 + lg * 4 + i;
                int c = col_base + wn * 32 + tn * 16 + lr;
                if (r < M && c < N)
                    C[(size_t)r * N + c] = __float2bfloat16(acc[tm][tn][i]);
            }
        }
    }
}

torch::Tensor launch_gemm(
    torch::Tensor A, torch::Tensor B_shuffled,
    torch::Tensor scales_A, torch::Tensor scales_B,
    int M, int N, int K)
{
    auto C = torch::empty({M, N},
        torch::TensorOptions().dtype(torch::kBFloat16).device(A.device()));
    dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN);
    dim3 block(N_THREADS);
    hipLaunchKernelGGL(fp4_gemm_16x16x128_kernel,
        grid, block, 2 * LDS_SLICE, 0,
        (const unsigned char*)A.data_ptr(),
        (const unsigned char*)B_shuffled.data_ptr(),
        (const unsigned char*)scales_A.data_ptr(),
        (const unsigned char*)scales_B.data_ptr(),
        (hip_bfloat16*)C.data_ptr(), M, N, K);
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_gemm", &launch_gemm, "FP4 GEMM 16x16x128 MFMA");
}
"""

gemm_module = load_inline(
    name="fp4_gemm_16x16",
    cpp_sources=cpp_sources,
    cuda_sources=cuda_sources,
    functions=["launch_gemm"],
    extra_cuda_cflags=["-O3", "--offload-arch=gfx950", "-mllvm", "-amdgpu-early-inline-all=true"],
    verbose=False,
    with_cuda=True,
)

def custom_kernel(data):
    A = data["A"]
    B = data["B_shuffled"]
    sa = data["scales_A"]
    sb = data["scales_B"]
    M = data["M"]
    N = data["N"]
    K = data["K"]
    return gemm_module.launch_gemm(A, B, sa, sb, M, N, K)
