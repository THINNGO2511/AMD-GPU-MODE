#pragma once

#include <stdio.h>         // For printf
#include <hip/hip_runtime.h>
#include <hip/hip_fp8.h>
#include <hip/hip_fp16.h>
#include <hip/amd_detail/amd_warp_sync_functions.h>
#include <ck/utility/amd_buffer_addressing.hpp>
#include <ck/utility/data_type.hpp>
#include <hipcub/hipcub.hpp>
#include <hipblas-common/hipblas-common.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <hipblaslt/hipblaslt.h>
#include <rocwmma/rocwmma.hpp>
#include <ATen/hip/HIPContext.h>
#define PARAMETERIZE_LIBRARY

namespace wmma = rocwmma;
struct KernelTimer {
  KernelTimer(size_t ops, float *, float *) {
  }
  void start_timer(hipStream_t stream = 0) {
  }
  void stop_timer(hipStream_t stream = 0) {
  }
  
  void synchronize() {}
};

class KernelTimerScoped {

  public:
    KernelTimerScoped(std::vector<std::shared_ptr<KernelTimer>> &timers, size_t calc_ops, float *time, float *gflops,
                      hipStream_t stream = 0) {}

    ~KernelTimerScoped() {}
};

// Combined source code


// From gpu_libs.h
#ifdef TEST_ON_CUDA

namespace wmma = nvcuda::wmma;

#define LIB_CALL(call)                                                                                                 \
  do {                                                                                                                 \
    cudaError_t err = call;                                                                                            \
    if (err != cudaSuccess) {                                                                                          \
      abort();                                                                                                         \
    }                                                                                                                  \
  } while (0)

#define HOST_TYPE(x) cuda##x

#else

#ifndef HIP_HEADERS__
#define HIP_HEADERS__
#endif

namespace wmma = rocwmma;

#define LIB_CALL(call)                                                                                                 \
  do {                                                                                                                 \
    hipError_t err = call;                                                                                             \
    if (err != hipSuccess) {                                                                                           \
      abort();                                                                                                         \
    }                                                                                                                  \
  } while (0)

#define HOST_TYPE(x) hip##x

#endif

// From gpu_types.h
#pragma once
#ifdef TEST_ON_CUDA
#define __FP8_TYPE __nv_fp8_e4m3
#define __FP8x4_TYPE __nv_fp8x4_e4m3
#define __BF16_TYPE __nv_bfloat16
#define __BF16x2_TYPE __nv_bfloat162

#else
#ifdef TEST_ON_RDNA4
#define __FP8_TYPE __hip_fp8_e4m3
#define __FP8x4_TYPE __hip_fp8x4_e4m3
constexpr const inline int WAVE_SIZE = 32;
constexpr const inline int XCD_SWIZZLE = 1;
#else
#define __FP8_TYPE __hip_fp8_e4m3_fnuz
#define __FP8x4_TYPE __hip_fp8x4_e4m3_fnuz
constexpr const inline int WAVE_SIZE = 64;
constexpr const inline int XCD_SWIZZLE = 8;
#endif

#define __BF16_TYPE __hip_bfloat16
#define __BF16x2_TYPE __hip_bfloat162
#define __FP16_TYPE __half
#define __INT16_TYPE int16_t

#endif

#define __FP32_TYPE float

// From metrics.h
#pragma once

struct PerfMetricEntry {
    char name[20];
    float time;
    float gflops;
};

struct PerfMetrics {
    int count = 0;
    PerfMetricEntry entries[20];
};

// From arithmetic.h
#ifndef ARITHMETIC_H
#define ARITHMETIC_H

template <int x, int y> constexpr __device__ __host__ inline int exact_div() {
  static_assert(x % y == 0);
  static_assert(x >= y);
  return x / y;
}

constexpr __device__ __host__ inline int ceil_div(int x, int y) { return (x + y - 1) / y; }

#endif

// From gemm_kernel.cpp
#ifndef GEMM_KERNEL
#define GEMM_KERNEL

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"

namespace gemm_kernel {

template <typename data_type, int BATCH_SIZE> __device__ inline void read_batch(data_type *dst, const data_type *src) {
    if constexpr ((sizeof(data_type) * BATCH_SIZE) == 2 * sizeof(ulong4)) {
        *(reinterpret_cast<ulong4 *>(dst) + 0) = *(reinterpret_cast<const ulong4 *>(src) + 0);
        *(reinterpret_cast<ulong4 *>(dst) + 1) = *(reinterpret_cast<const ulong4 *>(src) + 1);
    } else if constexpr ((sizeof(data_type) * BATCH_SIZE) == sizeof(ulong4)) {
        *reinterpret_cast<ulong4 *>(dst) = *reinterpret_cast<const ulong4 *>(src);
    } else if constexpr (sizeof(data_type) * BATCH_SIZE == sizeof(ulong2)) {
        *reinterpret_cast<ulong2 *>(dst) = *reinterpret_cast<const ulong2 *>(src);
    } else if constexpr (sizeof(data_type) * BATCH_SIZE == sizeof(ulong1)) {
        *reinterpret_cast<ulong1 *>(dst) = *reinterpret_cast<const ulong1 *>(src);
    } else if constexpr (sizeof(data_type) * BATCH_SIZE == sizeof(uint1)) {
        *reinterpret_cast<uint1 *>(dst) = *reinterpret_cast<const uint1 *>(src);
    } else {
#pragma unroll
        for (int b = 0; b < BATCH_SIZE; ++b) {
            dst[b] = src[b];
        }
    }
}

template <typename data_type, int BATCH_SIZE> __device__ inline void zero_batch(data_type *dst) {
    if constexpr ((sizeof(data_type) * BATCH_SIZE) == sizeof(ulong4)) {
        *reinterpret_cast<ulong4 *>(dst) = ulong4{};
    } else if constexpr (sizeof(data_type) * BATCH_SIZE == sizeof(ulong2)) {
        *reinterpret_cast<ulong2 *>(dst) = ulong2{};
    } else if constexpr (sizeof(data_type) * BATCH_SIZE == sizeof(ulong1)) {
        *reinterpret_cast<ulong1 *>(dst) = ulong1{};
    } else if constexpr (sizeof(data_type) * BATCH_SIZE == sizeof(uint1)) {
        *reinterpret_cast<uint *>(dst) = uint{};
    } else {
#pragma unroll
        for (int b = 0; b < BATCH_SIZE; ++b) {
            dst[b] = 0;
        }
    }
}

template <typename data_type, int DST_Y, int DST_X, int SRC_Y, int SRC_X, int BLOCK_DIM, int BATCH_SIZE>
__device__ inline void load_input(data_type dst[DST_Y][DST_X], const data_type src[SRC_Y][SRC_X], const int begin_x,
                                  const int begin_y) {
    static_assert(BATCH_SIZE > 0);
    /**
      Consider (SRC_X % DST_X == 0) && (SRC_Y % DST_Y == 0)
      Step 1:
        [   ][***][   ][   ]
        [   ][   ][   ][   ]
        [   ][   ][   ][   ]
        [   ][   ][   ][   ]
      Step 2:
        [   ][   ][   ][   ]
        [   ][***][   ][   ]
        [   ][   ][   ][   ]
        [   ][   ][   ][   ]
    */
    static_assert((SRC_X % BATCH_SIZE == 0) && (SRC_Y % BATCH_SIZE == 0));
    static_assert((DST_X % BATCH_SIZE == 0) && (DST_Y % BATCH_SIZE == 0));
    static_assert(BATCH_SIZE <= DST_X && DST_X % BATCH_SIZE == 0);
    const int begin_idx = threadIdx.x * BATCH_SIZE;
    const constexpr int total_elements = DST_X * DST_Y;
    const constexpr int elements_per_step = BLOCK_DIM * BATCH_SIZE;
// FIXME: loop unrolling
#pragma unroll
    for (int k = begin_idx; k < total_elements; k += elements_per_step) {
        int l_kx = k % DST_X;
        int l_ky = k / DST_X;
        int g_kx = l_kx + begin_x;
        int g_ky = l_ky + begin_y;
        auto *dst_flatten = &dst[l_ky][l_kx];
        // const auto *src_flatten = &src[g_ky][g_kx];
        // read_batch<data_type, BATCH_SIZE>(dst_flatten, src_flatten);
        if (((SRC_X % DST_X == 0) || (g_kx < SRC_X)) && ((SRC_Y % DST_Y == 0) || (g_ky < SRC_Y))) {
            const auto *src_flatten = &src[g_ky][g_kx];
            read_batch<data_type, BATCH_SIZE>(dst_flatten, src_flatten);
        } else {
            zero_batch<data_type, BATCH_SIZE>(dst_flatten);
        }
    }
}

template <int PM, int PN, int QM, int QN, int QK, int QUANT_SIZE, int BLOCK_SIZE, int BATCH_SIZE>
__device__ void load_scale(float s_s[PM][PN], const float sa[QK][QM], const float sb[QK][QN], const int m, const int n,
                           const int k) {
    constexpr int total_elements = PM * PN;
    constexpr int elements_per_step = BLOCK_SIZE * BATCH_SIZE;
    // static_assert(PN % BATCH_SIZE)

    const int begin_idx = threadIdx.x * BATCH_SIZE;
#pragma unroll
    for (int idx = begin_idx; idx < total_elements; idx += elements_per_step) {
        static_assert(BATCH_SIZE == 1);
        int i = idx / PN;
        int j = idx % PN;
        if (((QM % PM == 0) || (m + i < QM)) && ((QN % PN == 0) || ((n + j) / QUANT_SIZE < QN))) {
            s_s[i][j] = sa[k / QUANT_SIZE][(m + i)] * sb[k / QUANT_SIZE][(n) / QUANT_SIZE + j];
        } else {
            s_s[i][j] = 1.0f;
        }
    }
}

// don't use __builtin_readcyclecounter(), which would insert waitcnt
__device__ auto getclock() {
    uint64_t clk;
    asm volatile("s_memtime %0" : "=r"(clk));
    return clk;
}

template <typename Elem> __global__ void check_trans(const Elem *origin, const Elem *tranposed, int m, int n) {
    auto x = threadIdx.x + blockIdx.x * blockDim.x;
    auto y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x < m && y < n) {
        if (origin[x * n + y] != tranposed[y * m + x]) {
            printf("Error: %d %d\n", x, y);
        }
    }
}

template <typename in_data_type, typename acc_data_type, typename FragC, typename FragA, typename FragB, int PM, int PN,
          int BM, int BN, int BK, int FRAG_M, int FRAG_N, int FRAG_K, int WMMA_M, int WMMA_N, int WMMA_K, int WARP_M,
          int WARP_N, int BLOCK_SIZE, int BATCH_SIZE, int QUANT_SIZE>
__device__ void wmma_compute(const in_data_type s_a[BM][BK + 8], const in_data_type s_b[BN][BK + 8],
                             const float s_s[PN][PM], FragC frag_r[FRAG_M][FRAG_N], const int comp_c_frag_m,
                             const int comp_c_frag_n) {
    FragC frag_c[FRAG_M][FRAG_N];

#pragma unroll
    for (int i = 0; i < FRAG_M; i++) {
#pragma unroll
        for (int j = 0; j < FRAG_N; j++) {
            wmma::fill_fragment(frag_c[i][j], 0.0F);
        }
    }

#pragma unroll
    for (int k = 0; k < FRAG_K; ++k) {
#pragma unroll
        for (int i = 0; i < FRAG_M; i++) {
            FragA frag_a;
            int s_a_row = k * WMMA_K;
            int s_a_col = (comp_c_frag_m * FRAG_M + i) * WMMA_M;
            wmma::load_matrix_sync(frag_a, &s_a[s_a_col][s_a_row], BK + 8);
#pragma unroll
            for (int j = 0; j < FRAG_N; j++) {
                FragB frag_b;
                int s_b_row = k * WMMA_K;
                int s_b_col = (comp_c_frag_n * FRAG_N + j) * WMMA_N;
                wmma::load_matrix_sync(frag_b, &s_b[s_b_col][s_b_row], BK + 8);

                wmma::mma_sync(frag_c[i][j], frag_a, frag_b, frag_c[i][j]);
            }
        }
    }
#pragma unroll
    for (int i = 0; i < FRAG_M; i++) {
#pragma unroll
        for (int j = 0; j < FRAG_N; j++) {
#pragma unroll
            for (int k = 0; k < FragC::num_elements; ++k) {
#ifdef TEST_ON_RDNA4 // RDNA4, WAVE_SIZE = 32
                int m = ((threadIdx.x & 16) >> 1) | (k & 7) | (comp_c_frag_m * FRAG_M + i) * WMMA_M;
#else // CDNA3, WAVE_SIZE = 64
// int m = ((threadIdx.x & 48) >> 2) | (k & 3) | (comp_c_frag_m * FRAG_M + i) * WMMA_M;
#endif
                // int n = ((threadIdx.x & 15) | (comp_c_frag_n * FRAG_N + j) * WMMA_N) / QUANT_SIZE;
                auto lane = threadIdx.x % 64;
                int m, n;
                if constexpr (WMMA_M == 32) {
                    // C or D i: (8 * floor(GPR_num / 4) % 32) + 4 * floor(lane / 32) + (GPR_num % 4)
                    // C or D j: (lane % 32)
                    m = (8 * (k / 4) % 32) + 4 * (lane / 32) + (k % 4);
                    n = lane % 32;
                } else {
                    // C or D i: 4 * floor(lane / 16) + (GPR_num % 4)
                    // C or D j: (lane % 16)
                    m = 4 * (lane / 16) + (k % 4);
                    n = lane % 16;
                }
                m += (comp_c_frag_m * FRAG_M + i) * WMMA_M;
                n += (comp_c_frag_n * FRAG_N + j) * WMMA_N;
                n = n / QUANT_SIZE;
                // if(threadIdx.x == 192 && blockIdx.x ==0 && blockIdx.y == 0 && blockIdx.z == 0)
                // printf("m: %d, n: %d\n", m, n);
                float scale = s_s[n][m];
                frag_r[i][j].x[k] += (acc_data_type)scale * (acc_data_type)frag_c[i][j].x[k];
            }
        }
    }
}

__device__ rocwmma::bfloat16_t fast_f32tob16(float f) {
    union {
        float fp32;
        unsigned int u32;
    } u = {f};
    u.u32 += 0x7fff + ((u.u32 >> 16) & 1);
    auto ret = u.u32 >> 16;
    return reinterpret_cast<rocwmma::bfloat16_t &>(ret);
}

template <typename acc_data_type, typename out_data_type, typename FragC, typename FragOut, int WMMA_M, int WMMA_N,
          int BM, int BN, int M, int N, int FRAG_M, int FRAG_N>
__device__ inline void store_result(out_data_type c[M][N], FragC frag_r[FRAG_M][FRAG_N], const int m, const int n,
                                    const int comp_c_frag_m, const int comp_c_frag_n) {
#pragma unroll
    for (int i = 0; i < FRAG_M; i++) {
#pragma unroll
        for (int j = 0; j < FRAG_N; j++) {
            int frag_m = comp_c_frag_m * FRAG_M + i;
            int frag_n = comp_c_frag_n * FRAG_N + j;
            int row = m + frag_m * WMMA_M;
            int col = n + frag_n * WMMA_N;
            if (((M % BM == 0) || (row < M)) && ((N % BN == 0) || (col < N))) {
                out_data_type *c_ptr = &c[row][col];
                if constexpr (sizeof(acc_data_type) == sizeof(out_data_type)) { // split_k
                    auto lane = threadIdx.x % 64;
                    #pragma unroll
                    for (int k = 0; k < FragC::num_elements; ++k) {
                        int m, n;
                        if constexpr (WMMA_M == 32) {
                            // C or D i: (8 * floor(GPR_num / 4) % 32) + 4 * floor(lane / 32) + (GPR_num % 4)
                            // C or D j: (lane % 32)
                            m = (8 * (k / 4) % 32) + 4 * (lane / 32) + (k % 4);
                            n = lane % 32;
                        } else {
                            // C or D i: 4 * floor(lane / 16) + (GPR_num % 4)
                            // C or D j: (lane % 16)
                            m = 4 * (lane / 16) + (k % 4);
                            n = lane % 16;
                        }
                        c_ptr[m * N + n] = frag_r[i][j].x[k];;
                    }
                    
                    // wmma::store_matrix_sync(reinterpret_cast<out_data_type *>(c_ptr), frag_r[i][j], N,
                    //                         wmma::mem_row_major);
                } else if constexpr (sizeof(out_data_type) == sizeof(half)) {
                    FragOut frag_out;
                    static_assert(sizeof(half) == sizeof(out_data_type));
                    static_assert(FragOut::num_elements == FragC::num_elements);
                    for (int k = 0; k < FragOut::num_elements; ++k) {
                        auto reg = fast_f32tob16(frag_r[i][j].x[k]);
                        frag_out.x[k] = *reinterpret_cast<half *>(&reg);
                    }
                    wmma::store_matrix_sync(reinterpret_cast<half *>(c_ptr), frag_out, N, wmma::mem_row_major);
                } else {
                    static_assert(0, "Unsupported data type for output");
                }
            }
        }
    }
}

// a dummy template to allow inlcuding this file
template <int Splitk> __global__ void reduce(uint32_t m, uint32_t n, const float *c_splitk, __hip_bfloat16 *c) {
    auto tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= m * n) {
        return;
    }
    float4 sum{};
#pragma unroll
    for (auto i = 0; i < Splitk; ++i) {
        sum += *(float4 *)&c_splitk[i * (m * n) + tid * 4];
    }
    auto res =
        rocwmma::make_vector(fast_f32tob16(sum.x), fast_f32tob16(sum.y), fast_f32tob16(sum.z), fast_f32tob16(sum.w));
    *(decltype(res) *)&c[tid * 4] = res;
}

template<int M, int N, int SPLITK_FACTOR, int BLOCK_SIZE>
__launch_bounds__(BLOCK_SIZE)
__global__ void reduce_kernel(const float c_splitk[SPLITK_FACTOR][M][N], __hip_bfloat16 c[M][N]) {
    auto tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= M * N) {
        return;
    }
    float4 sum{};
    #pragma unroll
    for (auto i = 0; i < SPLITK_FACTOR; ++i) {
        sum += *(float4 *)&reinterpret_cast<const float*>(c_splitk)[i * (M * N) + tid * 4];
    }
    auto res =
        rocwmma::make_vector(fast_f32tob16(sum.x), fast_f32tob16(sum.y), fast_f32tob16(sum.z), fast_f32tob16(sum.w));
    *(decltype(res) *)&reinterpret_cast< __BF16_TYPE*>(c)[tid * 4] = res;
}

#ifdef PARAMETERIZE_LIBRARY
template <typename in_data_type,
          typename acc_data_type, // Accumulator type (e.g., float)
          typename out_data_type, // Output type (e.g., __hip_bfloat16)
          int M, int N, int K,    // Matrix dimensions
          int BM, int BN, int BK, // Tile dimensions
          int QUANT_SIZE,         // Quantization block size
          int BLOCK_SIZE,         // Block size
          int WARP_M, int WARP_N, // Warp dimensions
          int LDA, int LDB,
          int LOAD_BATCH_SIZE>    // Load batch size for vectorized memory operations
#else
using in_data_type = __FP8_TYPE;
using out_data_type = __BF16_TYPE;
using acc_data_type = float;
// constexpr int M = 4096, N = 4096, K = 4096;
constexpr int M = 6144, N = 4608, K = 7168;
constexpr int LDA = K, LDB = K;
// constexpr int M = 512, N = 512, K = 512;
constexpr int BM = 256, BN = 128, BK = 128;
constexpr int QUANT_SIZE = 128, BLOCK_SIZE = 512;
constexpr int LOAD_BATCH_SIZE = 16;
#ifdef TEST_ON_RDNA4 // RDNA4, WAVE_SIZE = 32
constexpr int WARP_M = 4, WARP_N = 2;
#else                // CDNA3, WAVE_SIZE = 64
constexpr int WARP_M = 4, WARP_N = 2;
#endif
#endif // End of parameterization
__global__ __launch_bounds__(BLOCK_SIZE) void gemm_kernel(
    const in_data_type a[M][LDA], const in_data_type b[N][LDB], out_data_type c[M][N],
    const float sa[ceil_div(K, QUANT_SIZE)][M / 1], // Assuming M is divisible by 1 (always true)
    const float sb[ceil_div(K, QUANT_SIZE)][ceil_div(N, QUANT_SIZE)]) {
    // --- Start: Derived parameters and constants ---
    constexpr int WMMA_M = 16; // Fixed WMMA dimension M
    constexpr int WMMA_N = 16; // Fixed WMMA dimension N
    constexpr int WMMA_K = 32; // Fixed WMMA dimension K (for FP8)

    // WARP_M/N define the 2D arrangement of warps in the block grid.
    // These might need adjustment based on BLOCK_DIM_X/Y strategy.
    // Using fixed values based on the non-parameterized version for now.
    // TODO: Derive WARP_M/N from BLOCK_DIM_X/Y if a flexible strategy is needed.
    constexpr int WARP_NUM = WARP_M * WARP_N; // Total warps per block

    // Assertion: Check if the assumed warp layout matches the block size
    static_assert(WARP_NUM * WAVE_SIZE == BLOCK_SIZE, "WARP_M * WARP_N * WAVE_SIZE must equal BLOCK_SIZE");

    // Fragments per warp
    constexpr int FRAG_M_PER_WARP = BM / WMMA_M / WARP_M;
    constexpr int FRAG_N_PER_WARP = BN / WMMA_N / WARP_N;
    constexpr int FRAG_K = BK / WMMA_K; // Fragments along K dimension tile

    static_assert(BM % (WMMA_M * WARP_M) == 0, "BM must be divisible by WMMA_M * WARP_M");
    static_assert(BN % (WMMA_N * WARP_N) == 0, "BN must be divisible by WMMA_N * WARP_N");
    static_assert(BK % WMMA_K == 0, "BK must be divisible by WMMA_K");
    static_assert(BK >= 32, "BK must be at least 32");
    // --- End: Derived parameters and constants ---

    constexpr int QM = M;                        // Dimension M for scale A
    constexpr int QN = ceil_div(N, QUANT_SIZE);  // Dimension N for scale B (quantized)
    constexpr int QK = ceil_div(K, QUANT_SIZE);  // Dimension K for scales (quantized)
    constexpr int PM = BM;                       // Block size M for scale A * B
    constexpr int PN = ceil_div(BN, QUANT_SIZE); // Block size N for scale A * B

    // Ensure derived fragment counts are positive
    static_assert(FRAG_M_PER_WARP > 0, "FRAG_M_PER_WARP must be positive");
    static_assert(FRAG_N_PER_WARP > 0, "FRAG_N_PER_WARP must be positive");
    static_assert(FRAG_K > 0, "FRAG_K must be positive");

    using FragA = wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, in_data_type, wmma::row_major>;
    using FragB = wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, in_data_type, wmma::col_major>;
    using FragC = wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, acc_data_type>;
    using FragOut = wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K,
                                   half>; // Output uses half for storage via bfloat16 reinterpret

    __shared__ in_data_type s_a[BM][BK + 8];
    __shared__ in_data_type s_b[BN][BK + 8];
    __shared__ acc_data_type s_s[PN][PM];           // Accumulator type for scales
    FragC frag_r[FRAG_M_PER_WARP][FRAG_N_PER_WARP]; // Accumulator fragments

    // handle splitk
    a = (decltype(a))((in_data_type *)a + blockIdx.z * K);
    b = (decltype(b))((in_data_type *)b + blockIdx.z * K);
    c += blockIdx.z * M;
    sa += blockIdx.z * QK;
    sb += blockIdx.z * QK;

    int tid = threadIdx.x;     // Linear thread ID within the block
    int wid = tid / WAVE_SIZE; // Warp ID within the block

    // Spilt and compute fragments
    constexpr int iteration_over_k = ceil_div(K, BK); // Use ceil_div for potentially non-divisible K
    static_assert(LOAD_BATCH_SIZE > 0, "LOAD_BATCH_SIZE must be positive");

    constexpr auto PIPELINE = true;
    // using LoadVec = rocwmma::VecT<float, LOAD_BATCH_SIZE / sizeof(float)>;
    using LoadVec = __attribute__((__vector_size__(LOAD_BATCH_SIZE))) float;
    static_assert(((BK * BM) % (BLOCK_SIZE * LOAD_BATCH_SIZE)) == 0,
                  "BK * BM must be divisible by BLOCK_SIZE * LOAD_BATCH_SIZE");
    static_assert(BK % LOAD_BATCH_SIZE == 0, "BK must be divisible by LOAD_BATCH_SIZE");
    LoadVec reg_a[BK * BM / BLOCK_SIZE / LOAD_BATCH_SIZE];
    LoadVec reg_b[BK * BN / BLOCK_SIZE / LOAD_BATCH_SIZE];
    constexpr auto PK = ceil_div(BK, QUANT_SIZE);
    static_assert(PK == 1, "PK must be 1 for now");
    float reg_sa[ceil_div(PM, BLOCK_SIZE)];
    float reg_sb[ceil_div(PN, BLOCK_SIZE)];

    // threadblock swizzle
    auto log_tile = 1;
    auto block_idx_x = blockIdx.x >> log_tile;
    auto block_idx_y = (blockIdx.y << log_tile) + ((blockIdx.x) & ((1 << (log_tile)) - 1));
    if (block_idx_x >= ceil_div(N, BN) || block_idx_y >= ceil_div(M, BM)) {
        return;
    }

    const int m = block_idx_y * BM;
    const int n = block_idx_x * BN;
    int k = 0;

    auto global2reg = [&]() {
#pragma unroll
        for (int reg = 0; reg < sizeof(reg_sa) / sizeof(float); reg++) {
            // NOTE: must iter over reg to make compiler unroll the loop
            // and thus be able to allocate reg_a on register instead of on scratch memroy
            int t = tid + reg * BLOCK_SIZE;
            // NOTE: don't branch here
            // if (t > PM) {
            //     break;
            // }
            int i = t / PM;
            int j = t % PM;
            reg_sa[reg] = sa[k / QUANT_SIZE][(m + j)];
        }
#pragma unroll
        for (int reg = 0; reg < sizeof(reg_sb) / sizeof(float); reg++) {
            // NOTE: must iter over reg to make compiler unroll the loop
            // and thus be able to allocate reg_a on register instead of on scratch memroy
            int t = tid + reg * BLOCK_SIZE;
            // NOTE: don't branch here
            // if (t > PN) {
            //     break;
            // }
            int i = t / PN;
            int j = t % PN;
            reg_sb[reg] = sb[k / QUANT_SIZE][(n) / QUANT_SIZE + j];
        }
#pragma unroll
        for (int reg = 0; reg < sizeof(reg_a) / sizeof(LoadVec); reg++) {
            // NOTE: must iter over reg to make compiler unroll the loop
            // and thus be able to allocate reg_a on register instead of on scratch memroy
            int t = tid * LOAD_BATCH_SIZE + reg * BLOCK_SIZE * LOAD_BATCH_SIZE;
            int i = t / BK;
            int j = t % BK;
            reg_a[reg] = *(LoadVec *)&a[m + i][k + j];
        }
#pragma unroll
        for (int reg = 0; reg < sizeof(reg_b) / sizeof(LoadVec); reg++) {
            // NOTE: must iter over reg to make compiler unroll the loop
            // and thus be able to allocate reg_a on register instead of on scratch memroy
            int t = tid * LOAD_BATCH_SIZE + reg * BLOCK_SIZE * LOAD_BATCH_SIZE;
            int i = t / BK;
            int j = t % BK;
            reg_b[reg] = *(LoadVec *)&b[n + i][k + j];
        }
    };

    auto reg2lds = [&]() {
#pragma unroll
        for (int rega = 0; rega < sizeof(reg_sa) / sizeof(float); rega++) {
            int ta = tid + rega * BLOCK_SIZE;
            int j = ta % PM;
#pragma unroll
            for (int regb = 0; regb < sizeof(reg_sb) / sizeof(float); regb++) {
                int tb = tid + regb * BLOCK_SIZE;
                int i = tb % PN;
                s_s[i][j] = reg_sa[rega] * reg_sb[regb];
            }
        }
#pragma unroll
        for (int reg = 0; reg < sizeof(reg_a) / sizeof(LoadVec); reg++) {
            int t = tid * LOAD_BATCH_SIZE + reg * BLOCK_SIZE * LOAD_BATCH_SIZE;
            int i = t / BK;
            int j = t % BK;
            *(LoadVec *)&s_a[i][j] = reg_a[reg];
        }
#pragma unroll
        for (int reg = 0; reg < sizeof(reg_b) / sizeof(LoadVec); reg++) {
            int t = tid * LOAD_BATCH_SIZE + reg * BLOCK_SIZE * LOAD_BATCH_SIZE;
            int i = t / BK;
            int j = t % BK;
            *(LoadVec *)&s_b[i][j] = reg_b[reg];
        }
    };

    if constexpr (PIPELINE) {
        global2reg();
    }

// Initialize the output accumulator fragments to zero
#pragma unroll
    for (int i = 0; i < FRAG_M_PER_WARP; i++) {
#pragma unroll
        for (int j = 0; j < FRAG_N_PER_WARP; j++) {
            wmma::fill_fragment(frag_r[i][j], 0.0f); // Use float literal
        }
    }

    if constexpr (!PIPELINE) {
        global2reg();
    }

    reg2lds();

    for (int bk = 1; bk < iteration_over_k; bk++) {
        k = bk * BK;

        // Calculate remaining K for boundary checks if needed (not currently used by load_input)
        // const int k_rem = K - k;

        // Load data into shared memory
        // load_input<in_data_type, BK, BM, K, M, BLOCK_SIZE, 32>(
        //     s_a, a, m, k);
        // load_input<in_data_type, BK, BN, K, N, BLOCK_SIZE, 32>(
        //     s_b, b, n, k);
        // Load scales into shared memory (using acc_data_type for s_s)
        // load_scale<PM, PN, QM, QN, QK, QUANT_SIZE, BLOCK_SIZE, 1>(
        //     s_s, sa, sb, m, n, k);

        if constexpr (PIPELINE) {
            global2reg();
        }

        __syncthreads();

        // Perform matrix multiplication using WMMA
        wmma_compute<in_data_type, acc_data_type, FragC, FragA, FragB, PM, PN, BM, BN, BK, FRAG_M_PER_WARP,
                     FRAG_N_PER_WARP, FRAG_K, WMMA_M, WMMA_N, WMMA_K, WARP_M, WARP_N, BLOCK_SIZE, LOAD_BATCH_SIZE,
                     QUANT_SIZE>( // Pass calculated BLOCK_SIZE and LOAD_BATCH_SIZE
            s_a, s_b, s_s, frag_r, wid / WARP_N, wid % WARP_N);
        __syncthreads();

        if constexpr (!PIPELINE) {
            global2reg();
        }

        // __builtin_amdgcn_sched_barrier(0);

        reg2lds();
    }
    __syncthreads();
    wmma_compute<in_data_type, acc_data_type, FragC, FragA, FragB, PM, PN, BM, BN, BK, FRAG_M_PER_WARP, FRAG_N_PER_WARP,
                 FRAG_K, WMMA_M, WMMA_N, WMMA_K, WARP_M, WARP_N, BLOCK_SIZE, LOAD_BATCH_SIZE,
                 QUANT_SIZE>( // Pass calculated BLOCK_SIZE and LOAD_BATCH_SIZE
        s_a, s_b, s_s, frag_r, wid / WARP_N, wid % WARP_N);
    // Store results from accumulator fragments to global memory
    store_result<acc_data_type, out_data_type, FragC, FragOut, WMMA_M, WMMA_N, BM, BN, M, N, FRAG_M_PER_WARP,
                 FRAG_N_PER_WARP>(c, frag_r, block_idx_y * BM, block_idx_x * BN, wid / WARP_N, wid % WARP_N);
};

}; // namespace gemm_kernel

#ifndef PARAMETERIZE_LIBRARY
// Define type aliases to match those in the namespace
using fp8_type = gemm_kernel::in_data_type;       // __hip_fp8_e4m3
using fp16_type = gemm_kernel::out_data_type;     // __hip_bfloat16
using acc_data_type = gemm_kernel::acc_data_type; // float

// Define constants to match those in the namespace
constexpr int M = gemm_kernel::M;   // 4096
constexpr int N = gemm_kernel::N;   // 4096
constexpr int K = gemm_kernel::K;   // 4096
constexpr int BM = gemm_kernel::BM; // 256
constexpr int BN = gemm_kernel::BN; // 128
constexpr int BK = gemm_kernel::BK; // 32
constexpr int BLOCK_SIZE = gemm_kernel::BLOCK_SIZE;
constexpr int QUANT_SIZE = gemm_kernel::QUANT_SIZE; // 128

// Define derived constants for the test
constexpr int QK = K / QUANT_SIZE;
constexpr int QM = M;
constexpr int QN = N / QUANT_SIZE;

// 检查HIP错误的辅助函数
#define CHECK_HIP_ERROR(val) check((val), #val, __FILE__, __LINE__)
template <typename T> void check(T err, const char *const func, const char *const file, const int line) {
    if (err != hipSuccess) {
        fprintf(stderr, "HIP Runtime Error at: %s:%d\n", file, line);
        fprintf(stderr, "%s %s\n", hipGetErrorString(err), func);
        exit(1);
    }
}

// Define a macro to check HIP errors
#define HIP_CALL(call)                                                                                                 \
    do {                                                                                                               \
        hipError_t err = call;                                                                                         \
        if (err != hipSuccess) {                                                                                       \
            fprintf(stderr, "HIP Error: %s at %s:%d\n", hipGetErrorString(err), __FILE__, __LINE__);                   \
            exit(EXIT_FAILURE);                                                                                        \
        }                                                                                                              \
    } while (0)

// CPU矩阵乘法实现，用于验证结果
void cpu_gemm(const fp8_type a[K][M], const fp8_type b[K][N], fp16_type c[M][N], const float sa[QK][QM],
              const float sb[QK][QN]) {
    float(*rc)[N] = new float[M][N];
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            rc[m][n] = 0.0f;
        }
    }
    for (int k = 0; k < K; ++k) {
        for (int m = 0; m < M; ++m) {
            for (int n = 0; n < N; ++n) {
                float scale = sa[k / QUANT_SIZE][m] * sb[k / QUANT_SIZE][n / QUANT_SIZE];
                rc[m][n] += (scale * (float)a[k][m] * (float)b[k][n]);
            }
        }
    }
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            c[m][n] = (fp16_type)rc[m][n];
        }
    }
    delete[] rc;
}

int main() {
    // 分配主机内存
    fp8_type(*h_a)[M] = new fp8_type[K][M];
    fp8_type(*h_b)[N] = new fp8_type[K][N];
    fp16_type(*h_c)[N] = new fp16_type[M][N];
    fp16_type(*h_c_ref)[N] = new fp16_type[M][N];

    // 分配量化比例因子的主机内存
    float(*h_sa)[QM] = new float[QK][QM];
    float(*h_sb)[QN] = new float[QK][QN];

    // 初始化输入数据
    for (int i = 0; i < K; ++i) {
        for (int j = 0; j < M; ++j) {
            h_a[i][j] = (fp8_type)((rand() % 10000) / 10000.0f);
        }
    }
    for (int i = 0; i < K; ++i) {
        for (int j = 0; j < N; ++j) {
            h_b[i][j] = (fp8_type)((rand() % 10000) / 10000.0f);
        }
    }

    // 初始化量化比例因子
    for (int i = 0; i < QK; ++i) {
        for (int j = 0; j < QM; ++j) {
            h_sa[i][j] = 1.0f;
        }
    }
    for (int i = 0; i < QK; ++i) {
        for (int j = 0; j < QN; ++j) {
            h_sb[i][j] = 1.0f;
        }
    }

    // 分配设备内存
    fp8_type(*d_a)[K];
    fp8_type(*d_b)[K];
    fp16_type(*d_c)[N];
    float(*d_sa)[QM];
    float(*d_sb)[QN];

    CHECK_HIP_ERROR(hipMalloc(&d_a, K * M * sizeof(fp8_type)));
    CHECK_HIP_ERROR(hipMalloc(&d_b, K * N * sizeof(fp8_type)));
    CHECK_HIP_ERROR(hipMalloc(&d_c, M * N * sizeof(fp16_type)));
    CHECK_HIP_ERROR(hipMalloc(&d_sa, QK * QM * sizeof(float)));
    CHECK_HIP_ERROR(hipMalloc(&d_sb, QK * QN * sizeof(float)));

    // 将数据从主机内存复制到设备内存
    CHECK_HIP_ERROR(hipMemcpy(d_a, h_a, K * M * sizeof(fp8_type), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_b, h_b, K * N * sizeof(fp8_type), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_sa, h_sa, QK * QM * sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_sb, h_sb, QK * QN * sizeof(float), hipMemcpyHostToDevice));

    // 计算网格和块大小 - 确保覆盖整个矩阵
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    dim3 block(BLOCK_SIZE);

    // 确保块大小是32的倍数，因为warp大小是32
    if (BLOCK_SIZE % 32 != 0) {
        printf("Error: Block size must be a multiple of warp size (32)\n");
        return 1;
    }

    // 检查设备是否支持所需的计算能力
    int deviceId;
    HIP_CALL(hipGetDevice(&deviceId));
    hipDeviceProp_t deviceProp;
    HIP_CALL(hipGetDeviceProperties(&deviceProp, deviceId));

    if (deviceProp.major < 7) {
        printf("Error: This kernel requires a GPU with compute capability 7.0 or higher\n");
        return 1;
    }

    printf("Running GEMM kernel with grid(%d,%d), block(%d)...\n", grid.x, grid.y, block.x);

    // Query and print kernel and device information
    printf("Querying kernel and device information...\n");

    // Get device properties
    HIP_CALL(hipGetDeviceProperties(&deviceProp, deviceId));
    printf("Device Name: %s\n", deviceProp.name);
    printf("Total Global Memory: %lu bytes\n", deviceProp.totalGlobalMem);
    printf("Shared Memory per Block: %lu bytes\n", deviceProp.sharedMemPerBlock);
    printf("Registers per Block: %d\n", deviceProp.regsPerBlock);
    printf("Warp Size: %d\n", deviceProp.warpSize);
    printf("Max Threads per Block: %d\n", deviceProp.maxThreadsPerBlock);
    printf("Max Threads per Multiprocessor: %d\n", deviceProp.maxThreadsPerMultiProcessor);
    printf("Number of Multiprocessors: %d\n", deviceProp.multiProcessorCount);

    // Query kernel attributes
    hipFuncAttributes funcAttr;
    HIP_CALL(hipFuncGetAttributes(&funcAttr, (const void *)gemm_kernel::gemm_kernel));
    printf("Kernel Attributes:\n");
    printf("  Shared Memory Size: %lu bytes\n", funcAttr.sharedSizeBytes);
    printf("  Number of Registers: %d\n", funcAttr.numRegs);
    printf("  Max Threads per Block: %d\n", funcAttr.maxThreadsPerBlock);
    printf("  Local Memory Size: %lu bytes\n", funcAttr.localSizeBytes);

    // 在启动内核前清零 C 矩阵
    CHECK_HIP_ERROR(hipMemset(d_c, 0, M * N * sizeof(fp16_type)));

    // 进行两次预热运行
    printf("Performing warmup runs...\n");
    gemm_kernel::gemm_kernel<<<grid, block>>>(d_a, d_b, d_c, d_sa, d_sb);
    CHECK_HIP_ERROR(hipDeviceSynchronize());
    gemm_kernel::gemm_kernel<<<grid, block>>>(d_a, d_b, d_c, d_sa, d_sb);
    CHECK_HIP_ERROR(hipDeviceSynchronize());

    // 声明和创建计时事件
    hipEvent_t start, stop;
    HIP_CALL(hipEventCreate(&start));
    HIP_CALL(hipEventCreate(&stop));

    // 在执行正式计时之前确保设备同步
    CHECK_HIP_ERROR(hipDeviceSynchronize());
    HIP_CALL(hipEventRecord(start));

    // Launch kernel
    printf("Launching kernel...\n");
    gemm_kernel::gemm_kernel<<<grid, block>>>(d_a, d_b, d_c, d_sa, d_sb);

    // 记录结束时间并计算执行时间
    HIP_CALL(hipEventRecord(stop));

    // 记录结束时间并计算执行时间
    HIP_CALL(hipEventSynchronize(stop));
    float milliseconds = 0;
    HIP_CALL(hipEventElapsedTime(&milliseconds, start, stop));
    printf("Kernel execution time: %f ms\n", milliseconds);

    // 检查HIP错误
    CHECK_HIP_ERROR(hipGetLastError());

    // 计算GPU性能指标
    double operations = 2.0 * M * N * K; // 每次乘加操作计为2个浮点操作
    double seconds = milliseconds / 1000.0;
    double tflops = (operations / seconds) / 1e12;
    printf("GPU Performance: %.2f TFLOPS\n", tflops);

    return 0;

    // 将结果从设备内存复制回主机内存
    CHECK_HIP_ERROR(hipMemcpy(h_c, d_c, M * N * sizeof(fp16_type), hipMemcpyDeviceToHost));

    // 计算参考结果
    printf("Computing reference result on CPU...\n");
    cpu_gemm(h_a, h_b, h_c_ref, h_sa, h_sb);

    // Print the first 10 values for comparison
    printf("First 10 values (GPU vs CPU):\n");
    int print_count = 0;
    for (int i = 0; i < M && print_count < 10; ++i) {
        for (int j = 0; j < N && print_count < 10; ++j) {
            printf("  [%d, %d]: GPU=%f, CPU=%f\n", i, j, (float)h_c[i][j], (float)h_c_ref[i][j]);
            print_count++;
        }
    }

    // 验证结果
    printf("Verifying results...\n");
    int errors = 0;
    float max_abs_diff = 0.0f;
    float max_rel_diff = 0.0f;
    struct ErrorInfo {
        int row, col;
        float gpu_val, cpu_val, abs_diff, rel_diff;
    };
    ErrorInfo first_10_errors[10];
    ErrorInfo max_10_errors[10] = {};

    // Add a configurable variable for the number of errors to output
    int max_errors_to_output = 10; // You can modify this value as needed

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float gpu_val = (float)h_c[i][j];
            float cpu_val = (float)h_c_ref[i][j];
            float abs_diff;
            float rel_diff;

            if (std::isnan(gpu_val) || std::isnan(cpu_val)) {
                abs_diff = INFINITY;
                rel_diff = INFINITY;
            } else {
                abs_diff = abs(gpu_val - cpu_val);
                rel_diff = abs_diff / (abs(cpu_val) + FLT_EPSILON);
            }

            // Track max absolute and relative differences
            max_abs_diff = fmaxf(max_abs_diff, abs_diff);
            max_rel_diff = fmaxf(max_rel_diff, rel_diff);

            // Record first 10 errors
            if (errors < max_errors_to_output && (rel_diff > 1e-2 || abs_diff > 1e-3)) {
                first_10_errors[errors] = {i, j, gpu_val, cpu_val, abs_diff, rel_diff};
            }

            // Track top 10 largest errors
            if (rel_diff > 1e-2 || abs_diff > 1e-3) {
                errors++;
                for (int k = 0; k < max_errors_to_output; ++k) {
                    if (abs_diff > max_10_errors[k].abs_diff) {
                        for (int l = max_errors_to_output - 1; l > k; --l) {
                            max_10_errors[l] = max_10_errors[l - 1];
                        }
                        max_10_errors[k] = {i, j, gpu_val, cpu_val, abs_diff, rel_diff};
                        break;
                    }
                }
            }
        }
    }

    // Print first 10 errors
    printf("First %d errors:\n", max_errors_to_output);
    for (int i = 0; i < fmin(errors, max_errors_to_output); ++i) {
        printf("Error at [%d, %d]: GPU=%f, CPU=%f, AbsDiff=%f, RelDiff=%f\n", first_10_errors[i].row,
               first_10_errors[i].col, first_10_errors[i].gpu_val, first_10_errors[i].cpu_val,
               first_10_errors[i].abs_diff, first_10_errors[i].rel_diff);
    }

    // Print top 10 largest errors
    printf("Top %d largest errors:\n", max_errors_to_output);
    for (int i = 0; i < max_errors_to_output && max_10_errors[i].abs_diff > 0; ++i) {
        printf("Error at [%d, %d]: GPU=%f, CPU=%f, AbsDiff=%f, RelDiff=%f\n", max_10_errors[i].row,
               max_10_errors[i].col, max_10_errors[i].gpu_val, max_10_errors[i].cpu_val, max_10_errors[i].abs_diff,
               max_10_errors[i].rel_diff);
    }

    printf("Max abs_diff: %f, Max rel_diff: %f\n", max_abs_diff, max_rel_diff);
    if (errors == 0) {
        printf("Test PASSED!\n");
    } else {
        printf("Test FAILED with %d errors\n", errors);
    }

    // 计算性能
    double flops = 2.0 * M * N * K;
    double gflops = (flops * 1e-9) / (milliseconds * 1e-3);
    printf("Performance: %.2f GFLOPS\n", gflops);

    // 释放内存
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
    delete[] h_c_ref;
    delete[] h_sa;
    delete[] h_sb;
    HIP_CALL(hipFree(d_a));
    HIP_CALL(hipFree(d_b));
    HIP_CALL(hipFree(d_c));
    HIP_CALL(hipFree(d_sa));
    HIP_CALL(hipFree(d_sb));
    HIP_CALL(hipEventDestroy(start));
    HIP_CALL(hipEventDestroy(stop));

    return 0;
}
#endif
#pragma clang diagnostic pop
#endif

// From transpose_kernel.cpp
#pragma once

namespace transpose_kernel {

template <typename Elem, int M, int N, int TILE_DIM, int BLOCK_SIZE, int VEC_SIZE>
__launch_bounds__(BLOCK_SIZE)
__global__ void transpose_kernel(Elem *odata, const Elem *idata) {
    constexpr auto TBLOCK_X = TILE_DIM / VEC_SIZE;
    constexpr auto TBLOCK_Y = BLOCK_SIZE / TBLOCK_X;

    // avoid read bank conflict
    // VEC_SIZE * (TILE_DIM + d) * sizeof(Elem) = TBLOCK_Y / (BLOCK_SIZE / WARP_SIZE) * sizeof(Elem) + 128k
    // each warp read row = TILE_DIM (in VEC_SIZE reads), col = TBLOCK_Y / (BLOCK_SIZE / WARP_SIZE)
    // warp 0                     warp 1
    // t0    t16    t32    t48    ...
    // ...
    // t1
    // ...
    // t15
    // don't know why padding to d as described above is not working, maybe gpu could merge contigious ds_read_u8 and
    // cause padding to be TBLOCK_Y / (BLOCK_SIZE / WARP_SIZE)
    constexpr auto PADDING = TBLOCK_Y / (BLOCK_SIZE / warpSize);
    __shared__ Elem tile[TILE_DIM][TILE_DIM + PADDING];

    int x = blockIdx.x * TILE_DIM + threadIdx.x * VEC_SIZE;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

// Load tile
#pragma unroll
    for (int i = 0; i < TILE_DIM; i += TBLOCK_Y) {
#pragma unroll
        for (int v = 0; v < VEC_SIZE; v++) {
            tile[threadIdx.y + i][threadIdx.x * VEC_SIZE + v] = idata[(y + i) * N + x + v];
        }
    }

    __syncthreads();

    // Transpose indices
    x = blockIdx.y * TILE_DIM + threadIdx.x * VEC_SIZE;
    y = blockIdx.x * TILE_DIM + threadIdx.y;

// Write tile
#pragma unroll
    for (int i = 0; i < TILE_DIM; i += TBLOCK_Y) {
#pragma unroll
        for (int v = 0; v < VEC_SIZE; v++) {
            odata[(y + i) * M + x + v] = tile[threadIdx.x * VEC_SIZE + v][threadIdx.y + i];
        }
    }
}

template <typename Elem, int M, int N, int TILE_DIM, int BLOCK_SIZE, int VEC_SIZE>
void launch_transpose(Elem *out, const Elem *in, hipStream_t stream = 0) {
    static_assert(TILE_DIM % VEC_SIZE == 0);
    constexpr auto TBLOCK_X = TILE_DIM / VEC_SIZE;
    static_assert(BLOCK_SIZE % TBLOCK_X == 0);
    constexpr auto TBLOCK_Y = BLOCK_SIZE / TBLOCK_X;
    static_assert(M % TILE_DIM == 0 && N % TILE_DIM == 0);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(transpose_kernel<Elem, M, N, TILE_DIM, BLOCK_SIZE, VEC_SIZE>),
        dim3(N / TILE_DIM, M / TILE_DIM), dim3(TBLOCK_X, TBLOCK_Y), 0, stream,
        out, in);
}

#define DISPATCH_TRANSPOSE(DIM_0, DIM_1, TILE_DIM, BLOCK_SIZE, VEC_SIZE) else if constexpr(IN_DIM_0 == DIM_0 && IN_DIM_1 == DIM_1) launch_transpose<__FP8_TYPE, IN_DIM_0, IN_DIM_1, TILE_DIM, BLOCK_SIZE, VEC_SIZE>(out, in, stream)

template <int DIM0, int DIM1>
struct unsupported_config {
    static_assert(DIM0 == -1, "Unsupported transpose configuration - check template parameters");
};

template <int IN_DIM_0, int IN_DIM_1>
void transpose_fp8(__FP8_TYPE *out, const __FP8_TYPE *in, hipStream_t stream = 0) {
    if constexpr (false /* dummy*/ ) static_assert(true);
    DISPATCH_TRANSPOSE(   256,   1024,     64,    256, 4); // Optimized: 2.71 µs (193.46 GB/s)
    DISPATCH_TRANSPOSE(   256,   6144,     64,    256, 4); // Optimized: 2.72 µs (1157.37 GB/s)
    DISPATCH_TRANSPOSE(   256,   7168,     64,    256, 8); // Optimized: 2.99 µs (1225.38 GB/s)
    DISPATCH_TRANSPOSE(   512,   1024,     64,    512, 4); // Optimized: 2.55 µs (411.21 GB/s)
    DISPATCH_TRANSPOSE(   512,   4096,     64,    256, 4); // Optimized: 3.01 µs (1394.85 GB/s)
    DISPATCH_TRANSPOSE(   512,   6144,     64,    512, 4); // Optimized: 3.58 µs (1755.43 GB/s)
    DISPATCH_TRANSPOSE(  1536,   1024,     64,   1024, 4); // Optimized: 2.78 µs (1130.74 GB/s)
    DISPATCH_TRANSPOSE(  1536,   3072,     64,    512, 4); // Optimized: 3.57 µs (2641.99 GB/s)
    DISPATCH_TRANSPOSE(  1536,   6144,    128,   1024, 8); // Optimized: 7.09 µs (2661.36 GB/s)
    DISPATCH_TRANSPOSE(  2048,   1024,     64,   1024, 4); // Optimized: 2.84 µs (1477.91 GB/s)
    DISPATCH_TRANSPOSE(  2048,   6144,    128,    512, 8); // Optimized: 8.94 µs (2816.23 GB/s)
    DISPATCH_TRANSPOSE(  2048,   7168,    128,    512, 8); // Optimized: 9.56 µs (3070.50 GB/s)
    DISPATCH_TRANSPOSE(  2304,   1024,     64,   1024, 4); // Optimized: 3.08 µs (1532.51 GB/s)
    DISPATCH_TRANSPOSE(  2304,   6144,    128,    512, 8); // Optimized: 9.30 µs (3043.93 GB/s)
    DISPATCH_TRANSPOSE(  2304,   7168,    128,    512, 8); // Optimized: 10.39 µs (3179.95 GB/s)
    DISPATCH_TRANSPOSE(  7168,    512,     64,    512, 4); // Optimized: 3.25 µs (2257.78 GB/s)
    DISPATCH_TRANSPOSE(  7168,    576,     64,    512, 4); // Optimized: 3.44 µs (2403.24 GB/s)
    DISPATCH_TRANSPOSE(  7168,   1024,     64,    256, 4); // Optimized: 5.07 µs (2892.62 GB/s)
    DISPATCH_TRANSPOSE(  7168,   1536,    128,   1024, 8); // Optimized: 7.72 µs (2851.97 GB/s)
    DISPATCH_TRANSPOSE(  7168,   4608,    128,    512, 8); // Optimized: 16.87 µs (3915.84 GB/s)
    DISPATCH_TRANSPOSE(  7168,   6144,    128,    256, 8); // Optimized: 21.59 µs (4079.12 GB/s)

    
    // else launch_transpose<__FP8_TYPE, IN_DIM_0, IN_DIM_1, 64, 512, 4>(out, in, stream);
    else static_assert(false);

}

} // namespace transpose_kernel

#ifndef PARAMETERIZE_LIBRARY
int main() {}
#endif

// From gemm_kernel_legacy.cpp
#pragma once

namespace gemm_kernel_legacy {

template <typename data_type, int BATCH_SIZE> 
__device__ inline void read_batch(data_type *dst, const data_type *src) {
    if constexpr ((sizeof(data_type) * BATCH_SIZE) == 2 * sizeof(ulong4)) {
        *(reinterpret_cast<ulong4 *>(dst) + 0) = *(reinterpret_cast<const ulong4 *>(src) + 0);
        *(reinterpret_cast<ulong4 *>(dst) + 1) = *(reinterpret_cast<const ulong4 *>(src) + 1);
    } else if constexpr ((sizeof(data_type) * BATCH_SIZE) == sizeof(ulong4)) {
        *reinterpret_cast<ulong4 *>(dst) = *reinterpret_cast<const ulong4 *>(src);
    } else if constexpr (sizeof(data_type) * BATCH_SIZE == sizeof(ulong2)) {
        *reinterpret_cast<ulong2 *>(dst) = *reinterpret_cast<const ulong2 *>(src);
    } else if constexpr (sizeof(data_type) * BATCH_SIZE == sizeof(ulong1)) {
        *reinterpret_cast<ulong1 *>(dst) = *reinterpret_cast<const ulong1 *>(src);
    } else if constexpr (sizeof(data_type) * BATCH_SIZE == sizeof(uint1)) {
        *reinterpret_cast<uint1 *>(dst) = *reinterpret_cast<const uint1 *>(src);
    } else {
        #pragma unroll
        for (int b = 0; b < BATCH_SIZE; ++b) {
            dst[b] = src[b];
        }
    }
}

template <typename data_type, int BATCH_SIZE> 
__device__ inline void zero_batch(data_type *dst) {
    if constexpr ((sizeof(data_type) * BATCH_SIZE) == sizeof(ulong4)) {
        *reinterpret_cast<ulong4 *>(dst) = ulong4{};
    } else if constexpr (sizeof(data_type) * BATCH_SIZE == sizeof(ulong2)) {
        *reinterpret_cast<ulong2 *>(dst) = ulong2{};
    } else if constexpr (sizeof(data_type) * BATCH_SIZE == sizeof(ulong1)) {
        *reinterpret_cast<ulong1 *>(dst) = ulong1{};
    } else if constexpr (sizeof(data_type) * BATCH_SIZE == sizeof(uint1)) {
        *reinterpret_cast<uint *>(dst) = uint{};
    } else {
        #pragma unroll
        for (int b = 0; b < BATCH_SIZE; ++b) {
            dst[b] = 0;
        }
    }
}

template <typename data_type, int DST_Y, int DST_X, int SRC_Y, int SRC_X, int BLOCK_DIM, int BATCH_SIZE>
__device__ inline void load_input(data_type dst[DST_Y][DST_X], const data_type src[SRC_Y][SRC_X],
                                         const int begin_x, const int begin_y) {
  static_assert(BATCH_SIZE > 0);
  /**
    Consider (SRC_X % DST_X == 0) && (SRC_Y % DST_Y == 0)
    Step 1:
      [   ][***][   ][   ]
      [   ][   ][   ][   ]
      [   ][   ][   ][   ]
      [   ][   ][   ][   ]
    Step 2:
      [   ][   ][   ][   ]
      [   ][***][   ][   ]
      [   ][   ][   ][   ]
      [   ][   ][   ][   ]
  */
  static_assert((SRC_X % BATCH_SIZE == 0) && (SRC_Y % BATCH_SIZE == 0));
  static_assert((DST_X % BATCH_SIZE == 0) && (DST_Y % BATCH_SIZE == 0));
  static_assert(BATCH_SIZE <= DST_X && DST_X % BATCH_SIZE == 0);
  const int begin_idx = threadIdx.x * BATCH_SIZE;
  const constexpr int total_elements = DST_X * DST_Y;
  const constexpr int elements_per_step = BLOCK_DIM * BATCH_SIZE;
  // FIXME: loop unrolling
  #pragma unroll
  for (int k = begin_idx; k < total_elements; k += elements_per_step) {
    int l_kx = k % DST_X;
    int l_ky = k / DST_X;
    int g_kx = l_kx + begin_x;
    int g_ky = l_ky + begin_y;
    auto *dst_flatten = &dst[l_ky][l_kx];
    // const auto *src_flatten = &src[g_ky][g_kx];
    // read_batch<data_type, BATCH_SIZE>(dst_flatten, src_flatten);
    if (((SRC_X % DST_X == 0) || (g_kx < SRC_X)) && ((SRC_Y % DST_Y == 0) || (g_ky < SRC_Y))) {
      const auto *src_flatten = &src[g_ky][g_kx];
      read_batch<data_type, BATCH_SIZE>(dst_flatten, src_flatten);
    } else {
      zero_batch<data_type, BATCH_SIZE>(dst_flatten);
    }
  }
}

template <int PM, int PN, int QM, int QN, int QK, int QUANT_SIZE, int BLOCK_SIZE, int BATCH_SIZE>
__device__ void load_scale(float s_s[PM][PN], const float sa[QK][QM], const float sb[QK][QN], 
                const int m, const int n, const int k) {
    constexpr int total_elements = PM * PN;
    constexpr int elements_per_step = BLOCK_SIZE * BATCH_SIZE;
    // static_assert(PN % BATCH_SIZE)
    
    const int begin_idx = threadIdx.x * BATCH_SIZE;
    #pragma unroll
    for (int idx = begin_idx; idx < total_elements; idx += elements_per_step) {
        static_assert(BATCH_SIZE == 1);
        int i = idx / PN;
        int j = idx % PN;
        if (((QM % PM == 0) || (m + i < QM)) && ((QN % PN == 0) || ((n + j) / QUANT_SIZE < QN))) {
            s_s[i][j] = sa[k / QUANT_SIZE][(m + i)] * sb[k / QUANT_SIZE][(n) / QUANT_SIZE + j];
        } else {
            s_s[i][j] = 1.0f;
        }
    }
    
}

template <typename in_data_type, typename acc_data_type,
    typename FragC, typename FragA, typename FragB, 
    int PM, int PN,
    int BM, int BN, int BK, 
    int FRAG_M, int FRAG_N, int FRAG_K, 
    int WMMA_M, int WMMA_N, int WMMA_K,
    int WARP_M, int WARP_N,
    int BLOCK_SIZE, int BATCH_SIZE, int QUANT_SIZE>
__device__ void wmma_compute(
    const in_data_type s_a[BK][BM],
    const in_data_type s_b[BK][BN],
    const float s_s[PM][PN],
    FragC frag_r[FRAG_M][FRAG_N],
    const int comp_c_frag_m,
    const int comp_c_frag_n
) {
    FragA frag_a[FRAG_K][FRAG_M]; 
    FragB frag_b[FRAG_K][FRAG_N];

    // Spilt k over BK
    for (int k = 0; k < FRAG_K; ++k) {
        #pragma unroll
        for (int i = 0; i < FRAG_M; ++i) {
            int s_a_row = k * WMMA_K;
            int s_a_col = (comp_c_frag_m * FRAG_M + i) * WMMA_M;
            wmma::load_matrix_sync(frag_a[k][i], &s_a[s_a_row][s_a_col], BM);
        }
        #pragma unroll
        for (int j = 0; j < FRAG_N; ++j) {
            int s_b_row = k * WMMA_K;
            int s_b_col = (comp_c_frag_n * FRAG_N + j) * WMMA_N;
            wmma::load_matrix_sync(frag_b[k][j], &s_b[s_b_row][s_b_col], BN);
        }
    }

    #pragma unroll
    for (int i = 0; i < FRAG_M; i++) {
        #pragma unroll
        for (int j = 0; j < FRAG_N; j++) {
            FragC frag_c;
            wmma::fill_fragment(frag_c, 0.0F);
            #pragma unroll
            for (int k = 0; k < FRAG_K; ++k) {
                wmma::mma_sync(frag_c, frag_a[k][i], frag_b[k][j], frag_c);
            }
            #pragma unroll
            for (int k = 0; k < FragC::num_elements; ++k) {
                #ifdef TEST_ON_RDNA4 // RDNA4, WAVE_SIZE = 32
                int m = ((threadIdx.x & 16) >> 1) | (k & 7) | (comp_c_frag_m * FRAG_M + i) * WMMA_M;
                #else // CDNA3, WAVE_SIZE = 64
                int m = ((threadIdx.x & 48) >> 2) | (k & 3) | (comp_c_frag_m * FRAG_M + i) * WMMA_M;
                #endif
                int n = ((threadIdx.x & 15) | (comp_c_frag_n * FRAG_N + j) * WMMA_N) / QUANT_SIZE;
                float scale = s_s[m][n];
                frag_r[i][j].x[k] += (acc_data_type)scale * (acc_data_type)frag_c.x[k];
            }  
        }
    }
}

template <typename acc_data_type, typename out_data_type, 
typename FragC, typename FragOut, int WMMA_M, int WMMA_N, 
int BM, int BN, int M, int N, int FRAG_M, int FRAG_N>
__device__ inline void store_result(
    out_data_type c[M][N], 
    FragC frag_r[FRAG_M][FRAG_N], 
    const int m,
    const int n,
    const int comp_c_frag_m, 
    const int comp_c_frag_n
) {
    #pragma unroll
    for (int i = 0; i < FRAG_M; i++) {
        #pragma unroll
        for (int j = 0; j < FRAG_N; j++) {
            int frag_m = comp_c_frag_m * FRAG_M + i;
            int frag_n = comp_c_frag_n * FRAG_N + j;
            int row = m + frag_m * WMMA_M;
            int col = n + frag_n * WMMA_N;
            if (((M % BM == 0) || (row < M)) && ((N % BN == 0) || (col < N))) {
                out_data_type *c_ptr = &c[row][col];
                if constexpr (sizeof(acc_data_type) == sizeof(out_data_type)) {
                    wmma::store_matrix_sync(reinterpret_cast<out_data_type*>(c_ptr), frag_r[i][j], N, wmma::mem_row_major);
                } else if constexpr (sizeof(out_data_type) == sizeof(half)) {
                    FragOut frag_out;
                    static_assert(sizeof(half) == sizeof(out_data_type));
                    static_assert(FragOut::num_elements == FragC::num_elements);
                    for (int k=0;k<FragOut::num_elements;++k) {
                        __hip_bfloat16 reg = frag_r[i][j].x[k];
                        frag_out.x[k] = *reinterpret_cast<half*>(&reg);
                    }
                    wmma::store_matrix_sync(reinterpret_cast<half*>(c_ptr), frag_out, N, wmma::mem_row_major);
                } else {
                    static_assert(0, "Unsupported data type for output");
                }

            }
        }
    }
}

// a dummy template to allow inlcuding this file
template<int Dummy=0>
__global__ void reduce(uint32_t m, uint32_t n, uint32_t splitk, const float *c_splitk, __hip_bfloat16 *c) {
    auto tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= m * n) {
        return;
    }
    float sum = 0;
    for (auto i = 0; i < splitk; ++i) {
        sum += c_splitk[i * (m * n) + tid];
    }
    c[tid] = sum;
}

    
#ifdef PARAMETERIZE_LIBRARY
template <
    typename in_data_type,
    typename acc_data_type, // Accumulator type (e.g., float)
    typename out_data_type, // Output type (e.g., __hip_bfloat16)
    int M, int N, int K,    // Matrix dimensions
    int BM, int BN, int BK, // Tile dimensions
    int QUANT_SIZE,         // Quantization block size
    int BLOCK_SIZE,         // Block size
    int WARP_M, int WARP_N // Warp dimensions
>
#else
using in_data_type = __FP8_TYPE;
using out_data_type = __BF16_TYPE;
using acc_data_type = float;
// constexpr int M = 4096, N = 4096, K = 4096;
constexpr int M = 96, N = 1024, K = 1024;
// constexpr int M = 512, N = 512, K = 512;
constexpr int BM = 64, BN = 256, BK = 32;
constexpr int QUANT_SIZE = 128, BLOCK_SIZE = 256;
#ifdef TEST_ON_RDNA4 // RDNA4, WAVE_SIZE = 32
constexpr int WARP_M = 4, WARP_N = 2;
#else // CDNA3, WAVE_SIZE = 64
constexpr int WARP_M = 2, WARP_N = 2;
#endif
#endif // End of parameterization
__global__ void gemm_kernel(
    const in_data_type a[K][M],
    const in_data_type b[K][N],
    out_data_type c[M][N],
    const float sa[ceil_div(K, QUANT_SIZE)][M / 1        ], // Assuming M is divisible by 1 (always true)
    const float sb[ceil_div(K, QUANT_SIZE)][ceil_div(N, QUANT_SIZE)]
) {
    // --- Start: Derived parameters and constants ---
    constexpr int WMMA_M = 16; // Fixed WMMA dimension M
    constexpr int WMMA_N = 16; // Fixed WMMA dimension N
    constexpr int WMMA_K = 32; // Fixed WMMA dimension K (for FP8)

    // WARP_M/N define the 2D arrangement of warps in the block grid.
    // These might need adjustment based on BLOCK_DIM_X/Y strategy.
    // Using fixed values based on the non-parameterized version for now.
    // TODO: Derive WARP_M/N from BLOCK_DIM_X/Y if a flexible strategy is needed.
    constexpr int WARP_NUM = WARP_M * WARP_N; // Total warps per block

    // Assertion: Check if the assumed warp layout matches the block size
    static_assert(WARP_NUM * WAVE_SIZE == BLOCK_SIZE, "WARP_M * WARP_N * WAVE_SIZE must equal BLOCK_SIZE");

    // Fragments per warp
    constexpr int FRAG_M_PER_WARP = BM / WMMA_M / WARP_M;
    constexpr int FRAG_N_PER_WARP = BN / WMMA_N / WARP_N;
    constexpr int FRAG_K = BK / WMMA_K; // Fragments along K dimension tile

    static_assert(BM % (WMMA_M * WARP_M) == 0, "BM must be divisible by WMMA_M * WARP_M");
    static_assert(BN % (WMMA_N * WARP_N) == 0, "BN must be divisible by WMMA_N * WARP_N");
    static_assert(BK % WMMA_K == 0, "BK must be divisible by WMMA_K");
    static_assert(BK >= 32, "BK must be at least 32");
    // --- End: Derived parameters and constants ---

    constexpr int QM = M; // Dimension M for scale A
    constexpr int QN = ceil_div(N, QUANT_SIZE); // Dimension N for scale B (quantized)
    constexpr int QK = ceil_div(K, QUANT_SIZE); // Dimension K for scales (quantized)
    constexpr int PM = BM; // Block size M for scale A * B
    constexpr int PN = ceil_div(BN, QUANT_SIZE); // Block size N for scale A * B

    // Ensure derived fragment counts are positive
    static_assert(FRAG_M_PER_WARP > 0, "FRAG_M_PER_WARP must be positive");
    static_assert(FRAG_N_PER_WARP > 0, "FRAG_N_PER_WARP must be positive");
    static_assert(FRAG_K > 0, "FRAG_K must be positive");

    using FragA = wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, in_data_type, wmma::col_major>;
    using FragB = wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, in_data_type, wmma::row_major>;
    using FragC = wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, acc_data_type>;
    using FragOut = wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half>; // Output uses half for storage via bfloat16 reinterpret

    __shared__ in_data_type  s_a[BK][BM];
    __shared__ in_data_type  s_b[BK][BN];
    __shared__ acc_data_type s_s[PM][PN]; // Accumulator type for scales
    FragC frag_r[FRAG_M_PER_WARP][FRAG_N_PER_WARP]; // Accumulator fragments

    // handle splitk
    a += blockIdx.z * K;
    b += blockIdx.z * K;
    c += blockIdx.z * M;
    sa += blockIdx.z * QK;
    sb += blockIdx.z * QK;

    int tid = threadIdx.x; // Linear thread ID within the block
    int wid = tid / WAVE_SIZE; // Warp ID within the block

    // Initialize the output accumulator fragments to zero
    #pragma unroll
    for (int i = 0; i < FRAG_M_PER_WARP; i++) {
        #pragma unroll
        for (int j = 0; j < FRAG_N_PER_WARP; j++) {
            wmma::fill_fragment(frag_r[i][j], 0.0f); // Use float literal
        }
    }

    // Spilt and compute fragments
    constexpr int iteration_over_k = ceil_div(K, BK); // Use ceil_div for potentially non-divisible K
    constexpr int LOAD_BATCH_SIZE = (2 * sizeof(float4) / sizeof(in_data_type)) > 0 ? (2 * sizeof(float4) / sizeof(in_data_type)) : 1; // Ensure batch size > 0
    static_assert(LOAD_BATCH_SIZE > 0, "LOAD_BATCH_SIZE must be positive");

    for (int bk = 0; bk < iteration_over_k; bk++) {
        const int m = blockIdx.y * BM;
        const int n = blockIdx.x * BN;
        const int k = bk * BK;

        // Calculate remaining K for boundary checks if needed (not currently used by load_input)
        // const int k_rem = K - k;

        // Load data into shared memory
        load_input<in_data_type, BK, BM, K, M, BLOCK_SIZE, LOAD_BATCH_SIZE>(
            s_a, a, m, k);
        load_input<in_data_type, BK, BN, K, N, BLOCK_SIZE, LOAD_BATCH_SIZE>(
            s_b, b, n, k);
        // Load scales into shared memory (using acc_data_type for s_s)
        load_scale<PM, PN, QM, QN, QK, QUANT_SIZE, BLOCK_SIZE, 1>(
            s_s, sa, sb, m, n, k);
        __syncthreads();

        // Perform matrix multiplication using WMMA
        wmma_compute<in_data_type, acc_data_type, FragC, FragA, FragB,
            PM, PN, BM, BN, BK, FRAG_M_PER_WARP, FRAG_N_PER_WARP, FRAG_K,
            WMMA_M, WMMA_N, WMMA_K,
            WARP_M, WARP_N,
            BLOCK_SIZE, LOAD_BATCH_SIZE, QUANT_SIZE>( // Pass calculated BLOCK_SIZE and LOAD_BATCH_SIZE
                s_a, s_b, s_s, frag_r, wid / WARP_N, wid % WARP_N);
        __syncthreads();
    }
    // Store results from accumulator fragments to global memory
    store_result<acc_data_type, out_data_type, FragC, FragOut,
        WMMA_M, WMMA_N, BM, BN, M, N, FRAG_M_PER_WARP, FRAG_N_PER_WARP>(
            c, frag_r, blockIdx.y * BM, blockIdx.x * BN,
            wid / WARP_N, wid % WARP_N);

};

}; // namespace gemm_kernel_legacy

// From gemm_launcher.cpp
#define PARAMETERIZE_LIBRARY
#undef PARAMETERIZE_LIBRARY

// Define LIB_CALL macro for error checking
// Don't print error message, since it will break gen_submission

std::vector<std::shared_ptr<KernelTimer>> timers;

using namespace std;

float *c_splitk = nullptr;
__FP8_TYPE *a_trans = nullptr;
__FP8_TYPE *b_trans = nullptr;
constexpr int MAX_MATRIX_M = 6144;
constexpr int MAX_MATRIX_N = 7168;
constexpr int MAX_MATRIX_K = 7168;
constexpr int MAX_SPLITK_FACTOR = 8;

void init_workspace() {
    LIB_CALL(HOST_TYPE(Malloc)(&c_splitk, MAX_MATRIX_M * MAX_MATRIX_N * sizeof(float) * MAX_SPLITK_FACTOR));
    LIB_CALL(HOST_TYPE(Malloc)(&a_trans, MAX_MATRIX_M * MAX_MATRIX_K * sizeof(__FP8_TYPE)));
    LIB_CALL(HOST_TYPE(Malloc)(&b_trans, MAX_MATRIX_N * MAX_MATRIX_K * sizeof(__FP8_TYPE)));
    // LIB_CALL(HOST_TYPE(StreamCreateWithFlags)(&job_stream0, HOST_TYPE(StreamNonBlocking)));
    // job_stream0 = 0;
}

template <int M, int N, int K, int BM, int BN, int BK, int WARP_M, int WARP_N, int BLOCK_SIZE, int QUANT_BLOCK_SIZE,
          int SPLITK_FACTOR, int LOAD_BATCH_SIZE = 16>
void launch_gemm(const __FP8_TYPE *a, const __FP8_TYPE *b, __BF16_TYPE *c, const float *as, const float *bs, HOST_TYPE(Stream_t) job_stream0) {
    static_assert(M <= MAX_MATRIX_M, "M exceeds maximum supported size");
    static_assert(N <= MAX_MATRIX_N, "N exceeds maximum supported size");
    static_assert(K <= MAX_MATRIX_K, "K exceeds maximum supported size");
    static_assert(SPLITK_FACTOR <= MAX_SPLITK_FACTOR, "SPLITK_FACTOR exceeds maximum supported size");
    if (__builtin_expect(c_splitk == nullptr, 0)) {
        init_workspace();
        LIB_CALL(hipDeviceSynchronize());
    }
    
    transpose_kernel::transpose_fp8<K, N>(b_trans, b, job_stream0);
    transpose_kernel::transpose_fp8<K, M>(a_trans, a, job_stream0);
    // transpose_kernel::launch_transpose<__FP8_TYPE, K, N, 64, 512, 4>(b_trans, b, job_stream0);
    // transpose_kernel::launch_transpose<__FP8_TYPE, K, M, 64, 512, 4>(a_trans, a, job_stream0);
    // Busy wait for 150 microseconds
    // auto start = std::chrono::high_resolution_clock::now();
    // while (std::chrono::duration_cast<std::chrono::microseconds>(
    //     std::chrono::high_resolution_clock::now() - start).count() < 150) {
    //     // Busy wait
    // }
    // be careful that blocksize < 1024, or there's a silent fault
    // gemm_kernel::check_trans<<<dim3(K / 32, M / 32), dim3(32, 32)>>>(a, a_trans, K, M);

    static_assert(K % SPLITK_FACTOR == 0, "K not divisible by SPLITK_FACTOR");
    dim3 grid(ceil_div(N, BN) << 1, ceil_div(M, BM) >> 1, SPLITK_FACTOR);
    static_assert(BLOCK_SIZE >= 32, "BLOCK_SIZE must be at least 32");
    dim3 block(BLOCK_SIZE);
    if constexpr (SPLITK_FACTOR == 1) {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(gemm_kernel::gemm_kernel<__FP8_TYPE, float, __BF16_TYPE, M, N, K, BM, BN, BK, QUANT_BLOCK_SIZE, BLOCK_SIZE, WARP_M, WARP_N, K, K, LOAD_BATCH_SIZE>),
            grid, block, 0, job_stream0, 
            reinterpret_cast<const __FP8_TYPE(*)[K]>(a_trans), 
            reinterpret_cast<const __FP8_TYPE(*)[K]>(b_trans),
            reinterpret_cast<__BF16_TYPE(*)[N]>(c), reinterpret_cast<const float(*)[M]>(as),
            reinterpret_cast<const float(*)[ceil_div(N, QUANT_BLOCK_SIZE)]>(bs)
        );
    } else {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(gemm_kernel::gemm_kernel<__FP8_TYPE, float, float, M, N, K / SPLITK_FACTOR, BM, BN, BK, QUANT_BLOCK_SIZE, BLOCK_SIZE, WARP_M, WARP_N, K, K, LOAD_BATCH_SIZE>),
            grid, block, 0, job_stream0,
            reinterpret_cast<const __FP8_TYPE(*)[K]>(a_trans), 
            reinterpret_cast<const __FP8_TYPE(*)[K]>(b_trans),
            reinterpret_cast<float(*)[N]>(c_splitk), reinterpret_cast<const float(*)[M]>(as),
            reinterpret_cast<const float(*)[ceil_div(N, QUANT_BLOCK_SIZE)]>(bs));
        constexpr uint32_t REDUCE_BLOCK = 256;
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(gemm_kernel::reduce_kernel<M, N, SPLITK_FACTOR, REDUCE_BLOCK>),
            ceil_div(M * N / 4, REDUCE_BLOCK), REDUCE_BLOCK, 0, job_stream0,
            reinterpret_cast<const float(*)[M][N]>(c_splitk), 
            reinterpret_cast<__BF16_TYPE(*)[N]>(c)
        );
        // constexpr uint32_t REDUCE_BLOCK = 256;
        // hipLaunchKernelGGL(
        //     HIP_KERNEL_NAME(gemm_kernel::reduce<SPLITK_FACTOR>),
        //     ceil_div(M * N / 4, REDUCE_BLOCK), REDUCE_BLOCK, 0, job_stream0,
        //     M, N, c_splitk, (__BF16_TYPE *)c);
    }
    // LIB_CALL(hipDeviceSynchronize());
    auto err = HOST_TYPE(GetLastError)();
    if (err != HOST_TYPE(Success)) {
        std::cerr << "Kernel execution failed.\n" << HOST_TYPE(GetErrorString)(err) << std::endl;
        abort();
    }
}

template <int M, int N, int K, int BM, int BN, int BK, int WARP_M, int WARP_N, int BLOCK_SIZE, int QUANT_BLOCK_SIZE, int SPLITK_FACTOR>
void launch_gemm_legacy(const __FP8_TYPE *a, const __FP8_TYPE *b, __BF16_TYPE *c, const float *as, const float *bs, HOST_TYPE(Stream_t) job_stream0) {
    static_assert(K % SPLITK_FACTOR == 0, "K not divisible by SPLITK_FACTOR");
    dim3 grid(ceil_div(N, BN), ceil_div(M, BM), SPLITK_FACTOR);
    static_assert(BLOCK_SIZE >= 32, "BLOCK_SIZE must be at least 32");
    dim3 block(BLOCK_SIZE);
    if (__builtin_expect(c_splitk == nullptr, 0)) {
        init_workspace();
        LIB_CALL(hipDeviceSynchronize());
    }

    if constexpr (SPLITK_FACTOR == 1) {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(gemm_kernel_legacy::gemm_kernel<__FP8_TYPE, float, __BF16_TYPE, M, N, K, BM, BN, BK, QUANT_BLOCK_SIZE, BLOCK_SIZE, WARP_M, WARP_N>),
            grid, block, 0, job_stream0,
            reinterpret_cast<const __FP8_TYPE (*)[M]>(a),
            reinterpret_cast<const __FP8_TYPE (*)[N]>(b),
            reinterpret_cast<__BF16_TYPE (*)[N]>(c),
            reinterpret_cast<const float (*)[M]>(as),
            reinterpret_cast<const float (*)[ceil_div(N, QUANT_BLOCK_SIZE)]>(bs)
        );
    } else {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(gemm_kernel_legacy::gemm_kernel<__FP8_TYPE, float, float, M, N, K / SPLITK_FACTOR, BM, BN, BK, QUANT_BLOCK_SIZE, BLOCK_SIZE, WARP_M, WARP_N>),
            grid, block, 0, job_stream0,
            reinterpret_cast<const __FP8_TYPE (*)[M]>(a),
            reinterpret_cast<const __FP8_TYPE (*)[N]>(b),
            reinterpret_cast<float (*)[N]>(c_splitk),
            reinterpret_cast<const float (*)[M]>(as),
            reinterpret_cast<const float (*)[ceil_div(N, QUANT_BLOCK_SIZE)]>(bs)
        );
        constexpr uint32_t REDUCE_BLOCK = 256;
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(gemm_kernel_legacy::reduce<0>),
            ceil_div(M * N, REDUCE_BLOCK), REDUCE_BLOCK, 0, job_stream0,
            M, N, SPLITK_FACTOR, c_splitk, (__BF16_TYPE *)c
        );
    }
    auto err = HOST_TYPE(GetLastError)();
    if (err != HOST_TYPE(Success)) {
        std::cerr << "Kernel execution failed.\n" << HOST_TYPE(GetErrorString)(err) << std::endl;
        abort();
    }
}

constexpr inline uint32_t pack_shape(uint32_t m, uint32_t n, uint32_t k) {
    // Pack m, n, k into a 32-bit integer
    // Use 8 bits for each dimension (supports 32-aligned values from 32 to 8192)
    // Divide each value by 32 to fit into 8 bits
    return ((m / 32) << 16) | ((n / 32) << 8) | (k / 32);
}
// int M, int N, int K, int BM, int BN, int BK, int WARP_M, int WARP_N, int BLOCK_SIZE, int QUANT_BLOCK_SIZE, int
// SPLITK_FACTOR, int LOAD_BATCH_SIZE
#define DISPATCH_GEMM(M, N, K, BM, BN, BK, WARP_M, WARP_N, BLOCK_SIZE, SPLITK_FACTOR, LOAD_BATCH_SIZE)                                  \
    case pack_shape_checked<M, N, K>(): {                                                                              \
        launch_gemm<M, N, K, BM, BN, BK, WARP_M, WARP_N, BLOCK_SIZE, 128, SPLITK_FACTOR, LOAD_BATCH_SIZE>(a_ptr, b_ptr, c_ptr, as_ptr, bs_ptr, job_stream0);          \
        break;                                                                                                         \
    }

#define DISPATCH_GEMM_LEGACY(M, N, K, BM, BN, BK, WARP_M, WARP_N, BLOCK_SIZE, SPLITK_FACTOR)                                                 \
    case pack_shape_checked<M, N, K>(): {                                                                                \
        launch_gemm_legacy<M, N, K, BM, BN, BK, WARP_M, WARP_N, BLOCK_SIZE, 128, SPLITK_FACTOR>(a_ptr, b_ptr, c_ptr, as_ptr, bs_ptr, job_stream0); \
        break;                                                                                                              \
    }

template <int M, int N, int K> constexpr inline uint32_t pack_shape_checked() {
    static_assert(M % 32 == 0, "M must be a multiple of 32");
    static_assert(N % 32 == 0, "N must be a multiple of 32");
    static_assert(K % 32 == 0, "K must be a multiple of 32");
    static_assert(M >= 32 && M <= 8192, "M must be between 32 and 8192");
    static_assert(N >= 32 && N <= 8192, "N must be between 32 and 8192");
    static_assert(K >= 32 && K <= 8192, "K must be between 32 and 8192");
    return pack_shape(M, N, K);
}

extern "C" {

void run(void *a, void *b, void *as, void *bs, void *c, int m, int n, int k, PerfMetrics *metrics, hipStream_t job_stream0) {
    // Cast pointers once
    const __FP8_TYPE *a_ptr = static_cast<const __FP8_TYPE *>(a);
    const __FP8_TYPE *b_ptr = static_cast<const __FP8_TYPE *>(b);
    __BF16_TYPE *c_ptr = static_cast<__BF16_TYPE *>(c);
    const float *as_ptr = static_cast<const float *>(as);
    const float *bs_ptr = static_cast<const float *>(bs);
    KernelTimerScoped timer(timers, 2LL * m * n * k,
        metrics ? &metrics->entries[0].time : nullptr,
        metrics ? &metrics->entries[0].gflops : nullptr, job_stream0);

    switch (pack_shape(m, n, k)) {
#ifdef TEST_ON_RDNA4 // RDNA4, WAVE_SIZE = 32
        // Test:      M,      N,      K,      BM,    BN,    BK,   WARP_M, WARP_N, BLOCK_SIZE,   SPLITK_FACTOR, LOAD_BATCH_SIZE
        DISPATCH_GEMM(64, 64, 128, 64, 64, 32, 1, 4, 128, 1, 16);
        DISPATCH_GEMM(64, 1536, 7168, 64, 128, 64, 4, 2, 256, 1, 16);
        DISPATCH_GEMM(64, 3072, 1536, 64, 128, 64, 4, 2, 256, 1, 16);
        DISPATCH_GEMM(64, 576, 7168, 64, 128, 64, 4, 2, 256, 1, 16);
        DISPATCH_GEMM(96, 7168, 256, 96, 256, 64, 2, 4, 256, 1, 16);
        DISPATCH_GEMM(96, 7168, 2048, 96, 256, 64, 2, 4, 256, 1, 16);
        DISPATCH_GEMM(96, 4608, 7168, 96, 256, 64, 2, 4, 256, 1, 16);
        DISPATCH_GEMM(128, 7168, 2304, 128, 128, 64, 4, 2, 256, 1, 16);
        DISPATCH_GEMM(128, 512, 7168, 128, 128, 64, 4, 2, 256, 1, 16);
        DISPATCH_GEMM(512, 4096, 512, 256, 128, 64, 4, 2, 256, 1, 16);
        DISPATCH_GEMM(512, 1536, 7168, 256, 128, 64, 4, 2, 256, 1, 16);
        // Benchmark: M,      N,      K,      BM,    BN,    BK,   WARP_M, WARP_N, BLOCK_SIZE,   SPLITK_FACTOR, LOAD_BATCH_SIZE
        DISPATCH_GEMM(1024, 1536, 7168, 128, 128, 64, 1, 4, 128, 4, 16); // Optimized: 0.49 ms (45.65 TFlops)
        DISPATCH_GEMM(1024, 3072, 1536, 256, 128, 32, 4, 2, 256, 1, 16); // Optimized: 0.19 ms (51.32 TFlops)
        DISPATCH_GEMM(1024, 576, 7168, 128, 64, 32, 4, 1, 128, 4, 16);   // Optimized: 0.30 ms (28.16 TFlops)
        DISPATCH_GEMM(1024, 7168, 256, 256, 128, 32, 4, 2, 256, 1, 16);  // Optimized: 0.08 ms (46.49 TFlops)
        DISPATCH_GEMM(1024, 7168, 2048, 256, 128, 32, 4, 2, 256, 1, 16); // Optimized: 0.49 ms (61.92 TFlops)
        DISPATCH_GEMM(1024, 4608, 7168, 128, 128, 32, 2, 2, 128, 1, 16); // Optimized: 0.99 ms (68.16 TFlops)
        DISPATCH_GEMM(1024, 7168, 2304, 256, 128, 32, 4, 2, 256, 1, 16); // Optimized: 0.51 ms (66.04 TFlops)
        DISPATCH_GEMM(1024, 512, 7168, 64, 128, 32, 2, 2, 128, 4, 16);   // Optimized: 0.26 ms (28.97 TFlops)
        DISPATCH_GEMM(1024, 4096, 512, 128, 256, 32, 2, 4, 256, 1, 16);  // Optimized: 0.08 ms (54.27 TFlops)
        DISPATCH_GEMM(6144, 1536, 7168, 256, 128, 32, 4, 2, 256, 1, 16); // Optimized: 1.76 ms (76.76 TFlops)
        DISPATCH_GEMM(6144, 3072, 1536, 256, 128, 32, 4, 2, 256, 1, 16); // Optimized: 0.88 ms (66.00 TFlops)
        DISPATCH_GEMM(6144, 576, 7168, 256, 128, 32, 4, 2, 256, 1, 16);  // Optimized: 0.84 ms (60.68 TFlops)
        DISPATCH_GEMM(6144, 7168, 256, 256, 128, 32, 4, 2, 256, 1, 16);  // Optimized: 0.49 ms (45.76 TFlops)
        DISPATCH_GEMM(6144, 7168, 2048, 256, 128, 32, 4, 2, 256, 1, 16); // Optimized: 2.17 ms (83.11 TFlops)
        DISPATCH_GEMM(6144, 4608, 7168, 256, 128, 32, 4, 2, 256, 1, 16); // Optimized: 4.56 ms (88.99 TFlops)
        DISPATCH_GEMM(6144, 7168, 2304, 256, 128, 32, 4, 2, 256, 1, 16); // Optimized: 2.41 ms (84.32 TFlops)
        DISPATCH_GEMM(6144, 512, 7168, 256, 128, 32, 4, 2, 256, 1, 16);  // Optimized: 0.67 ms (67.45 TFlops)
        DISPATCH_GEMM(6144, 4096, 512, 256, 128, 32, 4, 2, 256, 1, 16);  // Optimized: 0.51 ms (50.79 TFlops)
#else                                                                // CDNA3, WAVE_SIZE = 64
      // Benchmark:   M,    N,      K,      BM,    BN,    BK,   WARP_M, WARP_N, BLOCK_SZ, SPLITK_F, LOAD_BS
        DISPATCH_GEMM(1024, 1536,   7168,   256,   128,   128,   4,      2,      512,      4,        16); // #0
        DISPATCH_GEMM(1024, 3072,   1536,   256,   128,   128,   4,      2,      512,      2,        16); // #1
        DISPATCH_GEMM(1024, 576,    7168,   256,   128,   128,   4,      2,      512,      8,        16); // #2
        DISPATCH_GEMM(1024, 7168,   256,    256,   128,   128,   4,      2,      512,      1,        16); // #3
        DISPATCH_GEMM(1024, 7168,   2048,   256,   128,   128,   4,      2,      512,      1,        16); // #4
        DISPATCH_GEMM(1024, 4608,   7168,   256,   128,   128,   4,      2,      512,      2,        16); // #5
        DISPATCH_GEMM(1024, 7168,   2304,   256,   128,   128,   4,      2,      512,      1,        16); // #6
        DISPATCH_GEMM(1024, 512,    7168,   256,   128,   128,   4,      2,      512,      8,        16); // #7
        DISPATCH_GEMM(1024, 4096,   512,    256,   128,   128,   4,      2,      512,      1,        16); // #8
        DISPATCH_GEMM(6144, 1536,   7168,   256,   128,   128,   4,      2,      512,      1,        16); // #9
        DISPATCH_GEMM(6144, 3072,   1536,   256,   128,   128,   4,      2,      512,      1,        16); // #10
        DISPATCH_GEMM(6144, 576,    7168,   256,   128,   128,   4,      2,      512,      2,        16); // #11
        DISPATCH_GEMM(6144, 7168,   256,    256,   128,   128,   4,      2,      512,      1,        16); // #12
        DISPATCH_GEMM(6144, 7168,   2048,   256,   128,   128,   4,      2,      512,      1,        16); // #13
        DISPATCH_GEMM(6144, 4608,   7168,   256,   128,   128,   4,      2,      512,      1,        16); // #14
        DISPATCH_GEMM(6144, 7168,   2304,   256,   128,   128,   4,      2,      512,      1,        16); // #15
        DISPATCH_GEMM(6144, 512,    7168,   256,   128,   128,   4,      2,      512,      2,        16); // #16
        DISPATCH_GEMM(6144, 4096,   512,    256,   128,   128,   4,      2,      512,      1,        16); // #17
    // Test:                 M,    N,      K,      BM,    BN,    BK,   WARP_M, WARP_N, BLOCK_SZ,    SPLITK_F, 
        DISPATCH_GEMM_LEGACY(64,   64,     128,    64,    64,    32,   4,      2,      512,          1);
        DISPATCH_GEMM_LEGACY(64,   1536,   7168,   64,    128,   64,   4,      2,      512,          1);
        DISPATCH_GEMM_LEGACY(64,   3072,   1536,   64,    128,   64,   4,      2,      512,          1);
        DISPATCH_GEMM_LEGACY(64,   576,    7168,   64,    128,   64,   4,      2,      512,          1);
        DISPATCH_GEMM_LEGACY(96,   7168,   256,    96,    256,   64,   2,      4,      512,          1);
        DISPATCH_GEMM_LEGACY(96,   7168,   2048,   96,    256,   64,   2,      4,      512,          1);
        DISPATCH_GEMM_LEGACY(96,   4608,   7168,   96,    256,   64,   2,      4,      512,          1);
        DISPATCH_GEMM_LEGACY(128,  7168,   2304,   128,   128,   64,   4,      2,      512,          1);
        DISPATCH_GEMM_LEGACY(128,  512,    7168,   128,   128,   64,   4,      2,      512,          1);
        DISPATCH_GEMM_LEGACY(512,  4096,   512,    256,   128,   64,   4,      2,      512,          1);
        DISPATCH_GEMM_LEGACY(512,  1536,   7168,   256,   128,   64,   4,      2,      512,          1);
#endif
    default: {
        printf("Error: Unsupported shape M=%d, K=%d, N=%d\n", m, k, n);
        abort();
    }
    }
}
} // extern "C"



void fp8_mm(torch::Tensor a, torch::Tensor b, torch::Tensor as, torch::Tensor bs, torch::Tensor c) {
    int m = a.size(0);
    int n = b.size(0);
    int k = a.size(1);
    at::hip::HIPStream stream = at::hip::getCurrentHIPStream();
    hipStream_t raw_stream = stream.stream();
    run(
        (__FP8_TYPE*)a.data_ptr(), 
        (__FP8_TYPE*)b.data_ptr(), 
        as.data_ptr<float>(), 
        bs.data_ptr<float>(), 
        (__BF16_TYPE*)c.data_ptr(), 
        m, n, k, nullptr, raw_stream
    );
}

