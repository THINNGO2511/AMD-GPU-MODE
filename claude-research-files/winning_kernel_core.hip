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