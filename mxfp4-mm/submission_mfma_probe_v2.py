"""
MFMA FP4 Input Register Mapping Probe for gfx950 (MI355X)
============================================================
Determines the EXACT mapping: for each (thread_lane, nibble_index),
which A[row, k_col] and B[k_row, col] does it correspond to?

Strategy:
  Phase 1: 10 main probes to determine row/col mapping
  Phase 2: Per-nibble isolation (32 probes per target lane, 4 lanes for A, 4 for B)  
  Phase 3: Scale probe to verify scale application
  Phase 4: Cross-reference A×B probes to determine k-index mapping

The kernel runs all probes in sequence and outputs raw C matrices.
Python post-processing reconstructs the full mapping.
"""

import torch
import os
import sys
import tempfile

os.environ["PYTORCH_ROCM_ARCH"] = "gfx950"

cpp_sources = "torch::Tensor mfma_probe(torch::Tensor dummy);"

cuda_sources = r"""
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <torch/types.h>
#include <ATen/ATen.h>

typedef float  float16v  __attribute__((ext_vector_type(16)));
typedef int    int4v     __attribute__((ext_vector_type(4)));

#define SCALE_ONE 127
#define FP4_ZERO  0x0
#define FP4_ONE   0x2
#define NUM_PROBES 10

static __device__ __forceinline__ float16v do_mfma(
    int4v a, int4v b, float16v c, uint32_t scale_a, uint32_t scale_b)
{
    return __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
        a, b, c, 4, 4, 0, scale_a, 0, scale_b);
}

static __device__ __forceinline__ int4v fill_fp4(uint32_t nib)
{
    uint32_t w = nib | (nib << 4) | (nib << 8) | (nib << 12) |
                 (nib << 16) | (nib << 20) | (nib << 24) | (nib << 28);
    int4v v;
    v[0] = (int)w; v[1] = (int)w; v[2] = (int)w; v[3] = (int)w;
    return v;
}

static __device__ __forceinline__ int4v zero_fp4()
{
    int4v v; v[0] = 0; v[1] = 0; v[2] = 0; v[3] = 0; return v;
}

static __device__ __forceinline__ int4v set_single_nibble(int n, uint32_t nib)
{
    int4v v = zero_fp4();
    int word = n / 8;
    int pos  = n % 8;
    v[word] = (int)(((uint32_t)nib) << (pos * 4));
    return v;
}

static __device__ __forceinline__ void store_c(float16v c, float* out, int lane)
{
    for (int v = 0; v < 16; v++) {
        int row = 4 * (lane >> 5) + (lane & 3) + 8 * (v >> 2);
        int col = 4 * ((lane & 31) >> 2) + (v & 3);
        out[row * 32 + col] = c[v];
    }
}

static __device__ __forceinline__ float16v zero_c()
{
    float16v c; for (int i = 0; i < 16; i++) c[i] = 0.0f; return c;
}

/* ============================================================
 * MAIN PROBE KERNEL: 10 probes
 * ============================================================ */
__global__ __launch_bounds__(64, 1)
void mfma_probe_kernel(float* __restrict__ output)
{
    const int lane = threadIdx.x;
    
    for (int p = 0; p < NUM_PROBES; p++) {
        float16v c = zero_c();
        int4v a, b;
        uint32_t sa = SCALE_ONE, sb = SCALE_ONE;
        
        switch (p) {
        case 0: // BASELINE: A=all_1, B=all_1
            a = fill_fp4(FP4_ONE); b = fill_fp4(FP4_ONE); break;
        case 1: // Thread 0 A only
            a = (lane == 0) ? fill_fp4(FP4_ONE) : zero_fp4();
            b = fill_fp4(FP4_ONE); break;
        case 2: // Thread 1 A only
            a = (lane == 1) ? fill_fp4(FP4_ONE) : zero_fp4();
            b = fill_fp4(FP4_ONE); break;
        case 3: // Thread 32 A only
            a = (lane == 32) ? fill_fp4(FP4_ONE) : zero_fp4();
            b = fill_fp4(FP4_ONE); break;
        case 4: // Thread 33 A only
            a = (lane == 33) ? fill_fp4(FP4_ONE) : zero_fp4();
            b = fill_fp4(FP4_ONE); break;
        case 5: // B thread 0 only
            a = fill_fp4(FP4_ONE);
            b = (lane == 0) ? fill_fp4(FP4_ONE) : zero_fp4(); break;
        case 6: // B thread 1 only
            a = fill_fp4(FP4_ONE);
            b = (lane == 1) ? fill_fp4(FP4_ONE) : zero_fp4(); break;
        case 7: // B thread 32 only
            a = fill_fp4(FP4_ONE);
            b = (lane == 32) ? fill_fp4(FP4_ONE) : zero_fp4(); break;
        case 8: // B thread 33 only
            a = fill_fp4(FP4_ONE);
            b = (lane == 33) ? fill_fp4(FP4_ONE) : zero_fp4(); break;
        case 9: // A thread 0, nibble 0 only
            a = (lane == 0) ? set_single_nibble(0, FP4_ONE) : zero_fp4();
            b = fill_fp4(FP4_ONE); break;
        }
        
        c = do_mfma(a, b, c, sa, sb);
        store_c(c, output + p * 32 * 32, lane);
    }
}

/* ============================================================
 * NIBBLE ISOLATION KERNEL: 32 probes for one target lane
 * For A-mapping: isolate each nibble of target lane in A, B=all_1
 * ============================================================ */
__global__ __launch_bounds__(64, 1)
void mfma_a_nibble_kernel(float* __restrict__ output, int target_lane)
{
    const int lane = threadIdx.x;
    for (int nib = 0; nib < 32; nib++) {
        float16v c = zero_c();
        int4v a = (lane == target_lane) ? set_single_nibble(nib, FP4_ONE) : zero_fp4();
        int4v b = fill_fp4(FP4_ONE);
        c = do_mfma(a, b, c, SCALE_ONE, SCALE_ONE);
        store_c(c, output + nib * 32 * 32, lane);
    }
}

/* For B-mapping: isolate each nibble of target lane in B, A=all_1 */
__global__ __launch_bounds__(64, 1)
void mfma_b_nibble_kernel(float* __restrict__ output, int target_lane)
{
    const int lane = threadIdx.x;
    for (int nib = 0; nib < 32; nib++) {
        float16v c = zero_c();
        int4v a = fill_fp4(FP4_ONE);
        int4v b = (lane == target_lane) ? set_single_nibble(nib, FP4_ONE) : zero_fp4();
        c = do_mfma(a, b, c, SCALE_ONE, SCALE_ONE);
        store_c(c, output + nib * 32 * 32, lane);
    }
}

/* ============================================================
 * CROSS-REFERENCE KERNEL: A single nibble × B single nibble
 * Tests whether A[lane_a, nib_a] and B[lane_b, nib_b] share a k-index
 * If C != 0, they do. The non-zero (row, col) gives us both.
 * ============================================================ */
__global__ __launch_bounds__(64, 1)
void mfma_cross_kernel(float* __restrict__ output,
                       int a_lane, int a_nib, int b_lane, int b_nib)
{
    const int lane = threadIdx.x;
    float16v c = zero_c();
    int4v a = (lane == a_lane) ? set_single_nibble(a_nib, FP4_ONE) : zero_fp4();
    int4v b = (lane == b_lane) ? set_single_nibble(b_nib, FP4_ONE) : zero_fp4();
    c = do_mfma(a, b, c, SCALE_ONE, SCALE_ONE);
    store_c(c, output, lane);
}

/* ============================================================
 * SCALE PROBE: verify scale behavior
 * All A/B = 1.0, each thread gets unique A-scale
 * ============================================================ */
__global__ __launch_bounds__(64, 1)
void mfma_scale_kernel(float* __restrict__ output)
{
    const int lane = threadIdx.x;
    float16v c = zero_c();
    int4v a = fill_fp4(FP4_ONE);
    int4v b = fill_fp4(FP4_ONE);
    uint32_t sa = SCALE_ONE + (lane < 24 ? lane : 0);
    c = do_mfma(a, b, c, sa, SCALE_ONE);
    store_c(c, output, lane);
}

/* ============================================================
 * FULL A-MAPPING: All 64 lanes, 32 nibbles each = 2048 probes
 * Done lane-by-lane to save memory (32 probes per kernel call)
 * ============================================================ */
__global__ __launch_bounds__(64, 1)
void mfma_full_a_map_kernel(float* __restrict__ output, int target_lane)
{
    const int lane = threadIdx.x;
    for (int nib = 0; nib < 32; nib++) {
        float16v c = zero_c();
        int4v a = (lane == target_lane) ? set_single_nibble(nib, FP4_ONE) : zero_fp4();
        int4v b = fill_fp4(FP4_ONE);
        c = do_mfma(a, b, c, SCALE_ONE, SCALE_ONE);
        store_c(c, output + nib * 32 * 32, lane);
    }
}

/* FULL B-MAPPING: same for B */
__global__ __launch_bounds__(64, 1)
void mfma_full_b_map_kernel(float* __restrict__ output, int target_lane)
{
    const int lane = threadIdx.x;
    for (int nib = 0; nib < 32; nib++) {
        float16v c = zero_c();
        int4v a = fill_fp4(FP4_ONE);
        int4v b = (lane == target_lane) ? set_single_nibble(nib, FP4_ONE) : zero_fp4();
        c = do_mfma(a, b, c, SCALE_ONE, SCALE_ONE);
        store_c(c, output + nib * 32 * 32, lane);
    }
}

/* ============================================================
 * Python-callable: runs all probes
 * ============================================================ */
torch::Tensor mfma_probe(torch::Tensor dummy) {
    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(dummy.device());
    
    // Layout: [main_10 | scale_1 | a_map_64x32 | b_map_64x32 | cross_probes]
    // Main: 10 * 32*32
    // Scale: 32*32
    // A-map: 64 * 32 * 32*32 (each lane's 32 nibbles)
    // B-map: 64 * 32 * 32*32
    // Cross: 32*32 (a few spot checks)
    
    int main_sz = 10 * 32 * 32;
    int scale_sz = 32 * 32;
    int amap_sz = 64 * 32 * 32 * 32;
    int bmap_sz = 64 * 32 * 32 * 32;
    int cross_sz = 4 * 32 * 32;  // 4 cross probes
    int total = main_sz + scale_sz + amap_sz + bmap_sz + cross_sz;
    
    auto output = torch::zeros({total}, opts);
    float* ptr = output.data_ptr<float>();
    int off = 0;
    
    // Main probes
    hipLaunchKernelGGL(mfma_probe_kernel, dim3(1), dim3(64), 0, 0, ptr + off);
    off += main_sz;
    
    // Scale probe
    hipLaunchKernelGGL(mfma_scale_kernel, dim3(1), dim3(64), 0, 0, ptr + off);
    off += scale_sz;
    
    // Full A mapping (64 lanes)
    for (int lane = 0; lane < 64; lane++) {
        hipLaunchKernelGGL(mfma_full_a_map_kernel, dim3(1), dim3(64), 0, 0,
                           ptr + off + lane * 32 * 32 * 32, lane);
    }
    off += amap_sz;
    
    // Full B mapping (64 lanes)
    for (int lane = 0; lane < 64; lane++) {
        hipLaunchKernelGGL(mfma_full_b_map_kernel, dim3(1), dim3(64), 0, 0,
                           ptr + off + lane * 32 * 32 * 32, lane);
    }
    off += bmap_sz;
    
    // Cross probes: check if A(0,0) and B(0,n) share k-index
    for (int bn = 0; bn < 4; bn++) {
        hipLaunchKernelGGL(mfma_cross_kernel, dim3(1), dim3(64), 0, 0,
                           ptr + off + bn * 32 * 32,
                           0, 0, 0, bn);
    }
    off += cross_sz;
    
    hipDeviceSynchronize();
    return output;
}
"""

from torch.utils.cpp_extension import load_inline

def run_probe():
    print("=" * 70)
    print("MFMA FP4 INPUT REGISTER MAPPING PROBE v2")
    print("=" * 70)
    
    build_dir = tempfile.mkdtemp(prefix="mfma_probe_")
    
    try:
        module = load_inline(
            name="mfma_probe_v2",
            cpp_sources=cpp_sources,
            cuda_sources=cuda_sources,
            extra_cuda_cflags=["-O3", "--offload-arch=gfx950"],
            build_directory=build_dir,
            functions=["mfma_probe"],
            with_cuda=True,
            verbose=False,
        )
        print("[OK] Probe kernel compiled successfully")
    except Exception as e:
        print(f"[FAIL] Compilation error: {e}")
        return None
    
    dummy = torch.zeros(1, device="cuda")
    result = module.mfma_probe(dummy).cpu().numpy()
    
    import numpy as np
    
    off = 0
    main_sz = 10 * 32 * 32
    scale_sz = 32 * 32
    amap_sz = 64 * 32 * 32 * 32
    bmap_sz = 64 * 32 * 32 * 32
    cross_sz = 4 * 32 * 32
    
    main = result[off:off+main_sz].reshape(10, 32, 32); off += main_sz
    scale = result[off:off+scale_sz].reshape(32, 32); off += scale_sz
    amap = result[off:off+amap_sz].reshape(64, 32, 32, 32); off += amap_sz
    bmap = result[off:off+bmap_sz].reshape(64, 32, 32, 32); off += bmap_sz
    cross = result[off:off+cross_sz].reshape(4, 32, 32); off += cross_sz
    
    # ================================================================
    # ANALYZE BASELINE
    # ================================================================
    print("
--- P0: BASELINE (all 1.0) ---")
    p0 = main[0]
    print(f"  C[0,0]={p0[0,0]:.1f}  Mean={p0.mean():.1f}  Min={p0.min():.1f}  Max={p0.max():.1f}")
    baseline_val = p0[0,0]
    print(f"  {'OK' if abs(baseline_val - 64.0) < 1 else 'UNEXPECTED'}: expected 64.0")
    
    # ================================================================
    # ANALYZE PER-THREAD A PROBES (which rows does each thread own?)
    # ================================================================
    print("
--- P1-P4: Per-thread A row mapping ---")
    for pid, tlane in [(1, 0), (2, 1), (3, 32), (4, 33)]:
        p = main[pid]
        nz_rows = [r for r in range(32) if np.abs(p[r,:]).max() > 0.01]
        vals = {r: p[r,0] for r in nz_rows}
        print(f"  Thread {tlane:2d}: non-zero rows = {nz_rows}, val@col0 = {vals}")
    
    # ================================================================
    # ANALYZE PER-THREAD B PROBES (which cols does each thread own?)
    # ================================================================
    print("
--- P5-P8: Per-thread B col mapping ---")
    for pid, tlane in [(5, 0), (6, 1), (7, 32), (8, 33)]:
        p = main[pid]
        nz_cols = [c for c in range(32) if np.abs(p[:,c]).max() > 0.01]
        vals = {c: p[0,c] for c in nz_cols}
        print(f"  Thread {tlane:2d}: non-zero cols = {nz_cols}, val@row0 = {vals}")
    
    # ================================================================
    # FULL A MAPPING: for each (lane, nibble), find the row
    # ================================================================
    print("
" + "=" * 70)
    print("FULL A-INPUT MAPPING: (lane, nibble) -> A_row")
    print("=" * 70)
    
    a_row_map = np.full((64, 32), -1, dtype=np.int32)
    
    for lane in range(64):
        for nib in range(32):
            c = amap[lane, nib]  # 32x32 matrix
            row_sums = np.abs(c).sum(axis=1)
            nz_rows = np.where(row_sums > 0.01)[0]
            if len(nz_rows) == 1:
                a_row_map[lane, nib] = nz_rows[0]
            elif len(nz_rows) > 1:
                a_row_map[lane, nib] = nz_rows[0]  # take first (log warning)
    
    # Print summary
    for lane in [0, 1, 2, 3, 16, 31, 32, 33, 34, 63]:
        rows = a_row_map[lane, :]
        unique = sorted(set(rows[rows >= 0]))
        print(f"  Lane {lane:2d}: rows = {list(rows[:8])}... unique = {unique}")
    
    # Check hypothesis: row = lane % 32
    hypothesis_match = 0
    for lane in range(64):
        for nib in range(32):
            if a_row_map[lane, nib] == lane % 32:
                hypothesis_match += 1
    print(f"
  Hypothesis 'row = lane % 32': {hypothesis_match}/2048 match")
    
    # ================================================================
    # FULL B MAPPING: for each (lane, nibble), find the col
    # ================================================================
    print("
" + "=" * 70)
    print("FULL B-INPUT MAPPING: (lane, nibble) -> B_col")
    print("=" * 70)
    
    b_col_map = np.full((64, 32), -1, dtype=np.int32)
    
    for lane in range(64):
        for nib in range(32):
            c = bmap[lane, nib]  # 32x32 matrix
            col_sums = np.abs(c).sum(axis=0)
            nz_cols = np.where(col_sums > 0.01)[0]
            if len(nz_cols) == 1:
                b_col_map[lane, nib] = nz_cols[0]
            elif len(nz_cols) > 1:
                b_col_map[lane, nib] = nz_cols[0]
    
    for lane in [0, 1, 2, 3, 16, 31, 32, 33, 34, 63]:
        cols = b_col_map[lane, :]
        unique = sorted(set(cols[cols >= 0]))
        print(f"  Lane {lane:2d}: cols = {list(cols[:8])}... unique = {unique}")
    
    hypothesis_match = 0
    for lane in range(64):
        for nib in range(32):
            if b_col_map[lane, nib] == lane % 32:
                hypothesis_match += 1
    print(f"
  Hypothesis 'col = lane % 32': {hypothesis_match}/2048 match")
    
    # ================================================================
    # SCALE ANALYSIS
    # ================================================================
    print("
" + "=" * 70)
    print("SCALE PROBE ANALYSIS")
    print("=" * 70)
    
    print("  Scale probe C[row, 0] for rows 0-7:")
    for r in range(8):
        expected = 32.0 * (2**r + 1.0) if r < 24 else 64.0
        actual = scale[r, 0]
        match = "OK" if abs(actual - expected) / max(expected, 1) < 0.01 else "MISMATCH"
        print(f"    Row {r}: got {actual:.1f}, expected {expected:.1f} [{match}]")
    
    # ================================================================
    # CROSS-REFERENCE: k-index matching
    # ================================================================
    print("
" + "=" * 70)
    print("CROSS-REFERENCE: A(lane=0,nib=0) x B(lane=0,nib=N)")
    print("=" * 70)
    
    for bn in range(4):
        c = cross[bn]
        nz = np.abs(c) > 0.01
        nz_count = nz.sum()
        if nz_count > 0:
            nz_rows = np.where(np.abs(c).sum(axis=1) > 0.01)[0]
            nz_cols = np.where(np.abs(c).sum(axis=0) > 0.01)[0]
            val = c[nz_rows[0], nz_cols[0]] if len(nz_rows) > 0 and len(nz_cols) > 0 else 0
            print(f"  A(0,0) x B(0,{bn}): non-zero at rows={list(nz_rows)}, cols={list(nz_cols)}, val={val:.4f}")
            print(f"    -> k-index match! A nibble 0 and B nibble {bn} share k-index")
        else:
            print(f"  A(0,0) x B(0,{bn}): ALL ZERO (no k-index overlap)")
    
    # ================================================================
    # DETERMINE K-INDEX FROM A-NIBBLE PROBES
    # ================================================================
    # When A[lane,nib] is set and B=all_1:
    # C[row_mapped, j] = 1.0 for all j (since B[k,j]=1 for all k,j)
    # ALL columns get value 1.0, so we can't tell k from this alone.
    #
    # But the CROSS probes tell us:
    # A[la,na] × B[lb,nb] != 0  iff  A_k_col(la,na) == B_k_row(lb,nb)
    # 
    # Strategy: for each A nibble, find which B nibble it matches.
    # Then knowing B's k-row assignments, we know A's k-col.
    # But we don't know B's k-row either!
    #
    # Circular? No. If we enumerate ALL cross products, we get a 
    # 2048×2048 binary matrix. Each row has exactly one 1 (the matching B nibble).
    # This gives us a PERMUTATION from A-positions to B-positions.
    # Combined with row/col info, this fully determines the k-indices.
    #
    # But 2048*2048 = 4M cross probes is too many.
    # Instead: for each A nibble, binary-search the matching B nibble.
    # 2048 * 11 = 22528 probes. Still too many.
    #
    # BETTER: A has 64 k-columns (0..63). B has 64 k-rows (0..63).
    # Each (lane,nib) pair in A maps to one k-column.
    # We only need to find which of the 64 k-values each maps to.
    # That's 6 bits. We can use 6 binary probes over the B side:
    #
    # For bit b = 0..5:
    #   All threads: A = all nibbles 1.0
    #   B: nibbles with bit b set in their k-row-index have value 1.0, others 0
    #   But we don't know B's k-row-index!
    #
    # We CAN determine relative ordering:
    # Pick a "reference B set" of 32 B-nibbles (e.g., lane 0's 32 nibbles).
    # These map to 32 distinct k-rows (presumably k0..k31 or some permutation).
    # For each A nibble: cross with each of the 32 reference B nibbles.
    # The one that produces non-zero output identifies the matching k-row.
    # 2048 * 32 = 65536 probes. Still too many.
    #
    # Optimal: BINARY search. For each A nibble:
    # - Probe with B = (all of lane 0's nibbles 0-15). If non-zero, k is in first half.
    # - Then narrow down. 5 probes per A nibble = 5 * 2048 = 10240 probes.
    #
    # OR: Batch across A nibbles.
    # - Set ALL A nibbles to 1.0 for ALL threads.
    # - Set B = lane 0's nibbles 0-15 only (rest zero).
    # - C[i,j] = sum over k in {k-rows of B nibbles 0-15} A[i,k] * 1.0
    # - C[i,j] = (number of k-values in first-half-of-lane0 that exist in row i's range)
    # - This gives us a 32x32 matrix of counts. Not enough resolution.
    #
    # I think the most practical approach for the probe submission is:
    # 1. Run the 64+64 nibble-isolation probes to get A-row and B-col mapping
    # 2. Run a moderate number of cross probes to determine k-ordering
    # 3. If the hypothesis is correct (k = (lane//32)*32 + nib), verify with a few cross probes
    #
    # The FULL k-mapping can be determined from 32 carefully chosen cross probes
    # IF we know that each lane's 32 nibbles map to 32 consecutive k-values.
    
    print("
" + "=" * 70)
    print("MAPPING SUMMARY")
    print("=" * 70)
    
    # Print compact mapping tables
    print("
A-ROW MAP [64 lanes x 32 nibbles]:")
    print("(showing which matrix row each register nibble maps to)")
    for lane in range(64):
        rows = a_row_map[lane]
        row_str = ' '.join(f'{r:2d}' if r >= 0 else ' ?' for r in rows[:16])
        print(f"  lane {lane:2d}: {row_str} ...")
    
    print("
B-COL MAP [64 lanes x 32 nibbles]:")
    print("(showing which matrix column each register nibble maps to)")
    for lane in range(64):
        cols = b_col_map[lane]
        col_str = ' '.join(f'{c:2d}' if c >= 0 else ' ?' for c in cols[:16])
        print(f"  lane {lane:2d}: {col_str} ...")
    
    # Verify structure
    print("
" + "=" * 70)
    print("PATTERN ANALYSIS")
    print("=" * 70)
    
    # Check if A-row depends only on lane (not nibble)
    a_row_const = True
    for lane in range(64):
        unique_rows = set(a_row_map[lane, :])
        unique_rows.discard(-1)
        if len(unique_rows) > 1:
            a_row_const = False
            print(f"  Lane {lane}: A maps to MULTIPLE rows: {sorted(unique_rows)}")
            break
    if a_row_const:
        print("  A-row depends ONLY on lane (all nibbles of a lane -> same row)")
        row_per_lane = [a_row_map[l, 0] for l in range(64)]
        print(f"  Row mapping: {row_per_lane}")
    
    b_col_const = True
    for lane in range(64):
        unique_cols = set(b_col_map[lane, :])
        unique_cols.discard(-1)
        if len(unique_cols) > 1:
            b_col_const = False
            print(f"  Lane {lane}: B maps to MULTIPLE cols: {sorted(unique_cols)}")
            break
    if b_col_const:
        print("  B-col depends ONLY on lane (all nibbles of a lane -> same col)")
        col_per_lane = [b_col_map[l, 0] for l in range(64)]
        print(f"  Col mapping: {col_per_lane}")
    
    return {
        "main": main,
        "scale": scale,
        "a_row_map": a_row_map,
        "b_col_map": b_col_map,
        "cross": cross,
    }


def custom_kernel(data):
    """Entry point for evaluator. Runs probe then returns dummy output."""
    result = run_probe()
    M = data["A"].shape[0]
    N = data["B_q"].shape[0]
    return torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")


if __name__ == "__main__":
    run_probe()
