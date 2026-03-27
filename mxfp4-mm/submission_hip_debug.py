#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""Debug kernel: write intermediate values from thread (n=0,m=0) to a debug buffer."""
import torch, os, subprocess, ctypes, hashlib
from task import input_t, output_t

HIP_SRC = r"""
#include <hip/hip_runtime.h>

__device__ __forceinline__ float fp4_val(unsigned char nib) {
    const float lut[16] = {
        0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
        0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
    };
    return lut[nib];
}

__device__ __forceinline__ float bf16_f32(unsigned short x) {
    union { unsigned int i; float f; } u;
    u.i = (unsigned int)x << 16;
    return u.f;
}

__device__ __forceinline__ unsigned short f32_bf16(float x) {
    union { float f; unsigned int i; } u;
    u.f = x;
    u.i += ((u.i >> 16) & 1) + 0x7FFF;
    return (unsigned short)(u.i >> 16);
}

__device__ __forceinline__ float e8m0_f32(unsigned char e) {
    union { unsigned int i; float f; } u;
    u.i = (unsigned int)e << 23;
    return u.f;
}

__device__ __forceinline__ float compute_a_scale(float amax) {
    if (amax == 0.0f) return 0.0f;
    union { float f; unsigned int i; } u;
    u.f = amax;
    int be = (int)((u.i >> 23) & 0xFF);
    unsigned int mantissa = u.i & 0x7FFFFF;
    int se = be - ((mantissa > 0x400000) ? 1 : 2);
    if (se < 1) se = 1;
    if (se > 254) se = 254;
    u.i = (unsigned int)se << 23;
    return u.f;
}

__device__ __forceinline__ unsigned char qfp4(float v) {
    unsigned char s = (v < 0.0f) ? 8u : 0u;
    float a = fabsf(v);
    unsigned char m;
    if      (a < 0.25f) m = 0;
    else if (a < 0.75f) m = 1;
    else if (a < 1.25f) m = 2;
    else if (a < 1.75f) m = 3;
    else if (a < 2.5f)  m = 4;
    else if (a < 3.5f)  m = 5;
    else if (a < 5.0f)  m = 6;
    else                m = 7;
    return s | m;
}

// Debug kernel: only thread (n=0) writes debug info for m=0, kb=0..2
__global__ __launch_bounds__(256)
void mxfp4_debug(
    const unsigned short* __restrict__ A,
    const unsigned char* __restrict__ Bt,
    const unsigned char* __restrict__ Bs,
    float* __restrict__ dbg,  // debug buffer [256]
    int M, int N, int K)
{
    const int n = blockIdx.x * 256 + threadIdx.x;
    if (n != 0) return;  // only thread 0

    int nkb = K >> 5;
    int di = 0;  // debug index

    // Write basic info
    dbg[di++] = (float)M;   // [0]
    dbg[di++] = (float)N;   // [1]
    dbg[di++] = (float)K;   // [2]
    dbg[di++] = (float)nkb; // [3]

    // For kb=0, dump everything
    for (int kb = 0; kb < 3 && kb < nkb && di < 240; kb++) {
        unsigned char bs_byte = Bs[kb * N + 0];
        float bsc = e8m0_f32(bs_byte);
        dbg[di++] = (float)bs_byte;  // scale byte
        dbg[di++] = bsc;             // scale value

        // First 4 B bytes
        const unsigned char* bp = Bt + ((long long)kb * N + 0) * 16;
        for (int j = 0; j < 4 && di < 240; j++) {
            unsigned char byte = bp[j];
            dbg[di++] = (float)byte;           // raw byte
            dbg[di++] = fp4_val(byte & 0xF);   // lo dequant (no scale)
            dbg[di++] = fp4_val(byte >> 4);    // hi dequant (no scale)
        }

        // First 8 A values for m=0
        const unsigned short* ap = A + (long long)0 * K + kb * 32;
        for (int j = 0; j < 8 && di < 240; j++) {
            dbg[di++] = bf16_f32(ap[j]);
        }

        // Compute A scale for m=0
        float amax = 0.0f;
        float av[32];
        for (int j = 0; j < 32; j++) {
            av[j] = bf16_f32(ap[j]);
            amax = fmaxf(amax, fabsf(av[j]));
        }
        float asc = compute_a_scale(amax);
        float inv = (asc > 0.0f) ? (1.0f / asc) : 0.0f;
        dbg[di++] = amax;
        dbg[di++] = asc;

        // Compute dot product for m=0
        float bv[32];
        for (int j = 0; j < 16; j++) {
            unsigned char byte = bp[j];
            bv[2*j]   = fp4_val(byte & 0xF);
            bv[2*j+1] = fp4_val(byte >> 4);
        }
        float dot = 0.0f;
        for (int j = 0; j < 32; j++) {
            dot += fp4_val(qfp4(av[j] * inv)) * bv[j];
        }
        dbg[di++] = dot;
        dbg[di++] = dot * asc * bsc;
    }
    dbg[255] = (float)di;  // number of debug values written
}

extern "C" void launch_debug(
    void* A, void* Bt, void* Bs, void* dbg,
    int M, int N, int K)
{
    dim3 block(256);
    dim3 grid(1, 1);
    hipLaunchKernelGGL(mxfp4_debug, grid, block, 0, 0,
        (const unsigned short*)A, (const unsigned char*)Bt,
        (const unsigned char*)Bs, (float*)dbg, M, N, K);
}
"""

_lib = None

def _compile():
    global _lib
    if _lib is not None:
        return _lib
    h = hashlib.md5(HIP_SRC.encode()).hexdigest()[:8]
    src = f'/tmp/_mxfp4_dbg_{h}.hip'
    so = f'/tmp/_mxfp4_dbg_{h}.so'
    with open(src, 'w') as f:
        f.write(HIP_SRC)
    hipcc = 'hipcc'
    for p in ['/opt/rocm/bin/hipcc', '/opt/rocm/hip/bin/hipcc']:
        if os.path.exists(p):
            hipcc = p
            break
    r = subprocess.run(
        [hipcc, '-shared', '-fPIC', '-O3', '--offload-arch=gfx950',
         '-std=c++17', '-ffast-math', '-o', so, src],
        capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        raise RuntimeError(f"hipcc failed:\n{r.stderr}")
    _lib = ctypes.cdll.LoadLibrary(so)
    _lib.launch_debug.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int]
    _lib.launch_debug.restype = None
    return _lib

def _unshuffle_e8m0(s):
    s = s.view(torch.uint8)
    sm, sn = s.shape
    s = s.view(sm // 32, sn // 8, 4, 16, 2, 2)
    s = s.permute(0, 5, 3, 1, 4, 2).contiguous()
    return s.view(sm, sn)

_cache = {}
def custom_kernel(data: input_t) -> output_t:
    A, B, B_q, B_shuffle, B_scale_sh = data
    M, K = A.shape
    N = B.shape[0]

    lib = _compile()

    ck = (N, K)
    if ck not in _cache:
        bu = B_q.view(torch.uint8)
        Bt = bu.view(N, K // 32, 16).permute(1, 0, 2).contiguous()
        bs_raw = _unshuffle_e8m0(B_scale_sh)[:N, :].contiguous()
        Bs = bs_raw.view(torch.uint8).t().contiguous()
        _cache[ck] = (Bt, Bs)

    Bt, Bs = _cache[ck]

    # Debug buffer
    dbg = torch.zeros(256, dtype=torch.float32, device='cuda')

    lib.launch_debug(
        A.data_ptr(), Bt.data_ptr(), Bs.data_ptr(), dbg.data_ptr(),
        M, N, K)
    torch.cuda.synchronize()

    d = dbg.cpu().tolist()
    nwritten = int(d[255])
    print(f"=== DEBUG n=0, M={int(d[0])}, N={int(d[1])}, K={int(d[2])}, nkb={int(d[3])} ===")

    # Also compute expected values in Python for comparison
    bu = B_q.view(torch.uint8)
    bs_raw_py = _unshuffle_e8m0(B_scale_sh)[:N, :].contiguous().view(torch.uint8)

    FP4 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
           0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]

    import struct
    di = 4
    for kb in range(min(3, K // 32)):
        if di >= nwritten:
            break

        # Kernel values
        k_bs_byte = int(d[di]); di += 1
        k_bsc = d[di]; di += 1

        # Python values
        py_bs_byte = bs_raw_py[0, kb].item()
        py_bsc = struct.unpack('f', struct.pack('I', int(py_bs_byte) << 23))[0] if py_bs_byte > 0 else 0.0

        print(f"\n  kb={kb}: scale_byte kernel={k_bs_byte} python={py_bs_byte} | scale_val kernel={k_bsc:.6f} python={py_bsc:.6f}")

        # B bytes
        for j in range(4):
            if di + 2 >= nwritten: break
            k_byte = int(d[di]); di += 1
            k_lo = d[di]; di += 1
            k_hi = d[di]; di += 1

            py_byte = bu[0, kb*16+j].item()
            py_lo = FP4[py_byte & 0xF]
            py_hi = FP4[py_byte >> 4]

            match_byte = "OK" if k_byte == py_byte else "MISMATCH!"
            match_lo = "OK" if abs(k_lo - py_lo) < 1e-6 else "MISMATCH!"
            match_hi = "OK" if abs(k_hi - py_hi) < 1e-6 else "MISMATCH!"
            print(f"    B byte[{j}]: kernel={k_byte}({match_byte}) py={py_byte} | lo: k={k_lo:.3f}({match_lo}) py={py_lo:.3f} | hi: k={k_hi:.3f}({match_hi}) py={py_hi:.3f}")

        # A values
        a_vals = A[0, kb*32:kb*32+8].float().cpu().tolist()
        print(f"    A[0,{kb*32}:{kb*32+8}] python: {[f'{v:.4f}' for v in a_vals]}")
        kernel_a = []
        for j in range(8):
            if di >= nwritten: break
            kernel_a.append(d[di]); di += 1
        print(f"    A[0,{kb*32}:{kb*32+8}] kernel: {[f'{v:.4f}' for v in kernel_a]}")

        if di + 3 < nwritten:
            k_amax = d[di]; di += 1
            k_asc = d[di]; di += 1
            k_dot = d[di]; di += 1
            k_contrib = d[di]; di += 1
            print(f"    amax={k_amax:.6f} asc={k_asc:.6f} dot={k_dot:.6f} contrib={k_contrib:.6f}")

    # Use reference for correct output
    from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4
    bs_raw_full = _unshuffle_e8m0(B_scale_sh)
    C = gemm_a16wfp4(A, bu, bs_raw_full, dtype=torch.bfloat16)
    return C
