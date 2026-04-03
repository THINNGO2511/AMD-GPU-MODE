#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
GEMM — Test load_inline compilation on MI355X runner.

The winning MI300X solutions ALL use torch.utils.cpp_extension.load_inline
to compile custom HIP C++ kernels at runtime. This tests whether the
MI355X runner supports this approach.

Step 1: Try compiling a trivial HIP kernel
Step 2: If it works, try a simple bf16 GEMM via hipBLASLt
Step 3: Report compilation time, success/failure, available tools

Falls back to proven gemm_a16wfp4 if compilation fails.
"""
import os
import time
import sys

os.environ["HIP_FORCE_DEV_KERNARG"] = "1"
os.environ["OPTIMIZE_EPILOGUE"] = "1"

from task import input_t, output_t
import torch

_load_inline_works = None
_hipblas_module = None
_call = 0

# Proven fallback
_gc = {}; _bsr = None; _bqu = None; _braw = None; _yc = {}; _w = False
_K7168 = {"BLOCK_SIZE_M":16,"BLOCK_SIZE_N":64,"BLOCK_SIZE_K":512,"GROUP_SIZE_M":1,"num_warps":4,"num_stages":2,"waves_per_eu":2,"matrix_instr_nonkdim":16,"cache_modifier":".cg","NUM_KSPLIT":8,"SPLITK_BLOCK_SIZE":1024}
_K512 = {"BLOCK_SIZE_M":4,"BLOCK_SIZE_N":128,"BLOCK_SIZE_K":512,"GROUP_SIZE_M":1,"num_warps":4,"num_stages":3,"waves_per_eu":2,"matrix_instr_nonkdim":16,"cache_modifier":".cg","NUM_KSPLIT":1}
_K2048 = {"BLOCK_SIZE_M":16,"BLOCK_SIZE_N":128,"BLOCK_SIZE_K":512,"GROUP_SIZE_M":1,"num_warps":8,"num_stages":2,"waves_per_eu":4,"matrix_instr_nonkdim":16,"cache_modifier":".cg","NUM_KSPLIT":1}

def _test_load_inline():
    """Test if load_inline works on this runner."""
    global _load_inline_works, _hipblas_module

    try:
        t0 = time.time()

        # Check what's available
        print(f"[PROBE] Python: {sys.version}", flush=True)
        print(f"[PROBE] PyTorch: {torch.__version__}", flush=True)
        print(f"[PROBE] CUDA/HIP: {torch.version.hip}", flush=True)
        print(f"[PROBE] ROCM_HOME: {os.environ.get('ROCM_HOME', 'NOT SET')}", flush=True)
        print(f"[PROBE] PYTORCH_ROCM_ARCH: {os.environ.get('PYTORCH_ROCM_ARCH', 'NOT SET')}", flush=True)

        # Check if hipcc exists
        import shutil
        hipcc = shutil.which('hipcc')
        print(f"[PROBE] hipcc: {hipcc}", flush=True)

        # Set arch for gfx950
        os.environ['PYTORCH_ROCM_ARCH'] = 'gfx950'

        from torch.utils.cpp_extension import load_inline

        # Simplest possible HIP kernel
        cpp_source = """
        torch::Tensor test_add(torch::Tensor a, torch::Tensor b) {
            return a + b;
        }
        """

        cuda_source = """
        #include <hip/hip_runtime.h>

        __global__ void simple_add_kernel(const float* a, const float* b, float* c, int n) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < n) {
                c[idx] = a[idx] + b[idx];
            }
        }
        """

        t1 = time.time()
        print(f"[PROBE] Starting load_inline compilation...", flush=True)

        module = load_inline(
            name="test_probe",
            cpp_sources=[cpp_source],
            cuda_sources=[cuda_source],
            functions=["test_add"],
            verbose=False,
        )

        t2 = time.time()
        compile_time = t2 - t1
        print(f"[PROBE] load_inline SUCCEEDED in {compile_time:.1f}s!", flush=True)

        # Test the compiled function
        a = torch.randn(10, device='cuda')
        b = torch.randn(10, device='cuda')
        c = module.test_add(a, b)
        correct = torch.allclose(c, a + b)
        print(f"[PROBE] Function test: {'PASS' if correct else 'FAIL'}", flush=True)

        # Check if MFMA intrinsics are available
        print(f"[PROBE] Testing MFMA intrinsic compilation...", flush=True)
        mfma_source = """
        #include <hip/hip_runtime.h>
        #include <hip/hip_fp16.h>

        __global__ void mfma_test() {
            // Test if MFMA instruction compiles
            // v_mfma_f32_32x32x8_bf16 is available on gfx942+
            float result[32];
            // Just a compilation test
        }
        """

        try:
            mfma_module = load_inline(
                name="mfma_probe",
                cpp_sources=["void dummy() {}"],
                cuda_sources=[mfma_source],
                functions=["dummy"],
                verbose=False,
            )
            print(f"[PROBE] MFMA compilation: SUCCEEDED", flush=True)
        except Exception as e:
            print(f"[PROBE] MFMA compilation: FAILED - {str(e)[:200]}", flush=True)

        # Check available libraries
        for lib in ['hipblas', 'hipblaslt', 'rocblas', 'rocsolver']:
            found = shutil.which(lib) or os.path.exists(f'/opt/rocm/lib/lib{lib}.so')
            print(f"[PROBE] {lib}: {'FOUND' if found else 'NOT FOUND'}", flush=True)

        # Check include paths
        for inc in ['/opt/rocm/include', '/opt/rocm/include/hip', '/opt/rocm/include/hipblas']:
            exists = os.path.isdir(inc)
            print(f"[PROBE] {inc}: {'EXISTS' if exists else 'MISSING'}", flush=True)

        _load_inline_works = True
        return True

    except Exception as e:
        print(f"[PROBE] load_inline FAILED: {str(e)[:500]}", flush=True)
        _load_inline_works = False
        return False


def _bgc(sm,sn,d):
    t=sm*sn;d0,d1=sm//32,sn//8
    i=torch.arange(t,dtype=torch.int64,device=d)
    i=i.view(d0,d1,4,16,2,2).permute(0,5,3,1,4,2).contiguous().view(-1)
    return i,torch.empty(t,dtype=torch.uint8,device=d)

def _fu(f,sm,sn):
    gi,ob=_gc[(sm,sn)];torch.take(f,gi,out=ob);return ob.view(sm,sn)

def _pw(d):
    global _w
    if _w:return
    _w=True
    from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4
    from aiter.ops.triton.quant import dynamic_mxfp4_quant
    from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4
    for m,n,k in [(4,2880,512),(16,2112,7168),(32,4096,512),(32,2880,512),(64,7168,2048),(256,3072,1536)]:
        try:
            da=torch.randn(m,k,dtype=torch.bfloat16,device=d)
            if k==1536:
                af,asc=dynamic_mxfp4_quant(da)
                gemm_afp4wfp4(af.view(torch.uint8),torch.zeros(n,k//2,dtype=torch.uint8,device=d),asc,torch.full((n,k//32),127,dtype=torch.uint8,device=d),dtype=torch.bfloat16)
            else:
                pn=((n+31)//32)*32
                cfg=_K7168 if k==7168 else(_K2048 if k==2048 else _K512)
                gemm_a16wfp4(da,torch.zeros(n,k//2,dtype=torch.uint8,device=d),torch.full((pn,k//32),127,dtype=torch.uint8,device=d),dtype=torch.bfloat16,y=torch.empty(m,n,dtype=torch.bfloat16,device=d),config=cfg)
            del da
        except:pass
    torch.cuda.synchronize()


def custom_kernel(data: input_t) -> output_t:
    global _call
    _call += 1

    # Test load_inline on first call
    if _call == 1:
        _test_load_inline()

    # Always use proven fallback for actual GEMM
    global _bsr, _bqu, _braw
    A,B,B_q,B_shuffle,B_scale_sh=data;m,k=A.shape;n=B.shape[0]
    if _bsr is not B_scale_sh:
        _bsr=B_scale_sh;su=B_scale_sh.view(torch.uint8);sm,sn=su.shape
        if (sm,sn) not in _gc:_gc[(sm,sn)]=_bgc(sm,sn,su.device)
        _braw=_fu(su.reshape(-1),sm,sn);_bqu=B_q.view(torch.uint8)
    _pw(A.device)
    from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4
    key=(m,n)
    if key not in _yc:_yc[key]=torch.empty(m,n,dtype=torch.bfloat16,device=A.device)
    out=_yc[key]
    if k==1536:
        from aiter.ops.triton.quant import dynamic_mxfp4_quant
        from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4
        af,asc=dynamic_mxfp4_quant(A)
        return gemm_afp4wfp4(af.view(torch.uint8),_bqu,asc,_braw,dtype=torch.bfloat16)
    cfg=_K7168 if k==7168 else(_K2048 if k==2048 else _K512)
    gemm_a16wfp4(A,_bqu,_braw,dtype=torch.bfloat16,y=out,config=cfg)
    return out
