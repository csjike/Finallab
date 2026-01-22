import triton
import triton.language as tl
import torch
import torch_npu
import triton.testing
import time
import pandas as pd
from typing import Dict, List, Callable

DEV = "npu"
activation = "leaky_relu_custom"

AUTOTUNE_CONFIGS_FULL = [
    triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32}),  
    triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64}), 
    triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 128}),
    triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64}),
    triton.Config({"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 128}), 
    triton.Config({"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 64}),
]
AUTOTUNE_CONFIGS_BASE = [
    triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64}),
]

MATRIX_SHAPES = [
    (512, 512, 512),       # Small Square
    (2048, 2048, 2048),    # Medium Square
    (4096, 4096, 4096),    # Large Square
    (2048, 1024, 4096),    # Rectangular (K-dominant)
    (4096, 4096, 1024),    # Rectangular (M, N dominant)
]
NUM_TEST_RUNS = 100
NUM_WARMUP_RUNS = 20

@triton.jit
def leaky_relu_custom(x):
    return tl.where(x >= 0, x, 0.01 * x) + 1.0

def torch_matmul(a, b, activation=""):
    c = torch.matmul(a, b)
    if activation == "leaky_relu_custom":
        c = torch.where(c >= 0, c, 0.01 * c) + 1.0
    return c

@triton.jit
def _matmat_core_logic(
    a_ptr, b_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    """Calculates the dot product for one block and returns the accumulator and PIDs."""
    GROUP_SIZE_M: tl.constexpr = 1
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M) 
    
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    a_ptrs_base = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs_base = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    msk_m = offs_am < M
    msk_n = offs_bn < N
    
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a_ptrs = a_ptrs_base + k * BLOCK_SIZE_K * stride_ak
        b_ptrs = b_ptrs_base + k * BLOCK_SIZE_K * stride_bk
        
        a = tl.load(
            a_ptrs,
            mask=msk_m[:, None] & (offs_k[None, :] < K - k * BLOCK_SIZE_K),
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=msk_n[None, :] & (offs_k[:, None] < K - k * BLOCK_SIZE_K),
            other=0.0,
        )
        accumulator = tl.dot(a, b, accumulator)
    
    return accumulator, pid_m, pid_n, msk_m, msk_n

@triton.jit
def _matmat_single_writeback(
    c_ptr, accumulator, pid_m, pid_n, M, N, stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, ACTIVATION: tl.constexpr
):
    """Writes back C without NPU vector core parallelism."""
    if ACTIVATION == "leaky_relu_custom":
        accumulator = leaky_relu_custom(accumulator)
    c = accumulator.to(tl.float16)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)

@triton.jit
def _matmat_parallel_writeback(
    c_ptr, accumulator, pid_m, pid_n, M, N, stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, ACTIVATION: tl.constexpr
):
    """Writes back C with NPU vector core parallelism (tl.parallel)."""
    SUB_BLK_M: tl.constexpr = BLOCK_SIZE_M // 2
    for s in tl.parallel(0, 2, bind_sub_block=True):
        vec_sub_blk = tl.extract_slice(
            accumulator, (s * SUB_BLK_M, 0), (SUB_BLK_M, BLOCK_SIZE_N), (1, 1)
        )
        if ACTIVATION == "leaky_relu_custom":
            vec_sub_blk = leaky_relu_custom(vec_sub_blk)
        c_sub_blk = vec_sub_blk.to(tl.float16)

        offs_cm = pid_m * BLOCK_SIZE_M + s * SUB_BLK_M + tl.arange(0, SUB_BLK_M)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        tl.store(c_ptrs, c_sub_blk, mask=c_mask)

def create_kernel(name: str, autotune_configs: List[triton.Config], parallel: bool) -> Callable:
    """Dynamically creates a JIT kernel with specified autotune configs and parallelism."""
    
    @triton.autotune(
        configs=autotune_configs,
        key=["M", "N", "K"],
    )
    @triton.jit
    def kernel( 
        a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, ACTIVATION: tl.constexpr, 
        PARALLEL: tl.constexpr
    ):
        accumulator, pid_m, pid_n, msk_m, msk_n = _matmat_core_logic(
            a_ptr, b_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn,
            BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K
        )

        if PARALLEL:
            _matmat_parallel_writeback(
                c_ptr, accumulator, pid_m, pid_n, M, N, stride_cm, stride_cn,
                BLOCK_SIZE_M, BLOCK_SIZE_N, ACTIVATION
            )
        else:
            _matmat_single_writeback(
                c_ptr, accumulator, pid_m, pid_n, M, N, stride_cm, stride_cn,
                BLOCK_SIZE_M, BLOCK_SIZE_N, ACTIVATION
            )
    
    kernel.parallel_flag = parallel 
    
    return kernel

KERNEL_V1 = create_kernel("V1_Baseline", AUTOTUNE_CONFIGS_BASE, parallel=False)
KERNEL_V2 = create_kernel("V2_Opt_Autotune", AUTOTUNE_CONFIGS_FULL, parallel=False)
KERNEL_V3 = create_kernel("V3_Opt_NPU_Parallel", AUTOTUNE_CONFIGS_BASE, parallel=True)
KERNEL_V4 = create_kernel("V4_Opt_Full", AUTOTUNE_CONFIGS_FULL, parallel=True)

def matmul_wrapper(a, b, kernel_func):
    """Generic Python wrapper to launch a Matmul Triton kernel."""
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    
    M, K = a.shape 
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    
    is_parallel = getattr(kernel_func, 'parallel_flag', False)
    
    kernel_func[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        ACTIVATION=activation,
        PARALLEL=is_parallel
    )
    return c

EXPERIMENT_VERSIONS = {
    "V1_Baseline (Base Config | No Parallel)": KERNEL_V1,
    "V2_Opt_Autotune (Full Config | No Parallel)": KERNEL_V2,
    "V3_Opt_NPU_Parallel (Base Config | Parallel)": KERNEL_V3,
    "V4_Opt_Full (Full Config | Parallel)": KERNEL_V4,
}

def run_performance_test():
    torch.npu.set_device(0)
    torch.manual_seed(0)

    all_results = []
    
    start_event = torch.npu.Event(enable_timing=True)
    end_event = torch.npu.Event(enable_timing=True)
    
    print("--- Triton 算子优化多版本性能对比实验开始 ---")

    for M, K, N in MATRIX_SHAPES:
        print(f"\n--- 测试形状: M={M}, K={K}, N={N} ---")
        
        a = torch.randn((M, K), device=DEV, dtype=torch.float16)
        b = torch.randn((K, N), device=DEV, dtype=torch.float16)
        
        print(f"  > 预热 ({NUM_WARMUP_RUNS}次)...")
        for _ in range(NUM_WARMUP_RUNS):
            torch_matmul(a, b, activation)
            for kernel_func in EXPERIMENT_VERSIONS.values():
                 matmul_wrapper(a, b, kernel_func)
        torch.npu.synchronize()
        print("  > 预热完成。")

        shape_results = {"Shape": f"{M}x{K}x{N}"}
        reference_output = None
        
        for name, kernel_func in EXPERIMENT_VERSIONS.items():
            times_ms = []
            
            for _ in range(NUM_TEST_RUNS):
                start_event.record()
                output = matmul_wrapper(a, b, kernel_func) 
                end_event.record()
                end_event.synchronize()
                times_ms.append(start_event.elapsed_time(end_event))
            
            avg_time_ms = sum(times_ms) / NUM_TEST_RUNS
            shape_results[name] = f"{avg_time_ms:.3f} ms"
            
            if "V1_Baseline" in name:
                reference_output = output 
            
            print(f"  - {name}: {avg_time_ms:.3f} ms")


        torch_output_ref = torch_matmul(a, b, activation) 
        
        TOLERANCE_ATOL = 1e-2 
        TOLERANCE_RTOL = 1e-2
        is_close = torch.allclose(reference_output, torch_output_ref, atol=TOLERANCE_ATOL, rtol=TOLERANCE_RTOL, equal_nan=False)

        if not is_close:
            print(f"  警告: Triton 基线结果与 PyTorch 参照在容忍度 ({TOLERANCE_ATOL}) 外不一致。")
        else:
            print(f"  校验成功: Triton 基线结果与 PyTorch 参照一致。")

        all_results.append(shape_results)

    print("\n" + "=" * 100)
    print("Triton 算子优化性能量化结果")
    print("=" * 100)
    print("下表展示了四种优化策略在不同矩阵形状下的平均运行时间(ms)")
    
    df = pd.DataFrame(all_results)
    
    v1_times = df["V1_Baseline (Base Config | No Parallel)"].str.replace(' ms', '').astype(float)
    v4_times = df["V4_Opt_Full (Full Config | Parallel)"].str.replace(' ms', '').astype(float)
    df['V4/V1 提速比'] = (v1_times / v4_times).apply(lambda x: f"{x:.2f}x")
    
    df = df[['Shape', 
             'V1_Baseline (Base Config | No Parallel)', 
             'V2_Opt_Autotune (Full Config | No Parallel)', 
             'V3_Opt_NPU_Parallel (Base Config | Parallel)', 
             'V4_Opt_Full (Full Config | Parallel)', 
             'V4/V1 提速比']]

    print(df.to_string(index=False))


if __name__ == '__main__':
    print("通过四个版本对比 Autotune 和 NPU 向量核并行优化的效果")
    print("V1: 基线 | V2: Autotune | V3: NPU Parallel | V4: Autotune + NPU Parallel")
    run_performance_test()