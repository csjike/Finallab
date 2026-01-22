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
triton.Config({"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32}),
triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32}),
triton.Config({"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32}),
triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32}),
triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32}),
]
AUTOTUNE_CONFIGS_BASE = [
    triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64}),
]
@triton.jit
def leaky_relu_custom(x):
    return tl.where(x >= 0, x, 0.01 * x) + 1.0

@triton.jit
def relu_custom(x):
    pos = tl.maximum(x, 0.0)
    neg = tl.minimum(x, 0.0)
    return pos + 0.01 * neg + 1.0

def torch_matmul(a, b, activation=""):
    c = torch.matmul(a, b)
    if activation == "leaky_relu_custom":
        c = torch.where(c >= 0, c, 0.01 * c) + 1.0
    return c


@triton.jit
def matmul_splitk_stage1(
    a_ptr, b_ptr, c_partial_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_pk, stride_pm, stride_pn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    K_SPLITS: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    num_k_tiles = tl.cdiv(K, BLOCK_K)

    # split-K 核心：stride = K_SPLITS
    for k in range(pid_k, num_k_tiles, K_SPLITS):
        a_ptrs = a_ptr + offs_m[:, None] * stride_am + (k * BLOCK_K + offs_k)[None, :] * stride_ak
        b_ptrs = b_ptr + (k * BLOCK_K + offs_k)[:, None] * stride_bk + offs_n[None, :] * stride_bn

        a = tl.load(
            a_ptrs,
            mask=(offs_m[:, None] < M) & (k * BLOCK_K + offs_k[None, :] < K),
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=(offs_n[None, :] < N) & (k * BLOCK_K + offs_k[:, None] < K),
            other=0.0,
        )

        acc = tl.dot(a, b, acc)

    # 写 partial
    # compute base offset (in elements) for this (pid_k, pid_m, pid_n)
    base_off = pid_k * stride_pk + pid_m * stride_pm + pid_n * stride_pn
    c_ptrs = c_partial_ptr + base_off + offs_m[:, None] * stride_pm + offs_n[None, :] * stride_pn

    tl.store(
        c_ptrs,
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )

@triton.jit
def matmul_splitk_stage2(
    c_partial_ptr, c_ptr,
    M, N,
    stride_pk, stride_pm, stride_pn,
    stride_cm, stride_cn,
    K_SPLITS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(K_SPLITS):
        # compute base offset for this k,pid_m,pid_n
        base_off = k * stride_pk + pid_m * stride_pm + pid_n * stride_pn
        ptrs = c_partial_ptr + base_off + offs_m[:, None] * stride_pm + offs_n[None, :] * stride_pn
        acc += tl.load(ptrs, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N), other=0.0)

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(
        c_ptrs,
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )




def create_kernel_SplitK(
    name: str,
    autotune_configs: List[triton.Config],
    parallel: bool,
    k_splits: int = 1,
) -> Callable:
    """
    Returns a callable that performs matmul.
    If k_splits > 1, uses true two-stage split-K.
    """

    # =========================
    # Stage 1: split-K compute
    # =========================
    @triton.autotune(
        configs=autotune_configs,
        key=["M", "N", "K"],
    )
    @triton.jit
    def _stage1_kernel(
        a_ptr, b_ptr, c_partial_ptr,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_pk, stride_pm, stride_pn,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
        K_SPLITS: tl.constexpr,
    ):
        matmul_splitk_stage1(
            a_ptr, b_ptr,
            c_partial_ptr,
            M, N, K,
            stride_am, stride_ak,
            stride_bk, stride_bn,
            stride_pk, stride_pm, stride_pn,
            BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
            K_SPLITS,
        )

    # =========================
    # Stage 2: reduce kernel
    # =========================
    @triton.jit
    def _stage2_kernel(
        c_partial_ptr, c_ptr,
        M, N,
        stride_pk, stride_pm, stride_pn,
        stride_cm, stride_cn,
        K_SPLITS: tl.constexpr,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
    ):
        matmul_splitk_stage2(
            c_partial_ptr,
            c_ptr,
            M, N,
            stride_pk, stride_pm, stride_pn,
            stride_cm, stride_cn,
            K_SPLITS,
            BLOCK_SIZE_M, BLOCK_SIZE_N,
        )

    # =========================
    # Python wrapper (关键)
    # =========================
    def kernel(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor):
        #assert a.is_cuda and b.is_cuda and c.is_cuda
        assert a.dtype == b.dtype

        M, K = a.shape
        K2, N = b.shape
        assert K == K2

        # strides (in elements)
        stride_am, stride_ak = a.stride()
        stride_bk, stride_bn = b.stride()
        stride_cm, stride_cn = c.stride()

        BLOCK_M = autotune_configs[0].kwargs["BLOCK_SIZE_M"]
        BLOCK_N = autotune_configs[0].kwargs["BLOCK_SIZE_N"]
        BLOCK_K = autotune_configs[0].kwargs["BLOCK_SIZE_K"]

        grid_m = triton.cdiv(M, BLOCK_M)
        grid_n = triton.cdiv(N, BLOCK_N)

        # ==============
        # No split-K
        # ==============
        if k_splits == 1:
            grid = (grid_m, grid_n)
            _stage1_kernel[grid](
                a, b, c,
                M=M, N=N, K=K,
                stride_am=stride_am, stride_ak=stride_ak,
                stride_bk=stride_bk, stride_bn=stride_bn,
                stride_pk=0, stride_pm=stride_cm, stride_pn=stride_cn,
                K_SPLITS=1,
            )
            return

        # ===================
        # True split-K path
        # ===================
        # allocate partial buffer: [K_SPLITS, M, N] in blocks
        c_partial = torch.empty(
            (k_splits, M, N),
            device=c.device,
            dtype=torch.float32,
        )

        stride_pk, stride_pm, stride_pn = c_partial.stride()

        # ---- Stage 1 ----
        grid = (grid_m, grid_n, k_splits)
        _stage1_kernel[grid](
            a, b, c_partial,
            M=M, N=N, K=K,
            stride_am=stride_am, stride_ak=stride_ak,
            stride_bk=stride_bk, stride_bn=stride_bn,
            stride_pk=stride_pk, stride_pm=stride_pm, stride_pn=stride_pn,
            K_SPLITS=k_splits,
        )

        # ---- Stage 2 ----
        grid = (grid_m, grid_n)
        _stage2_kernel[grid](
            c_partial, c,
            M, N,
            stride_pk, stride_pm, stride_pn,
            stride_cm, stride_cn,
            k_splits,
            BLOCK_M, BLOCK_N,
        )

    kernel.parallel_flag = parallel
    kernel.k_splits = k_splits
    kernel.__name__ = name

    return kernel


def matmul_wrapper_splitK(a, b, kernel_func):
    """
    kernel_func is a Python callable returned by create_kernel_SplitK
    """
    assert a.shape[1] == b.shape[0]

    M, K = a.shape
    _, N = b.shape

    # 输出张量
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)

    # 直接调用 Python wrapper
    kernel_func(a, b, c)

    return c

