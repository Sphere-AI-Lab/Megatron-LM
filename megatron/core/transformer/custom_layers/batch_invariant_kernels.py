# Copyright 2025 Thinking Machines Lab
# The following code has been adapted
# from the following repo: https://github.com/thinking-machines-lab/batch_invariant_ops


import contextlib
import importlib
import importlib.util
import logging
from collections import namedtuple
from collections.abc import Callable
from typing import Any, Dict, List, Optional

import torch

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:
    from unittest.mock import MagicMock

    from megatron.core.utils import null_decorator

    triton = MagicMock()
    triton.jit = null_decorator
    tl = MagicMock()
    HAVE_TRITON = False

__all__ = [
    "set_batch_invariant_mode",
    "is_batch_invariant_mode_enabled",
    "disable_batch_invariant_mode",
    "enable_batch_invariant_mode",
    "matmul_fixed_order",
    "LinearFixedOrderFn",
]


_LOGGER = logging.getLogger(__name__)


def _matmul_launch_metadata(
    grid: Callable[..., Any], kernel: Any, args: Dict[str, Any]
) -> Dict[str, Any]:
    """Build launch metadata for Triton matmul kernels used in BIK matmul."""
    ret = {}
    m, n, k = args["M"], args["N"], args["K"]
    ret["name"] = f"{kernel.name} [M={m}, N={n}, K={k}]"
    if "tiles_per_update" in args:
        ret["name"] = (
            f"{kernel.name} [M={m}, N={n}, K={k}, tiles_per_update={args['tiles_per_update']:02}]"
        )
    if "c_ptr" in args:
        bytes_per_elem = args["c_ptr"].element_size()
    else:
        bytes_per_elem = 1 if args["FP8_OUTPUT"] else 2
    ret[f"flops{bytes_per_elem * 8}"] = 2.0 * m * n * k
    ret["bytes"] = bytes_per_elem * (m * k + n * k + m * n)
    return ret


@triton.jit
def _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS):
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (tile_id % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    return pid_m, pid_n


@triton.jit(launch_metadata=_matmul_launch_metadata)
def matmul_kernel_persistent(
    a_ptr,
    b_ptr,
    c_ptr,  #
    bias_ptr,
    M,
    N,
    K,  #
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,  #
    BLOCK_SIZE_N: tl.constexpr,  #
    BLOCK_SIZE_K: tl.constexpr,  #
    GROUP_SIZE_M: tl.constexpr,  #
    NUM_SMS: tl.constexpr,  #
    A_LARGE: tl.constexpr,
    B_LARGE: tl.constexpr,
    C_LARGE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    """Persistent matmul Triton kernel backing `matmul_persistent`."""
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n

    tile_id_c = start_pid - NUM_SMS

    offs_k_for_mask = tl.arange(0, BLOCK_SIZE_K)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, flatten=True):
        pid_m, pid_n = _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS)
        start_m = pid_m * BLOCK_SIZE_M
        start_n = pid_n * BLOCK_SIZE_N
        offs_am = start_m + tl.arange(0, BLOCK_SIZE_M)
        offs_bn = start_n + tl.arange(0, BLOCK_SIZE_N)
        if A_LARGE:
            offs_am = offs_am.to(tl.int64)
        if B_LARGE:
            offs_bn = offs_bn.to(tl.int64)
        offs_am = tl.where(offs_am < M, offs_am, 0)
        offs_bn = tl.where(offs_bn < N, offs_bn, 0)
        offs_am = tl.max_contiguous(tl.multiple_of(offs_am, BLOCK_SIZE_M), BLOCK_SIZE_M)
        offs_bn = tl.max_contiguous(tl.multiple_of(offs_bn, BLOCK_SIZE_N), BLOCK_SIZE_N)

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for ki in range(k_tiles):
            if A_LARGE or B_LARGE:
                offs_k = ki * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K).to(tl.int64)
            else:
                offs_k = ki * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
            a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
            b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

            a = tl.load(a_ptrs, mask=offs_k_for_mask[None, :] < K - ki * BLOCK_SIZE_K, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k_for_mask[:, None] < K - ki * BLOCK_SIZE_K, other=0.0)
            accumulator = tl.dot(a, b, accumulator)

        tile_id_c += NUM_SMS
        pid_m, pid_n = _compute_pid(tile_id_c, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS)
        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        if C_LARGE:
            offs_cm = offs_cm.to(tl.int64)
            offs_cn = offs_cn.to(tl.int64)
        c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        if HAS_BIAS:
            bias_ptrs = bias_ptr + offs_cn
            bias = tl.load(bias_ptrs, mask=offs_cn < N, other=0.0).to(tl.float32)
            accumulator += bias
        c = accumulator.to(c_ptr.dtype.element_ty)
        tl.store(c_ptrs, c, mask=c_mask)


def get_compute_units():
    """
    Returns the number of streaming multiprocessors (SMs) or equivalent compute units
    for the available accelerator. Assigns the value to NUM_SMS.
    """
    NUM_SMS = None
    device_type = getattr(torch.accelerator.current_accelerator(), "type", "cpu")

    # Use match/case for device-specific logic (Python 3.10+)
    match device_type:
        case "cuda":
            device_properties = torch.cuda.get_device_properties(0)
            NUM_SMS = device_properties.multi_processor_count
        case "xpu":
            device_properties = torch.xpu.get_device_properties(0)
            NUM_SMS = device_properties.max_compute_units
        case _:
            _LOGGER.warning("No CUDA or XPU device available. Using CPU.")
            # For CPU, you might want to use the number of CPU cores
            NUM_SMS = torch.get_num_threads()

    return NUM_SMS


def matmul_persistent(a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor | None = None):
    """Persistent matmul kernel used by batch-invariant GEMM."""
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.dtype == b.dtype, "Incompatible dtypes"
    assert (
        bias is None or bias.dim() == 1
    ), "Currently assuming bias is 1D, let Horace know if you run into this"

    NUM_SMS = get_compute_units()
    M, K = a.shape
    K, N = b.shape
    dtype = a.dtype
    # Allocates output.
    c = torch.empty((M, N), device=a.device, dtype=dtype)

    # 1D launch kernel where each block gets its own program.
    def grid(META):
        blocks_m = triton.cdiv(M, META["BLOCK_SIZE_M"])
        blocks_n = triton.cdiv(N, META["BLOCK_SIZE_N"])
        return (min(NUM_SMS, blocks_m * blocks_n),)

    configs = {
        torch.bfloat16: {
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 8,
            "num_stages": 3,
            "num_warps": 8,
        },
        torch.float16: {
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": 256,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 8,
            "num_stages": 3,
            "num_warps": 8,
        },
        torch.float32: {
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 32,
            "GROUP_SIZE_M": 8,
            "num_stages": 3,
            "num_warps": 8,
        },
    }
    matmul_kernel_persistent[grid](
        a,
        b,
        c,  #
        bias,
        M,
        N,
        K,  #
        a.stride(0),
        a.stride(1),  #
        b.stride(0),
        b.stride(1),  #
        c.stride(0),
        c.stride(1),  #
        NUM_SMS=NUM_SMS,  #
        A_LARGE=a.numel() > 2**31,
        B_LARGE=b.numel() > 2**31,
        C_LARGE=c.numel() > 2**31,
        HAS_BIAS=bias is not None,
        **configs[dtype],
    )
    return c


@triton.jit(launch_metadata=_matmul_launch_metadata)
def matmul_kernel_fixed_order(
    a_ptr,
    b_ptr,
    c_ptr,
    bias_ptr,
    M,
    K: tl.constexpr,
    N: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    """Fixed-order fp32 matmul kernel for DSV4 batch-invariant projections."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_SIZE_K):
        k_idxs = k0 + offs_k
        a = tl.load(
            a_ptr + offs_m[:, None] * K + k_idxs[None, :],
            mask=(offs_m[:, None] < M) & (k_idxs[None, :] < K),
            other=0.0,
        )
        b = tl.load(
            b_ptr + k_idxs[:, None] * N + offs_n[None, :],
            mask=(k_idxs[:, None] < K) & (offs_n[None, :] < N),
            other=0.0,
        )
        accumulator += tl.dot(a, b, input_precision="ieee")

    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
        accumulator += bias

    tl.store(
        c_ptr + offs_m[:, None] * N + offs_n[None, :],
        accumulator,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


def matmul_fixed_order(
    a: torch.Tensor,
    b: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    block_m: int | None = None,
    block_n: int | None = None,
    block_k: int = 64,
) -> torch.Tensor:
    """Batch-invariant fp32 matmul with a fixed K-reduction order."""
    assert a.dim() == 2 and b.dim() == 2, "matmul_fixed_order expects 2D tensors"
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.dtype == torch.float32 and b.dtype == torch.float32, (
        "matmul_fixed_order currently supports fp32 inputs only"
    )
    assert a.is_contiguous() and b.is_contiguous(), "inputs must be contiguous"
    assert bias is None or (bias.dim() == 1 and bias.dtype == torch.float32)

    M, K = a.shape
    _, N = b.shape
    if block_m is None and block_n is None:
        # Keep the faster small-row tile for stateful decode, and use a wider
        # N tile for prefill where compressor GEMMs dominate. The c4
        # compressor has N=1024 and benefits earlier than the c128 N=512 path.
        use_small_row_tile = M < 1280 if N <= 512 else M <= 512
        block_m, block_n = (128, 32) if use_small_row_tile else (64, 128)
    else:
        block_m = 128 if block_m is None else block_m
        block_n = 32 if block_n is None else block_n
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n))
    matmul_kernel_fixed_order[grid](
        a,
        b,
        c,
        bias,
        M,
        K,
        N,
        BLOCK_SIZE_M=block_m,
        BLOCK_SIZE_N=block_n,
        BLOCK_SIZE_K=block_k,
        HAS_BIAS=bias is not None,
        num_stages=3,
        num_warps=8,
    )
    return c


class LinearFixedOrderFn(torch.autograd.Function):
    """Fixed-order fp32 linear forward with torch-equivalent GEMM backward."""

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if x.dtype != torch.float32 or weight.dtype != torch.float32:
            raise RuntimeError("LinearFixedOrderFn requires fp32 input and weight.")
        if weight.dim() != 2:
            raise RuntimeError("LinearFixedOrderFn requires a 2D weight.")

        x_2d = x.reshape(-1, x.shape[-1]).contiguous()
        weight_2d = weight.contiguous()
        out_2d = matmul_fixed_order(x_2d, weight_2d.t().contiguous(), bias=bias)

        ctx.input_shape = x.shape
        ctx.has_bias = bias is not None
        ctx.save_for_backward(x_2d, weight_2d)
        return out_2d.reshape(*x.shape[:-1], weight.shape[0])

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x_2d, weight = ctx.saved_tensors
        grad_2d = grad_output.reshape(-1, grad_output.shape[-1]).contiguous()

        grad_x = grad_weight = grad_bias = None
        if ctx.needs_input_grad[0]:
            grad_x = grad_2d.matmul(weight).reshape(ctx.input_shape)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_2d.transpose(0, 1).matmul(x_2d)
        if ctx.has_bias and ctx.needs_input_grad[2]:
            grad_bias = grad_2d.sum(dim=0)

        return grad_x, grad_weight, grad_bias


@triton.jit
def _rms_norm_forward_rsigma_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    rsigma_ptr,
    input_row_stride: tl.constexpr,
    output_row_stride: tl.constexpr,
    n_cols: tl.constexpr,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0).to(tl.int64)
    row_start_ptr = input_ptr + row_idx * input_row_stride
    output_row_start_ptr = output_ptr + row_idx * output_row_stride

    sum_sq = tl.full((), 0.0, dtype=tl.float32)
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = col_idx < n_cols
        vals = tl.load(row_start_ptr + col_idx, mask=mask, other=0.0).to(tl.float32)
        sum_sq += tl.sum(tl.where(mask, vals * vals, 0.0))

    inv_rms = 1.0 / tl.sqrt(sum_sq / n_cols + eps)
    tl.store(rsigma_ptr + row_idx, inv_rms)

    for col_offset in range(0, n_cols, BLOCK_SIZE):
        col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = col_idx < n_cols
        vals = tl.load(row_start_ptr + col_idx, mask=mask, other=0.0)
        weight_vals = tl.load(weight_ptr + col_idx, mask=mask, other=1.0)
        out = vals.to(tl.float32) * inv_rms * weight_vals.to(tl.float32)
        tl.store(output_row_start_ptr + col_idx, out.to(vals.dtype), mask=mask)


def _rms_norm_with_rsigma(
    input_: torch.Tensor, weight: torch.Tensor, eps: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """RMSNorm forward that also returns per-row inverse RMS for backward."""
    assert weight.dim() == 1, "Weight must be 1-dimensional"
    assert input_.shape[-1] == weight.shape[0], (
        f"Input last dimension ({input_.shape[-1]}) must match "
        f"weight dimension ({weight.shape[0]})"
    )
    original_shape = input_.shape
    input_2d = input_.reshape(-1, input_.shape[-1]).contiguous()
    weight = weight.contiguous()
    n_rows, n_cols = input_2d.shape
    output = torch.empty_like(input_2d)
    rsigma = torch.empty((n_rows,), device=input_.device, dtype=torch.float32)
    _rms_norm_forward_rsigma_kernel[(n_rows,)](
        input_2d,
        weight,
        output,
        rsigma,
        input_2d.stride(0),
        output.stride(0),
        n_cols,
        eps,
        BLOCK_SIZE=1024,
    )
    return output.reshape(original_shape), rsigma.reshape(*original_shape[:-1], 1)


@triton.jit
def _rms_norm_backward_row_dot_kernel(
    x_ptr,
    grad_out_ptr,
    weight_ptr,
    row_dot_ptr,
    x_row_stride: tl.constexpr,
    grad_out_row_stride: tl.constexpr,
    n_cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0).to(tl.int64)
    x_row_start_ptr = x_ptr + row_idx * x_row_stride
    grad_out_row_start_ptr = grad_out_ptr + row_idx * grad_out_row_stride

    acc = tl.full((), 0.0, dtype=tl.float32)
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = col_idx < n_cols
        x_vals = tl.load(x_row_start_ptr + col_idx, mask=mask, other=0.0).to(tl.float32)
        grad_vals = tl.load(grad_out_row_start_ptr + col_idx, mask=mask, other=0.0).to(
            tl.float32
        )
        weight_vals = tl.load(weight_ptr + col_idx, mask=mask, other=0.0).to(tl.float32)
        acc += tl.sum(tl.where(mask, x_vals * grad_vals * weight_vals, 0.0))

    tl.store(row_dot_ptr + row_idx, acc)


@triton.jit
def _rms_norm_backward_dx_dw_partial_kernel(
    x_ptr,
    grad_out_ptr,
    weight_ptr,
    rsigma_ptr,
    row_dot_ptr,
    dx_ptr,
    partial_dw_ptr,
    x_row_stride: tl.constexpr,
    grad_out_row_stride: tl.constexpr,
    dx_row_stride: tl.constexpr,
    partial_dw_row_stride: tl.constexpr,
    n_rows: tl.constexpr,
    n_cols: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row_block = tl.program_id(0).to(tl.int64)
    col_block = tl.program_id(1)
    rows = row_block * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = col_block * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (rows[:, None] < n_rows) & (cols[None, :] < n_cols)

    x_vals = tl.load(
        x_ptr + rows[:, None] * x_row_stride + cols[None, :],
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    grad_vals = tl.load(
        grad_out_ptr + rows[:, None] * grad_out_row_stride + cols[None, :],
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    weight_vals = tl.load(weight_ptr + cols, mask=cols < n_cols, other=0.0).to(tl.float32)
    rsigma_vals = tl.load(rsigma_ptr + rows, mask=rows < n_rows, other=0.0).to(tl.float32)
    row_dot_vals = tl.load(row_dot_ptr + rows, mask=rows < n_rows, other=0.0).to(tl.float32)

    rsigma_2d = rsigma_vals[:, None]
    dx_vals = (
        grad_vals * weight_vals[None, :] * rsigma_2d
        - x_vals * (rsigma_2d * rsigma_2d * rsigma_2d) * row_dot_vals[:, None] / n_cols
    )
    tl.store(
        dx_ptr + rows[:, None] * dx_row_stride + cols[None, :],
        dx_vals,
        mask=mask,
    )

    partial_dw = tl.sum(tl.where(mask, grad_vals * x_vals * rsigma_2d, 0.0), axis=0)
    tl.store(
        partial_dw_ptr + row_block * partial_dw_row_stride + cols,
        partial_dw,
        mask=cols < n_cols,
    )


@triton.jit
def _rms_norm_backward_dw_reduce_kernel(
    partial_dw_ptr,
    grad_weight_ptr,
    partial_dw_row_stride: tl.constexpr,
    n_row_blocks: tl.constexpr,
    n_cols: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    col_block = tl.program_id(0)
    cols = col_block * BLOCK_N + tl.arange(0, BLOCK_N)
    row_offsets = tl.arange(0, BLOCK_R)
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for row_start in range(0, n_row_blocks, BLOCK_R):
        rows = row_start + row_offsets
        vals = tl.load(
            partial_dw_ptr + rows[:, None] * partial_dw_row_stride + cols[None, :],
            mask=(rows[:, None] < n_row_blocks) & (cols[None, :] < n_cols),
            other=0.0,
        )
        acc += tl.sum(vals, axis=0)

    tl.store(grad_weight_ptr + cols, acc, mask=cols < n_cols)


def _rms_norm_backward_fused(
    grad_output: torch.Tensor,
    x: torch.Tensor,
    weight_eff: torch.Tensor,
    rsigma: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    original_shape = x.shape
    x_2d = x.reshape(-1, x.shape[-1]).contiguous()
    grad_2d = grad_output.reshape(-1, grad_output.shape[-1]).contiguous()
    weight_eff = weight_eff.contiguous()
    rsigma_1d = rsigma.reshape(-1).contiguous()
    n_rows, n_cols = x_2d.shape

    if n_rows == 0:
        return torch.empty_like(x), torch.zeros_like(weight_eff)

    row_dot = torch.empty((n_rows,), device=x.device, dtype=torch.float32)
    dx_2d = torch.empty_like(x_2d)

    _rms_norm_backward_row_dot_kernel[(n_rows,)](
        x_2d,
        grad_2d,
        weight_eff,
        row_dot,
        x_2d.stride(0),
        grad_2d.stride(0),
        n_cols,
        BLOCK_SIZE=1024,
    )

    block_m = 16
    block_n = 256
    block_r = 64
    n_row_blocks = triton.cdiv(n_rows, block_m)
    partial_dw = torch.empty((n_row_blocks, n_cols), device=x.device, dtype=torch.float32)

    _rms_norm_backward_dx_dw_partial_kernel[
        (n_row_blocks, triton.cdiv(n_cols, block_n))
    ](
        x_2d,
        grad_2d,
        weight_eff,
        rsigma_1d,
        row_dot,
        dx_2d,
        partial_dw,
        x_2d.stride(0),
        grad_2d.stride(0),
        dx_2d.stride(0),
        partial_dw.stride(0),
        n_rows,
        n_cols,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=8,
    )

    grad_weight = torch.empty_like(weight_eff)
    _rms_norm_backward_dw_reduce_kernel[(triton.cdiv(n_cols, block_n),)](
        partial_dw,
        grad_weight,
        partial_dw.stride(0),
        n_row_blocks,
        n_cols,
        BLOCK_N=block_n,
        BLOCK_R=block_r,
        num_warps=8,
    )

    return dx_2d.reshape(original_shape), grad_weight


@triton.jit
def _log_softmax_kernel(
    input_ptr, output_ptr, input_row_stride, output_row_stride, n_cols, BLOCK_SIZE: tl.constexpr
):
    """
    Compute log_softmax along the last dimension of a 2D tensor.
    Each block handles one row of the input tensor.
    """
    # Get the row index for this block
    row_idx = tl.program_id(0).to(tl.int64)

    # Compute base pointers for input and output rows
    row_start_ptr = input_ptr + row_idx * input_row_stride
    output_row_start_ptr = output_ptr + row_idx * output_row_stride

    # Step 1: Find maximum value in the row for numerical stability
    max_val = -float("inf")
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = col_idx < n_cols

        # Load values
        vals = tl.load(row_start_ptr + col_idx, mask=mask, other=-float("inf"))

        # Update maximum
        max_val = tl.max(tl.maximum(vals, max_val))

    # Step 2: Compute sum of exp(x - max_val)
    sum_exp = 0.0
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = col_idx < n_cols

        # Load values
        vals = tl.load(row_start_ptr + col_idx, mask=mask, other=0.0)

        # Compute exp(x - max_val) and accumulate
        exp_vals = tl.exp(vals - max_val)
        sum_exp += tl.sum(tl.where(mask, exp_vals, 0.0))

    # Compute log(sum_exp)
    log_sum_exp = tl.log(sum_exp)

    # Step 3: Compute final log_softmax values: x - max_val - log_sum_exp
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = col_idx < n_cols

        # Load values
        vals = tl.load(row_start_ptr + col_idx, mask=mask)

        # Compute log_softmax
        output = vals - max_val - log_sum_exp

        # Store results
        tl.store(output_row_start_ptr + col_idx, output, mask=mask)


def log_softmax(input: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Compute log_softmax using Triton kernel.

    Args:
        input: Input tensor
        dim: Dimension along which to compute log_softmax (only -1 or last dim supported)
    >> Stashed changes
    Returns:
        Tensor with log_softmax applied along the specified dimension
    """
    if dim != -1 and dim != input.ndim - 1:
        raise ValueError("This implementation only supports log_softmax along the last dimension")
    # Flatten all dimensions except the last one
    original_shape = input.shape
    input_2d = input.reshape(-1, input.shape[-1])
    input_2d = input_2d.contiguous()

    n_rows, n_cols = input_2d.shape

    # Allocate output tensor
    output = torch.empty_like(input_2d)

    # Choose block size based on the number of columns
    BLOCK_SIZE = 1024

    # Launch kernel with one block per row
    grid = (n_rows,)
    _log_softmax_kernel[grid](
        input_2d, output, input_2d.stride(0), output.stride(0), n_cols, BLOCK_SIZE=BLOCK_SIZE
    )
    # Reshape output back to original shape
    return output.reshape(original_shape)


@triton.jit
def mean_kernel(
    input_ptr,
    output_ptr,
    input_stride0,
    input_stride1,
    input_stride2,
    output_stride0,
    output_stride1,
    M,  # size before reduction dim
    N,  # size of reduction dim
    K,  # size after reduction dim
    BLOCK_SIZE: tl.constexpr,
):
    """
    Kernel for computing mean along a single dimension.
    Input is viewed as (M, N, K) where N is the dimension being reduced.
    """
    # Program ID gives us which output element we're computing
    pid = tl.program_id(0)

    # Compute output indices
    m_idx = pid // K
    k_idx = pid % K

    # Bounds check
    if m_idx >= M or k_idx >= K:
        return

    # Accumulate sum across reduction dimension
    acc = 0.0
    for n_start in range(0, N, BLOCK_SIZE):
        n_offsets = n_start + tl.arange(0, BLOCK_SIZE)
        mask = n_offsets < N

        # Calculate input indices
        input_idx = m_idx * input_stride0 + n_offsets * input_stride1 + k_idx * input_stride2

        # Load and accumulate
        vals = tl.load(input_ptr + input_idx, mask=mask, other=0.0)
        acc += tl.sum(vals)

    # Compute mean and store
    mean_val = acc / N
    output_idx = m_idx * output_stride0 + k_idx * output_stride1
    tl.store(output_ptr + output_idx, mean_val)


def mean_dim(
    input: torch.Tensor, dim: int, keepdim: bool = False, dtype: torch.dtype | None = None
) -> torch.Tensor:
    """
    Triton implementation of torch.mean with single dimension reduction.

    Args:
        input: Input tensor
        dim: Single dimension along which to compute mean
        keepdim: Whether to keep the reduced dimension
        dtype: Output dtype. If None, uses input dtype (or float32 for integer inputs)

    Returns:
        Tensor with mean values along specified dimension
    """
    # Validate inputs
    assert input.is_cuda, "Input must be a CUDA tensor"
    assert (
        -input.ndim <= dim < input.ndim
    ), f"Invalid dimension {dim} for tensor with {input.ndim} dimensions"

    # Handle negative dim
    if dim < 0:
        dim = dim + input.ndim

    # Handle dtype
    if dtype is None:
        if input.dtype in [torch.int8, torch.int16, torch.int32, torch.int64]:
            dtype = torch.float32
        else:
            dtype = input.dtype

    # Convert input to appropriate dtype if needed
    if input.dtype != dtype:
        input = input.to(dtype)

    # Get input shape and strides
    shape = list(input.shape)

    # Calculate dimensions for kernel
    M = 1
    for i in range(dim):
        M *= shape[i]

    N = shape[dim]

    K = 1
    for i in range(dim + 1, len(shape)):
        K *= shape[i]

    # Reshape input to 3D view (M, N, K)
    input_3d = input.reshape(M, N, K)

    # Create output shape
    if keepdim:
        output_shape = shape.copy()
        output_shape[dim] = 1
    else:
        output_shape = shape[:dim] + shape[dim + 1 :]

    # Create output tensor
    output = torch.empty(output_shape, dtype=dtype, device=input.device)

    # Reshape output for kernel
    if keepdim:
        output_2d = output.reshape(M, 1, K).squeeze(1)
    else:
        output_2d = output.reshape(M, K)

    # Launch kernel
    grid = (M * K,)
    BLOCK_SIZE = 1024

    mean_kernel[grid](
        input_3d,
        output_2d,
        input_3d.stride(0),
        input_3d.stride(1),
        input_3d.stride(2),
        output_2d.stride(0),
        output_2d.stride(1) if output_2d.ndim > 1 else 0,
        M,
        N,
        K,
        BLOCK_SIZE,
    )

    return output


def mm_batch_invariant(a, b):
    """Batch-invariant replacement for `aten::mm` using a persistent matmul kernel."""
    return matmul_persistent(a, b)


def addmm_batch_invariant(bias, a, b):
    """Batch-invariant replacement for `aten::addmm` using a persistent matmul kernel."""
    return matmul_persistent(a, b, bias=bias)


def _log_softmax_batch_invariant(input, dim, _half_to_float):
    assert not _half_to_float, "not implemented"
    return log_softmax(input, dim=dim)


def mean_batch_invariant(input, dim, keepdim=False, dtype: torch.dtype | None = None):
    """Batch-invariant replacement for `aten::mean.dim` over one or more dimensions."""
    assert dtype is None or dtype == torch.float32, f"unsupported dtype: {dtype}"
    if len(dim) == 1:
        return mean_dim(input, dim[0], keepdim=keepdim)
    else:
        assert input.dtype in {
            torch.float16,
            torch.bfloat16,
            torch.float32,
        }, "only float types supported for now"
        n_elems = 1
        for d in dim:
            n_elems *= input.shape[d]
        return torch.sum(input, dim=dim, keepdim=keepdim, dtype=torch.float32) / n_elems


AttentionBlockSize = namedtuple("AttentionBlockSize", ["block_m", "block_n"])


def get_batch_invariant_attention_block_size() -> AttentionBlockSize:
    """Return the (block_m, block_n) tiling used for batch-invariant attention."""
    return AttentionBlockSize(block_m=16, block_n=16)


_batch_invariant_MODE = False
_batch_invariant_LIB = None
_TE_GENERAL_GEMM_ORIG = None
_TE_RMSNORM_ORIG_FWD = None
_MEG_TE_GENERAL_GEMM_ORIG = None
_TE_RMSNORM_FUNC_ORIGS: Dict[str, Any] = {}
_TE_GEMM_FUNC_ORIGS: Dict[str, Any] = {}


def _import_module_if_available(name: str):
    spec = importlib.util.find_spec(name)
    if spec is None:
        return None
    return importlib.import_module(name)


def _te_patch_for_batch_invariant():
    """Patch Transformer Engine modules to use batch-invariant GEMM and RMSNorm.

    This monkey-patches TE's GEMM and RMSNorm entry points to dispatch to the
    batch-invariant implementations when batch-invariant mode is enabled.
    Safe no-op if TE is unavailable.
    """
    global _TE_GENERAL_GEMM_ORIG, _TE_RMSNORM_ORIG_FWD, _MEG_TE_GENERAL_GEMM_ORIG
    import transformer_engine.pytorch as te
    import transformer_engine.pytorch.cpp_extensions as te_cpp

    # Patch general_gemm once
    if _TE_GENERAL_GEMM_ORIG is None and hasattr(te_cpp, "general_gemm"):
        _TE_GENERAL_GEMM_ORIG = te_cpp.general_gemm
        te_cpp.general_gemm = _te_general_gemm_patched

    # Also patch the symbol imported inside TE's module.linear
    # (from ..cpp_extensions import general_gemm)
    import transformer_engine.pytorch.module.linear as te_linear_mod

    if hasattr(te_linear_mod, "general_gemm"):
        if "module.linear.general_gemm" not in _TE_GEMM_FUNC_ORIGS:
            _TE_GEMM_FUNC_ORIGS["module.linear.general_gemm"] = te_linear_mod.general_gemm
            te_linear_mod.general_gemm = _te_general_gemm_patched

    # Also patch the symbol imported inside TE's module.layernorm_linear
    import transformer_engine.pytorch.module.layernorm_linear as te_layernorm_linear_mod

    if hasattr(te_layernorm_linear_mod, "general_gemm"):
        if "module.layernorm_linear.general_gemm" not in _TE_GEMM_FUNC_ORIGS:
            _TE_GEMM_FUNC_ORIGS["module.layernorm_linear.general_gemm"] = (
                te_layernorm_linear_mod.general_gemm
            )
            te_layernorm_linear_mod.general_gemm = _te_general_gemm_patched

    # Also patch the symbol imported into Megatron's TE wrapper module
    import megatron.core.extensions.transformer_engine as meg_te

    if _MEG_TE_GENERAL_GEMM_ORIG is None and hasattr(meg_te, "general_gemm"):
        _MEG_TE_GENERAL_GEMM_ORIG = meg_te.general_gemm
        meg_te.general_gemm = _te_general_gemm_patched

    # Patch RMSNorm.forward once (class may be on te or te.pytorch)
    rms_cls = getattr(te, "RMSNorm", None)
    if rms_cls is None:
        rms_cls = getattr(te, "pytorch", None)
        rms_cls = getattr(rms_cls, "RMSNorm", None)
    if rms_cls is not None and _TE_RMSNORM_ORIG_FWD is None and hasattr(rms_cls, "forward"):
        _TE_RMSNORM_ORIG_FWD = rms_cls.forward
        rms_cls.forward = _te_rmsnorm_forward_patched

    # Patch TE module-level RMSNorm functions used by fused LayerNormLinear
    import transformer_engine.pytorch.module.layernorm as te_layernorm_mod

    def _make_rmsnorm_patched(orig_func):
        # Module-level helpers (e.g. transformer_engine.pytorch.module.layernorm.rmsnorm)
        # do not go through the RMSNorm class, so we also wrap those functions here.
        def _patched(*args, **kwargs):
            # If batch-invariant mode is off, use original
            if not is_batch_invariant_mode_enabled():
                return orig_func(*args, **kwargs)

            # Extract x, weight, eps from args/kwargs per TE signatures
            x = args[0] if len(args) > 0 else kwargs.get("x")
            weight = args[1] if len(args) > 1 else kwargs.get("weight")
            eps = (args[2] if len(args) > 2 else None) if "eps" not in kwargs else kwargs.get("eps")
            if eps is None:
                eps = 1e-5
            if x is None or weight is None:
                return orig_func(*args, **kwargs)

            y = rmsnorm_batch_invariant(x, weight, float(eps))
            # Match TE behavior: cast output to parameter dtype
            return y.to(weight.dtype)

        return _patched

    for name in ("rmsnorm", "rmsnorm_forward", "rmsnorm_fwd"):
        if hasattr(te_layernorm_mod, name) and name not in _TE_RMSNORM_FUNC_ORIGS:
            orig = getattr(te_layernorm_mod, name)
            _TE_RMSNORM_FUNC_ORIGS[name] = orig
            setattr(te_layernorm_mod, name, _make_rmsnorm_patched(orig))


def _te_unpatch_for_batch_invariant():
    """Restore original Transformer Engine functions if they were patched."""
    global _TE_GENERAL_GEMM_ORIG, _TE_RMSNORM_ORIG_FWD, _MEG_TE_GENERAL_GEMM_ORIG
    te_cpp = _import_module_if_available("transformer_engine.pytorch.cpp_extensions")
    te = _import_module_if_available("transformer_engine.pytorch")
    if te_cpp is None or te is None:
        _TE_GENERAL_GEMM_ORIG = None
        _TE_RMSNORM_ORIG_FWD = None
        _MEG_TE_GENERAL_GEMM_ORIG = None
        return

    if _TE_GENERAL_GEMM_ORIG is not None and hasattr(te_cpp, "general_gemm"):
        te_cpp.general_gemm = _TE_GENERAL_GEMM_ORIG
        _TE_GENERAL_GEMM_ORIG = None

    rms_cls = getattr(te, "RMSNorm", None)
    if rms_cls is None:
        rms_cls = getattr(te, "pytorch", None)
        rms_cls = getattr(rms_cls, "RMSNorm", None)
    if rms_cls is not None and _TE_RMSNORM_ORIG_FWD is not None:
        rms_cls.forward = _TE_RMSNORM_ORIG_FWD
        _TE_RMSNORM_ORIG_FWD = None

    meg_te = _import_module_if_available("megatron.core.extensions.transformer_engine")
    if (
        meg_te is not None
        and _MEG_TE_GENERAL_GEMM_ORIG is not None
        and hasattr(meg_te, "general_gemm")
    ):
        meg_te.general_gemm = _MEG_TE_GENERAL_GEMM_ORIG
        _MEG_TE_GENERAL_GEMM_ORIG = None
    elif meg_te is None:
        _MEG_TE_GENERAL_GEMM_ORIG = None

    # Restore TE module-level RMSNorm functions
    te_layernorm_mod = _import_module_if_available("transformer_engine.pytorch.module.layernorm")
    if te_layernorm_mod is not None:
        for name, orig in list(_TE_RMSNORM_FUNC_ORIGS.items()):
            if hasattr(te_layernorm_mod, name):
                setattr(te_layernorm_mod, name, orig)
            _TE_RMSNORM_FUNC_ORIGS.pop(name, None)
    else:
        _TE_RMSNORM_FUNC_ORIGS.clear()

    # Restore TE module.linear imported symbol for general_gemm if patched
    te_linear_mod = _import_module_if_available("transformer_engine.pytorch.module.linear")
    key = "module.linear.general_gemm"
    if (
        te_linear_mod is not None
        and key in _TE_GEMM_FUNC_ORIGS
        and hasattr(te_linear_mod, "general_gemm")
    ):
        te_linear_mod.general_gemm = _TE_GEMM_FUNC_ORIGS[key]
        _TE_GEMM_FUNC_ORIGS.pop(key, None)
    else:
        _TE_GEMM_FUNC_ORIGS.pop(key, None)

    # Restore TE module.layernorm_linear imported symbol for general_gemm if patched
    te_layernorm_linear_mod = _import_module_if_available(
        "transformer_engine.pytorch.module.layernorm_linear"
    )
    key = "module.layernorm_linear.general_gemm"
    if (
        te_layernorm_linear_mod is not None
        and key in _TE_GEMM_FUNC_ORIGS
        and hasattr(te_layernorm_linear_mod, "general_gemm")
    ):
        te_layernorm_linear_mod.general_gemm = _TE_GEMM_FUNC_ORIGS[key]
        _TE_GEMM_FUNC_ORIGS.pop(key, None)
    else:
        _TE_GEMM_FUNC_ORIGS.pop(key, None)


def _extract_te_gemm_args(args: tuple, kwargs: Dict[str, Any]):
    """Utility to parse TE general_gemm flexible signature.

    Returns (A, B, out_dtype, layout, out, bias, grad).
    """
    A = args[0] if len(args) > 0 else kwargs.get("A")
    B = args[1] if len(args) > 1 else kwargs.get("B")
    out_dtype = kwargs.get("out_dtype")
    layout = kwargs.get("layout", "TN")
    out = kwargs.get("out")
    bias = kwargs.get("bias")
    grad = kwargs.get("grad", False)
    return A, B, out_dtype, layout, out, bias, grad


def _is_supported_dtype_for_bik(t: torch.dtype) -> bool:
    return t in {torch.float16, torch.bfloat16, torch.float32}


class BatchInvariantTEGemmFn(torch.autograd.Function):
    """Autograd function implementing batch-invariant TE GEMM."""

    @staticmethod
    def forward(
        ctx,
        A: torch.Tensor,
        B: torch.Tensor,
        bias: Optional[torch.Tensor],
        out_dtype: Optional[torch.dtype],
        layout: str,
    ):
        """Forward pass computing batch-invariant TE GEMM.

        Respects TE's flexible `layout` semantics, flattens leading dimensions of
        the input as needed, applies optional bias, and casts to `out_dtype`.
        """
        assert isinstance(layout, str) and len(layout) == 2, f"Unsupported layout: {layout}"
        transa = layout[0].upper() == "T"
        transb = layout[1].upper() == "T"

        opA = A.transpose(0, 1).contiguous() if transa else A.contiguous()  # [K, O] or [I, O]
        opB = B.transpose(0, 1).contiguous() if transb else B.contiguous()  # [..., K]

        # Flatten opA to 2D if needed (weight tensors should be 2D, but validate)
        if opA.dim() > 2:
            opA = opA.reshape(-1, opA.shape[-1])
        elif opA.dim() < 2:
            raise ValueError(f"opA has insufficient dimensions: {opA.shape}")
        assert opA.dim() == 2, f"opA must be 2D for matmul_persistent, got shape {opA.shape}"

        # Flatten all leading dims of opB except the last feature dim to match TE behavior
        if opB.dim() >= 2:
            leading_shape = opB.shape[:-1]
            K = opB.shape[-1]
            opB_2d = opB.reshape(-1, K)
        else:
            leading_shape = ()
            opB_2d = opB

        # Perform GEMM: (N_total, K) @ (K, O) -> (N_total, O)
        base_2d = matmul_persistent(opB_2d, opA, bias=None)

        # Reshape back to original leading dims with output features at the end
        out = base_2d.reshape(*leading_shape, base_2d.shape[-1])

        # Add bias after reshaping to match output structure
        if bias is not None:
            out = out + bias

        if out_dtype is not None:
            out = out.to(out_dtype)

        # Save for backward
        ctx.transa = transa
        ctx.transb = transb
        ctx.leading_shape = leading_shape
        ctx.bias_present = bias is not None
        ctx.save_for_backward(A, B)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """Backward pass for batch-invariant TE GEMM.

        Computes gradients w.r.t. A, B, and optional bias while mirroring the
        reshaping/layout logic used in the forward pass.
        """
        A, B = ctx.saved_tensors
        transa = ctx.transa
        transb = ctx.transb
        leading_shape = ctx.leading_shape

        # Reconstruct opA/opB for gradients
        opA = A.transpose(0, 1).contiguous() if transa else A  # [K, O]
        opB = B.transpose(0, 1).contiguous() if transb else B  # [..., K]

        # Flatten grad_output to 2D to mirror forward flatten
        if grad_output.dim() >= 2 and isinstance(leading_shape, tuple) and len(leading_shape) > 0:
            N_total = 1
            for s in leading_shape:
                N_total *= s
            grad_out_2d = grad_output.reshape(N_total, grad_output.shape[-1])
        else:
            grad_out_2d = grad_output

        if transa and not transb and A.dim() == 2 and opB.dim() >= 2:
            # Common DSV4 path: forward computes B_flat @ A.T where A is the
            # [out, in] weight and B is the token matrix. Avoid materializing
            # B.T.contiguous(), which is very expensive at long sequence.
            opB_2d = opB.reshape(-1, opB.shape[-1])
            if grad_output.dim() >= 2:
                grad_out_2d = grad_output.reshape(
                    opB_2d.shape[0], grad_output.shape[-1]
                ).contiguous()

            dA = None
            dB = None
            if ctx.needs_input_grad[0]:
                dA = grad_out_2d.transpose(0, 1).matmul(opB_2d)
            if ctx.needs_input_grad[1]:
                dB_2d = grad_out_2d.matmul(A)
                dB = dB_2d.reshape_as(B)

            if ctx.bias_present:
                dbias = grad_out_2d.sum(dim=0)
            else:
                dbias = None
            return dA, dB, dbias, None, None

        # Y = B_flat @ A -> dB_flat = dY @ A^T ; dA = B_flat^T @ dY
        d_opB_2d = grad_out_2d.matmul(opA.transpose(0, 1).contiguous())
        d_opA = opB.reshape(-1, opB.shape[-1]).transpose(0, 1).contiguous().matmul(grad_out_2d)

        # Reshape d_opB back to original opB shape
        d_opB = (
            d_opB_2d.reshape(*leading_shape, d_opB_2d.shape[-1])
            if grad_output.dim() >= 2
            else d_opB_2d
        )

        # Map back to dA, dB based on trans flags
        if transa:
            dA = d_opA.transpose(0, 1).contiguous()
        else:
            dA = d_opA

        if transb:
            dB = d_opB.transpose(0, 1).contiguous()
        else:
            dB = d_opB

        # Bias grad along last dimension of Y, if bias was added in forward
        if ctx.bias_present:
            dbias = grad_output.reshape(-1, grad_output.shape[-1]).sum(dim=0)
        else:
            dbias = None

        return dA, dB, dbias, None, None


def _te_general_gemm_patched(*args, **kwargs) -> List[torch.Tensor]:
    """
    Batch-invariant replacement for TE general_gemm.
    Returns a list of tensors to match TE's API: (gemm_out, bias_grad, gelu_input, extra_output)
    """
    global _TE_GENERAL_GEMM_ORIG
    # If original not captured, do nothing
    if _TE_GENERAL_GEMM_ORIG is None:
        raise RuntimeError("TE general_gemm original not captured; patching order issue")

    A, B, out_dtype, layout, out, bias, grad = _extract_te_gemm_args(args, kwargs)
    extra_output = kwargs.get("extra_output", None)
    ub = kwargs.get("ub", None)
    ub_type = kwargs.get("ub_type", None)
    bulk_overlap = kwargs.get("bulk_overlap", False)

    # Guardrails: validate inputs
    if A is None or B is None:
        raise ValueError("Batch-invariant GEMM requires A and B tensors.")
    if (not A.is_cuda) or (not B.is_cuda):
        raise RuntimeError("Batch-invariant GEMM requires CUDA tensors.")
    if not _is_supported_dtype_for_bik(A.dtype) or not _is_supported_dtype_for_bik(B.dtype):
        raise RuntimeError(f"Unsupported dtype for batch-invariant GEMM: {A.dtype}, {B.dtype}")

    # Disallow GEMM-comm overlap in batch-invariant mode
    if extra_output is not None or ub is not None or ub_type is not None or bulk_overlap:
        raise RuntimeError(
            "Batch-invariant GEMM does not support Userbuffers/overlap "
            "(extra_output/ub/ub_type/bulk_overlap)."
        )

    # Compute via autograd-aware function matching TE's layout semantics
    result = BatchInvariantTEGemmFn.apply(A, B, bias if not grad else None, out_dtype, layout)

    bias_grad = None
    if grad and bias is not None:
        # Flatten B to 2D and sum over batch/sequence dimension (first dim)
        B_flat = B.reshape(-1, B.shape[-1]) if B.dim() > 2 else B
        bias_grad = B_flat.sum(dim=0)  # Sum over batch/sequence, keeping output dim

    if out is not None:
        out.copy_(result)
        # TE expects (gemm_out, bias_grad, gelu_input, extra_output)
        return (out, bias_grad, None, extra_output)
    return (result, bias_grad, None, extra_output)


class BatchInvariantRMSNormFn(torch.autograd.Function):
    """Autograd function implementing batch-invariant RMSNorm."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, eps: float, zero_centered_gamma: bool):
        """Forward pass for batch-invariant RMSNorm.

        Normalizes `x` using an RMSNorm-style statistic computed via `mean_dim`,
        applies affine `weight`, and stores intermediate rsigma for backward.
        """
        if not x.is_cuda:
            raise RuntimeError("Batch-invariant RMSNorm requires CUDA tensors.")
        if not _is_supported_dtype_for_bik(x.dtype):
            raise RuntimeError(f"Unsupported dtype for batch-invariant RMSNorm: {x.dtype}")
        weight_eff = weight + 1.0 if zero_centered_gamma else weight

        # Forward follows the raw persistent kernel and saves the exact per-row
        # inverse RMS used by that kernel for the fused backward.
        out, rsigma = _rms_norm_with_rsigma(x, weight_eff, eps)

        # Save for backward
        ctx.eps = eps
        ctx.zero_centered_gamma = zero_centered_gamma
        ctx.rsigma = rsigma

        ctx.save_for_backward(x, weight, rsigma)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """Backward pass for batch-invariant RMSNorm.

        Computes gradients w.r.t. input and weight while matching TE's fp32
        accumulation and reduction behavior for numerical stability.
        """
        x, weight, rsigma = ctx.saved_tensors
        w_eff = (weight + 1.0) if ctx.zero_centered_gamma else weight

        dx, g_w = _rms_norm_backward_fused(grad_output, x, w_eff, rsigma)
        g_w = g_w.to(weight.dtype)

        return dx, g_w, None, None


def rmsnorm_batch_invariant(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Batch-invariant RMSNorm wrapper that delegates to autograd-aware implementation.

    This provides a simple functional interface while using the optimized BatchInvariantRMSNormFn
    which has better numerics (fp32 precision in forward/backward).
    """
    # Delegate to the autograd function with zero_centered_gamma=False (standard RMSNorm)
    return BatchInvariantRMSNormFn.apply(x, weight, eps, False)


def _te_rmsnorm_forward_patched(self, x: torch.Tensor) -> torch.Tensor:
    """Patched TE RMSNorm.forward that routes to batch-invariant
    implementation with autograd support.
    """
    weight = getattr(self, "weight", None)
    if weight is None:
        raise RuntimeError("Batch-invariant RMSNorm requires affine weight.")
    eps = getattr(self, "eps", 1e-5)
    zero_centered_gamma = getattr(self, "zero_centered_gamma", False)
    return BatchInvariantRMSNormFn.apply(x, weight, eps, zero_centered_gamma)


def is_batch_invariant_mode_enabled():
    """Return True if global batch-invariant mode is currently enabled."""
    return _batch_invariant_MODE


def enable_batch_invariant_mode():
    """Enable global batch-invariant mode and patch Aten/TE kernels."""
    global _batch_invariant_MODE, _batch_invariant_LIB
    if _batch_invariant_MODE:
        return
    dispatch_key = getattr(torch.accelerator.current_accelerator(), "type", "cpu").upper()
    _batch_invariant_MODE = True
    _batch_invariant_LIB = torch.library.Library("aten", "IMPL")
    _batch_invariant_LIB.impl("aten::mm", mm_batch_invariant, dispatch_key)
    _batch_invariant_LIB.impl("aten::addmm", addmm_batch_invariant, dispatch_key)
    _batch_invariant_LIB.impl("aten::_log_softmax", _log_softmax_batch_invariant, dispatch_key)
    _batch_invariant_LIB.impl("aten::mean.dim", mean_batch_invariant, dispatch_key)
    # Also patch Transformer Engine kernels when available
    _te_patch_for_batch_invariant()


def disable_batch_invariant_mode():
    """Disable global batch-invariant mode and restore original kernels."""
    global _batch_invariant_MODE, _batch_invariant_LIB
    if _batch_invariant_LIB is not None:
        _batch_invariant_LIB._destroy()
    _batch_invariant_MODE = False
    _batch_invariant_LIB = None
    # Restore Transformer Engine kernels if previously patched
    _te_unpatch_for_batch_invariant()


@contextlib.contextmanager
def set_batch_invariant_mode(enabled: bool = True):
    """Context manager to toggle global batch-invariant mode.

    When `enabled` is True, batch-invariant kernels are enabled for the duration of
    the context; when False, they are disabled for the duration. This implementation
    is re-entrant and correctly restores the previous state even under nesting.
    """
    global _batch_invariant_MODE, _batch_invariant_LIB
    # Save the previous on/off state so we can correctly restore it, even under
    # nested usage or when toggling from True->False inside an outer True scope.
    prev_enabled = _batch_invariant_MODE

    # Apply the requested state only if it differs from the current one.
    if enabled and not prev_enabled:
        enable_batch_invariant_mode()
    elif not enabled and prev_enabled:
        disable_batch_invariant_mode()

    try:
        yield
    finally:
        # Restore the previous state. If we turned BIK on at entry, turn it off here.
        # If we turned it off at entry (inside an outer True scope), turn it back on.
        if enabled and not prev_enabled:
            disable_batch_invariant_mode()
        elif not enabled and prev_enabled:
            enable_batch_invariant_mode()
