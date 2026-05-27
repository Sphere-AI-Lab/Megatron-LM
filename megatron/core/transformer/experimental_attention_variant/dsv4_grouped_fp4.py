# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Grouped FP4 linear helpers for DeepSeek V4 frozen routed experts.

The forward TileLang kernel mirrors SGLang's DSV4 CUDA-graph grouped FP4
GEMM: rows are already grouped into 32-row expert-aligned blocks and padding
rows use expert id -1. Backward follows the existing ``DSV4Linear`` frozen
base contract: only ``grad_input`` is returned, while the quantized base weight
and scale remain frozen.
"""

import os
from functools import lru_cache
from typing import Iterable, Optional

import torch
import torch.nn as nn
import tilelang
import tilelang.language as T
import triton
import triton.language as tl

from megatron.core.transformer.experimental_attention_variant import dsv4_kernels
from megatron.core.transformer.experimental_attention_variant.dsv4_linear import (
    _DEFAULT_SCALE_DTYPE,
    _DEFAULT_SCALE_FMT,
    _DefaultDtypeBF16,
    _FP8_BLOCK_SIZE,
    _dequantize_fp4_weight_bf16,
)

_DSV4_FP4_GEMM_BACKEND_ENV = "DSV4_FP4_GEMM_BACKEND"
_VALID_DSV4_FP4_GEMM_BACKENDS = ("auto", "deepgemm", "tilelang")
_DSV4_FP4_GEMM_BACKEND = (
    os.environ.get(_DSV4_FP4_GEMM_BACKEND_ENV, "auto").strip().lower() or "auto"
)
if _DSV4_FP4_GEMM_BACKEND not in _VALID_DSV4_FP4_GEMM_BACKENDS:
    raise ValueError(
        f"{_DSV4_FP4_GEMM_BACKEND_ENV} must be one of "
        f"{', '.join(_VALID_DSV4_FP4_GEMM_BACKENDS)}, got "
        f"{_DSV4_FP4_GEMM_BACKEND!r}"
    )

_deep_gemm_official_import_error = None
if _DSV4_FP4_GEMM_BACKEND == "tilelang":
    _deep_gemm_official = None
else:
    try:
        import deep_gemm_official as _deep_gemm_official
    except Exception as exc:
        _deep_gemm_official_import_error = exc
        _deep_gemm_official = None


tilelang.set_log_level("WARNING")

_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
}

_FP8 = "float8_e4m3"
_FP4 = "float4_e2m1fn"
_FE8M0 = "float8_e8m0fnu"
_BF16 = "bfloat16"
_FP32 = "float32"
_INT32 = "int32"
_FP4_BLOCK_SIZE = 32

_DEEP_GEMM_OFFICIAL_GROUPED_FP8_FP4_GEMM = (
    getattr(_deep_gemm_official, "m_grouped_fp8_fp4_gemm_nt_contiguous", None)
    if _deep_gemm_official is not None
    else None
)
if (
    _DSV4_FP4_GEMM_BACKEND == "deepgemm"
    and _DEEP_GEMM_OFFICIAL_GROUPED_FP8_FP4_GEMM is None
):
    if _deep_gemm_official_import_error is not None:
        raise RuntimeError(
            f"{_DSV4_FP4_GEMM_BACKEND_ENV}=deepgemm requires importing "
            "deep_gemm_official"
        ) from _deep_gemm_official_import_error
    raise RuntimeError(
        f"{_DSV4_FP4_GEMM_BACKEND_ENV}=deepgemm requires "
        "deep_gemm_official.m_grouped_fp8_fp4_gemm_nt_contiguous"
    )


def has_deep_gemm_official_fp8_fp4() -> bool:
    return _DEEP_GEMM_OFFICIAL_GROUPED_FP8_FP4_GEMM is not None


def _align_to(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _empty_deep_gemm_scale_layout(
    rows: int,
    scale_cols: int,
    device: torch.device,
) -> torch.Tensor:
    aligned_rows = _align_to(rows, 4)
    aligned_scale_cols = _align_to(scale_cols, 4)
    return torch.empty(
        (aligned_scale_cols // 4, aligned_rows),
        device=device,
        dtype=torch.int32,
    ).mT[:rows, :]


def _pack_fp8_e8m0_scale_for_deep_gemm_torch(scale_u8: torch.Tensor) -> torch.Tensor:
    num_groups, mn, scale_k = scale_u8.shape
    aligned_mn = _align_to(mn, 4)
    aligned_scale_k = _align_to(scale_k, 4)
    padded = torch.zeros(
        (num_groups, aligned_mn, aligned_scale_k),
        device=scale_u8.device,
        dtype=torch.uint8,
    )
    padded[:, :mn, :scale_k] = scale_u8
    packed = (
        padded.view(-1)
        .view(dtype=torch.int32)
        .view(num_groups, aligned_mn, aligned_scale_k // 4)
    )
    tma_layout = torch.empty(
        (num_groups, aligned_scale_k // 4, aligned_mn),
        device=scale_u8.device,
        dtype=torch.int32,
    ).mT
    tma_layout[:, :, :] = packed
    return tma_layout[:, :mn, :]


def pack_fp8_e8m0_scale_for_deep_gemm(scale: torch.Tensor) -> torch.Tensor:
    """Pack FP8 E8M0 scales into DeepGEMM's TMA-aligned int32 layout."""

    assert scale.dtype == torch.float8_e8m0fnu, (
        "DSV4 DeepGEMM FP4 scale packer expects float8_e8m0fnu scales"
    )
    assert scale.dim() in (2, 3), "scale must be [mn, k] or [groups, mn, k]"

    remove_dim = scale.dim() == 2
    scale_u8 = scale.contiguous().view(torch.uint8)
    if remove_dim:
        scale_u8 = scale_u8.unsqueeze(0)

    packed_scale = _pack_fp8_e8m0_scale_for_deep_gemm_torch(scale_u8)
    return packed_scale.squeeze(0) if remove_dim else packed_scale


@triton.jit
def _dsv4_deep_gemm_act_quant_kernel(
    x_ptr,
    y_ptr,
    s_ptr,
    m: tl.constexpr,
    k: tl.constexpr,
    stride_xm: tl.constexpr,
    stride_xk: tl.constexpr,
    stride_ym: tl.constexpr,
    stride_yk: tl.constexpr,
    stride_sm: tl.constexpr,
    stride_sk: tl.constexpr,
    num_scale_cols: tl.constexpr,
    block_m: tl.constexpr,
    group_size: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_pack = tl.program_id(1)

    rows = pid_m * block_m + tl.arange(0, block_m)
    cols = tl.arange(0, group_size)
    row_mask = rows < m
    packed = tl.zeros((block_m,), dtype=tl.uint32)

    for pack_idx in tl.static_range(0, 4):
        scale_col = pid_pack * 4 + pack_idx
        k_offsets = scale_col * group_size + cols
        valid_group = scale_col < num_scale_cols
        x = tl.load(
            x_ptr + rows[:, None] * stride_xm + k_offsets[None, :] * stride_xk,
            mask=row_mask[:, None] & valid_group,
            other=0.0,
        ).to(tl.float32)

        amax = tl.maximum(tl.max(tl.abs(x), axis=1), 1.0e-4)
        scale_unrounded = amax * (1.0 / 448.0)
        bits = scale_unrounded.to(tl.uint32, bitcast=True)
        mantissa = bits & 0x7FFFFF
        exp = ((bits >> 23) & 0xFF).to(tl.int32) - 127
        exp_rounded = exp + (mantissa != 0).to(tl.int32)
        scale_bits = ((exp_rounded + 127) & 0xFF).to(tl.uint32) << 23
        scale = scale_bits.to(tl.float32, bitcast=True)

        y = tl.minimum(tl.maximum(x / scale[:, None], -448.0), 448.0)
        tl.store(
            y_ptr + rows[:, None] * stride_ym + k_offsets[None, :] * stride_yk,
            y.to(tl.float8e4nv),
            mask=row_mask[:, None] & valid_group,
        )

        scale_byte = ((exp_rounded + 127) & 0xFF).to(tl.uint32)
        scale_byte = tl.where(valid_group, scale_byte, 0)
        packed = packed | (scale_byte << (pack_idx * 8))

    tl.store(
        s_ptr + rows * stride_sm + pid_pack * stride_sk,
        packed.to(tl.int32),
        mask=row_mask,
    )


def dsv4_deep_gemm_act_quant(
    x: torch.Tensor,
    block_size: int = _FP8_BLOCK_SIZE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize activations and emit DeepGEMM's packed E8M0 scale layout."""

    assert block_size == _FP8_BLOCK_SIZE, (
        "DeepGEMM official DSV4 act quant uses 128-wide blocks"
    )
    assert x.is_contiguous(), "input activations must be contiguous"
    k = x.size(-1)
    assert k % block_size == 0, "activation K must be divisible by 128"
    m = x.numel() // k
    x_2d = x.view(m, k)
    y = torch.empty_like(x_2d, dtype=torch.float8_e4m3fn)
    scale = _empty_deep_gemm_scale_layout(m, k // block_size, x.device)

    if m > 0:
        grid = (triton.cdiv(m, 32), triton.cdiv(k // block_size, 4))
        _dsv4_deep_gemm_act_quant_kernel[grid](
            x_2d,
            y,
            scale,
            m,
            k,
            x_2d.stride(0),
            x_2d.stride(1),
            y.stride(0),
            y.stride(1),
            scale.stride(0),
            scale.stride(1),
            k // block_size,
            block_m=32,
            group_size=block_size,
            num_warps=4,
            num_stages=1,
        )
    return y.view_as(x), scale


@lru_cache(maxsize=None)
@tilelang.jit(pass_configs=_PASS_CONFIGS)
def _grouped_fp4_gemm_kernel(
    num_experts: int,
    n: int,
    k_dim: int,
    out_dtype: str = _BF16,
    accum_dtype: str = _FP32,
    scale_dtype: str = _FE8M0,
):
    """Expert-major FP8 activation x FP4 weight GEMM.

    This is the official DSV4 fp4_gemm tile shape with the expert and output
    axes flattened on the weight tensors. Rows must be grouped by expert in
    32-row aligned segments, which is what the SGLang DSV4 CUDA-graph MoE path
    prepares. Padding rows use expert id -1 and are written as zero.
    """

    m = T.symbolic("m")
    act_group_size = 128
    weight_group_size = 32
    block_m = 32
    block_n = 128
    block_k = 32
    n_sub = act_group_size // block_k

    @T.prim_func
    def kernel(
        a: T.Tensor[(m, k_dim), _FP8],
        b: T.Tensor[(num_experts * n, k_dim), _FP4],
        c: T.Tensor[(m, n), out_dtype],
        scales_a: T.Tensor[(m, T.ceildiv(k_dim, act_group_size)), scale_dtype],
        scales_b: T.Tensor[
            (num_experts * n, T.ceildiv(k_dim, weight_group_size)), scale_dtype
        ],
        pos_to_expert: T.Tensor[(m,), _INT32],
    ):
        with T.Kernel(T.ceildiv(n, block_n), T.ceildiv(m, block_m), threads=128) as (
            bx,
            by,
        ):
            a_shared = T.alloc_shared((block_m, block_k), _FP8)
            b_fp4_shared = T.alloc_shared((block_n, block_k), _FP4)
            b_shared = T.alloc_shared((block_n, block_k), _FP8)
            c_shared = T.alloc_shared((block_m, block_n), out_dtype)
            c_local = T.alloc_fragment((block_m, block_n), accum_dtype)
            c_local_accum = T.alloc_fragment((block_m, block_n), accum_dtype)
            scale_a_frag = T.alloc_fragment((block_m,), _FP32)
            scale_b_frag = T.alloc_fragment((block_n,), _FP32)

            T.use_swizzle(panel_size=10)
            expert = pos_to_expert[by * block_m]
            T.clear(c_local)
            T.clear(c_local_accum)

            if expert >= 0:
                k_iters = T.ceildiv(k_dim, block_k)
                for kk in T.Pipelined(k_iters, num_stages=2):
                    T.copy(a[by * block_m, kk * block_k], a_shared)
                    T.copy(b[expert * n + bx * block_n, kk * block_k], b_fp4_shared)
                    for i, j in T.Parallel(block_n, block_k):
                        b_shared[i, j] = T.Cast(
                            _FP8, T.Cast(_FP32, b_fp4_shared[i, j])
                        )

                    for i in T.Parallel(block_n):
                        scale_b_frag[i] = T.Cast(
                            _FP32, scales_b[expert * n + bx * block_n + i, kk]
                        )
                    for i in T.Parallel(block_m):
                        scale_a_frag[i] = T.Cast(
                            _FP32, scales_a[by * block_m + i, kk // n_sub]
                        )

                    T.gemm(a_shared, b_shared, c_local, transpose_B=True)

                    for i, j in T.Parallel(block_m, block_n):
                        c_local_accum[i, j] += (
                            c_local[i, j] * scale_a_frag[i] * scale_b_frag[j]
                        )
                    T.clear(c_local)

            T.copy(c_local_accum, c_shared)
            T.copy(c_shared, c[by * block_m, bx * block_n])

    return kernel


def grouped_fp4_gemm(
    a: torch.Tensor,
    a_s: torch.Tensor,
    b: torch.Tensor,
    b_s: torch.Tensor,
    pos_to_expert: torch.Tensor,
    scale_dtype: Optional[torch.dtype] = _DEFAULT_SCALE_DTYPE,
) -> torch.Tensor:
    """Run the grouped DSV4 FP4 GEMM on already-quantized activations."""

    assert a.is_contiguous() and b.is_contiguous(), "input tensors must be contiguous"
    assert a.dtype == torch.float8_e4m3fn, a.dtype
    assert b.dtype == torch.float4_e2m1fn_x2, b.dtype
    assert b.dim() == 3, "grouped FP4 weights must be [experts, out, in//2]"
    assert pos_to_expert.is_contiguous(), "pos_to_expert must be contiguous"

    k_dim = a.size(-1)
    m = a.numel() // k_dim
    n = b.size(1)
    assert pos_to_expert.numel() == m, "pos_to_expert length must match input rows"

    c = a.new_empty(*a.size()[:-1], n, dtype=torch.get_default_dtype())
    if has_deep_gemm_official_fp8_fp4():
        grouped_layout = pos_to_expert.view(-1)
        if grouped_layout.dtype != torch.int32:
            grouped_layout = grouped_layout.to(torch.int32)
        _DEEP_GEMM_OFFICIAL_GROUPED_FP8_FP4_GEMM(
            (a.view(m, k_dim), a_s),
            (b.view(torch.int8), b_s),
            c.view(m, n),
            grouped_layout,
            recipe_a=(1, _FP8_BLOCK_SIZE),
            recipe_b=(1, _FP4_BLOCK_SIZE),
            compiled_dims="nk",
            disable_ue8m0_cast=True,
        )
        return c

    assert a_s.is_contiguous() and b_s.is_contiguous(), (
        "scaling factor tensors must be contiguous"
    )
    assert b_s.dim() == 3, "grouped FP4 scales must be [experts, out, in//32]"
    tl_scale_dtype = _FE8M0 if scale_dtype == torch.float8_e8m0fnu else _FP32
    kernel = _grouped_fp4_gemm_kernel(b.size(0), n, k_dim, scale_dtype=tl_scale_dtype)
    kernel(
        a.view(m, k_dim),
        b.view(b.size(0) * b.size(1), b.size(2)),
        c.view(m, n),
        a_s.view(m, -1),
        b_s.view(b_s.size(0) * b_s.size(1), b_s.size(2)),
        pos_to_expert.view(-1).to(torch.int32),
    )
    return c


@lru_cache(maxsize=None)
@tilelang.jit(pass_configs=_PASS_CONFIGS)
def _grouped_fp4_grad_input_kernel(
    num_experts: int,
    n: int,
    k_dim: int,
    out_dtype: str = _BF16,
    accum_dtype: str = _FP32,
    scale_dtype: str = _FE8M0,
):
    """Compute grad_input = grad_output @ dequant_fp4(weight)."""

    m = T.symbolic("m")
    weight_group_size = 32
    block_m = 32
    block_n = 128
    block_k = 128
    k_scale_sub = block_k // weight_group_size

    @T.prim_func
    def kernel(
        grad_output: T.Tensor[(m, n), _BF16],
        weight: T.Tensor[(num_experts * n, k_dim), _FP4],
        grad_input: T.Tensor[(m, k_dim), out_dtype],
        scales_b: T.Tensor[
            (num_experts * n, T.ceildiv(k_dim, weight_group_size)), scale_dtype
        ],
        pos_to_expert: T.Tensor[(m,), _INT32],
    ):
        with T.Kernel(
            T.ceildiv(k_dim, block_k), T.ceildiv(m, block_m), threads=128
        ) as (bx, by):
            go_shared = T.alloc_shared((block_m, block_n), _BF16)
            w_fp4_shared = T.alloc_shared((block_n, block_k), _FP4)
            w_shared = T.alloc_shared((block_n, block_k), _BF16)
            gi_shared = T.alloc_shared((block_m, block_k), out_dtype)
            gi_local = T.alloc_fragment((block_m, block_k), accum_dtype)
            gi_local_accum = T.alloc_fragment((block_m, block_k), accum_dtype)

            T.use_swizzle(panel_size=10)
            expert = pos_to_expert[by * block_m]
            T.clear(gi_local)
            T.clear(gi_local_accum)

            if expert >= 0:
                n_iters = T.ceildiv(n, block_n)
                for nn in T.Pipelined(n_iters, num_stages=2):
                    T.copy(grad_output[by * block_m, nn * block_n], go_shared)
                    T.copy(
                        weight[expert * n + nn * block_n, bx * block_k],
                        w_fp4_shared,
                    )
                    for i, j in T.Parallel(block_n, block_k):
                        scale = T.Cast(
                            _FP32,
                            scales_b[
                                expert * n + nn * block_n + i,
                                bx * k_scale_sub + j // weight_group_size,
                            ],
                        )
                        w_shared[i, j] = T.Cast(
                            _BF16, T.Cast(_FP32, w_fp4_shared[i, j]) * scale
                        )

                    T.gemm(go_shared, w_shared, gi_local, transpose_B=False)

                    for i, j in T.Parallel(block_m, block_k):
                        gi_local_accum[i, j] += gi_local[i, j]
                    T.clear(gi_local)

            T.copy(gi_local_accum, gi_shared)
            T.copy(gi_shared, grad_input[by * block_m, bx * block_k])

    return kernel


def grouped_fp4_linear_backward_fast(
    grad_output: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    pos_to_expert: torch.Tensor,
    input_dtype: torch.dtype,
    scale_dtype: Optional[torch.dtype] = _DEFAULT_SCALE_DTYPE,
) -> torch.Tensor:
    """Fused grouped ``grad_input`` kernel for frozen DSV4 FP4 weights."""

    assert grad_output.dtype == torch.bfloat16, grad_output.dtype
    assert weight.dtype == torch.float4_e2m1fn_x2, weight.dtype
    assert scale.is_contiguous(), "grouped FP4 scales must be contiguous"
    assert weight.is_contiguous(), "grouped FP4 weights must be contiguous"
    assert pos_to_expert.is_contiguous(), "pos_to_expert must be contiguous"

    n = weight.size(1)
    k_dim = weight.size(-1) * 2
    m = grad_output.numel() // n
    assert grad_output.size(-1) == n, "grad_output last dimension must match weight out"
    assert k_dim % 128 == 0, "DSV4 grouped FP4 fast backward requires K % 128 == 0"
    assert n % 128 == 0, "DSV4 grouped FP4 fast backward requires N % 128 == 0"
    assert pos_to_expert.numel() == m, "pos_to_expert length must match input rows"

    tl_scale_dtype = _FE8M0 if scale_dtype == torch.float8_e8m0fnu else _FP32
    grad_input = torch.empty(
        *grad_output.shape[:-1],
        k_dim,
        device=grad_output.device,
        dtype=input_dtype,
    )
    kernel = _grouped_fp4_grad_input_kernel(
        weight.size(0),
        n,
        k_dim,
        out_dtype=T.dtype(input_dtype),
        scale_dtype=tl_scale_dtype,
    )
    kernel(
        grad_output.contiguous().view(m, n),
        weight.view(weight.size(0) * weight.size(1), weight.size(2)),
        grad_input.view(m, k_dim),
        scale.view(scale.size(0) * scale.size(1), scale.size(2)),
        pos_to_expert.view(-1).to(torch.int32),
    )
    return grad_input


def _dequantize_fp8_activation_bf16(
    a: torch.Tensor,
    a_s: torch.Tensor,
) -> torch.Tensor:
    k_dim = a.size(-1)
    m = a.numel() // k_dim
    a_bf16 = a.view(m, k_dim).to(torch.bfloat16)
    scale = a_s.view(m, -1).to(torch.bfloat16)
    scale = scale.repeat_interleave(_FP8_BLOCK_SIZE, dim=1)[:, :k_dim]
    return a_bf16 * scale


def _dequantize_grouped_fp4_weight_bf16(
    weight: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    assert weight.is_contiguous(), "grouped FP4 weights must be contiguous"
    assert scale.is_contiguous(), "grouped FP4 scales must be contiguous"
    num_experts, out_features, packed_in = weight.shape
    flat_weight = weight.view(num_experts * out_features, packed_in)
    flat_scale = scale.view(num_experts * out_features, scale.size(-1))
    w_bf16 = _dequantize_fp4_weight_bf16(flat_weight, flat_scale)
    return w_bf16.view(num_experts, out_features, packed_in * 2)


def grouped_fp4_gemm_torch_reference(
    a: torch.Tensor,
    a_s: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    pos_to_expert: torch.Tensor,
) -> torch.Tensor:
    """Torch reference for the quantized grouped FP4 forward.

    The input is already FP8-quantized. This reference dequantizes the FP8
    activation and FP4 expert weights, then runs one torch matmul per selected
    expert. It is intended for parity tests, not the hot path.
    """

    k_dim = a.size(-1)
    m = a.numel() // k_dim
    n = weight.size(1)
    a_bf16 = _dequantize_fp8_activation_bf16(a, a_s)
    w_bf16 = _dequantize_grouped_fp4_weight_bf16(weight, scale)
    pos = pos_to_expert.view(-1).to(torch.int64)
    out = torch.zeros((m, n), device=a.device, dtype=torch.get_default_dtype())
    for expert in range(weight.size(0)):
        rows = pos == expert
        if rows.any():
            out[rows] = (a_bf16[rows] @ w_bf16[expert].t()).to(out.dtype)
    return out.view(*a.size()[:-1], n)


def grouped_fp4_linear_torch_surrogate(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    pos_to_expert: torch.Tensor,
) -> torch.Tensor:
    """Torch linear surrogate used to define the frozen-base backward."""

    in_features = weight.size(-1) * 2
    m = x.numel() // in_features
    n = weight.size(1)
    x_2d = x.reshape(m, in_features)
    w_bf16 = _dequantize_grouped_fp4_weight_bf16(weight, scale)
    pos = pos_to_expert.view(-1).to(torch.int64)
    out = x.new_zeros((m, n))
    for expert in range(weight.size(0)):
        rows = pos == expert
        if rows.any():
            y = x_2d[rows].to(w_bf16.dtype) @ w_bf16[expert].t()
            out[rows] = y.to(x.dtype)
    return out.view(*x.size()[:-1], n)


def grouped_fp4_linear_backward_torch_reference(
    grad_output: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    pos_to_expert: torch.Tensor,
    input_dtype: torch.dtype,
) -> torch.Tensor:
    """Torch reference for ``grad_input`` with frozen grouped FP4 weights."""

    n = weight.size(1)
    m = grad_output.numel() // n
    in_features = weight.size(-1) * 2
    go = grad_output.reshape(m, n)
    w_bf16 = _dequantize_grouped_fp4_weight_bf16(weight, scale)
    pos = pos_to_expert.view(-1).to(torch.int64)
    grad_input = torch.zeros(
        (m, in_features),
        device=grad_output.device,
        dtype=input_dtype,
    )
    for expert in range(weight.size(0)):
        rows = pos == expert
        if rows.any():
            gi = go[rows].to(w_bf16.dtype) @ w_bf16[expert]
            grad_input[rows] = gi.to(input_dtype)
    return grad_input.reshape(*grad_output.size()[:-1], in_features)


def _can_use_fast_backward(
    grad_output: torch.Tensor,
    weight: torch.Tensor,
    pos_to_expert: torch.Tensor,
    input_dtype: torch.dtype,
) -> bool:
    n = weight.size(1)
    k_dim = weight.size(-1) * 2
    return (
        not _args_request_slow_backward()
        and input_dtype == torch.bfloat16
        and grad_output.dtype == torch.bfloat16
        and grad_output.is_cuda
        and weight.is_cuda
        and pos_to_expert.is_cuda
        and n % 128 == 0
        and k_dim % 128 == 0
        and pos_to_expert.numel() == grad_output.numel() // n
    )


def _args_request_slow_backward() -> bool:
    try:
        from megatron.training.global_vars import get_args

        args = get_args()
    except (AssertionError, ImportError):
        return False
    return bool(getattr(args, "dsv4_slow_backward", False))


def _grouped_fp4_linear_forward(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    pos_to_expert: torch.Tensor,
    scale_dtype: torch.dtype = _DEFAULT_SCALE_DTYPE,
) -> torch.Tensor:
    with _DefaultDtypeBF16():
        if has_deep_gemm_official_fp8_fp4():
            x_q, x_s = dsv4_deep_gemm_act_quant(x.contiguous(), _FP8_BLOCK_SIZE)
            scale_for_kernel = pack_fp8_e8m0_scale_for_deep_gemm(scale)
        else:
            x_q, x_s = dsv4_kernels.act_quant(
                x.contiguous(),
                _FP8_BLOCK_SIZE,
                _DEFAULT_SCALE_FMT,
                scale_dtype,
            )
            scale_for_kernel = scale.contiguous()
        return grouped_fp4_gemm(
            x_q,
            x_s,
            weight.contiguous(),
            scale_for_kernel,
            pos_to_expert.contiguous(),
            scale_dtype,
        )


class _DSV4GroupedFP4LinearFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        scale: torch.Tensor,
        pos_to_expert: torch.Tensor,
        scale_dtype: torch.dtype,
    ) -> torch.Tensor:
        ctx.save_for_backward(weight, scale, pos_to_expert)
        ctx.x_dtype = x.dtype
        return _grouped_fp4_linear_forward(x, weight, scale, pos_to_expert, scale_dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        weight, scale, pos_to_expert = ctx.saved_tensors
        grad_input = None
        if ctx.needs_input_grad[0]:
            if _can_use_fast_backward(grad_output, weight, pos_to_expert, ctx.x_dtype):
                grad_input = grouped_fp4_linear_backward_fast(
                    grad_output.contiguous(),
                    weight.contiguous(),
                    scale.contiguous(),
                    pos_to_expert.contiguous(),
                    ctx.x_dtype,
                )
            else:
                grad_input = grouped_fp4_linear_backward_torch_reference(
                    grad_output,
                    weight,
                    scale,
                    pos_to_expert,
                    ctx.x_dtype,
                )
        return grad_input, None, None, None, None


def grouped_fp4_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    pos_to_expert: torch.Tensor,
    scale_dtype: torch.dtype = _DEFAULT_SCALE_DTYPE,
) -> torch.Tensor:
    """Autograd-aware grouped FP4 linear for frozen DSV4 routed experts."""

    if torch.is_grad_enabled() and x.requires_grad:
        return _DSV4GroupedFP4LinearFn.apply(
            x, weight, scale, pos_to_expert, scale_dtype
        )
    return _grouped_fp4_linear_forward(x, weight, scale, pos_to_expert, scale_dtype)


def stack_fp4_expert_linear_params(
    experts: Iterable[nn.Module],
    linear_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stack one FP4 linear's weight/scale from each expert into grouped layout."""

    weights = []
    scales = []
    for expert in experts:
        linear = getattr(expert, linear_name)
        if linear.weight.dtype != torch.float4_e2m1fn_x2:
            raise RuntimeError(
                f"grouped FP4 {linear_name} requires torch.float4_e2m1fn_x2, "
                f"got {linear.weight.dtype}"
            )
        weights.append(linear.weight)
        scales.append(linear.scale)
    return torch.stack(weights).contiguous(), torch.stack(scales).contiguous()


__all__ = [
    "dsv4_deep_gemm_act_quant",
    "grouped_fp4_gemm",
    "grouped_fp4_gemm_torch_reference",
    "grouped_fp4_linear",
    "grouped_fp4_linear_backward_fast",
    "grouped_fp4_linear_backward_torch_reference",
    "grouped_fp4_linear_torch_surrogate",
    "has_deep_gemm_official_fp8_fp4",
    "pack_fp8_e8m0_scale_for_deep_gemm",
    "stack_fp4_expert_linear_params",
]
