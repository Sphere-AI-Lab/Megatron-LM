# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Routed expert OFT rotation for DeepSeek V4 grouped MoE.

Megatron's DeepEP path hands the expert core tokens that are already grouped by
local expert. This module applies a per-expert block-diagonal OFT rotation to
that grouped layout without Python expert loops or boolean indexing, so the
forward path can be captured by CUDA graphs.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _dsv4_routed_oft_forward_kernel(
    x_ptr,
    stride_xm,
    stride_xk,
    out_ptr,
    stride_om,
    stride_ok,
    r_ptr,
    stride_re,
    stride_rb,
    stride_ri,
    stride_rj,
    pos_to_expert_ptr,
    num_rows: tl.constexpr,
    OFT_BLOCK_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    TILE_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_blk = tl.program_id(1)

    row_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int64)
    token_mask = row_offsets < num_rows
    row_experts = tl.load(
        pos_to_expert_ptr + row_offsets,
        mask=token_mask,
        other=-1,
    ).to(tl.int64)
    expert = tl.load(pos_to_expert_ptr + pid_m * BLOCK_M).to(tl.int64)
    valid_rows = token_mask & (expert >= 0) & (row_experts == expert)
    k_base = pid_blk * OFT_BLOCK_SIZE

    accum = tl.zeros((BLOCK_M, OFT_BLOCK_SIZE), dtype=tl.float32)
    if expert >= 0:
        for k_off in range(0, OFT_BLOCK_SIZE, TILE_K):
            k_inner = k_off + tl.arange(0, TILE_K).to(tl.int64)
            k_mask = k_inner < OFT_BLOCK_SIZE
            x_k = k_base + k_inner
            x_ptrs = x_ptr + row_offsets[:, None] * stride_xm + x_k[None, :] * stride_xk
            x_tile = tl.load(x_ptrs, mask=valid_rows[:, None] & k_mask[None, :], other=0.0)

            r_cols = tl.arange(0, OFT_BLOCK_SIZE).to(tl.int64)
            r_ptrs = (
                r_ptr
                + expert * stride_re
                + pid_blk * stride_rb
                + k_inner[:, None] * stride_ri
                + r_cols[None, :] * stride_rj
            )
            r_tile = tl.load(r_ptrs, mask=k_mask[:, None], other=0.0)
            accum += tl.dot(x_tile, r_tile, input_precision="ieee")

    out_k = k_base + tl.arange(0, OFT_BLOCK_SIZE).to(tl.int64)
    out_ptrs = out_ptr + row_offsets[:, None] * stride_om + out_k[None, :] * stride_ok
    tl.store(out_ptrs, accum.to(out_ptr.dtype.element_ty), mask=token_mask[:, None])


@triton.jit
def _dsv4_routed_oft_grad_r_kernel(
    x_ptr,
    stride_xm,
    stride_xk,
    grad_y_ptr,
    stride_gm,
    stride_gk,
    grad_r_ptr,
    stride_re,
    stride_rb,
    stride_ri,
    stride_rj,
    pos_to_expert_ptr,
    num_m_blocks: tl.constexpr,
    OFT_BLOCK_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
    MAX_BLOCK_SEARCH_ITERS: tl.constexpr,
):
    expert_idx = tl.program_id(0).to(tl.int64)
    block_idx = tl.program_id(1).to(tl.int64)

    row_offsets = tl.arange(0, OFT_BLOCK_SIZE).to(tl.int64)
    col_offsets = tl.arange(0, OFT_BLOCK_SIZE).to(tl.int64)
    token_offsets = tl.arange(0, BLOCK_M).to(tl.int64)
    k_base = block_idx * OFT_BLOCK_SIZE

    # pos_to_expert is expert-major by construction. Find this expert's
    # contiguous row-block range, matching SGLang's routed OFT grad-R path.
    lo = tl.full((), 0, dtype=tl.int64)
    hi = tl.full((), num_m_blocks, dtype=tl.int64)
    for _ in range(0, MAX_BLOCK_SEARCH_ITERS):
        active = lo < hi
        mid = (lo + hi) // 2
        val = tl.load(
            pos_to_expert_ptr + mid * BLOCK_M,
            mask=active,
            other=expert_idx,
        ).to(tl.int64)
        val = tl.where(val < 0, expert_idx + 1, val)
        go_right = val < expert_idx
        lo = tl.where(active & go_right, mid + 1, lo)
        hi = tl.where(active & ~go_right, mid, hi)
    start_block = lo

    lo = start_block
    hi = tl.full((), num_m_blocks, dtype=tl.int64)
    for _ in range(0, MAX_BLOCK_SEARCH_ITERS):
        active = lo < hi
        mid = (lo + hi) // 2
        val = tl.load(
            pos_to_expert_ptr + mid * BLOCK_M,
            mask=active,
            other=expert_idx + 1,
        ).to(tl.int64)
        val = tl.where(val < 0, expert_idx + 1, val)
        go_right = val <= expert_idx
        lo = tl.where(active & go_right, mid + 1, lo)
        hi = tl.where(active & ~go_right, mid, hi)
    end_block = lo

    acc = tl.zeros((OFT_BLOCK_SIZE, OFT_BLOCK_SIZE), dtype=tl.float32)
    m_block = start_block
    while m_block < end_block:
        rows = m_block * BLOCK_M + token_offsets
        row_experts = tl.load(pos_to_expert_ptr + rows).to(tl.int64)
        valid_rows = row_experts == expert_idx

        x_ptrs = x_ptr + rows[:, None] * stride_xm + (k_base + row_offsets[None, :]) * stride_xk
        grad_ptrs = (
            grad_y_ptr + rows[:, None] * stride_gm + (k_base + col_offsets[None, :]) * stride_gk
        )
        x_tile = tl.load(x_ptrs, mask=valid_rows[:, None], other=0.0)
        grad_tile = tl.load(grad_ptrs, mask=valid_rows[:, None], other=0.0)
        acc += tl.dot(tl.trans(x_tile), grad_tile, input_precision="ieee")
        m_block += 1

    out_ptrs = (
        grad_r_ptr
        + expert_idx * stride_re
        + block_idx * stride_rb
        + row_offsets[:, None] * stride_ri
        + col_offsets[None, :] * stride_rj
    )
    tl.store(out_ptrs, acc.to(OUT_DTYPE))


def _tile_k(block_size: int) -> int:
    if block_size % 64 == 0:
        return 64
    if block_size % 32 == 0:
        return 32
    return min(64, block_size)


def _check_inputs(
    x: torch.Tensor,
    oft_r: torch.Tensor,
    pos_to_expert: torch.Tensor,
    block_m: int,
) -> None:
    if x.dim() != 2:
        raise ValueError("x must be a 2D tensor")
    if oft_r.dim() != 4:
        raise ValueError("oft_r must have shape (num_experts, num_blocks, bs, bs)")
    if oft_r.shape[2] != oft_r.shape[3]:
        raise ValueError("oft_r blocks must be square")
    if x.shape[1] != oft_r.shape[1] * oft_r.shape[2]:
        raise ValueError("x hidden dimension must equal num_blocks * block_size")
    if pos_to_expert.numel() != x.shape[0]:
        raise ValueError("pos_to_expert length must match x rows")
    if x.shape[0] % block_m != 0:
        raise ValueError(
            f"DSV4 routed OFT expects rows padded to block_m={block_m}; got {x.shape[0]}"
        )
    if pos_to_expert.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"pos_to_expert must be int32/int64, got {pos_to_expert.dtype}")


def _empty_like_rotation_output(x: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x)


def dsv4_routed_oft_rotation_forward(
    x: torch.Tensor,
    oft_r: torch.Tensor,
    pos_to_expert: torch.Tensor,
    *,
    block_m: int = 32,
) -> torch.Tensor:
    """Apply per-row expert OFT rotation to grouped expert rows."""
    if x.numel() == 0:
        return x
    _check_inputs(x, oft_r, pos_to_expert, block_m)

    x = x.contiguous()
    pos_to_expert = pos_to_expert.contiguous()
    out = _empty_like_rotation_output(x)

    _, num_blocks, block_size, _ = oft_r.shape
    grid = (triton.cdiv(x.shape[0], block_m), num_blocks)
    _dsv4_routed_oft_forward_kernel[grid](
        x,
        x.stride(0),
        x.stride(1),
        out,
        out.stride(0),
        out.stride(1),
        oft_r,
        oft_r.stride(0),
        oft_r.stride(1),
        oft_r.stride(2),
        oft_r.stride(3),
        pos_to_expert,
        num_rows=x.shape[0],
        OFT_BLOCK_SIZE=block_size,
        BLOCK_M=block_m,
        TILE_K=_tile_k(block_size),
    )
    return out


def dsv4_routed_oft_rotation_grad_r(
    x: torch.Tensor,
    grad_y: torch.Tensor,
    oft_r: torch.Tensor,
    pos_to_expert: torch.Tensor,
    *,
    block_m: int = 32,
) -> torch.Tensor:
    if x.numel() == 0:
        return torch.zeros_like(oft_r)
    _check_inputs(x, oft_r, pos_to_expert, block_m)
    if grad_y.shape != x.shape:
        raise ValueError("grad_y shape must match x shape")

    x = x.contiguous()
    grad_y = grad_y.contiguous()
    pos_to_expert = pos_to_expert.contiguous()
    oft_r_c = oft_r.contiguous()
    grad_r = torch.empty_like(oft_r_c)

    dtype_map = {
        torch.float16: tl.float16,
        torch.bfloat16: tl.bfloat16,
        torch.float32: tl.float32,
    }
    if oft_r_c.dtype not in dtype_map:
        raise TypeError(f"unsupported oft_r dtype: {oft_r_c.dtype}")

    num_experts, num_blocks, block_size, _ = oft_r_c.shape
    num_m_blocks = x.shape[0] // block_m
    _dsv4_routed_oft_grad_r_kernel[(num_experts, num_blocks)](
        x,
        x.stride(0),
        x.stride(1),
        grad_y,
        grad_y.stride(0),
        grad_y.stride(1),
        grad_r,
        grad_r.stride(0),
        grad_r.stride(1),
        grad_r.stride(2),
        grad_r.stride(3),
        pos_to_expert,
        num_m_blocks=num_m_blocks,
        OFT_BLOCK_SIZE=block_size,
        BLOCK_M=block_m,
        OUT_DTYPE=dtype_map[oft_r_c.dtype],
        MAX_BLOCK_SEARCH_ITERS=max(1, math.ceil(math.log2(num_m_blocks + 1))),
    )
    return grad_r


class _DSV4RoutedOFTRotationFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        oft_r: torch.Tensor,
        pos_to_expert: torch.Tensor,
        block_m: int,
    ) -> torch.Tensor:
        x_c = x.contiguous()
        oft_r_c = oft_r.contiguous()
        pos_c = pos_to_expert.contiguous()
        ctx.save_for_backward(x_c, oft_r_c, pos_c)
        ctx.block_m = int(block_m)
        return dsv4_routed_oft_rotation_forward(
            x_c,
            oft_r_c,
            pos_c,
            block_m=ctx.block_m,
        )

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x, oft_r, pos_to_expert = ctx.saved_tensors
        grad_x = grad_r = None
        grad_output = grad_output.contiguous()
        if ctx.needs_input_grad[0]:
            grad_x = dsv4_routed_oft_rotation_forward(
                grad_output,
                oft_r.transpose(-1, -2),
                pos_to_expert,
                block_m=ctx.block_m,
            )
        if ctx.needs_input_grad[1]:
            grad_r = dsv4_routed_oft_rotation_grad_r(
                x,
                grad_output,
                oft_r,
                pos_to_expert,
                block_m=ctx.block_m,
            )
        return grad_x, grad_r, None, None


def dsv4_routed_oft_rotation(
    x: torch.Tensor,
    oft_r: torch.Tensor,
    pos_to_expert: torch.Tensor,
    *,
    block_m: int = 32,
) -> torch.Tensor:
    """Autograd-aware routed OFT rotation for grouped DSV4 expert rows."""
    return _DSV4RoutedOFTRotationFn.apply(x, oft_r, pos_to_expert, block_m)
