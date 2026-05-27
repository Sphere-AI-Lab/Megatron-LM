# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Kernel-backed Cayley-Neumann transform for DSV4 OFT.

This mirrors Megatron-Bridge's OFT Triton path:
R = I + 2Q + 2Q^2 + 2Q^3 + Q^4 for skew-symmetric Q.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

NUM_TERMS = 5
_TRITON_MAX_BLOCK_SIZE_FP32 = 32
_TRITON_MAX_BLOCK_SIZE_FP16 = 128


def _torch_cayley_neumann(q_skew: torch.Tensor, num_terms: int = NUM_TERMS) -> torch.Tensor:
    b, block_size, _ = q_skew.shape
    rotation = torch.eye(
        block_size,
        device=q_skew.device,
        dtype=q_skew.dtype,
    ).repeat(b, 1, 1)
    if num_terms > 1:
        rotation.add_(q_skew, alpha=2.0)
        if num_terms > 2:
            q_squared = torch.bmm(q_skew, q_skew)
            rotation.add_(q_squared, alpha=2.0)
            q_power = q_squared
            for _ in range(3, num_terms - 1):
                q_power = torch.bmm(q_power, q_skew)
                rotation.add_(q_power, alpha=2.0)
            q_power = torch.bmm(q_power, q_skew)
            rotation.add_(q_power)
    return rotation


@triton.jit
def _cayley_fwd_kernel(
    q_ptr,
    r_ptr,
    stride_b,
    stride_r,
    stride_c,
    BLOCK_SIZE: tl.constexpr,
):
    bid = tl.program_id(0)
    rows = tl.arange(0, BLOCK_SIZE)
    cols = tl.arange(0, BLOCK_SIZE)

    q_base = q_ptr + bid * stride_b
    q = tl.load(q_base + rows[:, None] * stride_r + cols[None, :] * stride_c)

    eye = (rows[:, None] == cols[None, :]).to(q.dtype)
    rotation = eye + 2.0 * q

    q2 = tl.dot(q, q, input_precision="ieee").to(q.dtype)
    rotation = rotation + 2.0 * q2

    q3 = tl.dot(q2, q, input_precision="ieee").to(q.dtype)
    rotation = rotation + 2.0 * q3

    q4 = tl.dot(q3, q, input_precision="ieee").to(q.dtype)
    rotation = rotation + q4

    r_base = r_ptr + bid * stride_b
    tl.store(
        r_base + rows[:, None] * stride_r + cols[None, :] * stride_c,
        rotation.to(q.dtype),
    )


@triton.jit
def _cayley_bwd_kernel(
    grad_r_ptr,
    q_ptr,
    grad_q_ptr,
    stride_b,
    stride_r,
    stride_c,
    BLOCK_SIZE: tl.constexpr,
):
    bid = tl.program_id(0)
    rows = tl.arange(0, BLOCK_SIZE)
    cols = tl.arange(0, BLOCK_SIZE)

    q_base = q_ptr + bid * stride_b
    q = tl.load(q_base + rows[:, None] * stride_r + cols[None, :] * stride_c)
    grad_r_base = grad_r_ptr + bid * stride_b
    grad_r = tl.load(grad_r_base + rows[:, None] * stride_r + cols[None, :] * stride_c)

    q_t = tl.trans(q)
    g_prev = grad_r
    acc = grad_r

    for _ in range(3):
        g_k = (2.0 * grad_r + tl.dot(g_prev, q_t, input_precision="ieee")).to(grad_r.dtype)
        g_prev = g_k
        acc = (g_k + tl.dot(q_t, acc, input_precision="ieee")).to(grad_r.dtype)

    grad_q_base = grad_q_ptr + bid * stride_b
    tl.store(
        grad_q_base + rows[:, None] * stride_r + cols[None, :] * stride_c,
        acc.to(grad_r.dtype),
    )


def cayley_neumann_fwd(q_skew: torch.Tensor, num_terms: int = NUM_TERMS) -> torch.Tensor:
    assert num_terms == NUM_TERMS, f"Only num_terms={NUM_TERMS} supported, got {num_terms}"
    num_blocks, block_size, _ = q_skew.shape
    rotation = torch.empty_like(q_skew)
    _cayley_fwd_kernel[(num_blocks,)](
        q_skew,
        rotation,
        q_skew.stride(0),
        q_skew.stride(1),
        q_skew.stride(2),
        BLOCK_SIZE=block_size,
    )
    return rotation


def cayley_neumann_bwd(
    grad_r: torch.Tensor,
    q_skew: torch.Tensor,
    num_terms: int = NUM_TERMS,
) -> torch.Tensor:
    assert num_terms == NUM_TERMS, f"Only num_terms={NUM_TERMS} supported, got {num_terms}"
    num_blocks, block_size, _ = q_skew.shape
    grad_q = torch.empty_like(q_skew)
    _cayley_bwd_kernel[(num_blocks,)](
        grad_r,
        q_skew,
        grad_q,
        q_skew.stride(0),
        q_skew.stride(1),
        q_skew.stride(2),
        BLOCK_SIZE=block_size,
        num_warps=4 if block_size <= 32 else 8,
        num_stages=1,
    )
    return grad_q


class CayleyNeumannFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q_skew: torch.Tensor, num_terms: int) -> torch.Tensor:
        ctx.save_for_backward(q_skew)
        ctx.num_terms = num_terms
        return cayley_neumann_fwd(q_skew, num_terms)

    @staticmethod
    def backward(ctx, grad_r: torch.Tensor):
        (q_skew,) = ctx.saved_tensors
        grad_q = cayley_neumann_bwd(grad_r.contiguous(), q_skew, ctx.num_terms)
        return grad_q, None


def cayley_neumann(q_skew: torch.Tensor, num_terms: int = NUM_TERMS) -> torch.Tensor:
    block_size = q_skew.shape[-1]
    if block_size < 16 or block_size > 256:
        raise ValueError(f"DSV4 OFT block_size must be in [16, 256], got {block_size}")

    max_bs = (
        _TRITON_MAX_BLOCK_SIZE_FP32
        if q_skew.dtype == torch.float32
        else _TRITON_MAX_BLOCK_SIZE_FP16
    )
    if block_size > max_bs:
        return _torch_cayley_neumann(q_skew, num_terms)
    return CayleyNeumannFunction.apply(q_skew, num_terms)


__all__ = [
    "cayley_neumann",
    "cayley_neumann_fwd",
    "cayley_neumann_bwd",
]
