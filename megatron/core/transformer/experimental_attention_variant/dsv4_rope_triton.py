"""Fused fp32-precision RoPE-splice kernel for DSV4 MLA.

Replaces the eager ``_splice_rotary`` in ``deepseek_v4.py`` which materialises
four intermediate tensors per call (fp32 upcast, complex64 product, bf16 cast,
full-size cat). This kernel keeps fp32 arithmetic in registers and emits a
single bf16 output, avoiding those intermediates. Numerically equivalent to
the eager path.

Batch-invariant by construction: element-wise per (token, head); fixed
constexpr tile sizes; no atomics; no cross-batch reductions.
"""

from typing import Tuple

import torch
import triton
import triton.language as tl


@triton.jit
def _splice_rotary_kernel(
    x_ptr,                 # [N, HEAD_DIM] input (bf16 or fp16 or fp32)
    out_ptr,               # [N, HEAD_DIM] output (same dtype as x)
    freqs_real_ptr,        # [seqlen, ROPE_DIM/2] fp32
    freqs_imag_ptr,        # [seqlen, ROPE_DIM/2] fp32
    N,
    seqlen,
    inner_stride,
    HEAD_DIM: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    STATIC_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_STATIC: tl.constexpr,
    BLOCK_ROPE_HALF: tl.constexpr,
    INVERSE: tl.constexpr,
):
    pid = tl.program_id(0)
    n_offs = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = n_offs < N

    # Static dims: byte-copy in tiles of BLOCK_STATIC.
    if STATIC_DIM > 0:
        for s_start in tl.static_range(0, STATIC_DIM, BLOCK_STATIC):
            s_offs = s_start + tl.arange(0, BLOCK_STATIC)
            s_mask = s_offs < STATIC_DIM
            ptrs = x_ptr + n_offs[:, None] * HEAD_DIM + s_offs[None, :]
            x_st = tl.load(
                ptrs, mask=n_mask[:, None] & s_mask[None, :], other=0.0
            )
            tl.store(
                out_ptr + n_offs[:, None] * HEAD_DIM + s_offs[None, :],
                x_st,
                mask=n_mask[:, None] & s_mask[None, :],
            )

    # RoPE dims: pair (real, imag) at offsets STATIC_DIM+2i / STATIC_DIM+2i+1.
    rope_half = ROPE_DIM // 2
    pair_offs = tl.arange(0, BLOCK_ROPE_HALF)
    pair_mask = pair_offs < rope_half
    real_col = STATIC_DIM + 2 * pair_offs
    imag_col = real_col + 1

    seq_idx = (n_offs // inner_stride) % seqlen  # int32

    x_real = tl.load(
        x_ptr + n_offs[:, None] * HEAD_DIM + real_col[None, :],
        mask=n_mask[:, None] & pair_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    x_imag = tl.load(
        x_ptr + n_offs[:, None] * HEAD_DIM + imag_col[None, :],
        mask=n_mask[:, None] & pair_mask[None, :],
        other=0.0,
    ).to(tl.float32)

    f_real = tl.load(
        freqs_real_ptr + seq_idx[:, None] * rope_half + pair_offs[None, :],
        mask=n_mask[:, None] & pair_mask[None, :],
        other=0.0,
    )
    f_imag = tl.load(
        freqs_imag_ptr + seq_idx[:, None] * rope_half + pair_offs[None, :],
        mask=n_mask[:, None] & pair_mask[None, :],
        other=0.0,
    )

    if INVERSE:
        f_imag = -f_imag

    out_real = x_real * f_real - x_imag * f_imag
    out_imag = x_real * f_imag + x_imag * f_real

    tl.store(
        out_ptr + n_offs[:, None] * HEAD_DIM + real_col[None, :],
        out_real,
        mask=n_mask[:, None] & pair_mask[None, :],
    )
    tl.store(
        out_ptr + n_offs[:, None] * HEAD_DIM + imag_col[None, :],
        out_imag,
        mask=n_mask[:, None] & pair_mask[None, :],
    )


def _next_pow2(x: int) -> int:
    p = 1
    while p < x:
        p <<= 1
    return p


def _resolve_layout(x: torch.Tensor) -> Tuple[int, int]:
    """(seqlen, inner_stride) for DSV4's two supported layouts.

    - 4D [batch, seqlen, heads, head_dim]: heads share freqs_cis at seq pos.
      Flat n = b*S*H + s*H + h -> seq_idx(n) = (n // H) % S; inner_stride = H.
    - 3D [batch, seqlen, head_dim]: inner_stride = 1.
    - 2D [seqlen, head_dim]: treated as 3D with batch=1.
    """
    if x.ndim == 4:
        return x.shape[1], x.shape[2]
    if x.ndim == 3:
        return x.shape[1], 1
    if x.ndim == 2:
        return x.shape[0], 1
    raise ValueError(f"splice_rotary_triton: unsupported x.ndim={x.ndim}")


def _splice_rotary_forward(
    x: torch.Tensor, rope_dim: int, freqs_cis: torch.Tensor, inverse: bool
) -> torch.Tensor:
    head_dim = x.shape[-1]
    assert rope_dim <= head_dim and rope_dim % 2 == 0
    assert freqs_cis.is_complex(), "freqs_cis must be complex (complex64/128)"
    static_dim = head_dim - rope_dim
    rope_half = rope_dim // 2

    seqlen, inner_stride = _resolve_layout(x)

    x_c = x.contiguous()
    out = torch.empty_like(x_c)
    N = x_c.numel() // head_dim

    freqs_real = freqs_cis.real.contiguous()
    freqs_imag = freqs_cis.imag.contiguous()
    if freqs_real.ndim != 2:
        freqs_real = freqs_real.reshape(-1, rope_half)
        freqs_imag = freqs_imag.reshape(-1, rope_half)
    assert freqs_real.shape[-1] == rope_half, (
        f"freqs_cis last dim {freqs_real.shape[-1]} != rope_dim/2 = {rope_half}"
    )
    assert freqs_real.shape[0] >= seqlen, (
        f"freqs_cis seqlen {freqs_real.shape[0]} < x seqlen {seqlen}"
    )

    BLOCK_N = 64
    BLOCK_STATIC = 64
    BLOCK_ROPE_HALF = _next_pow2(rope_half)

    grid = (triton.cdiv(N, BLOCK_N),)
    _splice_rotary_kernel[grid](
        x_c, out,
        freqs_real, freqs_imag,
        N, seqlen, inner_stride,
        HEAD_DIM=head_dim,
        ROPE_DIM=rope_dim,
        STATIC_DIM=static_dim,
        BLOCK_N=BLOCK_N,
        BLOCK_STATIC=BLOCK_STATIC,
        BLOCK_ROPE_HALF=BLOCK_ROPE_HALF,
        INVERSE=inverse,
    )
    return out.view(x.shape)


class _SpliceRotaryFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, rope_dim, freqs_cis, inverse):
        ctx.save_for_backward(freqs_cis)
        ctx.rope_dim = int(rope_dim)
        ctx.inverse = bool(inverse)
        return _splice_rotary_forward(x, ctx.rope_dim, freqs_cis, ctx.inverse)

    @staticmethod
    def backward(ctx, dy):
        (freqs_cis,) = ctx.saved_tensors
        # RoPE is unitary; adjoint = inverse rotation. Flip the inverse flag.
        dx = _splice_rotary_forward(
            dy.contiguous(), ctx.rope_dim, freqs_cis, not ctx.inverse
        )
        return dx, None, None, None


def splice_rotary_triton(
    x: torch.Tensor,
    rope_dim: int,
    freqs_cis: torch.Tensor,
    inverse: bool = False,
) -> torch.Tensor:
    """Drop-in replacement for ``deepseek_v4._splice_rotary``.

    Applies RoPE (fp32 math, in-register) to the trailing ``rope_dim`` features
    of ``x`` and returns a new tensor; static prefix is copied through.
    """
    return _SpliceRotaryFn.apply(x, rope_dim, freqs_cis, inverse)
