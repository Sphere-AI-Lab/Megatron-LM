"""Deterministic block-128 fp8 GEMM, drop-in for TileLang fp8_gemm.

Interface (matches DSV4-Flash's fp8_gemm in the official
DeepSeek-V4-Flash ``inference/kernel.py``):

    C[M, N] = A[M, K] @ B[N, K].T  with block-128 fp8 scaling on both A and B.

A is fp8_e4m3fn [M, K], B is fp8_e4m3fn [N, K].  Scales are stored with
one value per 128 elements along K (and along N for B).

Determinism strategy
====================

* **One CTA per output tile** (block_M=32, block_N=128) - no split-K, no
  cross-CTA atomic reduction. The scheduler is single-program-per-tile.
* **K-loop has fixed program order** - each output tile owns its full K
  reduction, so no split-K atomics or cross-CTA accumulation are involved.
* **`tl.dot` with default fp32 accumulator** - Triton emits sync
  `mma.sync.aligned.m16n8k32` on Hopper / sm_8x and the deterministic
  tcgen05 sync variant on Blackwell. Because there's a single warpgroup
  per CTA and a serial K-loop, the fragment arrival order is fixed by
  the IR.
* **Per-K-block scale multiplication is fp32**, applied to the partial
  product *after* the dot but *before* the outer fp32 accumulator add.
  Since K-iter order is serial, accumulator order is bit-stable.

This is a correctness-first deterministic baseline. Expect ~30-50%
slower than TileLang's pipelined fp8_gemm; the bench script measures it.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------
@triton.jit
def _det_fp8_gemm_kernel(
    A_ptr, B_ptr, C_ptr, sa_ptr, sb_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    stride_sam, stride_sak,
    stride_sbn, stride_sbk,
    BLOCK_M: tl.constexpr,    # 32
    BLOCK_N: tl.constexpr,    # 128
    BLOCK_K: tl.constexpr,    # 128 (= scale group size)
):
    """One CTA computes one (BLOCK_M, BLOCK_N) output tile.

    A: [M, K] fp8_e4m3, row-major.
    B: [N, K] fp8_e4m3, row-major; used as B.T inside the dot.
    sa: [M, K/128] fp32 (per-row K-block scale).
    sb: [N/128, K/128] fp32 (per-N-block, per-K-block scale).
    C: [M, N] bf16.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_row_ptrs = A_ptr + offs_m[:, None] * stride_am
    b_row_ptrs = B_ptr + offs_n[:, None] * stride_bn

    mask_m = offs_m < M
    mask_n = offs_n < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    n_k_iters = tl.cdiv(K, BLOCK_K)
    # Serial K-loop: one CTA owns the full output tile accumulation.
    for kit in range(0, n_k_iters):
        k_offs = kit * BLOCK_K + offs_k
        mask_k = k_offs < K

        a_ptrs = a_row_ptrs + k_offs[None, :] * stride_ak
        b_ptrs = b_row_ptrs + k_offs[None, :] * stride_bk
        a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=mask_n[:, None] & mask_k[None, :], other=0.0)

        # tl.dot(a, b.T): a is [BLOCK_M, BLOCK_K] fp8, b.T is [BLOCK_K, BLOCK_N] fp8.
        # fp32 accumulator inside the dot keeps the fragment-level reduction stable.
        partial = tl.dot(a, b.T, out_dtype=tl.float32)

        # Per-block scale correction. Scales are fp32.
        sa = tl.load(sa_ptr + offs_m * stride_sam + kit * stride_sak,
                     mask=mask_m, other=0.0).to(tl.float32)
        sb = tl.load(sb_ptr + pid_n * stride_sbn + kit * stride_sbk).to(tl.float32)

        acc += partial * sa[:, None] * sb

    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc.to(tl.bfloat16),
             mask=mask_m[:, None] & mask_n[None, :])


# ---------------------------------------------------------------------------
# Wrapper - interface-compatible with kernel.fp8_gemm
# ---------------------------------------------------------------------------
_BLOCK_N = 128
_BLOCK_K = 128


def _pick_block_m(M: int) -> int:
    """Choose BLOCK_M.

    Pinned to a single value across all M so that bit-stable per-row output
    does not depend on caller-driven batch size. Variable BLOCK_M (and the
    coupled num_warps) changed Triton's mma.sync layout enough to produce
    1-ULP bf16 rounding diffs for the same input row when called with
    different M (observed in DSV4 NaiveEP DP-MoE shared expert running on
    a scheduler-dependent gathered global).
    """
    return 32


def det_fp8_gemm(
    a: torch.Tensor,
    a_s: torch.Tensor,
    b: torch.Tensor,
    b_s: torch.Tensor,
    scale_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """C = A @ B.T  with block-128 fp8 scaling.  bf16 output.

    A: [..., K] fp8_e4m3
    B: [N, K]   fp8_e4m3
    a_s: [..., K/128]   ue8m0 or fp32
    b_s: [N/128, K/128] ue8m0 or fp32
    """
    assert a.is_contiguous() and b.is_contiguous()
    assert a_s.is_contiguous() and b_s.is_contiguous()
    assert a.dtype == torch.float8_e4m3fn and b.dtype == torch.float8_e4m3fn
    assert b.dim() == 2 and b.size(1) == a.size(-1)

    K = a.size(-1)
    M = a.numel() // K
    N = b.size(0)
    assert K % _BLOCK_K == 0, f"K={K} must be a multiple of {_BLOCK_K}"

    a2d = a.view(M, K)
    sa2d = a_s.view(M, -1).to(torch.float32).contiguous()
    sb_f32 = b_s.to(torch.float32).contiguous()
    assert sa2d.size(1) == K // _BLOCK_K
    assert sb_f32.dim() == 2 and sb_f32.size(0) == triton.cdiv(N, _BLOCK_N)
    assert sb_f32.size(1) == K // _BLOCK_K

    c = a.new_empty(*a.shape[:-1], N, dtype=torch.bfloat16)
    c2d = c.view(M, N)

    block_m = _pick_block_m(M)
    # num_warps tuned by tile size: smaller tile means fewer warps.
    if block_m <= 16:
        num_warps = 2
    elif block_m <= 32:
        num_warps = 4
    else:
        num_warps = 8
    grid = (triton.cdiv(M, block_m), triton.cdiv(N, _BLOCK_N))
    # num_stages=3: deterministic per-iter accumulation order is preserved
    # (the K-loop reduction is still a single python-level for-loop in IR).
    # The 3-stage async cp.async pipeline only overlaps loads, never reorders
    # the mma to fp32 acc chain.
    _det_fp8_gemm_kernel[grid](
        a2d, b, c2d, sa2d, sb_f32,
        M, N, K,
        a2d.stride(0), a2d.stride(1),
        b.stride(0), b.stride(1),
        c2d.stride(0), c2d.stride(1),
        sa2d.stride(0), sa2d.stride(1),
        sb_f32.stride(0), sb_f32.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=_BLOCK_N,
        BLOCK_K=_BLOCK_K,
        num_warps=num_warps,
        num_stages=3,
    )
    return c


# ===========================================================================
# Deterministic act_quant - matches kernel.act_quant in DSV4-Flash
# ===========================================================================
#
# Per-128-block fp8 e4m3 quantization with optional ue8m0 (power-of-2)
# scale rounding.  TileLang's version uses a pipelined warp-cooperative
# absmax reduction that has been observed to drift across CG replays.
# Pure abs/max is associative, so the math is deterministic; this version
# uses a serial single-warp tile schedule to make that determinism survive
# CG capture + replay.
# ---------------------------------------------------------------------------


@triton.jit
def _det_act_quant_kernel(
    X_ptr, Y_ptr, S_ptr,
    M, N,
    stride_xm, stride_xn,
    stride_ym, stride_yn,
    stride_sm, stride_sn,
    BLOCK_M: tl.constexpr,    # 32 rows per CTA
    GROUP: tl.constexpr,      # 128 block_size
    ROUND_SCALE: tl.constexpr,
):
    """One CTA quantizes a (BLOCK_M, GROUP) tile.

    pid_m: row-block index. pid_n: column-block index (each block has GROUP
    columns and produces one scale per row).
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)        # [BLOCK_M]
    offs_n = pid_n * GROUP + tl.arange(0, GROUP)            # [GROUP]

    mask_m = offs_m < M
    mask_n = offs_n < N

    x_ptrs = X_ptr + offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn
    x = tl.load(x_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0.0).to(tl.float32)

    # Per-row absmax over the GROUP columns. tl.max is associative and
    # implemented as a deterministic tree reduction in Triton IR.
    amax = tl.max(tl.abs(x), axis=1)                        # [BLOCK_M]
    amax = tl.maximum(amax, 1e-4)

    fp8_max_inv = 1.0 / 448.0
    if ROUND_SCALE:
        # Round to nearest power of 2 (ue8m0): scale = 2^ceil(log2(amax * fp8_max_inv)).
        # Implemented via fp32 bit ops to avoid log/exp transcendentals.
        # exp = ceil(log2(s_unrounded)). For positive normal floats,
        # ceil(log2(v)) = (bits >> 23) - 127 if v is exactly a power of 2,
        # else (bits >> 23) - 126 (round up by 1).
        s_un = amax * fp8_max_inv
        bits = s_un.to(tl.uint32, bitcast=True)
        mantissa = bits & 0x7FFFFF
        exp = ((bits >> 23) & 0xFF).to(tl.int32) - 127
        # If mantissa != 0, round up.
        round_up = (mantissa != 0).to(tl.int32)
        exp_rounded = exp + round_up
        # Reconstruct power-of-2 fp32: bits = (exp_rounded + 127) << 23.
        new_bits = ((exp_rounded + 127) & 0xFF).to(tl.uint32) << 23
        scale = new_bits.to(tl.float32, bitcast=True)
    else:
        scale = amax * fp8_max_inv

    # Quantize: y = clamp(x / scale, -448, 448) to fp8_e4m3.
    y = (x / scale[:, None]).to(tl.float32)
    y = tl.minimum(tl.maximum(y, -448.0), 448.0)

    y_ptrs = Y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    tl.store(y_ptrs, y.to(tl.float8e4nv),
             mask=mask_m[:, None] & mask_n[None, :])

    s_ptrs = S_ptr + offs_m * stride_sm + pid_n * stride_sn
    tl.store(s_ptrs, scale, mask=mask_m)


def det_act_quant(
    x: torch.Tensor,
    block_size: int = 128,
    scale_fmt: str | None = None,
    scale_dtype: torch.dtype = torch.float32,
    inplace: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Block-128 fp8 quant.  Returns (q, scale).

    Mirrors kernel.act_quant signature.  scale_fmt='ue8m0' rounds scale to
    nearest power of 2.  scale_dtype controls the storage dtype of the
    returned scale tensor.
    """
    assert not inplace, "det_act_quant inplace path not implemented (not used by DSV4)"
    assert x.is_contiguous()
    K = x.size(-1)
    assert K % block_size == 0
    M = x.numel() // K
    x2d = x.view(M, K)

    y = torch.empty_like(x2d, dtype=torch.float8_e4m3fn)
    s_fp32 = torch.empty(M, K // block_size, dtype=torch.float32, device=x.device)

    grid_m = triton.cdiv(M, 32)
    grid_n = K // block_size
    _det_act_quant_kernel[(grid_m, grid_n)](
        x2d, y, s_fp32,
        M, K,
        x2d.stride(0), x2d.stride(1),
        y.stride(0), y.stride(1),
        s_fp32.stride(0), s_fp32.stride(1),
        BLOCK_M=32,
        GROUP=block_size,
        ROUND_SCALE=(scale_fmt == "ue8m0"),
        num_warps=4,
        num_stages=1,
    )

    if scale_dtype == torch.float8_e8m0fnu:
        s_out = s_fp32.to(torch.float8_e8m0fnu)
    elif scale_dtype == torch.float32:
        s_out = s_fp32
    else:
        raise ValueError(f"Unsupported scale_dtype {scale_dtype}")

    return y.view(*x.shape[:-1], K), s_out.view(*x.shape[:-1], K // block_size)
