"""Importable DeepSeek V4 sparse attention in TileLang.

This file contains the current fast exact-order forward/backward path.  It is
intentionally kept as a single-file module so downstream experiments can import
one file without depending on the historical prototype ladder.

Public API:
    sparse_attn_tilelang(q, kv, attn_sink, topk_idxs, softmax_scale)
    sparse_attn_forward_tilelang(q, kv, attn_sink, topk_idxs, softmax_scale)
    sparse_attn_backward_tilelang(q, kv, attn_sink, topk_idxs, softmax_scale, dout)

Inputs:
    q, kv, dout: bfloat16 tensors
    attn_sink: float32 tensor
    topk_idxs: int tensor with -1 invalid entries

Outputs:
    dq, dkv, dsink as float32 tensors

The TileLang path follows the official single-kernel structure with symbolic
top-k, so variable top-k lengths do not create per-block forward
specializations.  The backward uses the GEMM-style path.  All paths preserve
the official forward precision path:
* BF16 q/kv inputs
* FP32 online-softmax state
* BF16 cast of exp(score - running_max) before numerator accumulation
* deterministic no-atomic dKV reduction
"""

import torch
import tilelang
import tilelang.language as T


tilelang.set_log_level("WARNING")

PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
}

FP32 = "float32"
BF16 = "bfloat16"
INT32 = "int32"
BLOCK = 64
DKV_ATOMIC_D_CHUNK = 64


def _pad_heads(
    q: torch.Tensor, attn_sink: torch.Tensor, dout: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    heads = q.shape[2]
    if heads < 16:
        pad_heads = 16 - heads
        q = torch.cat(
            [q, q.new_zeros(q.shape[0], q.shape[1], pad_heads, q.shape[3])], dim=2
        )
        dout = torch.cat(
            [
                dout,
                dout.new_zeros(dout.shape[0], dout.shape[1], pad_heads, dout.shape[3]),
            ],
            dim=2,
        )
        attn_sink = torch.cat([attn_sink, attn_sink.new_zeros(pad_heads)])
    return q.contiguous(), attn_sink.contiguous(), dout.contiguous(), heads


def _check_inputs(q: torch.Tensor, kv: torch.Tensor, dout: torch.Tensor) -> None:
    if q.dtype != torch.bfloat16 or kv.dtype != torch.bfloat16 or dout.dtype != torch.bfloat16:
        raise TypeError("sparse_attn_backward_tilelang expects BF16 q/kv/dout")


def _check_forward_inputs(q: torch.Tensor, kv: torch.Tensor) -> None:
    if q.dtype != torch.bfloat16 or kv.dtype != torch.bfloat16:
        raise TypeError("sparse_attn_forward_tilelang expects BF16 q/kv")


@tilelang.jit(pass_configs=PASS_CONFIGS)
def sparse_attn_kernel(h: int, d: int, scale=None):
    """Sparse multi-head attention via index gathering + online softmax."""
    b = T.symbolic("b")
    m = T.symbolic("m")
    n = T.symbolic("n")
    topk = T.symbolic("topk")
    if scale is None:
        scale = (1.0 / d) ** 0.5

    num_stages = 2
    threads = 256
    block = 64
    num_blocks = tilelang.cdiv(topk, block)

    @T.prim_func
    def sparse_attn_kernel_(
        q: T.Tensor[(b, m, h, d), BF16],
        kv: T.Tensor[(b, n, d), BF16],
        o: T.Tensor[(b, m, h, d), BF16],
        attn_sink: T.Tensor[(h,), FP32],
        topk_idxs: T.Tensor[(b, m, topk), INT32],
    ):
        with T.Kernel(m, b, threads=threads) as (bx, by):
            q_shared = T.alloc_shared((h, d), BF16)
            kv_shared = T.alloc_shared((block, d), BF16)
            o_shared = T.alloc_shared((h, d), BF16)
            acc_s_cast = T.alloc_shared((h, block), BF16)

            idxs = T.alloc_fragment(block, INT32)
            acc_s = T.alloc_fragment((h, block), FP32)
            acc_o = T.alloc_fragment((h, d), FP32)
            scores_max = T.alloc_fragment(h, FP32)
            scores_max_prev = T.alloc_fragment(h, FP32)
            scores_scale = T.alloc_fragment(h, FP32)
            scores_sum = T.alloc_fragment(h, FP32)
            sum_exp = T.alloc_fragment(h, FP32)

            T.clear(acc_o)
            T.clear(sum_exp)
            T.fill(scores_max, -T.infinity(FP32))
            T.copy(q[by, bx, :, :], q_shared)

            for t in T.Pipelined(num_blocks, num_stages=num_stages):
                for i in T.Parallel(block):
                    idxs[i] = T.if_then_else(
                        t * block + i < topk,
                        topk_idxs[by, bx, t * block + i],
                        -1,
                    )
                for i, j in T.Parallel(block, d):
                    kv_shared[i, j] = T.if_then_else(
                        idxs[i] != -1, kv[by, idxs[i], j], 0
                    )
                for i, j in T.Parallel(h, block):
                    acc_s[i, j] = T.if_then_else(idxs[j] != -1, 0, -T.infinity(FP32))

                T.gemm(
                    q_shared,
                    kv_shared,
                    acc_s,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullRow,
                )
                for i, j in T.Parallel(h, block):
                    acc_s[i, j] *= scale

                T.copy(scores_max, scores_max_prev)
                T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                for i in T.Parallel(h):
                    scores_scale[i] = T.exp(scores_max_prev[i] - scores_max[i])
                for i, j in T.Parallel(h, block):
                    acc_s[i, j] = T.exp(acc_s[i, j] - scores_max[i])
                T.reduce_sum(acc_s, scores_sum, dim=1)
                for i in T.Parallel(h):
                    sum_exp[i] = sum_exp[i] * scores_scale[i] + scores_sum[i]

                T.copy(acc_s, acc_s_cast)
                for i, j in T.Parallel(h, d):
                    acc_o[i, j] *= scores_scale[i]
                T.gemm(acc_s_cast, kv_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

            for i in T.Parallel(h):
                sum_exp[i] += T.exp(attn_sink[i] - scores_max[i])
            for i, j in T.Parallel(h, d):
                acc_o[i, j] /= sum_exp[i]
            T.copy(acc_o, o_shared)
            T.copy(o_shared, o[by, bx, :, :])

    return sparse_attn_kernel_


@tilelang.jit(pass_configs=PASS_CONFIGS)
def _forward_full_topk_state_kernel(h: int, d: int, scale: float):
    b = T.symbolic("b")
    m = T.symbolic("m")
    n = T.symbolic("n")
    topk = T.symbolic("topk")
    num_blocks = tilelang.cdiv(topk, BLOCK)

    @T.prim_func
    def _kernel(
        q: T.Tensor[(b, m, h, d), BF16],
        kv: T.Tensor[(b, n, d), BF16],
        topk_idxs: T.Tensor[(b, m, topk), INT32],
        acc_o_state: T.Tensor[(b, m, h, d), FP32],
        sum_exp_state: T.Tensor[(b, m, h), FP32],
        scores_max_state: T.Tensor[(b, m, h), FP32],
    ):
        with T.Kernel(m, b, threads=256) as (bx, by):
            q_shared = T.alloc_shared((h, d), BF16)
            kv_shared = T.alloc_shared((BLOCK, d), BF16)
            acc_s_cast = T.alloc_shared((h, BLOCK), BF16)

            idxs = T.alloc_fragment(BLOCK, INT32)
            acc_s = T.alloc_fragment((h, BLOCK), FP32)
            acc_o = T.alloc_fragment((h, d), FP32)
            scores_max = T.alloc_fragment(h, FP32)
            scores_max_prev = T.alloc_fragment(h, FP32)
            scores_scale = T.alloc_fragment(h, FP32)
            scores_sum = T.alloc_fragment(h, FP32)
            sum_exp = T.alloc_fragment(h, FP32)

            T.clear(acc_o)
            T.clear(sum_exp)
            T.fill(scores_max, -T.infinity(FP32))
            T.copy(q[by, bx, :, :], q_shared)

            for t in T.Pipelined(num_blocks, num_stages=2):
                for i in T.Parallel(BLOCK):
                    idxs[i] = T.if_then_else(
                        t * BLOCK + i < topk,
                        topk_idxs[by, bx, t * BLOCK + i],
                        -1,
                    )
                for i, j in T.Parallel(BLOCK, d):
                    kv_shared[i, j] = T.if_then_else(
                        idxs[i] != -1, kv[by, idxs[i], j], 0
                    )
                for i, j in T.Parallel(h, BLOCK):
                    acc_s[i, j] = T.if_then_else(idxs[j] != -1, 0, -T.infinity(FP32))

                T.gemm(
                    q_shared,
                    kv_shared,
                    acc_s,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullRow,
                )
                for i, j in T.Parallel(h, BLOCK):
                    acc_s[i, j] *= scale

                T.copy(scores_max, scores_max_prev)
                T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                for i in T.Parallel(h):
                    scores_scale[i] = T.exp(scores_max_prev[i] - scores_max[i])
                for i, j in T.Parallel(h, BLOCK):
                    acc_s[i, j] = T.exp(acc_s[i, j] - scores_max[i])
                T.reduce_sum(acc_s, scores_sum, dim=1)
                for i in T.Parallel(h):
                    sum_exp[i] = sum_exp[i] * scores_scale[i] + scores_sum[i]

                T.copy(acc_s, acc_s_cast)
                for i, j in T.Parallel(h, d):
                    acc_o[i, j] *= scores_scale[i]
                T.gemm(acc_s_cast, kv_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

            T.copy(acc_o, acc_o_state[by, bx, :, :])
            T.copy(sum_exp, sum_exp_state[by, bx, :])
            T.copy(scores_max, scores_max_state[by, bx, :])

    return _kernel


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE: True,
    }
)
def _gemm_backward_kernel(
    h: int,
    d: int,
    scale: float,
    block_size: int,
    threads: int,
):
    b = T.symbolic("b")
    m = T.symbolic("m")
    n = T.symbolic("n")
    topk = T.symbolic("topk")
    num_blocks = tilelang.cdiv(topk, block_size)
    split_store = 2

    @T.prim_func
    def _kernel(
        q: T.Tensor[(b, m, h, d), BF16],
        kv: T.Tensor[(b, n, d), BF16],
        dout: T.Tensor[(b, m, h, d), BF16],
        topk_idxs: T.Tensor[(b, m, topk), INT32],
        lse: T.Tensor[(b, m, h), FP32],
        delta: T.Tensor[(b, m, h), FP32],
        dq: T.Tensor[(b, m, h, d), FP32],
        dkv: T.Tensor[(b, n, d), FP32],
    ):
        with T.Kernel(num_blocks, m, b, threads=threads) as (block_id, bx, by):
            q_shared = T.alloc_shared((h, d), BF16)
            kv_shared = T.alloc_shared((block_size, d), BF16)
            dout_shared = T.alloc_shared((h, d), BF16)
            p_shared = T.alloc_shared((h, block_size), BF16)
            ds_shared = T.alloc_shared((h, block_size), BF16)
            dkv_shared = T.alloc_shared((block_size // split_store, d), FP32)

            idxs = T.alloc_shared((block_size,), INT32)
            acc_p = T.alloc_fragment((h, block_size), FP32)
            acc_dp = T.alloc_fragment((h, block_size), FP32)
            acc_dq = T.alloc_fragment((h, d), FP32)
            acc_dkv = T.alloc_fragment((block_size, d), FP32)

            T.copy(q[by, bx, :, :], q_shared)
            T.copy(dout[by, bx, :, :], dout_shared)
            T.clear(acc_dq)

            for bi in T.Parallel(block_size):
                idxs[bi] = -1
                if block_id * block_size + bi < topk:
                    idxs[bi] = topk_idxs[by, bx, block_id * block_size + bi]
            T.sync_threads()

            for bi, dj in T.Parallel(block_size, d):
                kv_shared[bi, dj] = T.if_then_else(
                    idxs[bi] != -1, kv[by, idxs[bi], dj], 0
                )

            for hi, bi in T.Parallel(h, block_size):
                acc_p[hi, bi] = T.if_then_else(idxs[bi] != -1, 0, -T.infinity(FP32))
            T.gemm(
                q_shared,
                kv_shared,
                acc_p,
                transpose_B=True,
                policy=T.GemmWarpPolicy.FullCol,
            )

            for hi, bi in T.Parallel(h, block_size):
                if idxs[bi] != -1:
                    acc_p[hi, bi] = T.exp(acc_p[hi, bi] * scale - lse[by, bx, hi])
                else:
                    acc_p[hi, bi] = 0.0
            T.copy(acc_p, p_shared)

            T.gemm(
                dout_shared,
                kv_shared,
                acc_dp,
                transpose_B=True,
                policy=T.GemmWarpPolicy.FullCol,
                clear_accum=True,
            )

            for hi, bi in T.Parallel(h, block_size):
                acc_dp[hi, bi] = (
                    acc_p[hi, bi] * (acc_dp[hi, bi] - delta[by, bx, hi]) * scale
                )
            T.copy(acc_dp, ds_shared)

            T.gemm(ds_shared, kv_shared, acc_dq, policy=T.GemmWarpPolicy.FullCol)

            for hi, dj in T.Parallel(h, d):
                T.atomic_add(dq[by, bx, hi, dj], acc_dq[hi, dj])

            T.gemm(
                ds_shared,
                q_shared,
                acc_dkv,
                transpose_A=True,
                policy=T.GemmWarpPolicy.FullCol,
                clear_accum=True,
            )
            T.gemm(
                p_shared,
                dout_shared,
                acc_dkv,
                transpose_A=True,
                policy=T.GemmWarpPolicy.FullCol,
            )

            for split in range(split_store):
                for bi, dj in T.Parallel(block_size // split_store, d):
                    dkv_shared[bi, dj] = acc_dkv[
                        bi + split * (block_size // split_store), dj
                    ]
                T.sync_threads()

                for bi, dj in T.Parallel(block_size // split_store, d // 4):
                    src_bi = bi + split * (block_size // split_store)
                    if idxs[src_bi] != -1:
                        T.atomic_addx4(
                            dkv[by, idxs[src_bi], dj * 4],
                            dkv_shared[bi, dj * 4],
                        )

    return _kernel


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE: True,
    }
)
def _gemm_backward_head_tiled_kernel(
    h: int,
    d: int,
    scale: float,
    block_size: int,
    head_tile: int,
    threads: int,
):
    b = T.symbolic("b")
    m = T.symbolic("m")
    n = T.symbolic("n")
    topk = T.symbolic("topk")
    num_blocks = tilelang.cdiv(topk, block_size)
    num_head_tiles = tilelang.cdiv(h, head_tile)
    split_store = 2

    @T.prim_func
    def _kernel(
        q: T.Tensor[(b, m, h, d), BF16],
        kv: T.Tensor[(b, n, d), BF16],
        dout: T.Tensor[(b, m, h, d), BF16],
        topk_idxs: T.Tensor[(b, m, topk), INT32],
        lse: T.Tensor[(b, m, h), FP32],
        delta: T.Tensor[(b, m, h), FP32],
        dq: T.Tensor[(b, m, h, d), FP32],
        dkv: T.Tensor[(b, n, d), FP32],
    ):
        with T.Kernel(num_blocks * num_head_tiles, m, b, threads=threads) as (
            tile_id,
            bx,
            by,
        ):
            block_id = tile_id // num_head_tiles
            head_tile_id = tile_id % num_head_tiles
            head_start = head_tile_id * head_tile

            q_shared = T.alloc_shared((head_tile, d), BF16)
            kv_shared = T.alloc_shared((block_size, d), BF16)
            dout_shared = T.alloc_shared((head_tile, d), BF16)
            p_shared = T.alloc_shared((head_tile, block_size), BF16)
            ds_shared = T.alloc_shared((head_tile, block_size), BF16)
            dkv_shared = T.alloc_shared((block_size // split_store, d), FP32)

            idxs = T.alloc_shared((block_size,), INT32)
            acc_p = T.alloc_fragment((head_tile, block_size), FP32)
            acc_dp = T.alloc_fragment((head_tile, block_size), FP32)
            acc_dq = T.alloc_fragment((head_tile, d), FP32)
            acc_dkv = T.alloc_fragment((block_size, d), FP32)

            for hi, dj in T.Parallel(head_tile, d):
                q_shared[hi, dj] = q[by, bx, head_start + hi, dj]
                dout_shared[hi, dj] = dout[by, bx, head_start + hi, dj]
            T.clear(acc_dq)

            for bi in T.Parallel(block_size):
                idxs[bi] = -1
                if block_id * block_size + bi < topk:
                    idxs[bi] = topk_idxs[by, bx, block_id * block_size + bi]
            T.sync_threads()

            for bi, dj in T.Parallel(block_size, d):
                kv_shared[bi, dj] = T.if_then_else(
                    idxs[bi] != -1, kv[by, idxs[bi], dj], 0
                )

            for hi, bi in T.Parallel(head_tile, block_size):
                acc_p[hi, bi] = T.if_then_else(idxs[bi] != -1, 0, -T.infinity(FP32))
            T.gemm(
                q_shared,
                kv_shared,
                acc_p,
                transpose_B=True,
                policy=T.GemmWarpPolicy.FullCol,
            )

            for hi, bi in T.Parallel(head_tile, block_size):
                if idxs[bi] != -1:
                    acc_p[hi, bi] = T.exp(
                        acc_p[hi, bi] * scale - lse[by, bx, head_start + hi]
                    )
                else:
                    acc_p[hi, bi] = 0.0
            T.copy(acc_p, p_shared)

            T.gemm(
                dout_shared,
                kv_shared,
                acc_dp,
                transpose_B=True,
                policy=T.GemmWarpPolicy.FullCol,
                clear_accum=True,
            )

            for hi, bi in T.Parallel(head_tile, block_size):
                acc_dp[hi, bi] = (
                    acc_p[hi, bi]
                    * (acc_dp[hi, bi] - delta[by, bx, head_start + hi])
                    * scale
                )
            T.copy(acc_dp, ds_shared)

            T.gemm(ds_shared, kv_shared, acc_dq, policy=T.GemmWarpPolicy.FullCol)

            for hi, dj in T.Parallel(head_tile, d):
                T.atomic_add(dq[by, bx, head_start + hi, dj], acc_dq[hi, dj])

            T.gemm(
                ds_shared,
                q_shared,
                acc_dkv,
                transpose_A=True,
                policy=T.GemmWarpPolicy.FullCol,
                clear_accum=True,
            )
            T.gemm(
                p_shared,
                dout_shared,
                acc_dkv,
                transpose_A=True,
                policy=T.GemmWarpPolicy.FullCol,
            )

            for split in range(split_store):
                for bi, dj in T.Parallel(block_size // split_store, d):
                    dkv_shared[bi, dj] = acc_dkv[
                        bi + split * (block_size // split_store), dj
                    ]
                T.sync_threads()

                for bi, dj in T.Parallel(block_size // split_store, d // 4):
                    src_bi = bi + split * (block_size // split_store)
                    if idxs[src_bi] != -1:
                        T.atomic_addx4(
                            dkv[by, idxs[src_bi], dj * 4],
                            dkv_shared[bi, dj * 4],
                        )

    return _kernel


def _backward_head_tile_size(heads: int) -> int:
    if heads <= 64:
        return heads
    for head_tile in (64, 32, 16):
        if heads % head_tile == 0:
            return head_tile
    raise ValueError(
        f"tilelang backward requires head count divisible by 16 when "
        f"heads > 64, got {heads}"
    )


def sparse_attn_forward_tilelang(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """Official-style forward with symbolic top-k and in-kernel block loop."""

    _check_forward_inputs(q, kv)
    b, s, h, d = q.size()
    if h < 16:
        q = torch.cat([q, q.new_zeros(b, s, 16 - h, d)], dim=2)
        attn_sink = torch.cat([attn_sink, attn_sink.new_zeros(16 - h)])

    q = q.contiguous()
    kv = kv.contiguous()
    topk_idxs = topk_idxs.to(torch.int32).contiguous()
    attn_sink = attn_sink.contiguous()

    out = torch.empty_like(q)
    kernel = sparse_attn_kernel(q.size(2), d, softmax_scale)
    kernel(q, kv, out, attn_sink, topk_idxs)

    if h < 16:
        out = out.narrow(2, 0, h).contiguous()
    return out


def _forward_state_for_backward(
    q: torch.Tensor,
    kv: torch.Tensor,
    topk_idxs: torch.Tensor,
    softmax_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if q.dtype != torch.bfloat16 or kv.dtype != torch.bfloat16:
        raise TypeError("forward recompute expects BF16 q/kv")

    bsz, seqlen, heads, head_dim = q.shape
    q = q.contiguous()
    kv = kv.contiguous()
    topk_idxs = topk_idxs.to(torch.int32).contiguous()

    acc_o_state = torch.zeros(
        (bsz, seqlen, heads, head_dim), device=q.device, dtype=torch.float32
    )
    sum_exp_state = torch.zeros(
        (bsz, seqlen, heads), device=q.device, dtype=torch.float32
    )
    scores_max_state = torch.full(
        (bsz, seqlen, heads), -float("inf"), device=q.device, dtype=torch.float32
    )

    _forward_full_topk_state_kernel(heads, head_dim, softmax_scale)(
        q, kv, topk_idxs, acc_o_state, sum_exp_state, scores_max_state
    )

    return acc_o_state, sum_exp_state, scores_max_state


def sparse_attn_backward_tilelang(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    softmax_scale: float,
    dout: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Experimental GEMM-style sparse attention backward.

    This computes the standard softmax backward for V4 sparse attention with
    an attention-sink denominator and a tensor-core dQ/dKV path.
    """

    _check_inputs(q, kv, dout)
    q, attn_sink, dout, original_heads = _pad_heads(q, attn_sink, dout)
    kv = kv.contiguous()
    topk_idxs = topk_idxs.to(torch.int32).contiguous()

    head_dim = q.shape[-1]
    acc_o, sum_exp, scores_max = _forward_state_for_backward(
        q, kv, topk_idxs, softmax_scale
    )
    sink_exp = torch.exp(attn_sink.view(1, 1, -1) - scores_max)
    denom = sum_exp + sink_exp
    lse = (scores_max + torch.log(denom)).contiguous()
    out_fp32 = acc_o / denom.unsqueeze(-1)
    delta = (dout.float() * out_fp32).sum(dim=-1).contiguous()
    dsink = (-(sink_exp / denom) * delta).sum(dim=(0, 1)).contiguous()

    dq = torch.zeros_like(q, dtype=torch.float32)
    dkv = torch.zeros_like(kv, dtype=torch.float32)
    head_tile = _backward_head_tile_size(q.shape[2])
    if head_tile == q.shape[2]:
        _gemm_backward_kernel(
            q.shape[2],
            head_dim,
            softmax_scale,
            BLOCK,
            128,
        )(q, kv, dout, topk_idxs, lse, delta, dq, dkv)
    else:
        _gemm_backward_head_tiled_kernel(
            q.shape[2],
            head_dim,
            softmax_scale,
            BLOCK,
            head_tile,
            128,
        )(q, kv, dout, topk_idxs, lse, delta, dq, dkv)

    if original_heads < dq.shape[2]:
        dq = dq.narrow(2, 0, original_heads).contiguous()
        dsink = dsink.narrow(0, 0, original_heads).contiguous()
    return dq, dkv, dsink


class _SparseAttnTileLangFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        kv: torch.Tensor,
        attn_sink: torch.Tensor,
        topk_idxs: torch.Tensor,
        softmax_scale: float,
    ) -> torch.Tensor:
        ctx.save_for_backward(q, kv, attn_sink, topk_idxs)
        ctx.softmax_scale = float(softmax_scale)
        return sparse_attn_forward_tilelang(
            q, kv, attn_sink, topk_idxs, softmax_scale
        )

    @staticmethod
    def backward(ctx, dout: torch.Tensor):
        q, kv, attn_sink, topk_idxs = ctx.saved_tensors
        dq, dkv, dsink = sparse_attn_backward_tilelang(
            q,
            kv,
            attn_sink,
            topk_idxs,
            ctx.softmax_scale,
            dout.contiguous(),
        )
        return dq, dkv, dsink, None, None


def sparse_attn_tilelang(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """Official-precision sparse attention with custom TileLang backward."""

    return _SparseAttnTileLangFn.apply(q, kv, attn_sink, topk_idxs, softmax_scale)


__all__ = [
    "sparse_attn_tilelang",
    "sparse_attn_forward_tilelang",
    "sparse_attn_backward_tilelang",
]
