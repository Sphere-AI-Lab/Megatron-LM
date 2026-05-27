# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""DeepSeek V4 experimental attention variant.

This module is intentionally scoped to the native DeepSeek V4 attention structure:
duplicated low-rank Q/KV down projections, tensor-parallel Q/O up projections,
window plus compressed sparse attention, and grouped output projection.

Environment variables:
    MEGATRON_DSV4_ATTN_IMPL ("tilelang", default "tilelang"):
        Only the autograd-friendly TileLang sparse-attention training kernel is
        supported in the model path.
    MEGATRON_USE_KV_QAT ("0"|"1", default "1"):
        Enables in-place FP8/FP4 activation QAT on KV non-rope dims, indexer
        Q post-rotate, and compressor KV. The default matches the official
        inference (always on). When enabled, the activation tensors must
        have last dimensions divisible by FP8 block_size=64 and FP4
        block_size=32; mini-config tests that violate this should set "0".
    MEGATRON_DSV4_CANONICAL_RMSNORM ("0"|"1", default "1"):
        Uses a batch-invariant RMSNorm path so full prefill uses the same
        reduction shape as singleton decode.
    MEGATRON_DSV4_CANONICAL_COMPRESSOR_LINEAR ("0"|"1", default "0"):
        Computes compressor wkv/wgate through the selected batch-invariant
        canonical helper so full prefill emits the same compressed KV as
        stateful decode.
    MEGATRON_DSV4_CANONICAL_COMPRESSOR_LINEAR_IMPL ("fixed_order"|"torch", default "fixed_order"):
        Selects the compressor helper implementation. "fixed_order" uses
        batch-invariant fp32 Triton and "torch" uses vectorized GEMM.
    MEGATRON_DSV4_USE_TILEKERNELS ("0"|"1", default "1"):
        Uses DeepSeek TileKernels for the default V4 MHC pre/head and router
        path. Missing TileKernels or incompatible dtype/shape is a hard error
        so production does not drift onto a different numerical path.
    MEGATRON_DSV4_GATE_LINEAR_IMPL ("persistent"|"torch", default "persistent"):
        Default score-router projection implementation when TileKernels router
        is enabled. "persistent" is the batch-invariant production path and
        "torch" is the simple fallback/reference path.
    MEGATRON_DSV4_TILE_ROUTER_STE_BACKWARD ("0"|"1", default "1"):
        Keeps the persistent/TileKernels router forward values, but attaches a
        PyTorch backward path for the gate projection and selected top-k weights.
    MEGATRON_DSV4_PREFILL_FIXED_WINDOW_LAYOUT ("0"|"1", default "0"):
        Diagnostic mode for short compressed-attention prefill. When a full
        prefill sequence is shorter than the sliding window, pad the vanilla
        KV tier to window_size and place compressed KV after that, matching
        the stateful decode cache layout.
    MEGATRON_DSV4_RECOMPUTE_ATTN_ROUND ("0"|"1", default "0"):
        Checkpoints each V4 layer's attention hyper-connection round inside
        the normal full-layer checkpoint, trading extra attention replay for
        lower backward recompute peak memory.
"""

import math
import os
from functools import lru_cache
from types import SimpleNamespace
from typing import Dict, Optional, Sequence, Tuple

import tilelang
import tilelang.language as T
import torch
import torch.nn as nn
import torch.nn.functional as F
from fast_hadamard_transform import hadamard_transform

from megatron.core import tensor_parallel
from megatron.core.dist_checkpointing.mapping import ShardedStateDict, ShardedTensor
from megatron.core.dist_checkpointing.utils import replace_prefix_for_sharding
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.mappings import (
    copy_to_tensor_model_parallel_region,
    gather_from_sequence_parallel_region,
    reduce_from_tensor_model_parallel_region,
    reduce_scatter_to_sequence_parallel_region,
    scatter_to_sequence_parallel_region,
)
from megatron.core.transformer.experimental_attention_variant import dsv4_kernels
from megatron.core.transformer.experimental_attention_variant.dsv4_linear import (
    DSV4ColumnParallelLinear,
    DSV4Linear,
    DSV4RowParallelLinear,
)
from megatron.core.transformer.experimental_attention_variant.dsv4_q_rmsnorm_triton import (
    dsv4_q_rmsnorm,
)
from megatron.core.transformer.experimental_attention_variant.dsv4_oft_cayley import (
    cayley_neumann as dsv4_oft_cayley_neumann,
)
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_block import get_num_layers_to_build
from megatron.core.transformer.transformer_config import MLATransformerConfig
from megatron.core.transformer.transformer_layer import (
    TransformerLayer,
    get_transformer_layer_offset,
)
from megatron.core.transformer.utils import (
    ensure_metadata_has_dp_cp_group,
    make_sharded_tensors_for_checkpoint,
    sharded_state_dict_default,
)
from megatron.core.typed_torch import apply_module
from megatron.core.utils import get_pg_rank, make_viewless_tensor

tilelang.set_log_level("WARNING")

_DSV4_DENSE_LINEAR_BACKWARD_DTYPES = {
    torch.float16,
    torch.bfloat16,
    torch.float32,
}
_DSV4_QUANT_LINEAR_WEIGHT_DTYPES = {
    torch.float8_e4m3fn,
    torch.float4_e2m1fn_x2,
}


_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
}


def _dsv4_canonical_impl(env_name: str, default: str = "fixed_order") -> str:
    return os.environ.get(
        env_name,
        os.environ.get("MEGATRON_DSV4_CANONICAL_KERNEL_IMPL", default),
    ).lower()


def _dsv4_best_linear_impl(env_name: str) -> str:
    return os.environ.get(env_name, "persistent").lower()


def _dsv4_tile_router_ste_backward_enabled() -> bool:
    return os.environ.get("MEGATRON_DSV4_TILE_ROUTER_STE_BACKWARD", "1") != "0"


def _dsv4_is_cuda_graph_capturing() -> bool:
    try:
        from megatron.core.transformer.cuda_graphs import is_graph_capturing, is_graph_warmup
    except ImportError:
        return False
    return is_graph_capturing() or is_graph_warmup()


def _dsv4_dense_linear_backward_supported(x: torch.Tensor, weight: torch.Tensor) -> bool:
    return (
        x.is_floating_point()
        and weight.is_floating_point()
        and x.dtype in _DSV4_DENSE_LINEAR_BACKWARD_DTYPES
        and weight.dtype in _DSV4_DENSE_LINEAR_BACKWARD_DTYPES
    )


def _dsv4_reject_quant_linear_weight(weight: torch.Tensor, env_name: str) -> None:
    if weight.dtype in _DSV4_QUANT_LINEAR_WEIGHT_DTYPES:
        raise RuntimeError(
            f"{env_name} helper kernels only support dense linear weights; got "
            f"{weight.dtype}. Quantized DeepSeek V4 weights must use DSV4Linear's "
            "scale-aware FP8/FP4 path instead of being cast to a dense dtype."
        )


def _dsv4_attach_torch_linear_backward(
    forward_out: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    if (
        not torch.is_grad_enabled()
        or not (x.requires_grad or weight.requires_grad)
        or not _dsv4_dense_linear_backward_supported(x, weight)
    ):
        return forward_out
    backward_weight = weight
    if (
        backward_weight.dtype != x.dtype
        and backward_weight.is_floating_point()
        and x.is_floating_point()
    ):
        backward_weight = backward_weight.to(x.dtype)
    backward_out = F.linear(x, backward_weight)
    return backward_out + (forward_out - backward_out).detach()


def _dsv4_attach_tile_router_weight_backward(
    raw_scores: torch.Tensor,
    indices: torch.Tensor,
    tile_weights: torch.Tensor,
    score_func: str,
    route_scale: float,
) -> torch.Tensor:
    if (
        not _dsv4_tile_router_ste_backward_enabled()
        or not torch.is_grad_enabled()
        or not raw_scores.requires_grad
    ):
        return tile_weights

    if score_func == "sigmoid":
        scores = raw_scores.sigmoid()
    elif score_func == "sqrtsoftplus":
        scores = F.softplus(raw_scores).sqrt()
    else:
        raise RuntimeError(
            f"Unsupported DSV4 TileKernels router score function {score_func!r}."
        )

    valid = indices >= 0
    safe_indices = indices.masked_fill(~valid, 0)
    selected = scores.gather(1, safe_indices)
    selected = selected.masked_fill(~valid, 0.0)
    denom = selected.sum(dim=-1, keepdim=True) + 1e-20
    backward_weights = selected / denom * float(route_scale)
    backward_weights = backward_weights.masked_fill(~valid, 0.0)
    return backward_weights + (tile_weights - backward_weights).detach()


def _dsv4_persistent_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    env_name: str,
) -> torch.Tensor:
    from megatron.core.transformer.custom_layers.batch_invariant_kernels import (
        BatchInvariantTEGemmFn,
        matmul_persistent,
    )

    _dsv4_reject_quant_linear_weight(weight, env_name)
    if (
        torch.is_grad_enabled()
        and (x.requires_grad or weight.requires_grad)
        and _dsv4_dense_linear_backward_supported(x, weight)
        and x.dtype == weight.dtype
    ):
        return BatchInvariantTEGemmFn.apply(weight, x, None, None, "TN")

    flat = x.reshape(-1, x.shape[-1]).contiguous()
    if flat.shape[0] == 0:
        return x.new_empty(*x.shape[:-1], weight.shape[0])
    weight_t = weight.t().contiguous()
    if weight_t.dtype != flat.dtype:
        weight_t = weight_t.to(flat.dtype)
    out = matmul_persistent(flat, weight_t)
    out = out.reshape(*x.shape[:-1], weight.shape[0])
    return _dsv4_attach_torch_linear_backward(out, x, weight)


def _dsv4_canonical_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    env_name: str,
) -> torch.Tensor:
    _dsv4_reject_quant_linear_weight(weight, env_name)
    impl = _dsv4_canonical_impl(env_name)
    if impl == "torch":
        return F.linear(x, weight)
    if impl == "fixed_order":
        from megatron.core.transformer.custom_layers.batch_invariant_kernels import (
            LinearFixedOrderFn,
            matmul_fixed_order,
        )

        flat = x.reshape(-1, x.shape[-1]).contiguous()
        if flat.shape[0] == 0:
            return x.new_empty(*x.shape[:-1], weight.shape[0])
        if flat.dtype != torch.float32:
            flat = flat.float()
        weight_for_linear = weight if weight.dtype == torch.float32 else weight.float()
        if (
            torch.is_grad_enabled()
            and (flat.requires_grad or weight_for_linear.requires_grad)
            and flat.dtype == torch.float32
            and weight_for_linear.dtype == torch.float32
        ):
            out = LinearFixedOrderFn.apply(flat, weight_for_linear, None)
        else:
            weight_t = weight_for_linear.t().contiguous()
            out = matmul_fixed_order(flat, weight_t)
        out = out.reshape(*x.shape[:-1], weight.shape[0])
        return out
    raise ValueError(
        f"Unsupported {env_name}={impl!r}; expected fixed_order or torch"
    )


def _dsv4_best_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    env_name: str,
) -> torch.Tensor:
    impl = _dsv4_best_linear_impl(env_name)
    if impl == "torch":
        return F.linear(x, weight)
    if impl == "persistent":
        return _dsv4_persistent_linear(x, weight, env_name)
    raise ValueError(
        f"Unsupported {env_name}={impl!r}; expected persistent or torch"
    )


def _dsv4_use_tilekernels() -> bool:
    return os.environ.get("MEGATRON_DSV4_USE_TILEKERNELS", "1") != "0"


@lru_cache(maxsize=None)
def _dsv4_tile_mhc_ops():
    try:
        from tile_kernels.modeling.mhc.ops.head_compute_mix import mhc_head_compute_mix
        from tile_kernels.modeling.mhc.ops.norm_fn import mhc_pre_norm_fn
        from tile_kernels.modeling.mhc.ops.pre_apply_mix import mhc_pre_apply_mix
        from tile_kernels.modeling.mhc.ops.pre_big_fuse import mhc_pre_big_fuse
        from tile_kernels.modeling.mhc.ops.pre_split_mixes import mhc_pre_split_mixes
        from tile_kernels.modeling.mhc.ops.sinkhorn import sinkhorn_normalize
    except Exception as exc:
        raise RuntimeError(
            "MEGATRON_DSV4_USE_TILEKERNELS=1 requires DeepSeek TileKernels "
            "installed in the active Python environment."
        ) from exc
    return (
        mhc_pre_big_fuse,
        mhc_pre_norm_fn,
        mhc_pre_split_mixes,
        sinkhorn_normalize,
        mhc_pre_apply_mix,
        mhc_head_compute_mix,
    )


@lru_cache(maxsize=None)
def _dsv4_tile_router_op():
    try:
        from tile_kernels.moe.top2_sum_gate_kernel import top2_sum_gate
    except Exception as exc:
        raise RuntimeError(
            "MEGATRON_DSV4_USE_TILEKERNELS=1 requires "
            "tile_kernels.moe.top2_sum_gate_kernel."
        ) from exc
    return top2_sum_gate


class _Dsv4MHCContiguousGradFn(torch.autograd.Function):
    """Pass-through forward that gives TileKernels MHC a dense output grad."""

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        return x

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        return grad_output.contiguous()


def _dsv4_mhc_contiguous_grad(x: torch.Tensor) -> torch.Tensor:
    if torch.is_grad_enabled() and x.requires_grad:
        return _Dsv4MHCContiguousGradFn.apply(x)
    return x


@tilelang.jit(pass_configs=_PASS_CONFIGS)
def _hc_split_sinkhorn_kernel(hc: int, sinkhorn_iters: int, eps: float):
    """DeepSeek V4 official TileLang Sinkhorn splitter for mHC mixes."""
    n = T.symbolic("n")
    mix_hc = (2 + hc) * hc
    threads = 64

    @T.prim_func
    def _kernel(
        mixes: T.Tensor[(n, mix_hc), T.float32],
        hc_scale: T.Tensor[(3,), T.float32],
        hc_base: T.Tensor[(mix_hc,), T.float32],
        pre: T.Tensor[(n, hc), T.float32],
        post: T.Tensor[(n, hc), T.float32],
        comb: T.Tensor[(n, hc, hc), T.float32],
    ):
        with T.Kernel(n, threads=threads) as i:
            mixes_shared = T.alloc_shared(mix_hc, T.float32)
            comb_frag = T.alloc_fragment((hc, hc), T.float32)
            T.copy(mixes[i, :], mixes_shared)

            for j in T.Parallel(hc):
                pre[i, j] = T.sigmoid(mixes_shared[j] * hc_scale[0] + hc_base[j]) + eps
            for j in T.Parallel(hc):
                post[i, j] = 2 * T.sigmoid(
                    mixes_shared[j + hc] * hc_scale[1] + hc_base[j + hc]
                )
            for j, k in T.Parallel(hc, hc):
                comb_frag[j, k] = (
                    mixes_shared[j * hc + k + hc * 2] * hc_scale[2]
                    + hc_base[j * hc + k + hc * 2]
                )

            row_sum = T.alloc_fragment(hc, T.float32)
            col_sum = T.alloc_fragment(hc, T.float32)
            row_max = T.alloc_fragment(hc, T.float32)

            T.reduce_max(comb_frag, row_max, dim=1)
            for j, k in T.Parallel(hc, hc):
                comb_frag[j, k] = T.exp(comb_frag[j, k] - row_max[j])
            T.reduce_sum(comb_frag, row_sum, dim=1)
            for j, k in T.Parallel(hc, hc):
                comb_frag[j, k] = comb_frag[j, k] / row_sum[j] + eps

            T.reduce_sum(comb_frag, col_sum, dim=0)
            for j, k in T.Parallel(hc, hc):
                comb_frag[j, k] = comb_frag[j, k] / (col_sum[k] + eps)

            for _ in T.serial(sinkhorn_iters - 1):
                T.reduce_sum(comb_frag, row_sum, dim=1)
                for j, k in T.Parallel(hc, hc):
                    comb_frag[j, k] = comb_frag[j, k] / (row_sum[j] + eps)
                T.reduce_sum(comb_frag, col_sum, dim=0)
                for j, k in T.Parallel(hc, hc):
                    comb_frag[j, k] = comb_frag[j, k] / (col_sum[k] + eps)

            T.copy(comb_frag, comb[i, :, :])

    return _kernel


def hc_split_sinkhorn(
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int = 4,
    sinkhorn_iters: int = 20,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """DeepSeek V4 official mHC split/sinkhorn wrapper."""
    batch, seqlen, _ = mixes.size()
    pre = mixes.new_empty(batch, seqlen, hc_mult)
    post = mixes.new_empty(batch, seqlen, hc_mult)
    comb = mixes.new_empty(batch, seqlen, hc_mult, hc_mult)
    kernel = _hc_split_sinkhorn_kernel(hc_mult, sinkhorn_iters, eps)
    kernel(
        mixes.view(-1, (2 + hc_mult) * hc_mult),
        hc_scale,
        hc_base,
        pre.view(-1, hc_mult),
        post.view(-1, hc_mult),
        comb.view(-1, hc_mult, hc_mult),
    )
    return pre, post, comb


class HCHeadParams(MegatronModule):
    """DeepSeek V4 model-level mHC head parameters."""

    def __init__(self, config: MLATransformerConfig):
        super().__init__(config=config)
        hc_mult = config.dsv4_hc_mult
        hc_dim = hc_mult * config.hidden_size
        self.hc_head_fn = nn.Parameter(torch.empty(hc_mult, hc_dim, dtype=torch.float32))
        self.hc_head_base = nn.Parameter(torch.empty(hc_mult, dtype=torch.float32))
        self.hc_head_scale = nn.Parameter(torch.empty(1, dtype=torch.float32))

        for param in (self.hc_head_fn, self.hc_head_base, self.hc_head_scale):
            param._keep_fp32 = True


class DeepSeekV4HyperConnectionUtil:
    """Utility for DeepSeek V4 manifold HyperConnection operations."""

    def __init__(self, config: MLATransformerConfig):
        self.norm_eps = config.layernorm_epsilon
        self.hc_mult = config.dsv4_hc_mult
        self.hc_sinkhorn_iters = config.dsv4_hc_sinkhorn_iters
        self.hc_eps = config.dsv4_hc_eps

    def hc_pre_raw(
        self,
        x: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        shape, dtype = x.size(), x.dtype
        if _dsv4_use_tilekernels():
            if dtype != torch.bfloat16:
                raise RuntimeError(
                    "TileKernels MHC pre expects bf16 residual input; "
                    f"got {dtype}."
                )
            (
                mhc_pre_big_fuse,
                mhc_pre_norm_fn,
                mhc_pre_split_mixes,
                sinkhorn_normalize,
                mhc_pre_apply_mix,
                _,
            ) = _dsv4_tile_mhc_ops()
            hc_fn = hc_fn.contiguous()
            hc_scale = hc_scale.contiguous()
            hc_base = hc_base.contiguous()
            if not torch.is_grad_enabled() or not x.requires_grad:
                with torch.no_grad():
                    post, comb, y = mhc_pre_big_fuse(
                        x,
                        hc_fn,
                        hc_scale,
                        hc_base,
                        rms_eps=self.norm_eps,
                        mhc_pre_eps=self.hc_eps,
                        mhc_sinkhorn_eps=self.hc_eps,
                        mhc_post_mult_value=2.0,
                        sinkhorn_repeat=self.hc_sinkhorn_iters,
                        n_splits=16,
                    )
                return y.to(dtype), post.squeeze(-1), comb

            mixes = mhc_pre_norm_fn(
                x,
                hc_fn,
                None,
                self.norm_eps,
                fuse_grad_acc=False,
                n_splits=16,
            )
            pre, post, comb = mhc_pre_split_mixes(
                mixes,
                hc_scale,
                hc_base,
                self.hc_mult,
                2.0,
                self.hc_eps,
            )
            comb = sinkhorn_normalize(
                comb, repeat=self.hc_sinkhorn_iters, eps=self.hc_eps
            )
            y = mhc_pre_apply_mix(x, pre)
            y = _dsv4_mhc_contiguous_grad(y)
            return y.to(dtype), post.squeeze(-1), comb

        x_flat = x.flatten(2).float()

        rsqrt = torch.rsqrt(x_flat.square().mean(-1, keepdim=True) + self.norm_eps)
        mixes = F.linear(x_flat, hc_fn) * rsqrt
        pre, post, comb = hc_split_sinkhorn(
            mixes,
            hc_scale,
            hc_base,
            self.hc_mult,
            self.hc_sinkhorn_iters,
            self.hc_eps,
        )

        y = torch.sum(pre.unsqueeze(-1) * x_flat.view(shape), dim=2)
        return y.to(dtype), post, comb

    def hc_post_raw(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
    ) -> torch.Tensor:
        if _dsv4_use_tilekernels() and (
            x.is_cuda and residual.is_cuda and post.is_cuda and comb.is_cuda
        ):
            hidden = x.shape[-1]
            hc_mult = residual.shape[-2]
            n_tokens = x.numel() // hidden
            from tile_kernels.modeling.mhc.ops import post as post_ops

            out = post_ops.mhc_post(
                x.contiguous().reshape(1, n_tokens, hidden),
                residual.contiguous().reshape(1, n_tokens, hc_mult, hidden),
                post.contiguous().reshape(1, n_tokens, hc_mult, 1),
                comb.contiguous().reshape(1, n_tokens, hc_mult, hc_mult),
            )
            return out.reshape(*x.shape[:-1], hc_mult, hidden)

        y = post.unsqueeze(-1) * x.unsqueeze(-2)
        y = y + torch.sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)
        return y.type_as(x)

    def hc_head_raw(
        self,
        x: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
    ) -> torch.Tensor:
        shape, dtype = x.size(), x.dtype
        if _dsv4_use_tilekernels():
            if dtype != torch.bfloat16:
                raise RuntimeError(
                    "TileKernels MHC head expects bf16 residual input; "
                    f"got {dtype}."
                )
            (
                _,
                mhc_pre_norm_fn,
                _,
                _,
                mhc_pre_apply_mix,
                mhc_head_compute_mix,
            ) = _dsv4_tile_mhc_ops()
            mhc_mult3 = self.hc_mult * (2 + self.hc_mult)
            hc_fn = hc_fn.contiguous()
            if hc_fn.shape[0] < mhc_mult3:
                hc_fn = F.pad(hc_fn, (0, 0, 0, mhc_mult3 - hc_fn.shape[0]))
            mixes = mhc_pre_norm_fn(
                x,
                hc_fn,
                None,
                self.norm_eps,
                fuse_grad_acc=False,
                n_splits=16,
            )[..., : self.hc_mult].contiguous()
            pre = mhc_head_compute_mix(
                mixes,
                hc_scale.reshape(1).contiguous(),
                hc_base.contiguous(),
                self.hc_eps,
            )
            y = mhc_pre_apply_mix(x, pre.unsqueeze(-1))
            y = _dsv4_mhc_contiguous_grad(y)
            return y.to(dtype)

        x_flat = x.flatten(2).float()

        rsqrt = torch.rsqrt(x_flat.square().mean(-1, keepdim=True) + self.norm_eps)
        mixes = F.linear(x_flat, hc_fn) * rsqrt
        pre = torch.sigmoid(mixes * hc_scale + hc_base) + self.hc_eps

        y = torch.sum(pre.unsqueeze(-1) * x_flat.view(shape), dim=2)
        return y.to(dtype)

    def block_expand(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states.unsqueeze(2).repeat(1, 1, self.hc_mult, 1)

    def layer_pre(
        self,
        hidden_states: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = hidden_states.permute(1, 0, 2, 3).contiguous()
        x, post, comb = self.hc_pre_raw(x, hc_fn, hc_scale, hc_base)
        return x.permute(1, 0, 2).contiguous(), post, comb

    def layer_post(
        self,
        output_with_bias: torch.Tensor | tuple[torch.Tensor, torch.Tensor | None],
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
    ) -> torch.Tensor:
        if isinstance(output_with_bias, tuple):
            out, bias = output_with_bias
            assert bias is None, "DeepSeek V4 native layers do not support bias outputs."
        else:
            out = output_with_bias

        out = out.permute(1, 0, 2).contiguous()
        residual = residual.permute(1, 0, 2, 3).contiguous()
        hidden_states = self.hc_post_raw(out, residual, post, comb)
        return hidden_states.permute(1, 0, 2, 3).contiguous()

    def block_head(
        self,
        hidden_states: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
    ) -> torch.Tensor:
        x = hidden_states.permute(1, 0, 2, 3).contiguous()
        x = self.hc_head_raw(x, hc_fn, hc_scale, hc_base)
        return x.permute(1, 0, 2).contiguous()


class RMSNorm(nn.Module):
    """DeepSeek-style RMSNorm with fp32 weight and fp32 variance accumulation."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        canonical_impl = _dsv4_canonical_impl(
            "MEGATRON_DSV4_CANONICAL_RMSNORM_IMPL", default="persistent"
        )
        if (
            os.environ.get("MEGATRON_DSV4_CANONICAL_RMSNORM", "1") == "1"
            and canonical_impl in ("row", "exact")
        ):
            shape = x.shape
            flat = x.reshape(-1, shape[-1])
            if flat.numel() == 0:
                return x
            rows = []
            for row in flat:
                row = row.float()
                row = row * torch.rsqrt(row.square().mean(-1, keepdim=True) + self.eps)
                rows.append((self.weight * row).to(dtype))
            return torch.stack(rows, dim=0).reshape(shape)
        if (
            os.environ.get("MEGATRON_DSV4_CANONICAL_RMSNORM", "1") == "1"
            and canonical_impl in ("persistent", "batch_invariant", "triton", "accelerated")
        ):
            from megatron.core.transformer.custom_layers.batch_invariant_kernels import (
                rmsnorm_batch_invariant,
            )

            return rmsnorm_batch_invariant(x.float(), self.weight, self.eps).to(dtype)
        x = x.float()
        x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps)
        return (self.weight * x).to(dtype)


def rotate_activation(x: torch.Tensor) -> torch.Tensor:
    """Scaled Hadamard transform used by DeepSeek V4 compressed index paths."""
    assert x.dtype == torch.bfloat16, "DeepSeek V4 rotation currently only supports bf16"
    return hadamard_transform(x, scale=x.size(-1) ** -0.5)


@lru_cache(2)
def _precompute_freqs_cis(
    dim: int,
    seqlen: int,
    original_seq_len: int,
    base: float,
    factor: float,
    beta_fast: int,
    beta_slow: int,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    def find_correction_dim(num_rotations, rope_dim, rope_base, max_seq_len):
        return rope_dim * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (
            2 * math.log(rope_base)
        )

    def find_correction_range(low_rot, high_rot, rope_dim, rope_base, max_seq_len):
        low = math.floor(find_correction_dim(low_rot, rope_dim, rope_base, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, rope_dim, rope_base, max_seq_len))
        return max(low, 0), min(high, rope_dim - 1)

    def linear_ramp_factor(min_value, max_value, rope_dim):
        if min_value == max_value:
            max_value += 0.001
        linear_func = (torch.arange(rope_dim, dtype=torch.float32, device=device) - min_value) / (
            max_value - min_value
        )
        return torch.clamp(linear_func, 0, 1)

    freqs = 1.0 / (
        base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
    )
    if original_seq_len > 0:
        low, high = find_correction_range(beta_fast, beta_slow, dim, base, original_seq_len)
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    t = torch.arange(seqlen, device=device)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def _wrapped_precompute_freqs_cis(
    config: MLATransformerConfig,
    rope_head_dim: int,
    base: float,
    yarn_disabled: bool = False,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    max_seq_len = 65536
    original_seq_len = 0 if yarn_disabled else config.original_max_position_embeddings
    return _precompute_freqs_cis(
        dim=rope_head_dim,
        seqlen=max_seq_len,
        original_seq_len=original_seq_len,
        base=base,
        factor=config.rotary_scaling_factor,
        beta_fast=config.beta_fast,
        beta_slow=config.beta_slow,
        device=device,
    )


def _get_device_freqs_cis(
    module: nn.Module,
    *,
    config: MLATransformerConfig,
    rope_head_dim: int,
    base: float,
    yarn_disabled: bool,
    device: torch.device,
) -> torch.Tensor:
    """Return RoPE freqs generated on the compute device for SGLang parity.

    CPU-generated and CUDA-generated ``torch.polar`` tables differ by ~1e-7 in
    some entries. That is enough to flip bf16 rounding in DSV4 layer-0 RoPE and
    then amplify through later layers, so CUDA forwards should use CUDA-built
    freqs instead of a CPU-built buffer moved to CUDA by ``module.cuda()``.
    """
    if (
        os.environ.get("MEGATRON_DSV4_ROPE_DEVICE_PRECOMPUTE", "1") != "1"
        or device.type != "cuda"
    ):
        return module.freqs_cis

    cache = getattr(module, "_freqs_cis_device_cache", None)
    if cache is None or cache.device != device:
        cache = _wrapped_precompute_freqs_cis(
            config,
            rope_head_dim=rope_head_dim,
            base=base,
            yarn_disabled=yarn_disabled,
            device=device,
        )
        setattr(module, "_freqs_cis_device_cache", cache)
    return cache


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor, inverse: bool = False):
    """Apply RoPE to the last dimension of x. Functional: returns a new
    tensor; ``x`` is NOT mutated.

    Was previously in-place (``y = x; y.copy_(x_out)``) which broke
    autograd through the rotation when ``x`` carried a ``grad_fn`` —
    notably at the post-``sparse_attn_torch`` call site. All call sites assign
    the return value back via ``torch.cat`` to splice the rotated tail
    onto the static prefix; forward parity (top-1 = 50063, max abs
    diff = 0.0) is preserved because the math + dtype semantics are
    unchanged — only the storage location differs.
    """
    x_complex = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
    freqs_cis = freqs_cis.to(device=x.device)
    if inverse:
        freqs_cis = freqs_cis.conj()
    if x_complex.ndim == 3:
        freqs_cis = freqs_cis.view(1, x_complex.size(1), x_complex.size(-1))
    else:
        freqs_cis = freqs_cis.view(1, x_complex.size(1), 1, x_complex.size(-1))
    return torch.view_as_real(x_complex * freqs_cis).flatten(-2).to(x.dtype)


def _splice_rotary(x: torch.Tensor, rope_dim: int, freqs_cis: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    """Apply RoPE to the last ``rope_dim`` features of ``x`` and return a
    new tensor. Uses the fused Triton kernel by default (fp32 math kept in
    registers, single bf16 output allocation). Set
    ``MEGATRON_DSV4_ROPE_TRITON=0`` to fall back to the eager implementation.
    """
    if os.environ.get("MEGATRON_DSV4_ROPE_TRITON", "1") == "1" and x.is_cuda:
        from megatron.core.transformer.experimental_attention_variant.dsv4_rope_triton import (
            splice_rotary_triton,
        )
        return splice_rotary_triton(x, rope_dim, freqs_cis, inverse=inverse)
    static_part = x[..., :-rope_dim]
    rope_part = apply_rotary_emb(x[..., -rope_dim:], freqs_cis, inverse=inverse)
    return torch.cat([static_part, rope_part], dim=-1)


def sparse_attn_torch(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    sm_scale: Optional[float] = None,
) -> torch.Tensor:
    """Reference sparse MQA attention with DeepSeek V4 attention sink."""
    dtype = q.dtype
    q = q.float()
    kv = kv.float()

    batch, seq_q, n_heads, head_dim = q.shape
    seq_k = kv.shape[1]
    assert (topk_idxs < seq_k).all(), "topk indices must be smaller than KV length"

    if sm_scale is None:
        sm_scale = head_dim**-0.5

    mask = topk_idxs != -1
    safe_idxs = topk_idxs.masked_fill(~mask, 0)
    batch_idx = torch.arange(batch, device=q.device).view(batch, 1, 1)
    kv_gathered = kv[batch_idx, safe_idxs]

    scores = torch.einsum("bmhd,bmkd->bmhk", q, kv_gathered) * sm_scale
    scores = scores.masked_fill(~mask.unsqueeze(2), float("-inf")).to(torch.float32)

    scores_max = scores.max(dim=-1).values
    exp_scores = torch.exp(scores - scores_max.unsqueeze(-1))
    numerator = torch.einsum("bmhk,bmkd->bmhd", exp_scores, kv_gathered)
    denominator = exp_scores.sum(dim=-1) + torch.exp(attn_sink.view(1, 1, n_heads) - scores_max)
    return (numerator / denominator.unsqueeze(-1)).to(dtype)


def all_gather_cp(tensor: torch.Tensor, dim: int, cp_group: torch.distributed.ProcessGroup):
    return torch.cat(torch.distributed.nn.functional.all_gather(tensor, group=cp_group), dim=dim)


def _merge_zigzag_cp_chunks(
    gathered: Sequence[torch.Tensor],
    *,
    cp_size: int,
    dim: int,
) -> torch.Tensor:
    """Merge Orbit THD zigzag CP chunks back into sample-local token order."""
    if cp_size <= 1:
        assert len(gathered) == 1
        return gathered[0]
    if len(gathered) != cp_size:
        raise ValueError(f"Expected {cp_size} gathered CP chunks, got {len(gathered)}")

    moved = [chunk.movedim(dim, 0) for chunk in gathered]
    local_len = moved[0].shape[0]
    if local_len % 2 != 0:
        raise ValueError(f"Zigzag CP local length must be even, got {local_len}")
    if any(chunk.shape[0] != local_len for chunk in moved):
        raise ValueError("All gathered zigzag CP chunks must have the same local length")

    chunk_len = local_len // 2
    parts: list[Optional[torch.Tensor]] = [None] * (2 * cp_size)
    for rank, chunk in enumerate(moved):
        parts[rank] = chunk[:chunk_len]
        parts[2 * cp_size - rank - 1] = chunk[chunk_len:]

    return torch.cat(parts, dim=0).movedim(0, dim)


def all_gather_zigzag_cp(
    tensor: torch.Tensor,
    *,
    dim: int,
    cp_size: int,
    cp_group: torch.distributed.ProcessGroup,
) -> torch.Tensor:
    gathered = torch.distributed.nn.functional.all_gather(tensor, group=cp_group)
    return _merge_zigzag_cp_chunks(gathered, cp_size=cp_size, dim=dim)


def get_q_positions_for_cp(
    seqlen_local: int,
    *,
    cp_size: int,
    cp_group: torch.distributed.ProcessGroup,
    device,
) -> torch.Tensor:
    if cp_size <= 1 or cp_group is None:
        return torch.arange(0, seqlen_local, device=device)
    cp_rank = cp_group.rank()
    start = cp_rank * seqlen_local
    return torch.arange(start, start + seqlen_local, device=device)


def get_zigzag_q_positions_for_cp(
    *,
    seqlen_global: int,
    cp_size: int,
    cp_rank: int,
    device,
) -> torch.Tensor:
    """Return sample-local positions for Orbit THD zigzag CP token order."""
    if cp_size <= 1:
        return torch.arange(0, seqlen_global, device=device)
    divisor = 2 * cp_size
    if seqlen_global % divisor != 0:
        raise ValueError(
            "Zigzag CP sequence length must be divisible by 2 * cp_size: "
            f"{seqlen_global=} {cp_size=}"
        )
    chunk_len = seqlen_global // divisor
    first_start = cp_rank * chunk_len
    second_start = (divisor - cp_rank - 1) * chunk_len
    return torch.cat(
        [
            torch.arange(first_start, first_start + chunk_len, device=device),
            torch.arange(second_start, second_start + chunk_len, device=device),
        ],
        dim=0,
    )


def get_zigzag_valid_token_mask_for_cp(
    *,
    seqlen_global: int,
    valid_seqlen: int,
    cp_size: int,
    cp_rank: int,
    device,
) -> torch.Tensor:
    """Return a CP-local bool mask for real tokens inside a padded zigzag sample."""
    if valid_seqlen < 0 or valid_seqlen > seqlen_global:
        raise ValueError(f"Invalid DSV4 valid length: {valid_seqlen=} {seqlen_global=}")
    q_positions = get_zigzag_q_positions_for_cp(
        seqlen_global=seqlen_global,
        cp_size=cp_size,
        cp_rank=cp_rank,
        device=device,
    )
    return q_positions < valid_seqlen


def _apply_dsv4_valid_token_mask(
    tensor: torch.Tensor,
    valid_mask: Optional[torch.Tensor],
    *,
    seq_dim: int = 0,
) -> torch.Tensor:
    """Detach and zero padded token rows while preserving real-token gradients."""
    if valid_mask is None:
        return tensor
    if valid_mask.numel() != tensor.size(seq_dim):
        raise ValueError(
            "DSV4 valid-token mask length must match the selected sequence dimension: "
            f"mask_len={valid_mask.numel()} seq_len={tensor.size(seq_dim)} seq_dim={seq_dim}"
        )
    valid_mask = valid_mask.to(device=tensor.device, dtype=torch.bool)
    if bool(valid_mask.all().item()):
        return tensor
    shape = [1] * tensor.dim()
    shape[seq_dim] = valid_mask.numel()
    valid_mask = valid_mask.view(shape)
    return torch.where(valid_mask, tensor, tensor.detach().new_zeros(tensor.shape))


def _dsv4_valid_cu_list(
    valid_cu_seqlens: Optional[torch.Tensor],
    cu_list: list[int],
) -> Optional[list[int]]:
    if valid_cu_seqlens is None:
        return None
    valid_cu = [int(x) for x in valid_cu_seqlens.detach().cpu().tolist()]
    if len(valid_cu) != len(cu_list):
        raise ValueError(
            "DSV4 valid cu_seqlens must have the same number of entries as padded "
            f"dsv4_cu_seqlens: valid={valid_cu} padded={cu_list}"
        )
    if valid_cu[0] != 0:
        raise ValueError(f"Invalid DSV4 valid cu_seqlens start: {valid_cu}")
    for idx, (valid_start, valid_end, padded_start, padded_end) in enumerate(
        zip(valid_cu[:-1], valid_cu[1:], cu_list[:-1], cu_list[1:], strict=True)
    ):
        valid_len = valid_end - valid_start
        padded_len = padded_end - padded_start
        if valid_len < 0 or valid_len > padded_len:
            raise ValueError(
                "Invalid DSV4 valid cu_seqlens entry: "
                f"idx={idx} valid_len={valid_len} padded_len={padded_len} "
                f"valid={valid_cu} padded={cu_list}"
            )
    return valid_cu


def _get_dsv4_packed_valid_token_mask(
    packed_seq_params,
    *,
    cp_size: int,
    cp_rank: int,
    seq_len: int,
    device,
    sequence_parallel: bool = False,
    tp_group: Optional[torch.distributed.ProcessGroup] = None,
) -> Optional[torch.Tensor]:
    if packed_seq_params is None or cp_size <= 1:
        return None
    cu_seqlens = getattr(packed_seq_params, "dsv4_cu_seqlens", None)
    valid_cu_seqlens = getattr(packed_seq_params, "dsv4_valid_cu_seqlens", None)
    if cu_seqlens is None or valid_cu_seqlens is None:
        return None

    cu_list = [int(x) for x in cu_seqlens.detach().cpu().tolist()]
    valid_cu = _dsv4_valid_cu_list(valid_cu_seqlens, cu_list)
    masks = []
    for padded_start, padded_end, valid_start, valid_end in zip(
        cu_list[:-1], cu_list[1:], valid_cu[:-1], valid_cu[1:], strict=True
    ):
        seqlen_global = padded_end - padded_start
        valid_seqlen = valid_end - valid_start
        if seqlen_global == 0:
            continue
        masks.append(
            get_zigzag_valid_token_mask_for_cp(
                seqlen_global=seqlen_global,
                valid_seqlen=valid_seqlen,
                cp_size=cp_size,
                cp_rank=cp_rank,
                device=device,
            )
        )

    full_mask = torch.cat(masks, dim=0) if masks else torch.empty(0, dtype=torch.bool, device=device)
    if sequence_parallel and tp_group is not None and tp_group.size() > 1:
        tp_size = tp_group.size()
        full_seq_len = seq_len * tp_size
        if full_mask.numel() > full_seq_len:
            raise ValueError(
                "DSV4 valid-token mask is longer than the gathered SP sequence: "
                f"mask_len={full_mask.numel()} full_seq_len={full_seq_len}"
            )
        if full_mask.numel() < full_seq_len:
            full_mask = F.pad(full_mask, (0, full_seq_len - full_mask.numel()), value=False)
        local_len = full_seq_len // tp_size
        start = tp_group.rank() * local_len
        return full_mask[start : start + local_len]

    if full_mask.numel() > seq_len:
        raise ValueError(
            "DSV4 valid-token mask is longer than the local sequence: "
            f"mask_len={full_mask.numel()} seq_len={seq_len}"
        )
    if full_mask.numel() < seq_len:
        full_mask = F.pad(full_mask, (0, seq_len - full_mask.numel()), value=False)
    return full_mask


def get_zigzag_compress_positions_for_cp(
    *,
    seqlen_global: int,
    ratio: int,
    cp_size: int,
    cp_rank: int,
    device,
) -> torch.Tensor:
    """Return sample-local compressed-group RoPE positions for zigzag CP order."""
    if ratio <= 0:
        raise ValueError(f"Compression ratio must be positive, got {ratio}")
    if cp_size <= 1:
        seqlen_compress = seqlen_global - (seqlen_global % ratio)
        return torch.arange(0, seqlen_compress, ratio, device=device)

    divisor = 2 * cp_size
    if seqlen_global % divisor != 0:
        raise ValueError(
            "Zigzag CP sequence length must be divisible by 2 * cp_size: "
            f"{seqlen_global=} {cp_size=}"
        )
    chunk_len = seqlen_global // divisor
    if chunk_len % ratio != 0:
        raise ValueError(
            "Zigzag CP chunk length must be divisible by compression ratio: "
            f"{chunk_len=} {ratio=}"
        )

    first_start = cp_rank * chunk_len
    second_start = (divisor - cp_rank - 1) * chunk_len
    return torch.cat(
        [
            torch.arange(first_start, first_start + chunk_len, ratio, device=device),
            torch.arange(second_start, second_start + chunk_len, ratio, device=device),
        ],
        dim=0,
    )


def get_window_topk_idxs_for_positions(
    q_positions: torch.Tensor,
    *,
    seqlen_kv: int,
    window_size: int,
    bsz: int,
    pad_short_to_window: bool = False,
) -> torch.Tensor:
    device = q_positions.device
    base = q_positions.unsqueeze(1)
    window_topk = (
        window_size if pad_short_to_window and seqlen_kv < window_size
        else min(seqlen_kv, window_size)
    )
    k_pos = (base - window_size + 1).clamp(0) + torch.arange(
        window_topk, device=device
    )
    topk_idxs = torch.where(k_pos > base, -1, k_pos)
    return topk_idxs.unsqueeze(0).expand(bsz, -1, -1)


def get_window_topk_idxs_cp(
    q_positions: torch.Tensor,
    *,
    window_size: int,
    cp_size: int,
    bsz: int,
    pad_short_to_window: bool = False,
) -> torch.Tensor:
    return get_window_topk_idxs_for_positions(
        q_positions,
        seqlen_kv=q_positions.shape[0] * cp_size,
        window_size=window_size,
        bsz=bsz,
        pad_short_to_window=pad_short_to_window,
    )


def get_compress_topk_idxs_for_positions(
    q_positions: torch.Tensor,
    *,
    seqlen_kv: int,
    ratio: int,
    bsz: int,
    kv_compress_offset: Optional[int] = None,
) -> torch.Tensor:
    device = q_positions.device
    offset = seqlen_kv if kv_compress_offset is None else kv_compress_offset
    k_group_idx = torch.arange(seqlen_kv // ratio, device=device).repeat(q_positions.shape[0], 1)
    q_first_invalid_group = (q_positions + 1).unsqueeze(1) // ratio
    invalid_mask = k_group_idx >= q_first_invalid_group
    compress_topk_idxs = torch.where(invalid_mask, -1, k_group_idx + offset)
    return compress_topk_idxs.unsqueeze(0).expand(bsz, -1, -1)


def get_compress_topk_idxs_cp(
    q_positions: torch.Tensor,
    *,
    ratio: int,
    cp_size: int,
    bsz: int,
    kv_compress_offset: Optional[int] = None,
) -> torch.Tensor:
    return get_compress_topk_idxs_for_positions(
        q_positions,
        seqlen_kv=q_positions.shape[0] * cp_size,
        ratio=ratio,
        bsz=bsz,
        kv_compress_offset=kv_compress_offset,
    )


def maybe_canonicalize_compress_topk_order(topk_idxs: torch.Tensor) -> torch.Tensor:
    if os.environ.get("MEGATRON_DSV4_CANONICAL_COMPRESS_TOPK_ORDER", "0") != "1":
        return topk_idxs
    valid = topk_idxs >= 0
    sentinel = torch.full_like(topk_idxs, torch.iinfo(topk_idxs.dtype).max)
    sort_keys = torch.where(valid, topk_idxs, sentinel)
    sorted_keys = torch.sort(sort_keys, dim=-1).values
    return torch.where(sorted_keys == sentinel, -1, sorted_keys)


def get_freqs_cis_for_cp(
    freqs_cis: torch.Tensor,
    seqlen_local: int,
    cp_size: int,
    cp_group: torch.distributed.ProcessGroup,
    stride: int = 1,
) -> torch.Tensor:
    if cp_size == 1 or cp_group is None:
        return freqs_cis[:seqlen_local:stride]
    cp_rank = cp_group.rank()
    start = cp_rank * seqlen_local
    return freqs_cis[start : start + seqlen_local : stride]


class _FakeQATFP8STE(torch.autograd.Function):
    """Out-of-place FP8 fused quant+dequant with straight-through gradient."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, block_size: int) -> torch.Tensor:
        x_q = x.detach().clone()
        dsv4_kernels.act_quant(x_q, block_size, "ue8m0", torch.float8_e8m0fnu, True)
        return x_q

    @staticmethod
    def backward(ctx, grad_out):
        return grad_out, None


class _FakeQATFP4STE(torch.autograd.Function):
    """Out-of-place FP4 fused quant+dequant with straight-through gradient."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, block_size: int) -> torch.Tensor:
        x_q = x.detach().clone()
        dsv4_kernels.fp4_act_quant(x_q, block_size, True)
        return x_q

    @staticmethod
    def backward(ctx, grad_out):
        return grad_out, None


def _maybe_fp8_simulate_qat(x: torch.Tensor, block_size: int) -> torch.Tensor:
    """FP8 fused quant+dequant on ``x`` to match ``inference/model.py``.

    The official inference module runs FP8 QAT on the non-rope dims of KV
    activations (see ``Attention.forward`` and ``Compressor.forward``). The
    simulation is gated on ``MEGATRON_USE_KV_QAT``: default ``"1"`` matches
    the official; ``"0"`` skips for smoke tests with non-conforming dims.

    Returns the quantised tensor. In autograd-tracked mode (grad enabled and
    ``x.requires_grad``) the op is out-of-place with a straight-through
    estimator so that autograd through frozen-quantised-base linears works.
    Otherwise the underlying kernel mutates ``x`` in-place (matches the
    official implementation byte-for-byte) and returns it. Callers must
    consume the return value because it is the ONLY tensor guaranteed to
    carry the simulated values.
    """
    if os.environ.get("MEGATRON_USE_KV_QAT", "1") == "0":
        return x
    if torch.is_grad_enabled() and x.requires_grad:
        return _FakeQATFP8STE.apply(x, block_size)
    return dsv4_kernels.act_quant(x, block_size, "ue8m0", torch.float8_e8m0fnu, True)


def _maybe_fp4_simulate_qat(x: torch.Tensor, block_size: int = 32) -> torch.Tensor:
    """FP4 fused quant+dequant on ``x`` to match Indexer FP4 simulation.

    Gated on ``MEGATRON_USE_KV_QAT`` (same env var as the FP8 helper). Same
    autograd contract as ``_maybe_fp8_simulate_qat``: out-of-place + STE in
    grad mode, in-place when grad is disabled.
    """
    if os.environ.get("MEGATRON_USE_KV_QAT", "1") == "0":
        return x
    if torch.is_grad_enabled() and x.requires_grad:
        return _FakeQATFP4STE.apply(x, block_size)
    return dsv4_kernels.fp4_act_quant(x, block_size, True)


def _overlap_transform(
    tensor: torch.Tensor,
    *,
    compress_ratio: int,
    head_dim: int,
    value=0,
) -> torch.Tensor:
    batch, groups, _, _ = tensor.size()
    new_tensor = tensor.new_full((batch, groups, 2 * compress_ratio, head_dim), value)
    new_tensor[:, :, compress_ratio:] = tensor[:, :, :, head_dim:]
    new_tensor[:, 1:, :compress_ratio] = tensor[:, :-1, :, :head_dim]
    return new_tensor


class DeepSeekV4Compressor(nn.Module):
    """DeepSeek V4 compressed-KV builder for C4/C128 layers."""

    def __init__(
        self,
        config: MLATransformerConfig,
        head_dim: int,
        compress_ratio: int,
        rotate: bool,
        cp_group: Optional[torch.distributed.ProcessGroup] = None,
    ):
        super().__init__()
        self.config = config
        self.dim = config.hidden_size
        self.head_dim = head_dim
        self.rope_head_dim = config.qk_pos_emb_head_dim
        self.nope_head_dim = head_dim - self.rope_head_dim
        self.compress_ratio = compress_ratio
        self.overlap = compress_ratio == 4
        self.rotate = rotate
        self.cp_group = cp_group
        self.cp_size = cp_group.size() if cp_group is not None else 1
        self.cp_rank = cp_group.rank() if cp_group is not None else 0

        coeff = 1 + int(self.overlap)
        self.ape = nn.Parameter(torch.empty(compress_ratio, coeff * head_dim, dtype=torch.float32))
        self.wkv = DSV4Linear(self.dim, coeff * head_dim, bias=False, dtype=torch.float32)
        self.wgate = DSV4Linear(self.dim, coeff * head_dim, bias=False, dtype=torch.float32)
        self.norm = RMSNorm(head_dim, config.layernorm_epsilon)

        if config.perform_initialization:
            torch.nn.init.zeros_(self.ape)

        for param in (self.ape, self.wkv.weight, self.wgate.weight):
            param._keep_fp32 = True

        freqs_cis = _wrapped_precompute_freqs_cis(
            config,
            rope_head_dim=self.rope_head_dim,
            base=config.dsv4_compress_rope_theta,
        )
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)
        self._rope_base = config.dsv4_compress_rope_theta
        self._yarn_disabled = False

    @staticmethod
    def _canonical_project(linear: nn.Module, x: torch.Tensor) -> torch.Tensor:
        if os.environ.get("MEGATRON_DSV4_CANONICAL_COMPRESSOR_LINEAR", "0") != "1":
            return linear(x)
        weight = linear.weight
        out = _dsv4_canonical_linear(
            x, weight, "MEGATRON_DSV4_CANONICAL_COMPRESSOR_LINEAR_IMPL"
        )
        if linear.bias is not None:
            out = out + linear.bias
        return out

    def overlap_transform_with_cp(
        self,
        tensor: torch.Tensor,
        value=0,
        *,
        cp_size: Optional[int] = None,
        cp_rank: Optional[int] = None,
        cp_group: Optional[torch.distributed.ProcessGroup] = None,
    ) -> torch.Tensor:
        cp_size = self.cp_size if cp_size is None else cp_size
        cp_rank = self.cp_rank if cp_rank is None else cp_rank
        cp_group = self.cp_group if cp_group is None else cp_group
        if cp_size == 1:
            return _overlap_transform(
                tensor, compress_ratio=self.compress_ratio, head_dim=self.head_dim, value=value
            )

        tensor = all_gather_cp(tensor, dim=1, cp_group=cp_group)
        tensor = _overlap_transform(
            tensor, compress_ratio=self.compress_ratio, head_dim=self.head_dim, value=value
        )
        groups_local = tensor.shape[1] // cp_size
        start = cp_rank * groups_local
        return tensor[:, start : start + groups_local]

    def forward_raw(
        self,
        x: torch.Tensor,
        *,
        cp_size: Optional[int] = None,
        cp_rank: Optional[int] = None,
        cp_group: Optional[torch.distributed.ProcessGroup] = None,
    ) -> torch.Tensor:
        cp_size = self.cp_size if cp_size is None else cp_size
        cp_rank = self.cp_rank if cp_rank is None else cp_rank
        cp_group = self.cp_group if cp_group is None else cp_group
        batch, seqlen_local, _ = x.size()
        ratio = self.compress_ratio
        dtype = x.dtype
        if seqlen_local < ratio:
            return x.new_zeros((batch, 0, self.head_dim), dtype=dtype)
        if cp_size == 1:
            seqlen_compress = seqlen_local - (seqlen_local % ratio)
        else:
            assert seqlen_local % ratio == 0, (
                f"DeepSeek V4 compressor with CP requires local sequence length divisible by ratio: "
                f"{seqlen_local=} {ratio=} cp_size={cp_size}"
            )
            seqlen_compress = seqlen_local
        if seqlen_compress == 0:
            return x.new_zeros((batch, 0, self.head_dim), dtype=dtype)

        x_fp32 = x.float()
        kv = self._canonical_project(self.wkv, x_fp32)
        score = self._canonical_project(self.wgate, x_fp32)
        if seqlen_compress != seqlen_local:
            kv = kv[:, :seqlen_compress]
            score = score[:, :seqlen_compress]

        kv = kv.unflatten(1, (-1, ratio))
        score = score.unflatten(1, (-1, ratio)) + self.ape

        if self.overlap:
            kv = self.overlap_transform_with_cp(
                kv, 0, cp_size=cp_size, cp_rank=cp_rank, cp_group=cp_group
            )
            score = self.overlap_transform_with_cp(
                score, float("-inf"), cp_size=cp_size, cp_rank=cp_rank, cp_group=cp_group
            )

        kv = (kv * score.softmax(dim=2)).sum(dim=2)
        kv = self.norm(kv.to(dtype))

        freqs_cis = get_freqs_cis_for_cp(
            _get_device_freqs_cis(
                self,
                config=self.config,
                rope_head_dim=self.rope_head_dim,
                base=self._rope_base,
                yarn_disabled=self._yarn_disabled,
                device=x.device,
            ),
            seqlen_compress,
            cp_size,
            cp_group,
            stride=ratio,
        )
        kv = _splice_rotary(kv, self.rope_head_dim, freqs_cis)

        if self.rotate:
            kv = rotate_activation(kv)
            kv = _maybe_fp4_simulate_qat(kv, 32)
        else:
            kv = kv.clone()
            rd = self.rope_head_dim
            kv_non_rope = _maybe_fp8_simulate_qat(kv[..., :-rd], 64)
            kv = torch.cat([kv_non_rope, kv[..., -rd:]], dim=-1)

        return kv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_bsd = x.permute(1, 0, 2).contiguous()
        k_bsd = self.forward_raw(x_bsd)
        return k_bsd.permute(1, 0, 2).contiguous()

    def forward_full_sequence(self, x: torch.Tensor) -> torch.Tensor:
        x_bsd = x.permute(1, 0, 2).contiguous()
        k_bsd = self.forward_raw(x_bsd, cp_size=1, cp_rank=0, cp_group=None)
        return k_bsd.permute(1, 0, 2).contiguous()

    def supports_zigzag_cp_full_sequence(self, seqlen_global: int, cp_size: int) -> bool:
        if cp_size <= 1:
            return True
        divisor = 2 * cp_size
        if seqlen_global % divisor != 0:
            return False
        chunk_len = seqlen_global // divisor
        return chunk_len >= self.compress_ratio and chunk_len % self.compress_ratio == 0

    def forward_zigzag_cp_full_sequence(
        self,
        x: torch.Tensor,
        *,
        seqlen_global: int,
        cp_size: Optional[int] = None,
        cp_rank: Optional[int] = None,
        cp_group: Optional[torch.distributed.ProcessGroup] = None,
    ) -> torch.Tensor:
        """Build full compressed KV from local zigzag CP tokens without gathering hidden."""
        cp_size = self.cp_size if cp_size is None else cp_size
        cp_rank = self.cp_rank if cp_rank is None else cp_rank
        cp_group = self.cp_group if cp_group is None else cp_group
        if cp_size == 1:
            return self.forward_full_sequence(x)
        if not self.supports_zigzag_cp_full_sequence(seqlen_global, cp_size):
            raise ValueError(
                "DSV4 zigzag CP local compressed KV requires each CP chunk to be "
                "aligned to the compression ratio: "
                f"{seqlen_global=} {cp_size=} ratio={self.compress_ratio}"
            )
        if cp_group is None:
            raise ValueError("DSV4 zigzag CP local compressed KV requires a CP group")

        divisor = 2 * cp_size
        chunk_len = seqlen_global // divisor
        group_chunk_len = chunk_len // self.compress_ratio
        expected_local_len = 2 * chunk_len
        if x.size(0) != expected_local_len:
            raise ValueError(
                "Invalid DSV4 zigzag CP local hidden length for compressed KV: "
                f"local_len={x.size(0)} expected={expected_local_len}"
            )

        x_bsd = x.permute(1, 0, 2).contiguous()
        batch, _, _ = x_bsd.size()
        ratio = self.compress_ratio
        dtype = x_bsd.dtype
        x_fp32 = x_bsd.float()

        kv = self._canonical_project(self.wkv, x_fp32).unflatten(1, (-1, ratio))
        score = self._canonical_project(self.wgate, x_fp32).unflatten(1, (-1, ratio)) + self.ape

        if self.overlap:
            chunk_last = torch.cat(
                [
                    kv[:, group_chunk_len - 1 : group_chunk_len],
                    kv[:, -1:],
                ],
                dim=1,
            )
            score_chunk_last = torch.cat(
                [
                    score[:, group_chunk_len - 1 : group_chunk_len],
                    score[:, -1:],
                ],
                dim=1,
            )
            global_chunk_last = all_gather_zigzag_cp(
                chunk_last, dim=1, cp_size=cp_size, cp_group=cp_group
            )
            global_score_chunk_last = all_gather_zigzag_cp(
                score_chunk_last, dim=1, cp_size=cp_size, cp_group=cp_group
            )

            kv_segments = []
            score_segments = []
            global_chunk_idxs = [cp_rank, divisor - cp_rank - 1]
            for local_chunk_idx, global_chunk_idx in enumerate(global_chunk_idxs):
                start = local_chunk_idx * group_chunk_len
                end = start + group_chunk_len
                kv_segment = kv[:, start:end]
                score_segment = score[:, start:end]

                if global_chunk_idx == 0:
                    prev_kv = kv.new_zeros((batch, 1, ratio, self.head_dim))
                    prev_score = score.new_full((batch, 1, ratio, self.head_dim), float("-inf"))
                else:
                    prev_kv = global_chunk_last[
                        :, global_chunk_idx - 1 : global_chunk_idx, :, : self.head_dim
                    ]
                    prev_score = global_score_chunk_last[
                        :, global_chunk_idx - 1 : global_chunk_idx, :, : self.head_dim
                    ]

                left_kv = torch.cat([prev_kv, kv_segment[:, :-1, :, : self.head_dim]], dim=1)
                right_kv = kv_segment[:, :, :, self.head_dim :]
                left_score = torch.cat(
                    [prev_score, score_segment[:, :-1, :, : self.head_dim]], dim=1
                )
                right_score = score_segment[:, :, :, self.head_dim :]

                kv_segments.append(torch.cat([left_kv, right_kv], dim=2))
                score_segments.append(torch.cat([left_score, right_score], dim=2))

            kv = torch.cat(kv_segments, dim=1)
            score = torch.cat(score_segments, dim=1)

        kv = (kv * score.softmax(dim=2)).sum(dim=2)
        kv = self.norm(kv.to(dtype))

        freqs_cis = _get_device_freqs_cis(
            self,
            config=self.config,
            rope_head_dim=self.rope_head_dim,
            base=self._rope_base,
            yarn_disabled=self._yarn_disabled,
            device=x.device,
        )
        compress_positions = get_zigzag_compress_positions_for_cp(
            seqlen_global=seqlen_global,
            ratio=ratio,
            cp_size=cp_size,
            cp_rank=cp_rank,
            device=x.device,
        )
        kv = _splice_rotary(
            kv, self.rope_head_dim, freqs_cis.index_select(0, compress_positions.to(torch.long))
        )

        if self.rotate:
            kv = rotate_activation(kv)
            kv = _maybe_fp4_simulate_qat(kv, 32)
        else:
            kv = kv.clone()
            rd = self.rope_head_dim
            kv_non_rope = _maybe_fp8_simulate_qat(kv[..., :-rd], 64)
            kv = torch.cat([kv_non_rope, kv[..., -rd:]], dim=-1)

        kv = all_gather_zigzag_cp(kv, dim=1, cp_size=cp_size, cp_group=cp_group)
        return kv.permute(1, 0, 2).contiguous()


class _DeepSeekV4Indexer(MegatronModule):
    """Torch fallback for DeepSeek V4 C4 compressed-KV top-k selection."""

    def __init__(self, config: MLATransformerConfig, pg_collection: ProcessGroupCollection):
        super().__init__(config=config)
        self.hidden_size = config.hidden_size
        self.q_lora_rank = config.q_lora_rank
        self.index_n_heads = config.dsa_indexer_n_heads
        self.index_head_dim = config.dsa_indexer_head_dim
        self.index_topk = config.dsa_indexer_topk
        self.rope_head_dim = config.qk_pos_emb_head_dim
        self.compress_ratio = 4
        self.pg_collection = pg_collection

        self.linear_wq_b = DSV4Linear(
            self.q_lora_rank,
            self.index_n_heads * self.index_head_dim,
            bias=False,
            dtype=torch.float8_e4m3fn,
        )
        self.linear_weights_proj = DSV4Linear(
            self.hidden_size,
            self.index_n_heads,
            bias=False,
            dtype=torch.bfloat16,
        )

        self.compressor = DeepSeekV4Compressor(
            config=config,
            head_dim=self.index_head_dim,
            compress_ratio=self.compress_ratio,
            rotate=True,
            cp_group=pg_collection.cp if hasattr(pg_collection, "cp") else None,
        )

        freqs_cis = _wrapped_precompute_freqs_cis(
            config,
            rope_head_dim=self.rope_head_dim,
            base=config.dsv4_compress_rope_theta,
        )
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)
        self._rope_base = config.dsv4_compress_rope_theta
        self._yarn_disabled = False

    def forward(self, x: torch.Tensor, qr: torch.Tensor) -> torch.Tensor:
        if self.config.sequence_parallel and self.pg_collection.tp.size() > 1:
            x = gather_from_sequence_parallel_region(x, group=self.pg_collection.tp)
            qr = gather_from_sequence_parallel_region(qr, group=self.pg_collection.tp)
        return self._forward_full_seq(x, qr)

    def _forward_full_seq(self, x: torch.Tensor, qr: torch.Tensor) -> torch.Tensor:
        """Indexer body, assuming ``x`` and ``qr`` are already full-sequence.

        ``forward`` calls this after applying the SP entry-gather; the DSV4
        packed-THD SP path (``DeepSeekV4Attention._forward_packed_thd_sp``)
        feeds already-gathered per-segment inputs and so calls this directly
        to avoid a no-op scatter/gather pair and (more importantly) the
        config-attribute mutation that would otherwise be needed to suppress
        ``forward``'s entry gather.
        """
        seqlen, batch, _ = x.size()
        x_bsd = x.permute(1, 0, 2).contiguous()
        qr_bsd = qr.permute(1, 0, 2).contiguous()
        q = self.linear_wq_b(qr_bsd).unflatten(-1, (self.index_n_heads, self.index_head_dim))

        cp_group = self.pg_collection.cp if hasattr(self.pg_collection, "cp") else None
        cp_size = cp_group.size() if cp_group is not None else 1
        freqs_cis = get_freqs_cis_for_cp(
            _get_device_freqs_cis(
                self,
                config=self.config,
                rope_head_dim=self.rope_head_dim,
                base=self._rope_base,
                yarn_disabled=self._yarn_disabled,
                device=q.device,
            ),
            seqlen,
            cp_size,
            cp_group,
            stride=1,
        )
        q = q.clone()
        q = _splice_rotary(q, self.rope_head_dim, freqs_cis)

        q = rotate_activation(q)
        q = _maybe_fp4_simulate_qat(q, block_size=32)

        k = self.compressor(x)
        weights = self.linear_weights_proj(x_bsd) * (self.index_n_heads**-0.5) * (self.index_head_dim**-0.5)

        if cp_size > 1 and cp_group is not None:
            k = all_gather_cp(k, dim=0, cp_group=cp_group)

        k_btd = k.permute(1, 0, 2).contiguous()
        scores = torch.einsum("bshd,btd->bsht", q, k_btd)
        scores = (scores.relu_() * weights.unsqueeze(-1)).sum(dim=2)

        q_positions = get_q_positions_for_cp(
            seqlen, cp_size=cp_size, cp_group=cp_group, device=x.device
        )
        valid_end = (q_positions + 1).unsqueeze(1) // self.compress_ratio
        kv_positions = torch.arange(scores.size(-1), device=x.device)
        scores = scores.masked_fill(
            kv_positions.view(1, 1, -1) >= valid_end.view(1, seqlen, 1),
            float("-inf"),
        )

        topk_k = min(self.index_topk, scores.size(-1))
        topk_idxs = scores.topk(topk_k, dim=-1).indices
        return torch.where(topk_idxs >= valid_end.view(1, seqlen, 1), -1, topk_idxs)

    def forward_with_zigzag_cp_kv(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        q_positions: torch.Tensor,
        *,
        seqlen_global: int,
        cp_size: int,
        cp_rank: int,
        cp_group: torch.distributed.ProcessGroup,
    ) -> torch.Tensor:
        """Indexer for packed CP: local queries score against gathered compressed keys."""
        seqlen, batch, _ = x.size()
        x_bsd = x.permute(1, 0, 2).contiguous()
        qr_bsd = qr.permute(1, 0, 2).contiguous()
        q = self.linear_wq_b(qr_bsd).unflatten(-1, (self.index_n_heads, self.index_head_dim))

        freqs_cis = _get_device_freqs_cis(
            self,
            config=self.config,
            rope_head_dim=self.rope_head_dim,
            base=self._rope_base,
            yarn_disabled=self._yarn_disabled,
            device=q.device,
        ).index_select(0, q_positions.to(torch.long))
        q = q.clone()
        q = _splice_rotary(q, self.rope_head_dim, freqs_cis)

        q = rotate_activation(q)
        q = _maybe_fp4_simulate_qat(q, block_size=32)

        k = self.compressor.forward_zigzag_cp_full_sequence(
            x,
            seqlen_global=seqlen_global,
            cp_size=cp_size,
            cp_rank=cp_rank,
            cp_group=cp_group,
        )
        weights = self.linear_weights_proj(x_bsd) * (self.index_n_heads**-0.5) * (self.index_head_dim**-0.5)

        k_btd = k.permute(1, 0, 2).contiguous()
        scores = torch.einsum("bshd,btd->bsht", q, k_btd)
        scores = (scores.relu_() * weights.unsqueeze(-1)).sum(dim=2)

        valid_end = (q_positions + 1).unsqueeze(1) // self.compress_ratio
        kv_positions = torch.arange(scores.size(-1), device=x.device)
        scores = scores.masked_fill(
            kv_positions.view(1, 1, -1) >= valid_end.view(1, seqlen, 1),
            float("-inf"),
        )

        topk_k = min(self.index_topk, scores.size(-1))
        topk_idxs = scores.topk(topk_k, dim=-1).indices
        return torch.where(topk_idxs >= valid_end.view(1, seqlen, 1), -1, topk_idxs)


class DeepSeekV4Gate(nn.Module):
    """V4 router: hash routing (first n_hash_layers) or score-based routing.

    Mirrors `inference/model.py:Gate`. `weight` is fp32; for score-based
    layers `bias` is fp32; for hash layers `tid2eid` int32 replaces the bias.
    Routing returns weights and indices that match the official semantics
    (sqrtsoftplus/sigmoid score normalization, biased top-k, route_scale).
    """

    def __init__(
        self,
        dim: int,
        n_routed_experts: int,
        n_activated: int,
        score_func: str,
        route_scale: float,
        vocab_size: int,
        is_hash_layer: bool,
    ):
        super().__init__()
        self.topk = n_activated
        self.score_func = score_func
        self.route_scale = route_scale
        self.is_hash_layer = is_hash_layer
        self.router_replay = None

        self.weight = nn.Parameter(torch.empty(n_routed_experts, dim, dtype=torch.float32))
        self.weight._keep_fp32 = True
        if is_hash_layer:
            self.tid2eid = nn.Parameter(
                torch.empty(vocab_size, n_activated, dtype=torch.int32),
                requires_grad=False,
            )
            self.bias = None
        else:
            self.bias = nn.Parameter(torch.empty(n_routed_experts, dtype=torch.float32))
            self.bias._keep_fp32 = True
            self.tid2eid = None

    def forward(self, x: torch.Tensor, input_ids: torch.Tensor):
        # x: [N, dim]; input_ids: [N]
        x = x.float()
        if _dsv4_use_tilekernels():
            scores = _dsv4_best_linear(
                x, self.weight, "MEGATRON_DSV4_GATE_LINEAR_IMPL"
            )
        else:
            scores = F.linear(x, self.weight)
        router_replay = getattr(self, "router_replay", None)
        if not self.is_hash_layer:
            if self.score_func not in ("sigmoid", "sqrtsoftplus"):
                raise RuntimeError(
                    "top2_sum_gate DSV4 router is the production path and "
                    f"requires sigmoid/sqrtsoftplus; got {self.score_func!r}."
                )
            top2_sum_gate = _dsv4_tile_router_op()
            tile_indices, tile_weights = top2_sum_gate(
                scores.contiguous(),
                self.bias.contiguous(),
                self.topk,
                0,
                0,
                False,
                0,
                float(self.route_scale),
                0,
                1,
                0,
                1,
                self.score_func,
            )
            if router_replay is None:
                weights = _dsv4_attach_tile_router_weight_backward(
                    scores,
                    tile_indices,
                    tile_weights,
                    self.score_func,
                    float(self.route_scale),
                )
                return weights, tile_indices

            def _compute_tile_topk(scores_for_replay, topk, num_groups=None, group_topk=None):
                del scores_for_replay, topk, num_groups, group_topk
                return tile_weights, tile_indices

            _, indices = router_replay.get_replay_topk(
                scores, self.topk, None, None, _compute_tile_topk
            )
            if indices.dtype != tile_indices.dtype:
                indices_for_match = indices.to(tile_indices.dtype)
            else:
                indices_for_match = indices
            matches = indices_for_match.unsqueeze(2) == tile_indices.unsqueeze(1)
            full_row_match = matches.any(dim=2).all(dim=1, keepdim=True)
            tile_pos = matches.to(torch.int64).argmax(dim=2)
            replay_ordered_tile_weights = tile_weights.gather(1, tile_pos)

            replay_scores = F.softplus(scores).sqrt() if self.score_func == "sqrtsoftplus" else scores.sigmoid()
            safe_indices = indices.masked_fill(indices < 0, 0).to(torch.int64)
            fallback_weights = replay_scores.gather(1, safe_indices)
            fallback_weights = fallback_weights / (fallback_weights.sum(dim=-1, keepdim=True) + 1e-20)
            fallback_weights = fallback_weights * float(self.route_scale)
            weights = torch.where(full_row_match, replay_ordered_tile_weights, fallback_weights)
            weights = _dsv4_attach_tile_router_weight_backward(
                scores,
                indices,
                weights,
                self.score_func,
                float(self.route_scale),
            )
            return weights, indices

        if self.score_func == "sigmoid":
            scores = scores.sigmoid()
        elif self.score_func == "sqrtsoftplus":
            scores = F.softplus(scores).sqrt()
        else:
            raise RuntimeError(
                f"Unsupported DSV4 router score function {self.score_func!r}."
            )
        original_scores = scores
        if self.bias is not None:
            scores = scores + self.bias
        if self.is_hash_layer:
            hash_indices = self.tid2eid[input_ids]
            if router_replay is None:
                indices = hash_indices
            else:
                def _compute_hash_topk(scores_for_replay, topk, num_groups=None, group_topk=None):
                    del topk, num_groups, group_topk
                    return scores_for_replay.gather(1, hash_indices), hash_indices

                _, indices = router_replay.get_replay_topk(
                    scores, self.topk, None, None, _compute_hash_topk
                )
        else:
            if router_replay is None:
                indices = scores.topk(self.topk, dim=-1)[1]
            else:
                def _compute_score_topk(scores_for_replay, topk, num_groups=None, group_topk=None):
                    del num_groups, group_topk
                    return torch.topk(scores_for_replay, k=topk, dim=-1)

                _, indices = router_replay.get_replay_topk(
                    scores, self.topk, None, None, _compute_score_topk
                )
        weights = original_scores.gather(1, indices)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        weights = weights * self.route_scale
        return weights, indices


class DeepSeekV4Attention(MegatronModule):
    """Native DeepSeek V4 attention block for Megatron-Core."""

    def __init__(
        self,
        config: MLATransformerConfig,
        submodules=None,
        layer_number: int = 1,
        attn_mask_type=None,
        attention_type: str = None,
        cp_comm_type: str = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ):
        super().__init__(config=config)
        del submodules, attn_mask_type, attention_type, cp_comm_type

        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=["tp", "cp"])
        self.pg_collection = pg_collection
        self.tp_group = self.pg_collection.tp
        self.cp_group = self.pg_collection.cp if hasattr(self.pg_collection, "cp") else None
        self.cp_size = self.cp_group.size() if self.cp_group is not None else 1

        self.layer_id = layer_number - 1
        self.dim = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.n_local_heads = self.n_heads // config.tensor_model_parallel_size
        self.q_lora_rank = config.q_lora_rank
        self.o_lora_rank = config.dsv4_o_lora_rank
        self.head_dim = config.kv_lora_rank
        self.rope_head_dim = config.qk_pos_emb_head_dim
        self.nope_head_dim = self.head_dim - self.rope_head_dim
        self.n_groups = config.dsv4_o_groups
        self.n_local_groups = self.n_groups // config.tensor_model_parallel_size
        self.window_size = config.dsv4_window_size
        self.compress_ratio = (
            config.dsv4_compress_ratios[self.layer_id] if config.dsv4_compress_ratios else 0
        )
        self.eps = config.layernorm_epsilon
        self.softmax_scale = self.head_dim**-0.5
        self.sequence_parallel = config.sequence_parallel

        self.attn_sink = nn.Parameter(torch.zeros(self.n_local_heads, dtype=torch.float32))
        self.attn_sink._keep_fp32 = True

        self.wq_a = DSV4Linear(
            self.dim, self.q_lora_rank, bias=False, dtype=torch.float8_e4m3fn,
        )
        self.q_norm = RMSNorm(self.q_lora_rank, eps=self.eps)
        self.wq_b = DSV4ColumnParallelLinear(
            self.q_lora_rank,
            self.n_heads * self.head_dim,
            bias=False,
            dtype=torch.float8_e4m3fn,
            tp_group=self.tp_group,
        )
        self.wkv = DSV4Linear(
            self.dim, self.head_dim, bias=False, dtype=torch.float8_e4m3fn,
        )
        self.kv_norm = RMSNorm(self.head_dim, eps=self.eps)

        self.wo_a = DSV4ColumnParallelLinear(
            self.n_heads * self.head_dim // self.n_groups,
            self.n_groups * self.o_lora_rank,
            bias=False,
            dtype=torch.bfloat16,
            tp_group=self.tp_group,
        )
        # Under SP=on the row-parallel reduction is fused into a single
        # ``reduce_scatter_to_sequence_parallel_region`` (replacing the legacy
        # "all_reduce + outer ``scatter_to_sequence_parallel_region``" pair).
        # The outer scatter on the main attention path below is dropped.
        # Note: this flag is harmless for the CP-only forward paths below
        # because their wo_b call also has the sequence dim at dim 0; the
        # only path the change is observable on is the main SP=on path.
        self.wo_b = DSV4RowParallelLinear(
            self.n_groups * self.o_lora_rank,
            self.dim,
            bias=False,
            dtype=torch.float8_e4m3fn,
            tp_group=self.tp_group,
            sequence_parallel=bool(getattr(config, "sequence_parallel", False)),
        )

        if self.compress_ratio:
            self.compressor = DeepSeekV4Compressor(
                config=config,
                head_dim=self.head_dim,
                compress_ratio=self.compress_ratio,
                rotate=False,
                cp_group=self.cp_group,
            )
            self.indexer = (
                _DeepSeekV4Indexer(config=config, pg_collection=pg_collection)
                if self.compress_ratio == 4
                else None
            )

        rope_base = config.dsv4_compress_rope_theta if self.compress_ratio else config.rotary_base
        yarn_disabled = (
            os.environ.get("MEGATRON_DSV4_CKPT_VERSION", "2604") == "0415"
            and not self.compress_ratio
        )
        freqs_cis = _wrapped_precompute_freqs_cis(
            config, rope_head_dim=self.rope_head_dim, base=rope_base, yarn_disabled=yarn_disabled
        )
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)
        self._rope_base = rope_base
        self._yarn_disabled = yarn_disabled

    def sharded_state_dict(
        self,
        prefix: str = "",
        sharded_offsets: tuple = (),
        metadata: Optional[dict] = None,
    ) -> ShardedStateDict:
        # Normalize metadata up-front so the ``attn_sink`` TP-shard injection
        # below runs regardless of whether the caller passed ``metadata=None``.
        # The previous gate (``metadata is not None and "dp_cp_group" in metadata``)
        # silently emitted ``attn_sink`` as replicated whenever a direct
        # caller forgot to supply ``dp_cp_group`` — the recursion path always
        # provides one via ``MegatronModule.sharded_state_dict`` and
        # ``sharded_state_dict_default`` (both call
        # ``ensure_metadata_has_dp_cp_group``), but direct invocations did not.
        metadata = ensure_metadata_has_dp_cp_group(metadata)
        ans = super().sharded_state_dict(prefix, sharded_offsets, metadata)
        ans.update(
            make_sharded_tensors_for_checkpoint(
                state_dict={"attn_sink": self.attn_sink},
                prefix=prefix,
                tensor_parallel_layers_axis_map={"attn_sink": 0},
                sharded_offsets=sharded_offsets,
                tp_group=self.tp_group,
                dp_cp_group=metadata["dp_cp_group"],
            )
        )
        return ans

    def _validate_zigzag_cp_local_kv(self, seqlen_global: int) -> None:
        if not self.compress_ratio:
            return
        if not self.compressor.supports_zigzag_cp_full_sequence(seqlen_global, self.cp_size):
            raise NotImplementedError(
                "DeepSeek V4 CP local-KV requires each zigzag CP chunk to be "
                "aligned to the compression ratio: "
                f"{seqlen_global=} cp_size={self.cp_size} ratio={self.compress_ratio}"
            )
        if self.indexer is not None and not self.indexer.compressor.supports_zigzag_cp_full_sequence(
            seqlen_global, self.cp_size
        ):
            raise NotImplementedError(
                "DeepSeek V4 CP local-KV indexer requires each zigzag CP chunk to be "
                "aligned to the compression ratio: "
                f"{seqlen_global=} cp_size={self.cp_size} ratio={self.compress_ratio}"
            )

    def _forward_unpacked_with_local_kv(
        self,
        local_hidden_states: torch.Tensor,
        q_positions: torch.Tensor,
        *,
        seqlen_global: int,
        valid_q_mask: Optional[torch.Tensor] = None,
        _skip_wo_b: bool = False,
    ) -> torch.Tensor:
        local_hidden_states = _apply_dsv4_valid_token_mask(local_hidden_states, valid_q_mask)
        x_q = local_hidden_states.permute(1, 0, 2).contiguous()
        bsz, seqlen_local, _ = x_q.size()

        freqs_cis = _get_device_freqs_cis(
            self,
            config=self.config,
            rope_head_dim=self.rope_head_dim,
            base=self._rope_base,
            yarn_disabled=self._yarn_disabled,
            device=x_q.device,
        )
        freqs_q = freqs_cis.index_select(0, q_positions.to(torch.long))
        rd = self.rope_head_dim

        qr = self.q_norm(self.wq_a(x_q))
        q = self.wq_b(qr).unflatten(-1, (self.n_local_heads, self.head_dim))
        q = dsv4_q_rmsnorm(q, self.eps)
        q = _splice_rotary(q, rd, freqs_q)

        kv_vanilla = self.kv_norm(self.wkv(x_q))
        kv_vanilla = _splice_rotary(kv_vanilla, rd, freqs_q)
        kv_non_rope = _maybe_fp8_simulate_qat(kv_vanilla[..., :-rd], 64)
        kv_vanilla = torch.cat([kv_non_rope, kv_vanilla[..., -rd:]], dim=-1)
        kv_vanilla = all_gather_zigzag_cp(
            kv_vanilla, dim=1, cp_size=self.cp_size, cp_group=self.cp_group
        )

        use_fixed_short_prefill_layout = (
            self.compress_ratio
            and seqlen_global < self.window_size
            and os.environ.get("MEGATRON_DSV4_PREFILL_FIXED_WINDOW_LAYOUT", "0") == "1"
        )
        kv_compress_offset = self.window_size if use_fixed_short_prefill_layout else seqlen_global
        topk_idxs = get_window_topk_idxs_for_positions(
            q_positions,
            seqlen_kv=seqlen_global,
            window_size=self.window_size,
            bsz=bsz,
            pad_short_to_window=use_fixed_short_prefill_layout,
        )

        kv_compress = None
        if self.compress_ratio:
            if self.indexer is not None:
                qr_sbd = qr.permute(1, 0, 2).contiguous()
                compress_topk_idxs = self.indexer.forward_with_zigzag_cp_kv(
                    local_hidden_states,
                    qr_sbd,
                    q_positions,
                    seqlen_global=seqlen_global,
                    cp_size=self.cp_size,
                    cp_rank=self.cp_group.rank(),
                    cp_group=self.cp_group,
                )
                q_first_invalid_group = (q_positions + 1).unsqueeze(1) // self.compress_ratio
                topk_idx_mask = (compress_topk_idxs >= q_first_invalid_group) | (
                    compress_topk_idxs < 0
                )
                compress_topk_idxs = torch.where(
                    topk_idx_mask, -1, compress_topk_idxs + kv_compress_offset
                )
            else:
                compress_topk_idxs = get_compress_topk_idxs_for_positions(
                    q_positions,
                    seqlen_kv=seqlen_global,
                    ratio=self.compress_ratio,
                    bsz=bsz,
                    kv_compress_offset=kv_compress_offset,
            )
            compress_topk_idxs = maybe_canonicalize_compress_topk_order(compress_topk_idxs)
            topk_idxs = torch.cat([topk_idxs, compress_topk_idxs], dim=-1)

            kv_compress_sbd = self.compressor.forward_zigzag_cp_full_sequence(
                local_hidden_states,
                seqlen_global=seqlen_global,
                cp_size=self.cp_size,
                cp_rank=self.cp_group.rank(),
                cp_group=self.cp_group,
            )
            kv_compress = kv_compress_sbd.permute(1, 0, 2).contiguous()

        if use_fixed_short_prefill_layout and kv_vanilla.size(1) < self.window_size:
            kv_vanilla = F.pad(kv_vanilla, (0, 0, 0, self.window_size - kv_vanilla.size(1)))

        kv = torch.cat([kv_vanilla, kv_compress], dim=1) if kv_compress is not None else kv_vanilla
        if self.tp_group is not None:
            kv = copy_to_tensor_model_parallel_region(kv, group=self.tp_group)

        attn_impl = os.environ.get("MEGATRON_DSV4_ATTN_IMPL", "tilelang")
        if attn_impl != "tilelang":
            raise ValueError(
                f"DeepSeek V4 Megatron sparse attention only supports "
                f"MEGATRON_DSV4_ATTN_IMPL=tilelang, got {attn_impl!r}"
            )
        from megatron.core.transformer.experimental_attention_variant.dsv4_sparse_attn_tilelang import (
            sparse_attn_tilelang,
        )
        out = sparse_attn_tilelang(
            q, kv, self.attn_sink, topk_idxs.int(), self.softmax_scale
        )

        out = _apply_dsv4_valid_token_mask(out, valid_q_mask, seq_dim=1)
        out = _splice_rotary(out, rd, freqs_q, inverse=True)
        out = out.flatten(-2)
        out = self.wo_a.apply_input_rotation(out)
        out = _apply_dsv4_valid_token_mask(out, valid_q_mask, seq_dim=1)
        out = out.view(bsz, seqlen_local, self.n_local_groups, -1)
        wo_a = self.wo_a.weight.view(self.n_local_groups, self.o_lora_rank, -1)
        out = torch.einsum("bsgd,grd->bsgr", out, wo_a)
        out = out.flatten(2)
        out = out.permute(1, 0, 2).contiguous()
        out = _apply_dsv4_valid_token_mask(out, valid_q_mask)
        if _skip_wo_b:
            return out
        out = self.wo_b(out)
        return _apply_dsv4_valid_token_mask(out, valid_q_mask)

    def _forward_packed_thd_zigzag_cp(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        valid_cu_seqlens: Optional[torch.Tensor] = None,
        *,
        _skip_wo_b: bool = False,
    ) -> torch.Tensor:
        cp_rank = self.cp_group.rank()
        cu_list = [int(x) for x in cu_seqlens.detach().cpu().tolist()]
        if cu_list[0] != 0:
            raise ValueError(f"Invalid DSV4 zigzag CP cu_seqlens start: {cu_list}")
        valid_cu_list = _dsv4_valid_cu_list(valid_cu_seqlens, cu_list)

        local_cu = []
        for value in cu_list:
            if value % self.cp_size != 0:
                raise ValueError(
                    "DSV4 zigzag CP cu_seqlens must be divisible by cp_size: "
                    f"value={value} cp_size={self.cp_size}"
                )
            local_cu.append(value // self.cp_size)
        if local_cu[-1] > hidden_states.size(0):
            raise ValueError(
                "Invalid DSV4 zigzag CP local length: "
                f"local_cu_end={local_cu[-1]}, hidden_len={hidden_states.size(0)}"
            )

        output_features = (
            self.n_local_groups * self.o_lora_rank if _skip_wo_b else hidden_states.size(-1)
        )
        output = hidden_states.new_zeros(
            hidden_states.size(0), hidden_states.size(1), output_features
        )
        if valid_cu_list is None:
            valid_pairs = zip(cu_list[:-1], cu_list[1:])
        else:
            valid_pairs = zip(valid_cu_list[:-1], valid_cu_list[1:])
        for (global_start, global_end, local_start, local_end, valid_pair) in zip(
            cu_list[:-1], cu_list[1:], local_cu[:-1], local_cu[1:], valid_pairs
        ):
            seqlen_global = global_end - global_start
            if seqlen_global == 0:
                continue
            valid_start, valid_end = valid_pair
            valid_seqlen = valid_end - valid_start
            if seqlen_global % (2 * self.cp_size) != 0:
                raise ValueError(
                    "DSV4 zigzag CP sample length must be divisible by 2 * cp_size: "
                    f"{seqlen_global=} cp_size={self.cp_size}"
                )
            local_hidden = hidden_states[local_start:local_end]
            q_positions = get_zigzag_q_positions_for_cp(
                seqlen_global=seqlen_global,
                cp_size=self.cp_size,
                cp_rank=cp_rank,
                device=hidden_states.device,
            )
            valid_q_mask = None
            if valid_cu_list is not None:
                valid_q_mask = get_zigzag_valid_token_mask_for_cp(
                    seqlen_global=seqlen_global,
                    valid_seqlen=valid_seqlen,
                    cp_size=self.cp_size,
                    cp_rank=cp_rank,
                    device=hidden_states.device,
                )
            self._validate_zigzag_cp_local_kv(seqlen_global)
            output[local_start:local_end] = self._forward_unpacked_with_local_kv(
                local_hidden,
                q_positions,
                seqlen_global=seqlen_global,
                valid_q_mask=valid_q_mask,
                _skip_wo_b=_skip_wo_b,
            )
        return output

    def _forward_packed_thd_zigzag_cp_sp(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        valid_cu_seqlens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.tp_group is None or self.tp_group.size() <= 1:
            raise RuntimeError(
                "DeepSeek V4 CP+SP requires a tensor-parallel group with size > 1."
            )
        hidden_cp_local = gather_from_sequence_parallel_region(
            hidden_states, tensor_parallel_output_grad=False, group=self.tp_group
        )
        pre_wob = self._forward_packed_thd_zigzag_cp(
            hidden_cp_local,
            cu_seqlens,
            valid_cu_seqlens,
            _skip_wo_b=True,
        )
        output = self.wo_b(pre_wob)
        valid_mask = _get_dsv4_packed_valid_token_mask(
            SimpleNamespace(dsv4_cu_seqlens=cu_seqlens, dsv4_valid_cu_seqlens=valid_cu_seqlens)
            if valid_cu_seqlens is not None
            else None,
            cp_size=self.cp_size,
            cp_rank=self.cp_group.rank(),
            seq_len=output.size(0),
            device=output.device,
            sequence_parallel=True,
            tp_group=self.tp_group,
        )
        return _apply_dsv4_valid_token_mask(output, valid_mask)

    def _forward_packed_thd(
        self,
        hidden_states: torch.Tensor,
        packed_seq_params,
        *,
        attention_mask=None,
        inference_context=None,
        rotary_pos_emb=None,
        rotary_pos_cos=None,
        rotary_pos_sin=None,
        rotary_pos_cos_sin=None,
        attention_bias=None,
        sequence_len_offset=None,
    ) -> torch.Tensor:
        if inference_context is not None or sequence_len_offset is not None:
            raise NotImplementedError(
                "DeepSeek V4 attention does not support inference_context or "
                "sequence_len_offset yet; chunked prefill/decode offsets would make "
                "RoPE positions and sparse top-k state incorrect."
            )

        dsv4_cu_seqlens = getattr(packed_seq_params, "dsv4_cu_seqlens", None)
        dsv4_valid_cu_seqlens = getattr(packed_seq_params, "dsv4_valid_cu_seqlens", None)
        if self.cp_size > 1:
            if hidden_states.size(1) != 1:
                raise NotImplementedError(
                    "DeepSeek V4 CP requires packed THD hidden states with shape "
                    "[tokens, 1, hidden]."
                )
            if dsv4_cu_seqlens is None:
                raise NotImplementedError(
                    "DeepSeek V4 CP requires packed THD with dsv4_cu_seqlens metadata. "
                    "Orbit allgather_cp=False provides this layout; allgather_cp=True "
                    "and generic Megatron THD CP are not supported yet."
                )
            if self.sequence_parallel:
                return self._forward_packed_thd_zigzag_cp_sp(
                    hidden_states, dsv4_cu_seqlens, dsv4_valid_cu_seqlens
                )
            return self._forward_packed_thd_zigzag_cp(
                hidden_states, dsv4_cu_seqlens, dsv4_valid_cu_seqlens
            )

        if (
            os.environ.get("MEGATRON_DSV4_PACKED_THD", "1") != "1"
            or hidden_states.size(1) != 1
        ):
            return self._forward_unpacked(
                hidden_states,
                attention_mask=attention_mask,
                inference_context=inference_context,
                rotary_pos_emb=rotary_pos_emb,
                rotary_pos_cos=rotary_pos_cos,
                rotary_pos_sin=rotary_pos_sin,
                rotary_pos_cos_sin=rotary_pos_cos_sin,
                attention_bias=attention_bias,
                packed_seq_params=None,
                sequence_len_offset=sequence_len_offset,
            )

        cu_seqlens = dsv4_cu_seqlens
        if cu_seqlens is None:
            cu_seqlens = packed_seq_params.cu_seqlens_q
        if cu_seqlens is None or cu_seqlens.numel() < 2:
            return self._forward_unpacked(
                hidden_states,
                attention_mask=attention_mask,
                inference_context=inference_context,
                rotary_pos_emb=rotary_pos_emb,
                rotary_pos_cos=rotary_pos_cos,
                rotary_pos_sin=rotary_pos_sin,
                rotary_pos_cos_sin=rotary_pos_cos_sin,
                attention_bias=attention_bias,
                packed_seq_params=None,
                sequence_len_offset=sequence_len_offset,
            )

        # Under SP=on we operate on the *global* (TP-replicated) sequence,
        # so cu_seqlens is checked against ``hidden_states.size(0) * tp_world``
        # in ``_forward_packed_thd_sp``. Under SP=off we keep the original
        # check (cu_seqlens describes the local hidden_states dim 0).
        if self.sequence_parallel and self.tp_group is not None and self.tp_group.size() > 1:
            return self._forward_packed_thd_sp(
                hidden_states,
                cu_seqlens,
                attention_mask=attention_mask,
                rotary_pos_emb=rotary_pos_emb,
                rotary_pos_cos=rotary_pos_cos,
                rotary_pos_sin=rotary_pos_sin,
                rotary_pos_cos_sin=rotary_pos_cos_sin,
                attention_bias=attention_bias,
            )

        cu_list = [int(x) for x in cu_seqlens.detach().cpu().tolist()]
        if cu_list[0] != 0 or cu_list[-1] > hidden_states.size(0):
            raise ValueError(
                "Invalid DSV4 packed THD cu_seqlens: "
                f"cu_seqlens={cu_list}, hidden_len={hidden_states.size(0)}"
            )
        if len(cu_list) == 2 and cu_list[-1] == hidden_states.size(0):
            return self._forward_unpacked(
                hidden_states,
                attention_mask=attention_mask,
                inference_context=inference_context,
                rotary_pos_emb=rotary_pos_emb,
                rotary_pos_cos=rotary_pos_cos,
                rotary_pos_sin=rotary_pos_sin,
                rotary_pos_cos_sin=rotary_pos_cos_sin,
                attention_bias=attention_bias,
                packed_seq_params=None,
                sequence_len_offset=sequence_len_offset,
            )

        output = hidden_states.new_zeros(hidden_states.shape)
        for start, end in zip(cu_list[:-1], cu_list[1:]):
            if end <= start:
                continue
            output[start:end] = self._forward_unpacked(
                hidden_states[start:end],
                attention_mask=attention_mask,
                inference_context=inference_context,
                rotary_pos_emb=rotary_pos_emb,
                rotary_pos_cos=rotary_pos_cos,
                rotary_pos_sin=rotary_pos_sin,
                rotary_pos_cos_sin=rotary_pos_cos_sin,
                attention_bias=attention_bias,
                packed_seq_params=None,
                sequence_len_offset=None,
            )
        return output

    def _forward_packed_thd_sp(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        *,
        attention_mask=None,
        rotary_pos_emb=None,
        rotary_pos_cos=None,
        rotary_pos_sin=None,
        rotary_pos_cos_sin=None,
        attention_bias=None,
    ) -> torch.Tensor:
        """Packed-THD attention forward under ``sequence_parallel=True``.

        Without this path the SP=on packed-THD attention would silently drop
        ``cu_seqlens`` and run a single causal attention over the entire
        concatenated microbatch, letting sample i (for i >= 1) attend back
        into samples 0..i-1. In production DSV4 + Orbit packed-THD this blew
        train-recompute log_probs to ~-15.8 nat/token (vs sglang's -4.8) with
        up to 67-nat single-token errors.

        Strategy:

        1. SP-gather hidden_states once:
           ``[S_local, B, H] -> [S_global, B, H]`` (replicated across TP).
        2. For each ``(start, end)`` segment from ``cu_seqlens``, call
           ``_forward_unpacked(_input_is_full_seq=True, _skip_wo_b=True)``
           on the segment slice. This bypasses the inner SP entry-gather,
           bypasses the DSA indexer's no-op scatter+gather roundtrip (which
           would also require segment-length divisibility by TP), and stops
           short of applying ``wo_b``.
        3. Apply ``wo_b`` once on the *full* assembled pre-wo_b output. Under
           SP=on ``wo_b`` reduce-scatters to ``[S_global/TP, B, H]``, exactly
           the SP=on output layout. One collective on the global sequence —
           no per-segment scatter (which would fail on non-TP-divisible
           segment lengths) and no extra outer ``scatter_to_sequence_parallel``.

        This is side-effect-free on ``self``: no attribute mutation, so
        ``torch.compile`` / Dynamo doesn't see writes and graph capture is
        consistent across calls. The Python-level segment loop and the
        ``cu_seqlens.cpu().tolist()`` sync are inherited from the SP=off
        packed-THD path (``_forward_packed_thd``) — neither path supports
        full CUDA-graph capture of variable-length packed batches today, but
        this wrapper does not make capture *worse* than SP=off.
        """
        if self.tp_group is None or self.tp_group.size() <= 1:
            raise RuntimeError(
                "_forward_packed_thd_sp called with no TP group; this path "
                "is only meaningful under tensor parallelism."
            )
        if cu_seqlens is None or cu_seqlens.numel() < 2:
            raise ValueError(
                "_forward_packed_thd_sp requires cu_seqlens with at least one segment."
            )

        hidden_full = gather_from_sequence_parallel_region(
            hidden_states, tensor_parallel_output_grad=False, group=self.tp_group
        )

        cu_list = [int(x) for x in cu_seqlens.detach().cpu().tolist()]
        if cu_list[0] != 0 or cu_list[-1] > hidden_full.size(0):
            raise ValueError(
                "Invalid DSV4 packed THD cu_seqlens for SP=on: "
                f"cu_seqlens={cu_list}, gathered_hidden_len={hidden_full.size(0)}"
            )

        # Assemble pre-wo_b output across segments. ``_forward_unpacked``
        # returns ``[seg_len, B, n_local_groups * o_lora_rank]`` when
        # ``_skip_wo_b=True`` — the same in-features layout ``wo_b`` expects.
        wob_in_features_local = self.n_local_groups * self.o_lora_rank
        pre_wob_full = hidden_full.new_zeros(
            (hidden_full.size(0), hidden_full.size(1), wob_in_features_local),
            dtype=hidden_full.dtype,
        )
        for start, end in zip(cu_list[:-1], cu_list[1:]):
            if end <= start:
                continue
            pre_wob_full[start:end] = self._forward_unpacked(
                hidden_full[start:end],
                attention_mask=attention_mask,
                inference_context=None,
                rotary_pos_emb=rotary_pos_emb,
                rotary_pos_cos=rotary_pos_cos,
                rotary_pos_sin=rotary_pos_sin,
                rotary_pos_cos_sin=rotary_pos_cos_sin,
                attention_bias=attention_bias,
                packed_seq_params=None,
                sequence_len_offset=None,
                _input_is_full_seq=True,
                _skip_wo_b=True,
            )

        return self.wo_b(pre_wob_full)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask=None,
        inference_context=None,
        rotary_pos_emb=None,
        rotary_pos_cos=None,
        rotary_pos_sin=None,
        rotary_pos_cos_sin=None,
        attention_bias=None,
        packed_seq_params=None,
        sequence_len_offset=None,
    ) -> torch.Tensor:
        if packed_seq_params is not None and getattr(packed_seq_params, "qkv_format", None) == "thd":
            return self._forward_packed_thd(
                hidden_states,
                packed_seq_params,
                attention_mask=attention_mask,
                inference_context=inference_context,
                rotary_pos_emb=rotary_pos_emb,
                rotary_pos_cos=rotary_pos_cos,
                rotary_pos_sin=rotary_pos_sin,
                rotary_pos_cos_sin=rotary_pos_cos_sin,
                attention_bias=attention_bias,
                sequence_len_offset=sequence_len_offset,
            )
        return self._forward_unpacked(
            hidden_states,
            attention_mask=attention_mask,
            inference_context=inference_context,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            rotary_pos_cos_sin=rotary_pos_cos_sin,
            attention_bias=attention_bias,
            packed_seq_params=packed_seq_params,
            sequence_len_offset=sequence_len_offset,
        )

    def _forward_unpacked(
        self,
        hidden_states: torch.Tensor,
        attention_mask=None,
        inference_context=None,
        rotary_pos_emb=None,
        rotary_pos_cos=None,
        rotary_pos_sin=None,
        rotary_pos_cos_sin=None,
        attention_bias=None,
        packed_seq_params=None,
        sequence_len_offset=None,
        *,
        _input_is_full_seq: bool = False,
        _skip_wo_b: bool = False,
    ) -> torch.Tensor:
        """Unpacked attention forward.

        ``_input_is_full_seq`` and ``_skip_wo_b`` are internal hooks for the
        packed-THD SP wrapper (``_forward_packed_thd_sp``), which feeds
        per-segment already-gathered hidden states and applies ``wo_b`` once
        on the full assembled output. Passing them as kwargs (rather than
        mutating ``self.sequence_parallel`` / ``wo_b.sequence_parallel`` /
        ``indexer.config.sequence_parallel``) keeps the forward
        side-effect-free, so ``torch.compile`` / Dynamo doesn't see attribute
        writes and graph capture stays consistent across calls.
        """
        if inference_context is not None or sequence_len_offset is not None:
            raise NotImplementedError(
                "DeepSeek V4 attention does not support inference_context or "
                "sequence_len_offset yet; chunked prefill/decode offsets would make "
                "RoPE positions and sparse top-k state incorrect."
            )
        if self.cp_size > 1:
            raise NotImplementedError(
                "DeepSeek V4 CP requires packed THD with dsv4_cu_seqlens metadata. "
                "Orbit allgather_cp=False provides this layout; allgather_cp=True "
                "and generic Megatron THD CP are not supported yet."
            )

        del (
            attention_mask,
            inference_context,
            rotary_pos_emb,
            rotary_pos_cos,
            rotary_pos_sin,
            rotary_pos_cos_sin,
            attention_bias,
            packed_seq_params,
            sequence_len_offset,
        )

        if self.sequence_parallel and not _input_is_full_seq:
            hidden_states = gather_from_sequence_parallel_region(
                hidden_states, tensor_parallel_output_grad=False, group=self.tp_group
            )

        x = hidden_states.permute(1, 0, 2).contiguous()
        bsz, seqlen_local, _ = x.size()
        freqs_cis = get_freqs_cis_for_cp(
            _get_device_freqs_cis(
                self,
                config=self.config,
                rope_head_dim=self.rope_head_dim,
                base=self._rope_base,
                yarn_disabled=self._yarn_disabled,
                device=x.device,
            ),
            seqlen_local,
            self.cp_size,
            self.cp_group,
        )
        rd = self.rope_head_dim

        qr = self.q_norm(self.wq_a(x))
        q = self.wq_b(qr).unflatten(-1, (self.n_local_heads, self.head_dim))
        q = dsv4_q_rmsnorm(q, self.eps)
        q = _splice_rotary(q, rd, freqs_cis)

        kv_vanilla = self.kv_norm(self.wkv(x))
        kv_vanilla = _splice_rotary(kv_vanilla, rd, freqs_cis)
        # FP8 QAT on non-rope dims only; rope dims stay bf16 (matches official Attention.forward).
        kv_non_rope = _maybe_fp8_simulate_qat(kv_vanilla[..., :-rd], 64)
        kv_vanilla = torch.cat([kv_non_rope, kv_vanilla[..., -rd:]], dim=-1)

        seqlen_global = seqlen_local * self.cp_size
        q_positions = get_q_positions_for_cp(
            seqlen_local,
            cp_size=self.cp_size,
            cp_group=self.cp_group,
            device=x.device,
        )
        use_fixed_short_prefill_layout = (
            self.compress_ratio
            and seqlen_global < self.window_size
            and os.environ.get("MEGATRON_DSV4_PREFILL_FIXED_WINDOW_LAYOUT", "0") == "1"
        )
        kv_compress_offset = self.window_size if use_fixed_short_prefill_layout else seqlen_global
        topk_idxs = get_window_topk_idxs_cp(
            q_positions,
            window_size=self.window_size,
            cp_size=self.cp_size,
            bsz=bsz,
            pad_short_to_window=use_fixed_short_prefill_layout,
        )

        kv_compress = None
        if self.compress_ratio:
            if self.indexer is not None:
                x_sbd = x.permute(1, 0, 2).contiguous()
                qr_sbd = qr.permute(1, 0, 2).contiguous()
                if self.sequence_parallel and not _input_is_full_seq:
                    # Standard SP path: scatter input, indexer.forward gathers
                    # it back. Net is identity but a real scatter+gather pair
                    # is wired into the autograd graph for the SP=on case.
                    x_sbd = scatter_to_sequence_parallel_region(x_sbd, group=self.tp_group)
                    qr_sbd = scatter_to_sequence_parallel_region(qr_sbd, group=self.tp_group)
                    compress_topk_idxs = self.indexer(x_sbd, qr_sbd)
                elif self.sequence_parallel and _input_is_full_seq:
                    # Packed-THD SP path: already-gathered (and possibly
                    # non-TP-divisible per-segment) input. Skip both the outer
                    # scatter and the indexer's entry gather to avoid
                    # divisibility constraints and unnecessary collectives.
                    compress_topk_idxs = self.indexer._forward_full_seq(x_sbd, qr_sbd)
                else:
                    compress_topk_idxs = self.indexer(x_sbd, qr_sbd)
                q_first_invalid_group = (q_positions + 1).unsqueeze(1) // self.compress_ratio
                topk_idx_mask = (compress_topk_idxs >= q_first_invalid_group) | (
                    compress_topk_idxs < 0
                )
                compress_topk_idxs = torch.where(
                    topk_idx_mask, -1, compress_topk_idxs + kv_compress_offset
                )
            else:
                compress_topk_idxs = get_compress_topk_idxs_cp(
                    q_positions,
                    ratio=self.compress_ratio,
                    cp_size=self.cp_size,
                    bsz=bsz,
                    kv_compress_offset=kv_compress_offset,
            )
            compress_topk_idxs = maybe_canonicalize_compress_topk_order(compress_topk_idxs)
            topk_idxs = torch.cat([topk_idxs, compress_topk_idxs], dim=-1)

            x_sbd = x.permute(1, 0, 2).contiguous()
            kv_compress_sbd = self.compressor(x_sbd)
            kv_compress = kv_compress_sbd.permute(1, 0, 2).contiguous()

        if self.cp_size > 1:
            kv_vanilla = all_gather_cp(kv_vanilla, dim=1, cp_group=self.cp_group)
            if kv_compress is not None:
                kv_compress = all_gather_cp(kv_compress, dim=1, cp_group=self.cp_group)

        if use_fixed_short_prefill_layout and kv_vanilla.size(1) < self.window_size:
            kv_vanilla = F.pad(kv_vanilla, (0, 0, 0, self.window_size - kv_vanilla.size(1)))

        kv = torch.cat([kv_vanilla, kv_compress], dim=1) if kv_compress is not None else kv_vanilla
        kv = copy_to_tensor_model_parallel_region(kv, group=self.tp_group)

        attn_impl = os.environ.get("MEGATRON_DSV4_ATTN_IMPL", "tilelang")
        if attn_impl != "tilelang":
            raise ValueError(
                f"DeepSeek V4 Megatron sparse attention only supports "
                f"MEGATRON_DSV4_ATTN_IMPL=tilelang, got {attn_impl!r}"
            )
        from megatron.core.transformer.experimental_attention_variant.dsv4_sparse_attn_tilelang import (
            sparse_attn_tilelang,
        )
        out = sparse_attn_tilelang(
            q, kv, self.attn_sink, topk_idxs.int(), self.softmax_scale
        )

        out = _splice_rotary(out, rd, freqs_cis, inverse=True)

        # OFT input-rotation hook for wo_a: identity for non-OFT-wrapped
        # wo_a, adapter rotation for DSV4OFTLinear-wrapped wo_a. Applied
        # *before* the n_local_groups split-view so the rotation
        # contracts over the full ``in_features = n_local_heads *
        # head_dim`` dimension that wo_a's weight expects (consistent
        # with how OFTRotationModule was constructed under
        # DSV4OFT.transform). Restores wo_a to OFT scope while
        # preserving the einsum-bypass that reads wo_a.weight directly.
        out = out.flatten(-2)  # (b, s, h*d)
        out = self.wo_a.apply_input_rotation(out)
        out = out.view(bsz, seqlen_local, self.n_local_groups, -1)
        wo_a = self.wo_a.weight.view(self.n_local_groups, self.o_lora_rank, -1)
        out = torch.einsum("bsgd,grd->bsgr", out, wo_a)
        out = out.flatten(2)
        # Permute to ``[s, b, g*r]`` BEFORE wo_b so its row-parallel reduction
        # has the sequence dim at dim 0. This lets wo_b's
        # ``reduce_scatter_to_sequence_parallel_region`` (under SP=on) scatter
        # along the sequence dim in a single collective — the legacy
        # "all_reduce on full + outer scatter" combo collapses into one call.
        # Under SP=off, wo_b does the standard all-reduce and the layout is
        # equivalent to the old "linear -> permute" ordering.
        out = out.permute(1, 0, 2).contiguous()
        if _skip_wo_b:
            # Packed-THD SP wrapper applies wo_b once on the assembled full
            # output (so wo_b's reduce-scatter under SP=on fires a single
            # collective on the global sequence, regardless of segment-length
            # divisibility).
            return out
        output = self.wo_b(out)
        return output


class DeepSeekV4Expert(nn.Module):
    """V4 SwiGLU FFN expert. dtype controls FP4/FP8/bf16 storage of w1/w2/w3.

    Mirrors `inference/model.py:Expert`. The forward computes
    ``w2(silu(clamp(w1(x))) * clamp(w3(x))) * weights``, where the clamps are
    only applied when ``swiglu_limit > 0`` (matching the official behavior;
    shared experts use ``swiglu_limit=0`` and skip the clamp).

    Tensor parallelism: with ``tp_group`` of size > 1, ``w1`` / ``w3`` shard
    ``inter_dim`` across the group (column-parallel) and ``w2`` shards
    ``inter_dim`` on its input side (row-parallel) and all-reduces the
    output. This matches the standard SwiGLU TP pattern and lets the V4
    bridge mappings (``ColumnParallelMapping`` for w1/w3, ``RowParallelMapping``
    for w2) load FP4 weights + scales correctly under TP > 1 instead of
    failing the per-rank shape check. With ``tp_group=None`` (or size 1)
    the Column/Row variants degenerate to bare ``DSV4Linear`` weights and
    forward, so the EP-only and single-rank paths remain bit-exact.
    """

    def __init__(
        self,
        dim: int,
        inter_dim: int,
        dtype: Optional[torch.dtype] = None,
        swiglu_limit: float = 0.0,
        tp_group: Optional["dist.ProcessGroup"] = None,
        expert_skip_comm: bool = False,
    ):
        super().__init__()
        self.w1 = DSV4ColumnParallelLinear(
            dim, inter_dim, bias=False, dtype=dtype, tp_group=tp_group
        )
        # ``expert_skip_comm`` mirrors stock Megatron's ``explicit_expert_comm``:
        # when set, w2 returns the partial (un-reduced) output and the caller
        # combines partials with a single outer ``reduce_scatter``. Used by
        # DSV4MoE under SP=on to implement the standard MoE token-combine
        # pattern (one RS at the end of the MoE block instead of an all_reduce
        # per expert + a final slice).
        self.w2 = DSV4RowParallelLinear(
            inter_dim, dim, bias=False, dtype=dtype, tp_group=tp_group,
            expert_skip_comm=expert_skip_comm,
        )
        self.w3 = DSV4ColumnParallelLinear(
            dim, inter_dim, bias=False, dtype=dtype, tp_group=tp_group
        )
        self.swiglu_limit = swiglu_limit

    def forward(
        self, x: torch.Tensor, weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        dtype = x.dtype
        if weights is not None:
            from megatron.core.transformer.experimental_attention_variant.dsv4_fused_route import (
                dsv4_clamp_silu_mul_preexpanded,
            )

            y = dsv4_clamp_silu_mul_preexpanded(
                self.w1(x),
                self.w3(x),
                weights,
                self.swiglu_limit,
                dtype,
            )
        else:
            gate = self.w1(x).float()
            up = self.w3(x).float()
            if self.swiglu_limit > 0:
                up = torch.clamp(up, min=-self.swiglu_limit, max=self.swiglu_limit)
                gate = torch.clamp(gate, max=self.swiglu_limit)
            y = (F.silu(gate) * up).to(dtype)
        return self.w2(y)

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        """Recurse into ``w1`` / ``w2`` / ``w3`` so each linear's
        ``sharded_state_dict`` runs (which is what TP-shards weight + scale
        when ``expt_tp`` > 1). DSV4Expert is a plain ``nn.Module``, so
        without this override ``sharded_state_dict_default`` falls through
        to the flat ``state_dict()`` path on the parent ``DeepSeekV4MoE``
        and the per-linear TP axis maps never get applied.
        """
        sharded_state_dict: ShardedStateDict = {}
        for name, module in self.named_children():
            sharded_state_dict.update(
                sharded_state_dict_default(
                    module,
                    f"{prefix}{name}.",
                    sharded_offsets,
                    metadata,
                    tp_group=getattr(module, "tp_group", None),
                )
            )
        return sharded_state_dict


class DeepSeekV4MoE(MegatronModule):
    """V4 MoE block: gate -> top-k routed experts (FP4) + 1 shared expert (FP8)."""

    def __init__(
        self,
        config: MLATransformerConfig,
        layer_id: int,
        is_hash_layer: bool,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ):
        super().__init__(config=config)
        if pg_collection is None:
            # Request ``expt_tp`` so standalone construction (non-Bridge) shards
            # experts on the correct group when
            # ``expert_tensor_parallel_size != tensor_model_parallel_size``.
            # The ``expt_tp`` resolution below also consults ``parallel_state``
            # for callers that pass a partial custom process-group collection.
            # ``expt_dp`` is the data-parallel group for expert params: at
            # ``EP > 1`` with ``DP > 1`` it identifies how many ranks
            # replicate the SAME local expert. ``sharded_state_dict`` uses
            # ``expt_dp_group.rank()`` as the third component of
            # ``replica_id`` so the dist-checkpointing validator accepts the
            # template — standard ``SequentialMLP`` follows the same pattern.
            pg_collection = ProcessGroupCollection.use_mpu_process_groups(
                required_pgs=["tp", "expt_tp", "expt_dp"]
            )
        self.pg_collection = pg_collection
        # Cache the expert-data-parallel group (matches standard
        # ``SequentialMLP.dp_group``). Prefer ``pg_collection.expt_dp``;
        # fall back to ``parallel_state.get_expert_data_parallel_group`` for
        # callers that build a custom ``pg_collection`` without ``expt_dp``
        # wired in. Without this fallback, expert-key ``replica_id[2]``
        # silently collapses to 0 (Codex high-severity finding).
        self.expt_dp_group = getattr(pg_collection, "expt_dp", None)
        if self.expt_dp_group is None:
            try:
                from megatron.core import parallel_state as _ps
                self.expt_dp_group = _ps.get_expert_data_parallel_group(check_initialized=False)
            except Exception:
                self.expt_dp_group = None
        # Resolve EP topology from pg_collection.ep when available, else fall
        # back to the global mpu state, else single-rank. The previous MVP
        # hard-coded ep_world=1 / ep_rank=0 which silently replicated all
        # experts on every rank when --expert-model-parallel-size>1: compute
        # was correct, but memory cost was ep_world× the expected shard.
        ep_group = getattr(pg_collection, "ep", None)
        if ep_group is not None and torch.distributed.is_initialized():
            ep_world = torch.distributed.get_world_size(group=ep_group)
            ep_rank = torch.distributed.get_rank(group=ep_group)
        else:
            try:
                from megatron.core import parallel_state as _ps
                ep_world = _ps.get_expert_model_parallel_world_size()
                ep_rank = _ps.get_expert_model_parallel_rank()
                # Pull the corresponding torch ProcessGroup for the forward
                # comm path; fall back to None (which forces ep_world=1 below).
                try:
                    ep_group = _ps.get_expert_model_parallel_group()
                except Exception:
                    ep_group = None
            except Exception:
                ep_world, ep_rank, ep_group = 1, 0, None
            if ep_group is None:
                ep_world, ep_rank = 1, 0
        self.ep_world = ep_world
        self.ep_rank = ep_rank
        self.ep_group = ep_group
        self.sequence_parallel = bool(getattr(config, "sequence_parallel", False))
        self.dim = config.hidden_size
        self.n_routed_experts = config.num_moe_experts
        assert self.n_routed_experts % ep_world == 0, (
            f"num_moe_experts={self.n_routed_experts} not divisible by "
            f"expert-model-parallel world={ep_world}"
        )
        self.n_local_experts = self.n_routed_experts // ep_world
        self.experts_start_idx = ep_rank * self.n_local_experts
        self.experts_end_idx = self.experts_start_idx + self.n_local_experts
        self.n_activated = config.moe_router_topk
        self.gate = DeepSeekV4Gate(
            dim=self.dim,
            n_routed_experts=self.n_routed_experts,
            n_activated=self.n_activated,
            score_func=config.moe_router_score_function,
            route_scale=config.moe_router_topk_scaling_factor,
            vocab_size=config.vocab_size,
            is_hash_layer=is_hash_layer,
        )
        expert_dtype = (
            torch.float4_e2m1fn_x2 if getattr(config, "dsv4_expert_dtype", "fp4") == "fp4"
            else torch.float8_e4m3fn
        )
        # Expert TP group: standard Megatron MoE convention is to route
        # expert sharding through ``pg_collection.expt_tp`` (the
        # expert-tensor-parallel sub-group). When ``pg_collection`` doesn't
        # carry ``expt_tp``, resolve it from ``parallel_state`` directly
        # instead of substituting the model-TP group. At ETP=1 with TP>1,
        # expert weights and checkpoint metadata should be unsharded across
        # expert TP; model-TP ranks are replicas, not expert-TP shards.
        # ``parallel_state.get_expert_tensor_parallel_group`` returns the
        # actual ETP group (size=1 at ETP=1, size=ETP at ETP>1).
        expt_tp_group = getattr(pg_collection, "expt_tp", None)
        if expt_tp_group is None:
            try:
                from megatron.core import parallel_state as _ps
                expt_tp_group = _ps.get_expert_tensor_parallel_group(
                    check_initialized=False
                )
            except Exception:
                expt_tp_group = None
        self.expt_tp_group = expt_tp_group
        # When expert TP > 1, fold the per-expert all_reduce + outer slice
        # into a single fp32 partial accumulator + outer reduce (RS under SP=on,
        # AR under SP=off) + single bf16 cast. Matches stock Megatron MoE's
        # "explicit_expert_comm + dispatcher combine" precision pattern: one
        # bf16 round-trip per MoE block instead of one per expert.
        expt_tp_world = (
            torch.distributed.get_world_size(group=expt_tp_group)
            if expt_tp_group is not None and torch.distributed.is_initialized()
            else 1
        )
        self._use_outer_tp_reduce = expt_tp_world > 1
        moe_inter = config.ffn_hidden_size
        self.experts = nn.ModuleList([
            DeepSeekV4Expert(
                self.dim,
                moe_inter,
                dtype=expert_dtype,
                swiglu_limit=config.dsv4_swiglu_limit or 0.0,
                tp_group=expt_tp_group,
                expert_skip_comm=self._use_outer_tp_reduce,
            )
            if self.experts_start_idx <= i < self.experts_end_idx else None
            for i in range(self.n_routed_experts)
        ])
        # Mark routed-expert params as expert-parallel so Megatron's DDP
        # (distributed_data_parallel.py:122) buckets them under
        # ``expert_parallel_params`` and allreduces them only over the EP-DP
        # group, not the full DP group. Without this, under full finetune
        # each rank's local-shard gradients would be allreduced against
        # other ranks' DIFFERENT-shard gradients — corrupting both. Gate
        # and shared_experts stay replicated so they keep the default
        # ``allreduce=True``.
        for expert in self.experts:
            if expert is None:
                continue
            for p in expert.parameters(recurse=True):
                p.allreduce = False
        # Shared expert uses FP8 storage (no swiglu_limit per official model).
        # Same expert TP group so its w1/w2/w3 shard alongside routed experts.
        # Shares ``expert_skip_comm`` with routed experts: under TP>1 the
        # shared-expert partial is summed with routed partials and the cross-TP
        # combine fires once at the end of the MoE block.
        shared_inter = config.moe_shared_expert_intermediate_size
        self.shared_experts = DeepSeekV4Expert(
            self.dim,
            shared_inter,
            dtype=torch.float8_e4m3fn,
            swiglu_limit=0.0,
            tp_group=expt_tp_group,
            expert_skip_comm=self._use_outer_tp_reduce,
        )
        self._grouped_fp4_param_cache = {}
        # Bridge-compatible compressed OFT skew parameters. Each tensor has
        # shape (n_local_experts, num_blocks, block_size * (block_size - 1) / 2).
        self.w1_oft_r: Optional[torch.Tensor] = None
        self.w2_oft_r: Optional[torch.Tensor] = None
        self.w3_oft_r: Optional[torch.Tensor] = None
        if expert_dtype == torch.float4_e2m1fn_x2:
            self._alias_grouped_fp4_expert_params()

        # Optional fused all-to-all dispatcher for the EP > 1 path.
        self.dispatcher_backend = self._normalize_dispatcher_backend(
            getattr(config, "dsv4_moe_dispatcher", "naive")
        )
        self._deepep = None
        if self.dispatcher_backend == "deepep" and ep_world > 1:
            expt_tp_world = (
                torch.distributed.get_world_size(group=expt_tp_group)
                if expt_tp_group is not None else 1
            )
            self._validate_deepep_dispatcher_config(config, expt_tp_world)
            if getattr(config, "dsv4_deepep_expert_alignment", None) is None:
                # The grouped FP4/OFT expert core indexes one expert per 32-row
                # block. DSV4 uses native FP4 expert weights even when the
                # generic Megatron fp4 mode is not enabled, so request DeepEP
                # expert padding explicitly for this backend.
                config.dsv4_deepep_expert_alignment = 32
            from megatron.core.transformer.moe.token_dispatcher import _DeepepManager
            self._deepep = _DeepepManager(
                group=ep_group,
                num_local_experts=self.n_local_experts,
                router_topk=self.n_activated,
                num_experts=self.n_routed_experts,
                config=config,
            )
        elif self.dispatcher_backend == "deepep":
            self.dispatcher_backend = "naive"

    @staticmethod
    def _shared_expert_model_tp_replica_rank(model_tp_group, expt_tp_group) -> int:
        """Extra model-TP replica rank when shared experts use expert TP."""
        if (
            not torch.distributed.is_initialized()
            or model_tp_group is None
            or expt_tp_group is None
        ):
            return 0

        model_tp_size = model_tp_group.size()
        expt_tp_size = expt_tp_group.size()
        if expt_tp_size >= model_tp_size:
            return 0

        assert model_tp_size % expt_tp_size == 0, (
            "DSV4 shared_experts use expt_tp_group inside model TP; expected "
            f"model_tp_size={model_tp_size} to be divisible by "
            f"expt_tp_size={expt_tp_size}."
        )

        model_tp_rank = torch.distributed.get_rank(group=model_tp_group)
        return model_tp_rank // expt_tp_size

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        """Sharded state dict for routed experts + gate + shared expert.

        Mirrors standard ``SequentialMLP.sharded_state_dict``:

        * ``singleton_local_shards=False`` (default; matches Megatron-Bridge's
          save default ``training/checkpointing.py:save_checkpoint``):
          all local experts emit the *same* sharded key ``experts.<name>``
          and the expert axis is encoded in ``sharded_offsets`` as
          ``(axis, global_idx, num_global_experts)``. Compact on-disk
          format used by Kimi / Qwen3 / DSv3 via the standard ``MoELayer``.
        * ``singleton_local_shards=True``: per-expert sharded keys
          ``experts.{global_idx}.<name>``. Selected by callers that need
          per-expert sharding (e.g. some PEFT paths).

        ``replica_id[2]`` is the ``expt_dp`` group rank — the data-parallel
        rank along which the SAME local expert is replicated. With
        ``expt_dp.size() > 1`` (e.g. ``EP × DP-replication > 1``) the per-DP
        replicas need distinct ids; with size 1 the rank is 0.

        Bridges the ``ModuleList`` recursion gap on ``self.experts`` (the
        default ``MegatronModule`` flat-``state_dict`` fallback both
        bypasses each ``DSV4Column/RowParallelLinear.sharded_state_dict``
        and computes ``replica_id`` from the wider ``dp_cp_group``, which
        EP slices into).
        """
        metadata = ensure_metadata_has_dp_cp_group(metadata)
        singleton_local_shards = metadata.get("singleton_local_shards", False)

        # ``replica_id[2]`` = expert-data-parallel rank. At ``expt_dp == 1``
        # (no DP × EP co-replication) this is ``0``; at ``DP > 1`` with
        # ``EP > 1`` the per-DP replicas need distinct ids 0..expt_dp-1 so
        # the dist-checkpointing validator's replica check passes.
        # ``self.expt_dp_group`` is wired in ``__init__`` via
        # ``pg_collection.expt_dp``.
        expt_dp_rank = (
            torch.distributed.get_rank(group=self.expt_dp_group)
            if (self.expt_dp_group is not None and torch.distributed.is_initialized())
            else 0
        )

        sharded_state_dict: ShardedStateDict = {}

        # Mirror ``SequentialMLP.sharded_state_dict``'s expert loop. DSV4
        # stores experts at *global* slots in ``self.experts`` (a sparse
        # ``nn.ModuleList`` with ``None`` at non-local slots), so we collapse
        # to the non-None entries with ``enumerate`` here. NOTE: standard
        # SequentialMLP uses ``local_experts.{local_idx}.`` as the *dict-key*
        # prefix (its module hierarchy actually has a ``self.local_experts``
        # ModuleList, so the dict key doubles as a valid module path).
        # DSV4MoE has no ``local_experts`` submodule — its actual module
        # path is ``experts.{global_idx}.`` — and downstream code
        # (``_materialize_loaded_meta_module_tensors`` etc.) treats dict
        # keys as module paths to look up parameters with
        # ``module.get_submodule``. So we use ``experts.{global_idx}.`` as
        # both dict-key prefix and (in singleton mode) the on-disk
        # ShardedTensor key prefix. In non-singleton mode the ShardedTensor
        # key is renamed to the shared ``experts.`` form via
        # ``replace_prefix_for_sharding`` — same on-disk layout as
        # SequentialMLP, just keyed differently in-process.
        local_experts = [e for e in self.experts if e is not None]
        num_global_experts = len(self.experts)
        local_expert_indices_offset = self.experts_start_idx

        for expert_local_idx, expert in enumerate(local_experts):
            expert_global_idx = local_expert_indices_offset + expert_local_idx
            expert_state_dict_prefix = f"{prefix}experts.{expert_global_idx}."
            if singleton_local_shards:
                expert_sharded_prefix = expert_state_dict_prefix
                expert_sharded_offsets = sharded_offsets
            else:
                expert_sharded_prefix = f"{prefix}experts."
                expert_sharded_offsets = (
                    *sharded_offsets,
                    (len(sharded_offsets), expert_global_idx, num_global_experts),
                )
            expert_state_dict = expert.sharded_state_dict(
                expert_state_dict_prefix, expert_sharded_offsets, metadata
            )
            if expert_sharded_prefix != expert_state_dict_prefix:
                replace_prefix_for_sharding(
                    expert_state_dict, expert_state_dict_prefix, expert_sharded_prefix
                )
            for sh_ten in expert_state_dict.values():
                r = sh_ten.replica_id
                assert (
                    len(r) == 3
                ), f"Expected (PP, TP, DP) replica_id, got {r}"
                sh_ten.replica_id = (*r[:2], expt_dp_rank)
            sharded_state_dict.update(expert_state_dict)

        # OFT params are direct MoE parameters with shape
        # ``(n_local_experts, num_blocks, n_elements)``. Unlike standard MoE
        # expert weights, the local-expert axis already exists in the tensor,
        # so the generic helper cannot express the EP offset without adding an
        # extra dimension. Keep this wrapper small and use rank offsets with
        # the same replica-id convention as the standard linear helpers.
        oft_params = {
            f"{p}_oft_r": getattr(self, f"{p}_oft_r", None) for p in ("w1", "w2", "w3")
        }
        if any(v is not None for v in oft_params.values()):
            prepend_axis_num = len(sharded_offsets)
            expert_axis = prepend_axis_num
            tp_axis = prepend_axis_num + 1
            expt_tp_size_oft = (
                self.expt_tp_group.size() if self.expt_tp_group is not None else 1
            )
            expt_tp_rank_oft = (
                torch.distributed.get_rank(group=self.expt_tp_group)
                if (self.expt_tp_group is not None and torch.distributed.is_initialized())
                else 0
            )
            model_tp_group_oft = getattr(self.pg_collection, "tp", None)
            model_tp_rank_oft = (
                torch.distributed.get_rank(group=model_tp_group_oft)
                if (model_tp_group_oft is not None and torch.distributed.is_initialized())
                else 0
            )
            for name, oft_r in oft_params.items():
                if oft_r is None:
                    continue
                rank_offsets = [
                    *sharded_offsets,
                    (expert_axis, self.ep_rank, self.ep_world),
                ]
                if name == "w2_oft_r":
                    rank_offsets.append((tp_axis, expt_tp_rank_oft, expt_tp_size_oft))
                    replica_id = (0, 0, expt_dp_rank)
                else:
                    replica_id = (0, model_tp_rank_oft, expt_dp_rank)
                sharded_state_dict[f"{prefix}{name}"] = ShardedTensor.from_rank_offsets(
                    f"{prefix}{name}",
                    oft_r,
                    *rank_offsets,
                    replica_id=replica_id,
                    prepend_axis_num=prepend_axis_num,
                )

        # Non-expert children:
        #   - ``gate`` (DSV4Gate, a plain ``nn.Module`` with no
        #     ``sharded_state_dict``) falls into ``sharded_state_dict_default``'s
        #     ELSE branch with the model-TP group we pass, so its replica_id's
        #     TP component varies across model-TP ranks — correct.
        #   - ``shared_experts`` uses DSV4 linears sharded on ``expt_tp_group``.
        #     When ETP < TP, this mirrors routed-expert bookkeeping: the
        #     leftover model-TP blocks are replicas of the same expert-TP shard.
        expt_tp_size = (
            self.expt_tp_group.size() if self.expt_tp_group is not None else 1
        )
        model_tp_group = self.pg_collection.tp
        model_tp_size = model_tp_group.size() if model_tp_group is not None else 1
        shared_needs_model_tp_replica = expt_tp_size < model_tp_size
        shared_model_tp_replica_rank = (
            self._shared_expert_model_tp_replica_rank(model_tp_group, self.expt_tp_group)
            if shared_needs_model_tp_replica
            else 0
        )

        for name, module in self.named_children():
            if module is self.experts:
                continue
            child_sd = sharded_state_dict_default(
                module,
                f"{prefix}{name}.",
                sharded_offsets,
                metadata,
                tp_group=model_tp_group,
            )
            if name == "shared_experts" and shared_needs_model_tp_replica:
                for sh_ten in child_sd.values():
                    r = sh_ten.replica_id
                    sh_ten.replica_id = (r[0], shared_model_tp_replica_rank, r[2])
            sharded_state_dict.update(child_sd)
        return sharded_state_dict

    def _expert_tp_world_size(self) -> int:
        expt_tp_group = getattr(self, "expt_tp_group", None)
        if expt_tp_group is None:
            return 1
        return expt_tp_group.size()

    def _model_tp_world_size(self) -> int:
        tp_group = getattr(self.pg_collection, "tp", None)
        if tp_group is None:
            return 1
        return tp_group.size()

    @staticmethod
    def _all_gather_expert_tp_sizes(
        local_tokens: int,
        group: "dist.ProcessGroup",
        device: torch.device,
    ) -> tuple[list[int], int]:
        world = group.size()
        local_size = torch.tensor([local_tokens], dtype=torch.int64, device=device)
        gathered_sizes = [torch.empty_like(local_size) for _ in range(world)]
        torch.distributed.all_gather(gathered_sizes, local_size, group=group)
        sizes = [int(size.item()) for size in gathered_sizes]
        rank = torch.distributed.get_rank(group=group)
        return sizes, sum(sizes[:rank])

    @staticmethod
    def _all_gather_expert_tp_rows(
        tensor: torch.Tensor,
        sizes: list[int],
        group: "dist.ProcessGroup",
    ) -> torch.Tensor:
        max_tokens = max(sizes) if sizes else 0
        if tensor.shape[0] == max_tokens:
            padded = tensor.contiguous()
        else:
            padded = tensor.new_empty((max_tokens, *tensor.shape[1:]))
            if tensor.shape[0] > 0:
                padded[: tensor.shape[0]].copy_(tensor)

        gathered = [torch.empty_like(padded) for _ in sizes]
        torch.distributed.all_gather(gathered, padded, group=group)
        return torch.cat(
            [part[:num_tokens] for part, num_tokens in zip(gathered, sizes)],
            dim=0,
        )

    def _maybe_gather_inputs_for_expert_tp(
        self,
        x: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[tuple[torch.Size, int, int]]]:
        """Replicate MoE rows before expert-TP linears when ranks own different rows.

        Expert tensor parallelism shards the expert intermediate dimension:
        w1/w3 are column-parallel and w2 is row-parallel. The w2 all-reduce is
        only legal when every expert-TP rank processes identical token rows in
        identical order. If expert TP spans normal DP ranks, or if sequence
        parallelism has split rows across TP ranks, gather rows first and slice
        the local rows back out after the MoE.
        """
        expt_tp_world = self._expert_tp_world_size()
        if (
            expt_tp_world <= 1
            or x.dim() != 3
            or (
                not self.sequence_parallel
                and expt_tp_world <= self._model_tp_world_size()
            )
        ):
            return x, input_ids, None

        bsz, seqlen_local, hidden = x.shape
        if input_ids.dim() == 1:
            if input_ids.numel() != bsz * seqlen_local:
                return x, input_ids, None
            input_ids = input_ids.view(bsz, seqlen_local)
        elif input_ids.dim() != 2 or tuple(input_ids.shape) != (bsz, seqlen_local):
            return x, input_ids, None

        x_flat = x.reshape(-1, hidden).contiguous()
        input_ids_flat = input_ids.reshape(-1).contiguous()
        sizes, local_start = self._all_gather_expert_tp_sizes(
            x_flat.shape[0],
            self.expt_tp_group,
            x_flat.device,
        )
        x_full = self._all_gather_expert_tp_rows(
            x_flat,
            sizes,
            self.expt_tp_group,
        )
        input_ids_full = self._all_gather_expert_tp_rows(
            input_ids_flat,
            sizes,
            self.expt_tp_group,
        )
        gather_meta = (x.shape, local_start, x_flat.shape[0])
        return x_full.view(1, -1, hidden), input_ids_full.view(1, -1), gather_meta

    def _maybe_slice_expert_tp_output(
        self,
        y: torch.Tensor,
        gather_meta: Optional[tuple[torch.Size, int, int]],
    ) -> torch.Tensor:
        if gather_meta is None:
            return y
        original_shape, local_start, local_tokens = gather_meta
        y_flat = y.reshape(-1, original_shape[-1])
        return y_flat[local_start : local_start + local_tokens].view(
            original_shape
        ).contiguous()

    @staticmethod
    def _normalize_dispatcher_backend(dispatcher_backend: str) -> str:
        valid_backends = ("naive", "deepep")
        if dispatcher_backend not in valid_backends:
            raise ValueError(
                f"Invalid dsv4_moe_dispatcher={dispatcher_backend!r}; "
                f"expected one of {valid_backends}."
            )
        return dispatcher_backend

    @staticmethod
    def _validate_deepep_dispatcher_config(
        config: MLATransformerConfig,
        expt_tp_world: int,
    ) -> None:
        if not getattr(config, "moe_permute_fusion", False):
            raise ValueError(
                "dsv4_moe_dispatcher='deepep' requires "
                "config.moe_permute_fusion=True; the unfused permute path is "
                "non-deterministic with non-uniform probs."
            )
        if expt_tp_world > 1:
            raise ValueError(
                "DSV4 DeepEP dispatcher currently supports "
                "expert_tensor_parallel_size == 1 only; got "
                f"expt_tp_world={expt_tp_world}. The MCore-style TPxEP "
                "expansion (group=tp_ep_group, router_topk*=tp) is not "
                "wired here yet."
            )

    def _iter_local_experts(self):
        for i in range(self.experts_start_idx, self.experts_end_idx):
            expert = self.experts[i]
            if expert is None:
                continue
            yield expert

    def invalidate_grouped_fp4_cache(self) -> None:
        self._grouped_fp4_param_cache = {}

    @staticmethod
    def _copy_parameter_runtime_attrs(src: nn.Parameter, dst: nn.Parameter) -> None:
        for attr in ("allreduce", "_keep_fp32"):
            if hasattr(src, attr):
                setattr(dst, attr, getattr(src, attr))

    @staticmethod
    def _params_are_grouped_storage_aliases(params) -> bool:
        if not params:
            return False
        first = params[0]
        storage_ptr = first.untyped_storage().data_ptr()
        base_offset = first.storage_offset()
        shape = tuple(first.shape)
        stride = tuple(first.stride())
        numel = first.numel()
        for idx, param in enumerate(params):
            if tuple(param.shape) != shape or tuple(param.stride()) != stride:
                return False
            if param.untyped_storage().data_ptr() != storage_ptr:
                return False
            if param.storage_offset() != base_offset + idx * numel:
                return False
        return True

    @staticmethod
    def _grouped_view_from_storage_aliases(params) -> torch.Tensor:
        first = params[0]
        return torch.as_strided(
            first,
            size=(len(params), *tuple(first.shape)),
            stride=(first.numel(), *tuple(first.stride())),
            storage_offset=first.storage_offset(),
        )

    def _local_expert_linears(self, linear_name: str):
        return [getattr(expert, linear_name) for expert in self._iter_local_experts()]

    @staticmethod
    def _validate_grouped_fp4_linear_name(linear_name: str) -> None:
        if linear_name not in {"w1", "w2", "w3"}:
            raise RuntimeError(f"Unsupported DSV4 routed expert linear: {linear_name}")

    def _validate_grouped_fp4_expert_linear_tensors(
        self,
        linear_name: str,
        grouped_weight: torch.Tensor,
        grouped_scale: torch.Tensor,
    ):
        self._validate_grouped_fp4_linear_name(linear_name)
        linears = self._local_expert_linears(linear_name)
        if not linears:
            raise RuntimeError("DSV4 grouped FP4 expected at least one local expert")

        first_weight = linears[0].weight
        first_scale = linears[0].scale
        if first_scale is None:
            raise RuntimeError(f"DSV4 grouped FP4 {linear_name} requires FP4 scales")

        expected_weight_shape = (len(linears), *tuple(first_weight.shape))
        expected_scale_shape = (len(linears), *tuple(first_scale.shape))
        if tuple(grouped_weight.shape) != expected_weight_shape:
            raise ValueError(
                f"DSV4 grouped FP4 {linear_name} weight shape mismatch: "
                f"expected {expected_weight_shape}, got {tuple(grouped_weight.shape)}"
            )
        if tuple(grouped_scale.shape) != expected_scale_shape:
            raise ValueError(
                f"DSV4 grouped FP4 {linear_name} scale shape mismatch: "
                f"expected {expected_scale_shape}, got {tuple(grouped_scale.shape)}"
            )
        if grouped_weight.dtype != torch.float4_e2m1fn_x2:
            raise ValueError(
                f"DSV4 grouped FP4 {linear_name} weight requires "
                f"torch.float4_e2m1fn_x2, got {grouped_weight.dtype}"
            )
        if grouped_scale.dtype != first_scale.dtype:
            raise ValueError(
                f"DSV4 grouped FP4 {linear_name} scale dtype mismatch: "
                f"expected {first_scale.dtype}, got {grouped_scale.dtype}"
            )
        if grouped_weight.device != grouped_scale.device:
            raise ValueError(
                f"DSV4 grouped FP4 {linear_name} weight/scale devices differ: "
                f"{grouped_weight.device} vs {grouped_scale.device}"
            )
        if not grouped_weight.is_contiguous():
            raise ValueError(f"DSV4 grouped FP4 {linear_name} weight must be contiguous")
        if not grouped_scale.is_contiguous():
            raise ValueError(f"DSV4 grouped FP4 {linear_name} scale must be contiguous")
        return linears

    @staticmethod
    def _set_linear_parameter(linear: nn.Module, name: str, param: nn.Parameter) -> None:
        linear.__dict__.pop(name, None)
        linear._buffers.pop(name, None)
        linear._modules.pop(name, None)
        linear._non_persistent_buffers_set.discard(name)
        linear._parameters[name] = param

    def set_grouped_fp4_expert_linear_tensors(
        self,
        linear_name: str,
        grouped_weight: torch.Tensor,
        grouped_scale: torch.Tensor,
    ) -> None:
        """Install grouped FP4 storage and expose each expert as a view.

        This is the no-copy setup/offload hook: callers provide the grouped
        weight and scale tensors, and each per-expert DSV4Linear parameter is
        rebound to the corresponding slice.
        """
        linears = self._validate_grouped_fp4_expert_linear_tensors(
            linear_name,
            grouped_weight,
            grouped_scale,
        )
        for idx, linear in enumerate(linears):
            old_weight = linear.weight
            old_scale = linear.scale
            new_weight = nn.Parameter(
                grouped_weight[idx],
                requires_grad=old_weight.requires_grad,
            )
            new_scale = nn.Parameter(
                grouped_scale[idx],
                requires_grad=old_scale.requires_grad,
            )
            self._copy_parameter_runtime_attrs(old_weight, new_weight)
            self._copy_parameter_runtime_attrs(old_scale, new_scale)
            self._set_linear_parameter(linear, "weight", new_weight)
            self._set_linear_parameter(linear, "scale", new_scale)
            linear.weight.scale = new_scale
        self.invalidate_grouped_fp4_cache()

    def set_grouped_fp4_expert_tensor(
        self,
        tensor_name: str,
        grouped_tensor: torch.Tensor,
    ) -> None:
        """Install one grouped FP4 expert tensor from an offload flat slice."""
        try:
            linear_name, tensor_kind = tensor_name.split(".", 1)
        except ValueError as exc:
            raise ValueError(
                "DSV4 grouped FP4 tensor name must be '<w1|w2|w3>.<weight|scale>', "
                f"got {tensor_name!r}"
            ) from exc
        self._validate_grouped_fp4_linear_name(linear_name)
        if tensor_kind not in {"weight", "scale"}:
            raise ValueError(
                "DSV4 grouped FP4 tensor kind must be 'weight' or 'scale', "
                f"got {tensor_kind!r}"
            )

        linears = self._local_expert_linears(linear_name)
        if not linears:
            raise RuntimeError("DSV4 grouped FP4 expected at least one local expert")
        first_tensor = getattr(linears[0], tensor_kind)
        if first_tensor is None:
            raise RuntimeError(
                f"DSV4 grouped FP4 {linear_name} requires FP4 {tensor_kind} tensors"
            )
        expected_shape = (len(linears), *tuple(first_tensor.shape))
        if tuple(grouped_tensor.shape) != expected_shape:
            raise ValueError(
                f"DSV4 grouped FP4 {linear_name}.{tensor_kind} shape mismatch: "
                f"expected {expected_shape}, got {tuple(grouped_tensor.shape)}"
            )
        expected_dtype = (
            torch.float4_e2m1fn_x2 if tensor_kind == "weight" else first_tensor.dtype
        )
        if grouped_tensor.dtype != expected_dtype:
            raise ValueError(
                f"DSV4 grouped FP4 {linear_name}.{tensor_kind} dtype mismatch: "
                f"expected {expected_dtype}, got {grouped_tensor.dtype}"
            )
        if not grouped_tensor.is_contiguous():
            raise ValueError(
                f"DSV4 grouped FP4 {linear_name}.{tensor_kind} must be contiguous"
            )

        for idx, linear in enumerate(linears):
            old_tensor = getattr(linear, tensor_kind)
            new_tensor = nn.Parameter(
                grouped_tensor[idx],
                requires_grad=old_tensor.requires_grad,
            )
            self._copy_parameter_runtime_attrs(old_tensor, new_tensor)
            self._set_linear_parameter(linear, tensor_kind, new_tensor)
            linear.weight.scale = linear.scale
        self.invalidate_grouped_fp4_cache()

    def grouped_fp4_expert_tensors(self) -> Dict[str, torch.Tensor]:
        """Return the six grouped FP4 expert tensors used by kernels/offload."""
        self.ensure_grouped_fp4_aliases()
        grouped = {}
        for linear_name in ("w1", "w2", "w3"):
            weight, scale = self._grouped_expert_linear_params(linear_name)
            grouped[f"{linear_name}.weight"] = weight
            grouped[f"{linear_name}.scale"] = scale
        return grouped

    def set_grouped_fp4_expert_tensors(
        self,
        grouped_tensors: Dict[str, torch.Tensor],
    ) -> None:
        """Install all grouped FP4 expert tensors from an offload/setup bundle."""
        expected = {
            f"{linear_name}.{kind}"
            for linear_name in ("w1", "w2", "w3")
            for kind in ("weight", "scale")
        }
        missing = expected.difference(grouped_tensors)
        extra = set(grouped_tensors).difference(expected)
        if missing or extra:
            raise ValueError(
                "DSV4 grouped FP4 tensor bundle mismatch: "
                f"missing={sorted(missing)}, extra={sorted(extra)}"
            )
        for linear_name in ("w1", "w2", "w3"):
            self.set_grouped_fp4_expert_linear_tensors(
                linear_name,
                grouped_tensors[f"{linear_name}.weight"],
                grouped_tensors[f"{linear_name}.scale"],
            )

    def _alias_grouped_fp4_expert_linear_params(self, linear_name: str) -> None:
        self._validate_grouped_fp4_linear_name(linear_name)
        linears = self._local_expert_linears(linear_name)
        if not linears:
            raise RuntimeError("DSV4 grouped FP4 expected at least one local expert")

        weights = [linear.weight for linear in linears]
        scales = [linear.scale for linear in linears]
        if any(weight.dtype != torch.float4_e2m1fn_x2 for weight in weights):
            raise RuntimeError(f"DSV4 grouped FP4 {linear_name} requires FP4 expert weights")
        if any(scale is None for scale in scales):
            raise RuntimeError(f"DSV4 grouped FP4 {linear_name} requires FP4 scales")

        if (
            self._params_are_grouped_storage_aliases(weights)
            and self._params_are_grouped_storage_aliases(scales)
        ):
            return

        weight_shape = tuple(weights[0].shape)
        scale_shape = tuple(scales[0].shape)
        if any(tuple(weight.shape) != weight_shape for weight in weights):
            raise RuntimeError(f"DSV4 grouped FP4 {linear_name} weight shapes differ")
        if any(tuple(scale.shape) != scale_shape for scale in scales):
            raise RuntimeError(f"DSV4 grouped FP4 {linear_name} scale shapes differ")

        grouped_weight = torch.empty(
            (len(weights), *weight_shape),
            device=weights[0].device,
            dtype=weights[0].dtype,
        )
        grouped_scale = torch.empty(
            (len(scales), *scale_shape),
            device=scales[0].device,
            dtype=scales[0].dtype,
        )
        with torch.no_grad():
            for idx, (weight, scale) in enumerate(zip(weights, scales)):
                grouped_weight[idx].copy_(weight)
                grouped_scale[idx].copy_(scale)

        for idx, linear in enumerate(linears):
            old_weight = linear.weight
            old_scale = linear.scale
            new_weight = nn.Parameter(
                grouped_weight[idx],
                requires_grad=old_weight.requires_grad,
            )
            new_scale = nn.Parameter(
                grouped_scale[idx],
                requires_grad=old_scale.requires_grad,
            )
            self._copy_parameter_runtime_attrs(old_weight, new_weight)
            self._copy_parameter_runtime_attrs(old_scale, new_scale)
            self._set_linear_parameter(linear, "weight", new_weight)
            self._set_linear_parameter(linear, "scale", new_scale)
            linear.weight.scale = new_scale

    def _alias_grouped_fp4_expert_params(self) -> None:
        for linear_name in ("w1", "w2", "w3"):
            self._alias_grouped_fp4_expert_linear_params(linear_name)
        self.invalidate_grouped_fp4_cache()

    def ensure_grouped_fp4_aliases(self) -> None:
        self._alias_grouped_fp4_expert_params()

    def _apply(self, fn):
        super()._apply(fn)
        if hasattr(self, "experts") and hasattr(self, "_grouped_fp4_param_cache"):
            local_experts = list(self._iter_local_experts())
            if local_experts:
                first = local_experts[0].w1.weight
                if first.dtype == torch.float4_e2m1fn_x2:
                    self.ensure_grouped_fp4_aliases()
        return self

    def _grouped_expert_linear_params(
        self,
        linear_name: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self._validate_grouped_fp4_linear_name(linear_name)
        self._alias_grouped_fp4_expert_linear_params(linear_name)
        linears = self._local_expert_linears(linear_name)
        weights = [linear.weight for linear in linears]
        scales = [linear.scale for linear in linears]
        return (
            self._grouped_view_from_storage_aliases(weights),
            self._grouped_view_from_storage_aliases(scales),
        )

    def _grouped_fp4_linear(
        self,
        x: torch.Tensor,
        linear_name: str,
        pos_to_expert: torch.Tensor,
    ) -> torch.Tensor:
        from megatron.core.transformer.experimental_attention_variant.dsv4_grouped_fp4 import (
            grouped_fp4_linear,
        )

        weight, scale = self._grouped_expert_linear_params(linear_name)
        return grouped_fp4_linear(
            x,
            weight,
            scale,
            pos_to_expert.contiguous(),
        )

    def _reduce_grouped_row_parallel_output(self, y: torch.Tensor) -> torch.Tensor:
        expt_tp_group = getattr(self, "expt_tp_group", None)
        if expt_tp_group is None or expt_tp_group.size() <= 1:
            return y
        out_dtype = y.dtype
        y = reduce_from_tensor_model_parallel_region(
            y.float(),
            group=expt_tp_group,
        )
        return y.to(out_dtype)

    def _dsv4_expert_oft_input_dim(self, proj: str) -> int:
        if proj in {"w1", "w3"}:
            return self.dim
        if proj == "w2":
            first_expert = next(iter(self._iter_local_experts()))
            return int(first_expert.w2.weight.shape[1] * 2)
        raise RuntimeError(f"Unsupported DSV4 expert OFT projection: {proj}")

    @staticmethod
    def _dsv4_oft_num_elements(block_size: int) -> int:
        return block_size * (block_size - 1) // 2

    def _dsv4_expert_oft_block_size(self, proj: str, oft_r: torch.Tensor) -> int:
        if oft_r.dim() != 3:
            raise ValueError(
                f"DSV4 {proj} OFT expects compressed oft_r with shape "
                "(num_local_experts, num_blocks, n_elements); got "
                f"{tuple(oft_r.shape)}"
            )
        if oft_r.shape[0] != self.n_local_experts:
            raise ValueError(
                f"DSV4 {proj} OFT expected {self.n_local_experts} local experts, "
                f"got {oft_r.shape[0]}"
            )

        input_dim = self._dsv4_expert_oft_input_dim(proj)
        num_blocks = int(oft_r.shape[1])
        if num_blocks <= 0 or input_dim % num_blocks != 0:
            raise ValueError(
                f"DSV4 {proj} OFT input dim {input_dim} is not divisible by "
                f"num_blocks {num_blocks}"
            )
        block_size = input_dim // num_blocks
        expected = self._dsv4_oft_num_elements(block_size)
        if oft_r.shape[2] != expected:
            raise ValueError(
                f"DSV4 {proj} OFT compressed size mismatch: block_size={block_size} "
                f"expects {expected} elements, got {oft_r.shape[2]}"
            )
        return block_size

    @staticmethod
    def _dsv4_oft_skew_symmetric(oft_r: torch.Tensor, block_size: int) -> torch.Tensor:
        flat = oft_r.reshape(-1, oft_r.shape[-1])
        rows, cols = torch.triu_indices(
            block_size,
            block_size,
            1,
            device=oft_r.device,
        )
        q = torch.zeros(
            flat.shape[0],
            block_size,
            block_size,
            device=oft_r.device,
            dtype=oft_r.dtype,
        )
        q[:, rows, cols] = flat
        q = q - q.transpose(-2, -1)
        return q.reshape(*oft_r.shape[:-1], block_size, block_size)

    @staticmethod
    def _dsv4_oft_cayley_neumann(q: torch.Tensor) -> torch.Tensor:
        block_size = q.shape[-1]
        q_flat = q.reshape(-1, block_size, block_size).contiguous()
        return dsv4_oft_cayley_neumann(q_flat, 5).reshape_as(q)

    def _dsv4_expert_oft_rotation(self, proj: str) -> torch.Tensor:
        oft_r = getattr(self, f"{proj}_oft_r")
        if oft_r is None:
            raise RuntimeError(f"DSV4 {proj} OFT rotation requested but no oft_r is registered")
        block_size = self._dsv4_expert_oft_block_size(proj, oft_r)
        q = self._dsv4_oft_skew_symmetric(oft_r, block_size)
        return self._dsv4_oft_cayley_neumann(q)

    def ensure_dsv4_expert_oft_r(
        self,
        proj: str,
        *,
        block_size: int,
        dtype: torch.dtype,
        sample: Optional[torch.Tensor] = None,
    ) -> bool:
        input_dim = self._dsv4_expert_oft_input_dim(proj)
        if input_dim % block_size != 0:
            raise ValueError(
                f"DSV4 {proj} OFT input dim {input_dim} is not divisible "
                f"by block_size {block_size}"
            )
        num_blocks = input_dim // block_size
        n_elements = self._dsv4_oft_num_elements(block_size)
        if sample is not None and tuple(sample.shape) != (num_blocks, n_elements):
            raise ValueError(
                f"DSV4 {proj} OFT shape mismatch: expected "
                f"({num_blocks}, {n_elements}), got {tuple(sample.shape)}"
            )

        attr = f"{proj}_oft_r"
        current = getattr(self, attr)
        shape = (self.n_local_experts, num_blocks, n_elements)
        device = next(iter(self._iter_local_experts())).w1.weight.device
        if current is not None:
            if tuple(current.shape) == shape and current.dtype == dtype and current.device == device:
                return False
            raise RuntimeError(
                "DSV4 expert OFT update would replace an existing R buffer: "
                f"proj={proj}, current_shape={tuple(current.shape)}, "
                f"current_dtype={current.dtype}, current_device={current.device}, "
                f"incoming_shape={shape}, incoming_dtype={dtype}, "
                f"incoming_device={device}."
            )

        buf = torch.zeros(shape, dtype=dtype, device=device)
        param = nn.Parameter(buf)
        param.allreduce = False
        setattr(self, attr, param)
        return True

    def has_dsv4_expert_oft(self) -> bool:
        return (
            self.w1_oft_r is not None
            or self.w2_oft_r is not None
            or self.w3_oft_r is not None
        )

    @staticmethod
    def _routed_expert_oft_rotation(
        x: torch.Tensor,
        oft_r: torch.Tensor,
        pos_to_expert: torch.Tensor,
    ) -> torch.Tensor:
        from megatron.core.transformer.experimental_attention_variant.dsv4_oft_rotation import (
            dsv4_routed_oft_rotation,
        )

        return dsv4_routed_oft_rotation(x, oft_r, pos_to_expert, block_m=32)

    def _prepare_grouped_deepep_inputs(
        self,
        permuted_x: torch.Tensor,
        permuted_probs: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        alignment: int = 32,
    ):
        probs = permuted_probs.reshape(-1)
        if probs.numel() != permuted_x.shape[0]:
            raise ValueError(
                "DSV4 DeepEP grouped core expected permuted_probs to be 1-D over "
                f"permuted rows; got {probs.numel()} values for "
                f"{permuted_x.shape[0]} rows"
            )

        if _dsv4_is_cuda_graph_capturing():
            expert_ids = torch.arange(
                self.n_local_experts,
                device=permuted_x.device,
                dtype=torch.int64,
            )
            pos_to_expert = torch.repeat_interleave(
                expert_ids,
                tokens_per_expert.to(torch.int64),
                output_size=permuted_x.shape[0],
            ).to(torch.int32)
            return permuted_x, probs, pos_to_expert, None

        counts = [int(v) for v in tokens_per_expert.detach().cpu().tolist()]
        aligned_counts = [((n + alignment - 1) // alignment) * alignment for n in counts]
        total_tokens = sum(counts)
        total_aligned = sum(aligned_counts)
        if total_tokens != permuted_x.shape[0]:
            raise ValueError(
                "DSV4 DeepEP grouped core expected tokens_per_expert to sum to "
                f"permuted_x rows; got {total_tokens} vs {permuted_x.shape[0]}"
            )

        if aligned_counts == counts:
            expert_ids = torch.arange(
                self.n_local_experts,
                device=permuted_x.device,
                dtype=torch.int64,
            )
            pos_to_expert = torch.repeat_interleave(
                expert_ids,
                tokens_per_expert.to(torch.int64),
                output_size=permuted_x.shape[0],
            ).to(torch.int32)
            return permuted_x, probs, pos_to_expert, None

        # Eager training can see naturally unaligned expert counts. Keep the
        # grouped FP4 path active by padding each expert segment to the kernel's
        # 32-row contract, then unpad before DeepEP combine.
        pos_to_expert = torch.full(
            (total_aligned,),
            -1,
            dtype=torch.int32,
            device=permuted_x.device,
        )
        grouped_x = permuted_x.new_zeros((total_aligned, permuted_x.shape[-1]))
        grouped_probs = probs.new_zeros((total_aligned,))

        segments = []
        src_offset = 0
        dst_offset = 0
        for expert_id, (count, aligned_count) in enumerate(zip(counts, aligned_counts)):
            if aligned_count > 0:
                pos_to_expert[dst_offset : dst_offset + aligned_count] = expert_id
            if count > 0:
                grouped_x[dst_offset : dst_offset + count].copy_(
                    permuted_x[src_offset : src_offset + count]
                )
                grouped_probs[dst_offset : dst_offset + count].copy_(
                    probs[src_offset : src_offset + count]
                )
                segments.append((src_offset, dst_offset, count))
            src_offset += count
            dst_offset += aligned_count
        return grouped_x, grouped_probs, pos_to_expert, segments

    def _restore_grouped_deepep_output(
        self,
        grouped_y: torch.Tensor,
        output_shape: torch.Size,
        segments,
    ) -> torch.Tensor:
        if segments is None or grouped_y.shape[0] == output_shape[0]:
            return grouped_y
        output = grouped_y.new_zeros(output_shape)
        for src_offset, dst_offset, count in segments:
            output[src_offset : src_offset + count].copy_(
                grouped_y[dst_offset : dst_offset + count]
            )
        return output

    def _forward_grouped_deepep_experts(
        self,
        permuted_x: torch.Tensor,
        permuted_probs: torch.Tensor,
        tokens_per_expert: torch.Tensor,
    ) -> torch.Tensor:
        if permuted_x.numel() == 0:
            return torch.zeros_like(permuted_x)
        from megatron.core.transformer.experimental_attention_variant.dsv4_grouped_fp4 import (
            has_deep_gemm_official_fp8_fp4,
        )

        grouped_x, grouped_probs, pos_to_expert, segments = self._prepare_grouped_deepep_inputs(
            permuted_x,
            permuted_probs,
            tokens_per_expert,
            alignment=128 if has_deep_gemm_official_fp8_fp4() else 32,
        )

        w1_input = grouped_x
        w3_input = grouped_x
        if self.w1_oft_r is not None:
            w1_input = self._routed_expert_oft_rotation(
                w1_input,
                self._dsv4_expert_oft_rotation("w1"),
                pos_to_expert,
            )
        if self.w3_oft_r is not None:
            w3_input = self._routed_expert_oft_rotation(
                w3_input,
                self._dsv4_expert_oft_rotation("w3"),
                pos_to_expert,
            )

        gate = self._grouped_fp4_linear(w1_input, "w1", pos_to_expert)
        up = self._grouped_fp4_linear(w3_input, "w3", pos_to_expert)
        first_expert = next(iter(self._iter_local_experts()))
        from megatron.core.transformer.experimental_attention_variant.dsv4_fused_route import (
            dsv4_clamp_silu_mul_preexpanded,
        )

        y = dsv4_clamp_silu_mul_preexpanded(
            gate,
            up,
            grouped_probs,
            first_expert.swiglu_limit,
            permuted_x.dtype,
        )
        if self.w2_oft_r is not None:
            y = self._routed_expert_oft_rotation(
                y,
                self._dsv4_expert_oft_rotation("w2"),
                pos_to_expert,
            )
        grouped_out = self._grouped_fp4_linear(y, "w2", pos_to_expert)
        grouped_out = self._reduce_grouped_row_parallel_output(grouped_out)
        return self._restore_grouped_deepep_output(
            grouped_out,
            permuted_x.shape,
            segments,
        )

    def _forward_fused_route_grouped_experts(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> torch.Tensor:
        if hidden_states.numel() == 0:
            return torch.zeros_like(hidden_states)

        from megatron.core.transformer.experimental_attention_variant.dsv4_fused_route import (
            dsv4_clamp_silu_mul_topk,
            expand_to_fused_route,
            get_fused_route_mapping,
            reduce_fused_topk_fp32,
        )
        from megatron.core.transformer.experimental_attention_variant.dsv4_grouped_fp4 import (
            has_deep_gemm_official_fp8_fp4,
        )

        topk_ids = topk_ids.to(torch.int64).contiguous()
        topk_weights = topk_weights.contiguous()
        pos_to_expert, pos_to_token_topk, token_topk_to_pos = get_fused_route_mapping(
            topk_ids,
            self.n_local_experts,
            alignment=128 if has_deep_gemm_official_fp8_fp4() else 32,
        )
        expanded_x = expand_to_fused_route(
            hidden_states,
            token_topk_to_pos,
            pos_to_expert,
        )

        w1_input = expanded_x
        w3_input = expanded_x
        if self.w1_oft_r is not None:
            w1_input = self._routed_expert_oft_rotation(
                w1_input,
                self._dsv4_expert_oft_rotation("w1"),
                pos_to_expert,
            )
        if self.w3_oft_r is not None:
            w3_input = self._routed_expert_oft_rotation(
                w3_input,
                self._dsv4_expert_oft_rotation("w3"),
                pos_to_expert,
            )

        gate = self._grouped_fp4_linear(w1_input, "w1", pos_to_expert)
        up = self._grouped_fp4_linear(w3_input, "w3", pos_to_expert)
        first_expert = next(iter(self._iter_local_experts()))
        y = dsv4_clamp_silu_mul_topk(
            gate,
            up,
            topk_weights,
            pos_to_token_topk,
            first_expert.swiglu_limit,
            hidden_states.dtype,
        )
        if self.w2_oft_r is not None:
            y = self._routed_expert_oft_rotation(
                y,
                self._dsv4_expert_oft_rotation("w2"),
                pos_to_expert,
            )
        expanded_out = self._grouped_fp4_linear(y, "w2", pos_to_expert)
        expanded_out = self._reduce_grouped_row_parallel_output(expanded_out)
        routed = reduce_fused_topk_fp32(expanded_out.contiguous(), token_topk_to_pos)
        return routed.to(hidden_states.dtype)

    def _forward_local_grouped_oft_experts(
        self,
        x: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        """Run local routed experts through the grouped FP4 core with MoE OFT.

        This is the naive/single-rank counterpart of the DeepEP expert core:
        build an expert-major local token layout, run the same grouped FP4/OFT
        kernels, then scatter-add outputs back to token order.
        """
        token_indices = []
        scatter_counts = []
        chunks = []
        probs = []
        counts = []
        for expert_id in range(self.experts_start_idx, self.experts_end_idx):
            idx, top = torch.where(indices == expert_id)
            counts.append(int(idx.numel()))
            if idx.numel() == 0:
                continue
            token_indices.append(idx)
            scatter_counts.append(int(idx.numel()))
            chunks.append(x[idx])
            probs.append(weights[idx, top])

        if not chunks:
            return torch.zeros_like(x, dtype=torch.float32)

        permuted_x = torch.cat(chunks, dim=0)
        permuted_probs = torch.cat(probs, dim=0)
        tokens_per_expert = torch.tensor(
            counts,
            device=x.device,
            dtype=torch.int64,
        )
        permuted_y = self._forward_grouped_deepep_experts(
            permuted_x,
            permuted_probs,
            tokens_per_expert,
        )

        y = torch.zeros_like(x, dtype=torch.float32)
        offset = 0
        for idx, count in zip(token_indices, scatter_counts):
            y.index_add_(0, idx, permuted_y[offset : offset + count].float())
            offset += count
        return y

    def _forward_naive_ep_routed(
        self,
        x: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        """Naive EP routed experts, matching SGLang's DP-gathered MoE path."""
        import torch.distributed as _dist
        from torch.distributed.nn import functional as _dist_nn

        n_local = x.shape[0]
        size_tensor = torch.tensor([n_local], device=x.device, dtype=torch.int64)
        size_list = [torch.empty_like(size_tensor) for _ in range(self.ep_world)]
        _dist.all_gather(size_list, size_tensor, group=self.ep_group)
        sizes = [int(s.item()) for s in size_list]
        max_n = max(sizes)

        if n_local == 0:
            x_pad = x.new_zeros((max_n, x.shape[1]))
            indices_pad = indices.new_zeros((max_n,) + indices.shape[1:])
            weights_pad = weights.new_zeros((max_n,) + weights.shape[1:])
        elif max_n > n_local:
            pad_count = max_n - n_local
            x_pad = torch.cat([x, x[:1].expand(pad_count, -1).contiguous()], dim=0)
            indices_pad = torch.cat(
                [indices, indices[:1].expand(pad_count, -1).contiguous()],
                dim=0,
            )
            weights_pad = torch.cat(
                [weights, weights[:1].expand(pad_count, -1).contiguous()],
                dim=0,
            )
        else:
            x_pad, indices_pad, weights_pad = x, indices, weights

        x_global = torch.cat(_dist_nn.all_gather(x_pad, group=self.ep_group), dim=0)
        indices_global = torch.cat(
            _dist_nn.all_gather(indices_pad, group=self.ep_group), dim=0
        )
        weights_global = torch.cat(
            _dist_nn.all_gather(weights_pad, group=self.ep_group), dim=0
        )

        if self.has_dsv4_expert_oft():
            y_global = self._forward_local_grouped_oft_experts(
                x_global,
                weights_global,
                indices_global,
            )
        else:
            y_global = torch.zeros_like(x_global, dtype=torch.float32)
            counts = torch.bincount(
                indices_global.flatten(), minlength=self.n_routed_experts
            ).tolist()
            for i in range(self.experts_start_idx, self.experts_end_idx):
                if counts[i] == 0:
                    continue
                expert = self.experts[i]
                idx, top = torch.where(indices_global == i)
                y_global[idx] = y_global[idx] + expert(
                    x_global[idx], weights_global[idx, top, None]
                )
        y_global = _dist_nn.all_reduce(y_global.float(), group=self.ep_group)
        start = self.ep_rank * max_n
        return y_global[start : start + n_local]

    def forward(self, x: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        x, input_ids, expert_tp_gather_meta = (
            self._maybe_gather_inputs_for_expert_tp(x, input_ids)
        )
        shape = x.size()
        x = x.view(-1, self.dim)
        weights, indices = self.gate(x, input_ids.flatten())

        if self.ep_world == 1:
            if self.has_dsv4_expert_oft():
                # OFT path keeps its own (per-expert-reduced) precision pattern
                # for now. Mixing OFT with skip_comm would require lifting the
                # outer reduce into the OFT helper too; out of scope here.
                assert not self._use_outer_tp_reduce, (
                    "DSV4 OFT MoE path is incompatible with skip_comm + outer "
                    "reduce. Either disable OFT or set expt_tp_world == 1."
                )
                y = self._forward_local_grouped_oft_experts(x, weights, indices)
            else:
                # Per-expert loop. Under ``_use_outer_tp_reduce`` the experts
                # return PARTIAL (un-reduced) bf16 outputs; we accumulate them
                # in fp32 across experts (and the shared expert below), then
                # do a single outer reduce (RS under SP=on, AR otherwise) and
                # a single bf16 cast at the very end. Matches stock Megatron
                # MoE's precision pattern.
                y = torch.zeros_like(x, dtype=torch.float32)
                counts = torch.bincount(
                    indices.flatten(), minlength=self.n_routed_experts
                ).tolist()
                for i in range(self.experts_start_idx, self.experts_end_idx):
                    if counts[i] == 0:
                        continue
                    expert = self.experts[i]
                    idx, top = torch.where(indices == i)
                    y[idx] = y[idx] + expert(x[idx], weights[idx, top, None])
            y = y + self.shared_experts(x)
            if self._use_outer_tp_reduce:
                # ``y`` here is fp32, partial across TP, full-sequence layout
                # ``[B*S, H]`` (after the ``view(-1, dim)`` above; for the SP=on
                # case the gather wrapper already gathered to full sequence).
                if self.sequence_parallel:
                    # SP=on: scatter along seq dim. The output is the per-rank
                    # SP shard already, so we bypass _maybe_slice_expert_tp_output
                    # below — the SP=on gather wrapper sets gather_meta such
                    # that the slice would otherwise re-shard incorrectly.
                    y = reduce_scatter_to_sequence_parallel_region(
                        y, group=self.expt_tp_group
                    )
                else:
                    y = reduce_from_tensor_model_parallel_region(
                        y, group=self.expt_tp_group
                    )
            y = y.type_as(x)
            if self._use_outer_tp_reduce and self.sequence_parallel:
                # Under SP=on the outer RS replaced the legacy "all_reduce +
                # outer slice" combo. Reshape using the ORIGINAL pre-gather
                # shape (the rank's SP shard).
                if expert_tp_gather_meta is not None:
                    original_shape = expert_tp_gather_meta[0]
                    y = y.view(original_shape).contiguous()
                else:
                    # SP=on with no gather wrapper: keep shape from view(-1, dim).
                    y = y.view(shape[0], -1, self.dim).contiguous()
                return y
            y = y.view(shape)
            y = self._maybe_slice_expert_tp_output(
                y,
                expert_tp_gather_meta,
            )
            return y

        if self.dispatcher_backend == "deepep" and self._deepep is not None:
            y = self._forward_deepep(x, weights, indices, shape)
            y = self._maybe_slice_expert_tp_output(
                y,
                expert_tp_gather_meta,
            )
            return y

        # ep_world>1, naive backend: SGLang gathers DP-sharded MoE inputs before
        # the routed experts, runs the local expert shard, then scatters the
        # all-reduced routed output back to the rank-local tokens.
        y = self._forward_naive_ep_routed(x, weights, indices)

        # Shared expert is replicated across EP ranks; compute on local input.
        y = y + self.shared_experts(x)
        y = y.type_as(x).view(shape)
        y = self._maybe_slice_expert_tp_output(
            y,
            expert_tp_gather_meta,
        )
        return y

    @staticmethod
    def _canonicalize_deepep_metadata(
        weights: torch.Tensor,
        indices: torch.Tensor,
        num_experts: int,
        router_topk: int,
        capacity_factor: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        probs = torch.zeros(
            weights.shape[0], num_experts, dtype=torch.float32, device=weights.device
        )
        probs.scatter_(1, indices.to(torch.int64), weights.float())
        token_probs, token_indices = torch.topk(probs, router_topk, dim=-1)
        if capacity_factor is not None:
            token_indices = token_indices.masked_fill(token_probs == 0, -1)
        return token_probs.contiguous(), token_indices.contiguous()

    def _forward_deepep(
        self,
        x: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
        shape: torch.Size,
    ) -> torch.Tensor:
        """DeepEP fused alltoall dispatch + combine path."""
        manager = self._deepep
        # Match _DeepepManager.setup_metadata's dense-probs -> torch.topk
        # canonicalization, without allocating the bool routing_map whose
        # values are unused by setup_metadata.
        manager.token_probs, manager.token_indices = self._canonicalize_deepep_metadata(
            weights,
            indices,
            self.n_routed_experts,
            self.n_activated,
            manager.capacity_factor,
        )

        dispatched_x = manager.dispatch(x)
        routed_y = self._forward_fused_route_grouped_experts(
            dispatched_x,
            manager.dispatched_indices,
            manager.dispatched_probs,
        )
        y = manager.combine(routed_y)
        y = y + self.shared_experts(x)
        return y.type_as(x).view(shape)

    @classmethod
    def from_official(cls, official_moe, args, layer_id):
        """Test helper: build an instance and copy state from an official `MoE` instance."""
        from megatron.core.transformer.transformer_config import MLATransformerConfig as _C

        # Note: we deliberately omit `experimental_attention_variant="dsv4"` here because
        # MLATransformerConfig.__post_init__ would then require the full set of attention-side
        # fields (dsv4_o_groups, dsv4_o_lora_rank, dsv4_window_size, dsv4_compress_rope_theta).
        # DeepSeekV4MoE doesn't read that flag — it only consumes hidden_size, num_moe_experts,
        # moe_router_*, ffn_hidden_size, moe_shared_expert_intermediate_size, dsv4_swiglu_limit.
        cfg = _C(
            num_layers=1, hidden_size=args.dim, num_attention_heads=args.n_heads,
            num_moe_experts=args.n_routed_experts, moe_router_topk=args.n_activated_experts,
            moe_router_score_function=args.score_func, moe_router_topk_scaling_factor=args.route_scale,
            ffn_hidden_size=args.moe_inter_dim,
            moe_shared_expert_intermediate_size=args.moe_inter_dim * args.n_shared_experts,
            params_dtype=torch.bfloat16, add_bias_linear=False,
            dsv4_swiglu_limit=args.swiglu_limit,
        )
        cfg.vocab_size = args.vocab_size
        is_hash = layer_id < args.n_hash_layers
        ours = cls(config=cfg, layer_id=layer_id, is_hash_layer=is_hash, pg_collection=ProcessGroupCollection(tp=None, cp=None))
        ours.gate.weight.data.copy_(official_moe.gate.weight.data)
        # Hash gates expose tid2eid (int32 lookup); score gates expose bias (fp32). Cross-checking
        # keeps a stale layer_id from silently AttributeError-ing inside the wrong branch.
        if is_hash:
            assert hasattr(official_moe.gate, "tid2eid") and official_moe.gate.tid2eid is not None, \
                f"layer_id={layer_id} < n_hash_layers={args.n_hash_layers} but official gate has no tid2eid"
            ours.gate.tid2eid.data.copy_(official_moe.gate.tid2eid.data)
        else:
            assert hasattr(official_moe.gate, "bias") and official_moe.gate.bias is not None, \
                f"layer_id={layer_id} >= n_hash_layers={args.n_hash_layers} but official gate has no bias"
            ours.gate.bias.data.copy_(official_moe.gate.bias.data)
        for i in range(args.n_routed_experts):
            if official_moe.experts[i] is None:
                continue
            # ``ours.experts[i]`` is None on non-owner ranks when this
            # helper runs under an EP>1 mpu state (the V4 MoE constructor
            # falls back to ``mpu.get_expert_model_parallel_world_size()``
            # when the supplied pg_collection has no ``ep`` field). Skip
            # those positions so the helper still works as a per-rank
            # local-shard copier.
            if ours.experts[i] is None:
                continue
            for which in ("w1", "w2", "w3"):
                src = getattr(official_moe.experts[i], which)
                dst = getattr(ours.experts[i], which)
                dst.weight.data.copy_(src.weight.data)
                if hasattr(src.weight, "scale"):
                    dst.weight.scale.data.copy_(src.weight.scale.data)
        for which in ("w1", "w2", "w3"):
            src = getattr(official_moe.shared_experts, which)
            dst = getattr(ours.shared_experts, which)
            dst.weight.data.copy_(src.weight.data)
            if hasattr(src.weight, "scale"):
                dst.weight.scale.data.copy_(src.weight.scale.data)
        return ours


class DeepSeekV4TransformerLayer(MegatronModule):
    """DeepSeek V4 transformer layer: per-layer mHC mixers around attention + MoE.

    Mirrors `inference/model.py:Block`. The hidden state is held in V4's
    `[s, b, hc, d]` layout (with `hc = config.dsv4_hc_mult` copies); each call
    performs two `layer_pre -> norm -> submodule -> layer_post` rounds (attention
    then FFN). The MoE module operates on `[b, s, d]`, so we permute around it.
    """

    def __init__(
        self,
        config: MLATransformerConfig,
        layer_number: int,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ):
        super().__init__(config=config)
        self.layer_number = layer_number
        self.layer_id = layer_number - 1

        self.hc_util = DeepSeekV4HyperConnectionUtil(config)
        self.hc_mult = config.dsv4_hc_mult
        hc_dim = self.hc_mult * config.hidden_size
        mix_hc = (2 + self.hc_mult) * self.hc_mult

        # Per-layer mHC parameters. fp32 storage matches `Block.__init__` under
        # `set_dtype(torch.float32)`; mark `_keep_fp32` to stay fp32 across casts.
        self.hc_attn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim, dtype=torch.float32))
        self.hc_attn_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
        self.hc_attn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))
        self.hc_ffn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim, dtype=torch.float32))
        self.hc_ffn_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
        self.hc_ffn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))
        for p in (
            self.hc_attn_fn, self.hc_attn_base, self.hc_attn_scale,
            self.hc_ffn_fn, self.hc_ffn_base, self.hc_ffn_scale,
        ):
            p._keep_fp32 = True

        # `attn_norm` and `ffn_norm` in the official Block — Megatron names mirror
        # the standard transformer-layer naming so existing dist-ckpt machinery
        # treats them uniformly.
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.layernorm_epsilon)
        self.pre_mlp_layernorm = RMSNorm(config.hidden_size, eps=config.layernorm_epsilon)

        self.self_attention = DeepSeekV4Attention(
            config=config, layer_number=layer_number, pg_collection=pg_collection
        )

        is_hash = self.layer_id < (config.dsv4_n_hash_layers or 0)
        self.mlp = DeepSeekV4MoE(
            config=config,
            layer_id=self.layer_id,
            is_hash_layer=is_hash,
            pg_collection=pg_collection,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        packed_seq_params=None,
    ) -> torch.Tensor:
        """Run one V4 transformer layer.

        ``hidden_states`` is laid out as ``[s, b, hc, d]`` (V4-native). The two
        hyper-connection rounds reduce to ``[s, b, d]`` for the inner submodule
        and re-expand back to ``[s, b, hc, d]``. The MoE expects ``[b, s, d]``
        natively, so we permute around it.
        """
        valid_token_mask = _get_dsv4_packed_valid_token_mask(
            packed_seq_params,
            cp_size=self.self_attention.cp_size,
            cp_rank=self.self_attention.cp_group.rank()
            if self.self_attention.cp_group is not None
            else 0,
            seq_len=hidden_states.size(0),
            device=hidden_states.device,
            sequence_parallel=self.self_attention.sequence_parallel,
            tp_group=self.self_attention.tp_group,
        )
        hidden_states = _apply_dsv4_valid_token_mask(hidden_states, valid_token_mask)

        recompute_attn_round = (
            self.training
            and torch.is_grad_enabled()
            and hidden_states.requires_grad
            and getattr(self.config, "recompute_granularity", None) == "full"
            and os.environ.get("MEGATRON_DSV4_RECOMPUTE_ATTN_ROUND", "0") == "1"
        )

        def checkpointed_attention_round(hidden_states):
            residual = hidden_states
            x_sbd, post, comb = self.hc_util.layer_pre(
                hidden_states, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base
            )
            x_sbd = self.input_layernorm(x_sbd)
            x_sbd = _apply_dsv4_valid_token_mask(x_sbd, valid_token_mask)
            x_sbd = self.self_attention(x_sbd, packed_seq_params=packed_seq_params)
            x_sbd = _apply_dsv4_valid_token_mask(x_sbd, valid_token_mask)
            hidden_states = self.hc_util.layer_post(x_sbd, residual, post, comb)
            return _apply_dsv4_valid_token_mask(hidden_states, valid_token_mask)

        if recompute_attn_round:
            hidden_states = tensor_parallel.checkpoint(
                checkpointed_attention_round, False, hidden_states
            )
        else:
            hidden_states = checkpointed_attention_round(hidden_states)

        # ---- ffn round ----
        residual = hidden_states
        x_sbd, post, comb = self.hc_util.layer_pre(
            hidden_states, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base
        )
        x_sbd = self.pre_mlp_layernorm(x_sbd)
        x_sbd = _apply_dsv4_valid_token_mask(x_sbd, valid_token_mask)
        # MoE is [b, s, d]-native; permute, run, permute back.
        x_bsd = x_sbd.permute(1, 0, 2).contiguous()
        x_bsd = self.mlp(x_bsd, input_ids)
        x_sbd = x_bsd.permute(1, 0, 2).contiguous()
        x_sbd = _apply_dsv4_valid_token_mask(x_sbd, valid_token_mask)
        hidden_states = self.hc_util.layer_post(x_sbd, residual, post, comb)
        return _apply_dsv4_valid_token_mask(hidden_states, valid_token_mask)

    @classmethod
    def from_official(cls, official_block, args, layer_id: int):
        """Test helper: build an instance and copy state from an official `Block`.

        ``args`` is an instance of `inference/model.py:ModelArgs`.
        """
        from megatron.core.transformer.transformer_config import MLATransformerConfig as _C

        cfg = _C(
            num_layers=args.n_layers,
            hidden_size=args.dim,
            num_attention_heads=args.n_heads,
            q_lora_rank=args.q_lora_rank,
            kv_lora_rank=args.head_dim,
            qk_head_dim=args.head_dim,
            qk_pos_emb_head_dim=args.rope_head_dim,
            v_head_dim=args.head_dim,
            add_bias_linear=False,
            params_dtype=torch.bfloat16,
            experimental_attention_variant="dsv4",
            dsv4_o_groups=args.o_groups,
            dsv4_o_lora_rank=args.o_lora_rank,
            dsv4_window_size=args.window_size,
            dsv4_compress_ratios=list(args.compress_ratios),
            dsv4_compress_rope_theta=args.compress_rope_theta,
            dsv4_hc_mult=args.hc_mult,
            dsv4_hc_sinkhorn_iters=args.hc_sinkhorn_iters,
            dsv4_hc_eps=args.hc_eps,
            dsv4_swiglu_limit=args.swiglu_limit,
            dsv4_n_hash_layers=args.n_hash_layers,
            num_moe_experts=args.n_routed_experts,
            moe_router_topk=args.n_activated_experts,
            moe_router_score_function=args.score_func,
            moe_router_topk_scaling_factor=args.route_scale,
            ffn_hidden_size=args.moe_inter_dim,
            moe_shared_expert_intermediate_size=args.moe_inter_dim * args.n_shared_experts,
            rotary_scaling_factor=args.rope_factor,
            original_max_position_embeddings=args.original_seq_len,
            # Indexer config fields are required when compress_ratio==4 fires.
            dsa_indexer_n_heads=args.index_n_heads,
            dsa_indexer_head_dim=args.index_head_dim,
            dsa_indexer_topk=args.index_topk,
            # V4 norm_eps is 1e-6 (vs Megatron's 1e-5 default). Without this
            # override, every RMSNorm in the layer normalises with a larger eps
            # than the official, producing detectable parity drift.
            layernorm_epsilon=args.norm_eps,
        )
        cfg.vocab_size = args.vocab_size
        layer = cls(
            config=cfg,
            layer_number=layer_id + 1,
            pg_collection=ProcessGroupCollection(tp=None, cp=None),
        )

        # mHC parameters (fp32) ----------------------------------------------------
        for name in (
            "hc_attn_fn", "hc_attn_base", "hc_attn_scale",
            "hc_ffn_fn", "hc_ffn_base", "hc_ffn_scale",
        ):
            getattr(layer, name).data.copy_(getattr(official_block, name).data)

        # Norms --------------------------------------------------------------------
        layer.input_layernorm.weight.data.copy_(official_block.attn_norm.weight.data)
        layer.pre_mlp_layernorm.weight.data.copy_(official_block.ffn_norm.weight.data)

        # Attention ----------------------------------------------------------------
        a, oa = layer.self_attention, official_block.attn
        for which in ("wq_a", "wq_b", "wkv", "wo_a", "wo_b"):
            d = getattr(a, which)
            s = getattr(oa, which)
            d.weight.data.copy_(s.weight.data)
            if hasattr(s.weight, "scale"):
                d.weight.scale.data.copy_(s.weight.scale.data)
        a.q_norm.weight.data.copy_(oa.q_norm.weight.data)
        a.kv_norm.weight.data.copy_(oa.kv_norm.weight.data)
        a.attn_sink.data.copy_(oa.attn_sink.data)

        if oa.compress_ratio:
            # Compressor ---------------------------------------------------
            a.compressor.ape.data.copy_(oa.compressor.ape.data)
            a.compressor.wkv.weight.data.copy_(oa.compressor.wkv.weight.data)
            a.compressor.wgate.weight.data.copy_(oa.compressor.wgate.weight.data)
            a.compressor.norm.weight.data.copy_(oa.compressor.norm.weight.data)
            # Indexer (only present at compress_ratio==4 layers) ----------
            if oa.indexer is not None:
                a.indexer.linear_wq_b.weight.data.copy_(oa.indexer.wq_b.weight.data)
                if hasattr(oa.indexer.wq_b.weight, "scale"):
                    a.indexer.linear_wq_b.weight.scale.data.copy_(
                        oa.indexer.wq_b.weight.scale.data
                    )
                a.indexer.linear_weights_proj.weight.data.copy_(
                    oa.indexer.weights_proj.weight.data
                )
                a.indexer.compressor.ape.data.copy_(oa.indexer.compressor.ape.data)
                a.indexer.compressor.wkv.weight.data.copy_(
                    oa.indexer.compressor.wkv.weight.data
                )
                a.indexer.compressor.wgate.weight.data.copy_(
                    oa.indexer.compressor.wgate.weight.data
                )
                a.indexer.compressor.norm.weight.data.copy_(
                    oa.indexer.compressor.norm.weight.data
                )

        # MoE FFN ------------------------------------------------------------------
        # Reuse DeepSeekV4MoE.from_official (it builds + copies state); replace
        # the freshly-allocated `layer.mlp` so we don't double-init experts.
        layer.mlp = DeepSeekV4MoE.from_official(
            official_block.ffn, args, layer_id
        ).cuda()
        return layer


class DeepSeekV4TransformerBlock(MegatronModule):
    """DeepSeek V4 decoder: ``block_expand`` -> N V4 layers -> ``hc_head`` -> norm.

    Mirrors `inference/model.py:Transformer.forward`:

    .. code-block:: python

        h = self.embed(input_ids)              # [b, s, d]
        h = h.unsqueeze(2).repeat(1, 1, hc, 1) # [b, s, hc, d]  (block_expand)
        for layer in self.layers:
            h = layer(h, start_pos, input_ids) # [b, s, hc, d]
        logits = self.head(h, hc_head_fn, hc_head_scale, hc_head_base, self.norm)
        # head() = hc_head -> norm -> get_logits = F.linear(x[:, -1].float(), W)

    Megatron's standard ``TransformerBlock`` doesn't do ``block_expand`` or hold
    the decoder-level mHC head; it also operates on ``[s, b, d]`` inputs. We
    accept ``[s, b, d]`` (Megatron convention), expand to ``[s, b, hc, d]`` for
    the inner V4 stack, and reduce back to ``[s, b, d]`` after ``block_head`` +
    ``final_layernorm``. Embedding + output_layer are owned by the GPT-style
    wrapper; the ``forward_full`` helper below stashes
    a copy locally so this test class can exercise the end-to-end forward.
    """

    def __init__(
        self,
        config: MLATransformerConfig,
        pg_collection: Optional[ProcessGroupCollection] = None,
        vp_stage: Optional[int] = None,
    ):
        super().__init__(config=config)
        if pg_collection is None:
            # Request ``expt_tp`` so standalone construction (non-Bridge) shards
            # routed and shared experts on the expert-tensor-parallel group when
            # ``expert_tensor_parallel_size != tensor_model_parallel_size``.
            # ``DeepSeekV4MoE`` also consults ``parallel_state`` if ``expt_tp``
            # is missing from a custom process-group collection.
            # ``pp`` is requested so ``sharded_state_dict`` can pass a real PP
            # rank to ``get_transformer_layer_offset`` for PP > 1 layouts; on
            # single-PP-stage runs the offset returned is 0.
            # ``expt_dp`` is needed by ``DeepSeekV4MoE.sharded_state_dict`` to
            # rewrite the third component of expert ``replica_id``; without it
            # every rank in the DP-replication-of-an-expert group collapses to
            # ``replica_id[2]=0`` and the dist-checkpointing validator rejects
            # the template (manifests at e.g. ``EP=2 + DP_replication=2``).
            pg_collection = ProcessGroupCollection.use_mpu_process_groups(
                required_pgs=["tp", "expt_tp", "pp", "expt_dp"]
            )
        self.pg_collection = pg_collection
        # ``vp_stage`` mirrors standard ``TransformerBlock``: needed for
        # ``get_transformer_layer_offset`` under virtual-pipeline schedules.
        # None is the correct value for non-VP layouts.
        self.vp_stage = vp_stage
        self.hc_util = DeepSeekV4HyperConnectionUtil(config)
        self.hc_head_params = HCHeadParams(config)

        pp_group = getattr(self.pg_collection, "pp", None)
        pp_rank = get_pg_rank(pp_group)
        offset = get_transformer_layer_offset(config, vp_stage, pp_rank)
        num_layers_to_build = get_num_layers_to_build(config, vp_stage, pp_rank)
        self.layers = nn.ModuleList(
            [
                DeepSeekV4TransformerLayer(
                    config=config,
                    layer_number=offset + i + 1,
                    pg_collection=pg_collection,
                )
                for i in range(num_layers_to_build)
            ]
        )
        self.final_layernorm = RMSNorm(config.hidden_size, eps=config.layernorm_epsilon)
        # Embedding + output_layer are owned by the GPTModel wrapper.
        # ``forward_full`` (test-only) stashes them via ``from_official``.
        self._embed_weight: Optional[torch.Tensor] = None
        self._head_weight: Optional[torch.Tensor] = None

    def sharded_state_dict(
        self,
        prefix: str = "",
        sharded_offsets: tuple = (),
        metadata: Optional[Dict] = None,
    ) -> ShardedStateDict:
        """Sharded state dict for the DSV4 decoder.

        Mirrors standard ``TransformerBlock.sharded_state_dict``:

        * ``non_homogeneous_layers=False`` (default): emit one shared
          ``layers.`` prefix per parameter, with a PP-style sharded offset
          ``(axis, global_layer_offset, num_layers)`` distinguishing layers.
          Compact on-disk format; the canonical layout standard Megatron
          MoE+MLA models (Kimi, Qwen3, DeepSeek V3 via the standard
          ``MoELayer``) use.
        * ``non_homogeneous_layers=True``: emit per-layer keys
          ``layers.{global_layer_offset}.`` — used when layer-level tensor
          structure differs between layers (e.g., MoE-freq-mixed configs).
          Auto-enabled when DSV4 has hash-routed layers
          (``dsv4_n_hash_layers > 0``) because ``DeepSeekV4Gate`` stores a
          different parameter set (``tid2eid`` instead of ``bias``) on
          hash-vs-expert layers, breaking the shared-prefix flow.

        Bridges the ``ModuleList``-recursion gap (default
        ``MegatronModule.sharded_state_dict`` falls through to
        ``sharded_state_dict_default``'s flat ``state_dict`` path on
        ``self.layers``, bypassing per-layer ``sharded_state_dict`` and so
        TP-sharding of e.g. ``attn_sink``). PP-aware via
        ``get_transformer_layer_offset`` so PP > 1 layouts get the right
        per-stage local-vs-global layer mapping; the call returns 0 at
        ``pipeline_model_parallel_size == 1``.
        """
        assert not sharded_offsets, "DeepSeekV4TransformerBlock expects no sharded_offsets"

        non_homogeneous_layers = (
            metadata is not None and metadata.get("non_homogeneous_layers", False)
        )
        # DSV4-specific: hash-routed layers have a different gate parameter
        # set than expert-routed layers, so the shared-prefix path can't
        # align keys cross-layer. Force per-layer prefixes whenever any hash
        # layer exists in this block.
        if (getattr(self.config, "dsv4_n_hash_layers", 0) or 0) > 0:
            non_homogeneous_layers = True
        # DSV4 also lets ``dsv4_compress_ratios`` vary per layer — different
        # ratios produce different attention-side parameter sets
        # (no compressor at ratio=0, compressor at ratio>0, compressor +
        # indexer at ratio==4). Mixed ratios → per-layer prefix required.
        _compress_ratios = getattr(self.config, "dsv4_compress_ratios", None)
        if _compress_ratios is not None and len(set(_compress_ratios)) > 1:
            non_homogeneous_layers = True
        # ``singleton_local_shards=True`` (standard convention) implies
        # per-layer prefix as well (see standard ``TransformerBlock``).
        if (metadata or {}).get("singleton_local_shards", False):
            non_homogeneous_layers = True

        # Per-PP-stage local->global layer mapping. Match standard
        # ``TransformerBlock`` rank resolution.
        pp_group = getattr(self.pg_collection, "pp", None)
        pp_rank = get_pg_rank(pp_group)
        offset = get_transformer_layer_offset(self.config, self.vp_stage, pp_rank)

        sharded_state_dict: ShardedStateDict = {}
        layer_prefix = f"{prefix}layers."
        num_layers = self.config.num_layers
        for layer in self.layers:
            # ``layer.layer_number`` is the 1-based global index;
            # ``layer.layer_number - 1 - offset`` is the local (in-stage)
            # ModuleList index when PP > 1. Standard
            # ``TransformerBlock.sharded_state_dict`` uses the same formula.
            global_layer_offset = layer.layer_number - 1
            state_dict_prefix = f"{layer_prefix}{global_layer_offset - offset}."
            if non_homogeneous_layers:
                sharded_prefix = f"{layer_prefix}{global_layer_offset}."
                layer_sharded_offsets: tuple = ()
            else:
                sharded_prefix = layer_prefix
                layer_sharded_offsets = (
                    (0, global_layer_offset, num_layers),
                )
            layer_sd = layer.sharded_state_dict(state_dict_prefix, layer_sharded_offsets, metadata)
            if sharded_prefix != state_dict_prefix:
                replace_prefix_for_sharding(layer_sd, state_dict_prefix, sharded_prefix)
            sharded_state_dict.update(layer_sd)

        # Non-layer children (hc_head_params, final_layernorm, …) — recurse
        # via the standard default. ``hc_util`` is a plain Python utility
        # (not an nn.Module), so ``named_children`` skips it.
        for name, module in self.named_children():
            if module is self.layers:
                continue
            sharded_state_dict.update(
                sharded_state_dict_default(
                    module,
                    f"{prefix}{name}.",
                    sharded_offsets,
                    metadata,
                    tp_group=self.pg_collection.tp,
                )
            )
        return sharded_state_dict

    def _maybe_scatter_input_ids_for_sequence_parallel(
        self, input_ids: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """Keep token ids aligned with sequence-parallel hidden states."""
        local_tokens = hidden_states.shape[0] * hidden_states.shape[1]
        if input_ids.numel() == local_tokens:
            return input_ids

        if (
            self.config.sequence_parallel
            and input_ids.dim() == 2
            and input_ids.shape[0] == hidden_states.shape[1]
            and input_ids.shape[1] % self.pg_collection.tp.size() == 0
        ):
            input_ids = (
                scatter_to_sequence_parallel_region(
                    input_ids.transpose(0, 1).contiguous(), group=self.pg_collection.tp
                )
                .transpose(0, 1)
                .contiguous()
            )
            if input_ids.numel() == local_tokens:
                return input_ids

        raise ValueError(
            "DeepSeekV4TransformerBlock input_ids are not aligned with hidden_states: "
            f"input_ids shape={tuple(input_ids.shape)}, hidden_states shape={tuple(hidden_states.shape)}."
        )

    def set_input_tensor(self, input_tensor: torch.Tensor) -> None:
        """PP-stage input wiring shim — DSV4 runs in a single PP stage so this
        is effectively a no-op. Stored on ``self.input_tensor`` to match the
        contract Megatron's standard ``TransformerBlock`` exposes."""
        self.input_tensor = input_tensor

    def _checkpointed_forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        packed_seq_params: Optional[object],
    ) -> torch.Tensor:
        """Forward DSV4 decoder layers with Megatron full activation recompute."""

        def custom(start: int, end: int):
            def custom_forward(hidden_states):
                for index in range(start, end):
                    hidden_states = self.layers[index](
                        hidden_states,
                        input_ids=input_ids,
                        packed_seq_params=packed_seq_params,
                    )
                return hidden_states

            return custom_forward

        def checkpoint_handler(forward_func, hidden_states):
            return tensor_parallel.checkpoint(
                forward_func,
                self.config.distribute_saved_activations,
                hidden_states,
            )

        num_layers = len(self.layers)
        if self.config.recompute_method == "uniform":
            layer_idx = 0
            while layer_idx < num_layers:
                chunk_end = min(layer_idx + self.config.recompute_num_layers, num_layers)
                hidden_states = checkpoint_handler(custom(layer_idx, chunk_end), hidden_states)
                layer_idx += self.config.recompute_num_layers
        elif self.config.recompute_method == "block":
            for layer_idx in range(num_layers):
                if layer_idx < self.config.recompute_num_layers:
                    hidden_states = checkpoint_handler(
                        custom(layer_idx, layer_idx + 1),
                        hidden_states,
                    )
                else:
                    hidden_states = custom(layer_idx, layer_idx + 1)(hidden_states)
        else:
            raise ValueError("Invalid activation recompute method.")

        return hidden_states

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        input_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Run the full DSV4 decoder stack.

        ``hidden_states`` enters as ``[s, b, d]`` (Megatron convention); we
        ``block_expand`` to ``[s, b, hc, d]``, run through the layer list, then
        ``block_head`` -> ``final_layernorm`` to reduce back to ``[s, b, d]``.

        Signature compatibility note: the standard
        ``GPTModel.forward`` calls ``self.decoder(hidden_states,
        attention_mask=..., rotary_pos_emb=..., inference_context=..., ...)`` and
        does not pass ``input_ids``. The V4 block has no use for
        ``attention_mask`` or rotary cos/sin (each ``DeepSeekV4Attention``
        derives its own RoPE/sliding-window state from ``input_ids``), so we
        accept them and silently ignore them. ``input_ids`` is required by the
        per-layer hash-gate path in ``DeepSeekV4Gate``; callers that drive the
        block via ``GPTModel`` must thread it through ``extra_block_kwargs``
        (or stash it on ``self._input_ids_cache`` before forward). If neither
        is provided we fall back to ``self._input_ids_cache``; if that's also
        missing we raise a clear error.
        """
        if input_ids is None:
            input_ids = getattr(self, "_input_ids_cache", None)
        if input_ids is None:
            raise ValueError(
                "DeepSeekV4TransformerBlock.forward requires input_ids: pass it "
                "via extra_block_kwargs={'input_ids': ...} when going through "
                "GPTModel.forward, or set ``self._input_ids_cache`` before forward."
            )
        packed_seq_params = kwargs.get("packed_seq_params")
        del attention_mask, kwargs  # explicit: other standard block kwargs are not used by V4
        input_ids = self._maybe_scatter_input_ids_for_sequence_parallel(input_ids, hidden_states)
        # block_expand operates on [b, s, d] -> [b, s, hc, d]; permute around it.
        h_bsd = hidden_states.permute(1, 0, 2).contiguous()
        h_bshd = self.hc_util.block_expand(h_bsd)  # [b, s, hc, d]
        h = h_bshd.permute(1, 0, 2, 3).contiguous()  # -> [s, b, hc, d]

        if self.config.recompute_granularity == "full" and self.training:
            h = make_viewless_tensor(inp=h, requires_grad=True, keep_graph=True)
            if not h.requires_grad:
                # PEFT freezes the prefix before the first adapter. Reentrant
                # checkpointing still needs a grad-carrying input to replay the
                # checkpoint body and accumulate adapter gradients.
                h.requires_grad_(True)
            h = self._checkpointed_forward(h, input_ids, packed_seq_params)
        else:
            for layer in self.layers:
                h = layer(h, input_ids=input_ids, packed_seq_params=packed_seq_params)

        # block_head expects [b, s, hc, d] and returns [b, s, d]; permute around it.
        h_bshd = h.permute(1, 0, 2, 3).contiguous()
        h_bsd = self.hc_util.block_head(
            h_bshd,
            self.hc_head_params.hc_head_fn,
            self.hc_head_params.hc_head_scale,
            self.hc_head_params.hc_head_base,
        )
        h_sbd = h_bsd.permute(1, 0, 2).contiguous()
        h_sbd = self.final_layernorm(h_sbd)
        return h_sbd

    def forward_full(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Test helper that reproduces ``Transformer.forward`` end-to-end.

        Mirrors ``ParallelHead.forward`` -> ``get_logits``: ``F.linear(x[:, -1]
        .float(), self._head_weight)`` on the last sequence position.
        """
        assert self._embed_weight is not None and self._head_weight is not None, (
            "forward_full requires _embed_weight/_head_weight to be set "
            "(use DeepSeekV4TransformerBlock.from_official)."
        )
        emb = F.embedding(input_ids, self._embed_weight)  # [b, s, d]
        h_sbd = emb.permute(1, 0, 2).contiguous()
        h_sbd = self.forward(h_sbd, input_ids=input_ids)
        # ParallelHead.get_logits: F.linear(x[:, -1].float(), head.weight).
        h_bsd = h_sbd.permute(1, 0, 2).contiguous()
        return F.linear(h_bsd[:, -1].float(), self._head_weight)

    @classmethod
    def from_official(cls, official_t, args):
        """Build a ``DeepSeekV4TransformerBlock`` and copy state from an official Transformer.

        ``official_t`` is an instance of ``inference/model.py:Transformer``;
        ``args`` is its corresponding ``ModelArgs``. Each layer is built via
        ``DeepSeekV4TransformerLayer.from_official`` (which already handles
        per-layer mHC + attention + MoE state) and then loaded into our
        ``self.layers[i]`` via ``load_state_dict``. We additionally copy the
        decoder-level mHC head, ``final_layernorm``, and stash the embedding
        and head weights so ``forward_full`` can exercise the end-to-end path.
        """
        from megatron.core.transformer.transformer_config import MLATransformerConfig as _C

        cfg = _C(
            num_layers=args.n_layers,
            hidden_size=args.dim,
            num_attention_heads=args.n_heads,
            q_lora_rank=args.q_lora_rank,
            kv_lora_rank=args.head_dim,
            qk_head_dim=args.head_dim,
            qk_pos_emb_head_dim=args.rope_head_dim,
            v_head_dim=args.head_dim,
            add_bias_linear=False,
            params_dtype=torch.bfloat16,
            experimental_attention_variant="dsv4",
            dsv4_o_groups=args.o_groups,
            dsv4_o_lora_rank=args.o_lora_rank,
            dsv4_window_size=args.window_size,
            dsv4_compress_ratios=list(args.compress_ratios),
            dsv4_compress_rope_theta=args.compress_rope_theta,
            dsv4_hc_mult=args.hc_mult,
            dsv4_hc_sinkhorn_iters=args.hc_sinkhorn_iters,
            dsv4_hc_eps=args.hc_eps,
            dsv4_swiglu_limit=args.swiglu_limit,
            dsv4_n_hash_layers=args.n_hash_layers,
            num_moe_experts=args.n_routed_experts,
            moe_router_topk=args.n_activated_experts,
            moe_router_score_function=args.score_func,
            moe_router_topk_scaling_factor=args.route_scale,
            ffn_hidden_size=args.moe_inter_dim,
            moe_shared_expert_intermediate_size=args.moe_inter_dim * args.n_shared_experts,
            rotary_scaling_factor=args.rope_factor,
            original_max_position_embeddings=args.original_seq_len,
            # Indexer config (required when any compress_ratio==4 layer fires).
            dsa_indexer_n_heads=args.index_n_heads,
            dsa_indexer_head_dim=args.index_head_dim,
            dsa_indexer_topk=args.index_topk,
            # V4 norm_eps is 1e-6 (vs Megatron's 1e-5 default).
            layernorm_epsilon=args.norm_eps,
        )
        cfg.vocab_size = args.vocab_size

        ours = cls(config=cfg, pg_collection=ProcessGroupCollection(tp=None, cp=None))

        # Decoder-level mHC head (fp32) ------------------------------------------
        ours.hc_head_params.hc_head_fn.data.copy_(official_t.hc_head_fn.data)
        ours.hc_head_params.hc_head_base.data.copy_(official_t.hc_head_base.data)
        ours.hc_head_params.hc_head_scale.data.copy_(official_t.hc_head_scale.data)

        # Final layernorm --------------------------------------------------------
        ours.final_layernorm.weight.data.copy_(official_t.norm.weight.data)

        # Per-layer state (delegate to the layer's from_official). DSV4Linear
        # registers its quantization scale as an nn.Parameter (also aliased onto
        # weight.scale), so state_dict()/load_state_dict() round-trips it via
        # the registered name; no separate scale walk needed.
        for i in range(cfg.num_layers):
            from_layer = DeepSeekV4TransformerLayer.from_official(
                official_t.layers[i], args, i
            ).cuda()
            ours.layers[i].load_state_dict(from_layer.state_dict())

        # Embedding + head weights are owned by the wrapper in production; for
        # the parity test we stash a clone so forward_full reproduces the
        # official Transformer.forward end-to-end.
        ours._embed_weight = official_t.embed.weight.data.clone()
        ours._head_weight = official_t.head.weight.data.clone()
        return ours


_DSV4_LM_HEAD_BI_ATTR = "_dsv4_lm_head_batch_invariant_wrapped"


def maybe_wrap_dsv4_lm_head_batch_invariant(model: nn.Module) -> bool:
    """Wrap ``model.output_layer.forward`` in ``set_batch_invariant_mode(True)``.

    Mirrors sglang's ``SGLANG_DSV4_CANONICAL_LM_HEAD`` knob. Default off
    (the BI matmul is slower than cuBLAS); enable with
    ``MEGATRON_DSV4_CANONICAL_LM_HEAD=1`` when bit-exact per-token logits
    across batch sizes are required (e.g., on-policy RL train/rollout match).

    Returns True if the wrap was applied, False otherwise. Idempotent.
    """
    if os.environ.get("MEGATRON_DSV4_CANONICAL_LM_HEAD", "0") != "1":
        return False
    output_layer = getattr(model, "output_layer", None)
    if output_layer is None or not hasattr(output_layer, "forward"):
        return False
    if getattr(output_layer.forward, _DSV4_LM_HEAD_BI_ATTR, False):
        return False
    from megatron.core.transformer.custom_layers.batch_invariant_kernels import (
        set_batch_invariant_mode,
    )

    orig = output_layer.forward

    def _bi_forward(*args, **kwargs):
        with set_batch_invariant_mode(True):
            return orig(*args, **kwargs)

    setattr(_bi_forward, _DSV4_LM_HEAD_BI_ATTR, True)
    output_layer.forward = _bi_forward
    return True
