# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Native FP4/FP8/bf16/fp32 frozen-base linear primitive for DeepSeek V4.

Mirrors the official inference `Linear` class
(the official DeepSeek-V4 ``inference/model.py``)
so checkpoint weight + scale layouts are identical. Forward dispatches to
the same kernels (`act_quant` + `fp8_gemm` for FP8, `act_quant` + `fp4_gemm`
for FP4, `F.linear` for bf16/fp32).

Backward: the quantized forward kernels (`fp8_gemm`,
`fp4_gemm`) are forward-only. To let gradients flow through a *frozen*
quantized base — the train↔rollout contract — `_quantized_linear`
is wrapped in `_DSV4FrozenQuantizedLinearFn`, a custom autograd Function
that:

* runs the same forward kernel,
* on backward, dequantizes the (frozen) weight to bf16
  (E4M3FN→bf16 + per-block-128 E8M0 scale broadcast for FP8;
  E2M1 unpack + per-block-32 E8M0 scale broadcast for FP4)
  and computes ``grad_input = grad_output @ W_bf16``,
* returns ``None`` for ``grad_weight`` and ``grad_scale`` since the base
  is frozen — saving the cost of an inverse FP8/FP4 quant kernel.

This is the gradient-correct backward path for OFT-on-frozen-base.
"""

import os
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from megatron.core.tensor_parallel.layers import set_tensor_model_parallel_attributes
from megatron.core.tensor_parallel.mappings import (
    reduce_from_tensor_model_parallel_region,
    reduce_scatter_to_sequence_parallel_region,
)
from megatron.core.transformer.experimental_attention_variant import dsv4_kernels
from megatron.core.transformer.utils import make_sharded_tensors_for_checkpoint

# Block sizes follow the official `inference/model.py` globals.
_FP8_BLOCK_SIZE = 128
_FP4_BLOCK_SIZE = 32
# Scale storage / format defaults match `dtype="fp8" scale_fmt="ue8m0" scale_dtype="fp8"`.
_DEFAULT_SCALE_FMT = "ue8m0"
_DEFAULT_SCALE_DTYPE = torch.float8_e8m0fnu


class _DefaultDtypeBF16:
    """Context manager: temporarily flip ``torch.get_default_dtype()`` to bf16.

    The official V4 ``fp8_gemm`` / ``fp4_gemm`` allocate their output buffer as
    ``a.new_empty(..., dtype=torch.get_default_dtype())`` and the tilelang
    kernel asserts ``c.dtype == bfloat16``. SGLang's inference path sets
    default dtype to bf16 globally, but Megatron training keeps it at fp32 —
    so on the train path we'd hand the kernel an fp32 ``c`` and crash. Flip
    just for the GEMM call (act_quant + matmul) and restore.
    """

    def __enter__(self):
        self._prev = torch.get_default_dtype()
        if self._prev is not torch.bfloat16:
            torch.set_default_dtype(torch.bfloat16)
        return self

    def __exit__(self, *_):
        if self._prev is not torch.bfloat16:
            torch.set_default_dtype(self._prev)


def _quantized_linear(
    x: torch.Tensor,
    weight: nn.Parameter,
    scale: Optional[nn.Parameter],
) -> torch.Tensor:
    if weight.dtype == torch.float4_e2m1fn_x2:
        # Activation block stays at 128 (FP8 quant) even when weight is FP4 block-32 —
        # this matches the official inference `linear()` semantics in
        # the official DeepSeek-V4 inference `model.py` (lines 108-119).
        with _DefaultDtypeBF16():
            x_q, s_x = dsv4_kernels.act_quant(
                x, _FP8_BLOCK_SIZE, _DEFAULT_SCALE_FMT, _DEFAULT_SCALE_DTYPE
            )
            return dsv4_kernels.fp4_gemm(x_q, s_x, weight, scale, _DEFAULT_SCALE_DTYPE)
    if weight.dtype == torch.float8_e4m3fn:
        with _DefaultDtypeBF16():
            if os.environ.get("MEGATRON_DSV4_DET_FP8_GEMM", "0") == "1":
                x_q, s_x = dsv4_kernels.det_act_quant(
                    x, _FP8_BLOCK_SIZE, _DEFAULT_SCALE_FMT, _DEFAULT_SCALE_DTYPE
                )
                return dsv4_kernels.det_fp8_gemm(
                    x_q, s_x, weight, scale, _DEFAULT_SCALE_DTYPE
                )
            x_q, s_x = dsv4_kernels.act_quant(
                x, _FP8_BLOCK_SIZE, _DEFAULT_SCALE_FMT, _DEFAULT_SCALE_DTYPE
            )
            return dsv4_kernels.fp8_gemm(x_q, s_x, weight, scale, _DEFAULT_SCALE_DTYPE)
    return F.linear(x, weight)


def _dequantize_fp8_weight_bf16(
    weight: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    """Dequantize an FP8-E4M3FN weight tensor to bf16 by direct cast +
    per-block-128 E8M0 scale broadcast. Used on the backward pass when
    the base is frozen, to compute ``grad_input = grad_output @ W_bf16``.

    Block layout matches ``fp8_gemm_kernel``: scale shape is
    ``(ceildiv(out, 128), ceildiv(in, 128))`` where each scale[i,j]
    governs the (128, 128) tile of weight at [i*128:(i+1)*128, j*128:(j+1)*128].
    """
    assert weight.dtype == torch.float8_e4m3fn, weight.dtype
    out_features, in_features = weight.shape
    # E4M3FN -> bf16 cast; preserves NaN/Inf at saturate boundary.
    w_bf16 = weight.to(torch.bfloat16)
    # E8M0 scale -> bf16 (loss-free since E8M0 is a power of 2 ≤ bf16 range).
    s_bf16 = scale.to(torch.bfloat16)
    # Repeat block scale to weight shape, then truncate any tail.
    s_expanded = s_bf16.repeat_interleave(_FP8_BLOCK_SIZE, dim=0).repeat_interleave(
        _FP8_BLOCK_SIZE, dim=1
    )
    s_expanded = s_expanded[:out_features, :in_features]
    return w_bf16 * s_expanded


# E2M1 (1 sign + 2 exp + 1 mantissa) lookup table: 16 codes -> bf16 value.
# Matches the IEEE-style decoding used by torch.float4_e2m1fn_x2 storage:
# bit 3 = sign, bits 2-1 = exponent, bit 0 = mantissa. Subnormal exponent
# = 0 means value = mantissa/2 * 2^(1-bias) with bias=1; normal values use
# 2^(exp-bias) * (1 + mantissa/2). The 16-entry table is a constant.
_E2M1_LUT_VALUES = (
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
)


def _dequantize_fp4_weight_bf16(
    weight: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    """Dequantize an FP4-E2M1 packed weight tensor to bf16.

    ``weight`` storage is ``torch.float4_e2m1fn_x2`` — two FP4 codes per
    byte, low nibble first. Logical shape is ``(out, in)`` while storage
    shape is ``(out, in // 2)``. Scale is per-block-32 along the in
    dimension, dtype E8M0, shape ``(out, ceildiv(in, 32))``.
    """
    assert weight.dtype == torch.float4_e2m1fn_x2, weight.dtype
    out_features, packed_in = weight.shape
    in_features = packed_in * 2
    # View as raw bytes (uint8) — 1 byte per 2 fp4 codes.
    packed = weight.view(torch.uint8)
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    # Interleave low/high to recover the unpacked-along-in layout.
    unpacked = torch.stack((low, high), dim=-1).reshape(out_features, in_features)
    lut = torch.tensor(
        _E2M1_LUT_VALUES, dtype=torch.bfloat16, device=weight.device
    )
    w_bf16 = lut[unpacked.long()]
    s_bf16 = scale.to(torch.bfloat16)
    s_expanded = s_bf16.repeat_interleave(_FP4_BLOCK_SIZE, dim=1)
    s_expanded = s_expanded[:, :in_features]
    return w_bf16 * s_expanded


class _DSV4FrozenQuantizedLinearFn(torch.autograd.Function):
    """Autograd Function for ``_quantized_linear`` with a *frozen* weight.

    Forward replays ``_quantized_linear`` (no extra cost). Backward
    dequantizes the weight to bf16 on the fly and computes
    ``grad_input = grad_output @ W_bf16``. Returns ``None`` for the
    weight + scale grads — the train↔rollout contract requires the quantized
    base to stay frozen, so we pay no inverse-quant cost.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(weight, scale)
        ctx.x_dtype = x.dtype
        return _quantized_linear(x, weight, scale)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        weight, scale = ctx.saved_tensors
        grad_input = None
        if ctx.needs_input_grad[0]:
            if weight.dtype == torch.float4_e2m1fn_x2:
                w_bf16 = _dequantize_fp4_weight_bf16(weight, scale)
            elif weight.dtype == torch.float8_e4m3fn:
                w_bf16 = _dequantize_fp8_weight_bf16(weight, scale)
            else:
                w_bf16 = weight
            # weight is [out, in]; flatten leading dims of grad_output to do
            # one matmul, then restore.
            go = grad_output
            orig_shape = go.shape
            go_2d = go.reshape(-1, orig_shape[-1]).to(w_bf16.dtype)
            gi_2d = go_2d @ w_bf16
            grad_input = gi_2d.reshape(*orig_shape[:-1], gi_2d.shape[-1]).to(ctx.x_dtype)
        return grad_input, None, None


def _quantized_linear_with_grad(
    x: torch.Tensor,
    weight: nn.Parameter,
    scale: Optional[nn.Parameter],
) -> torch.Tensor:
    """Autograd-aware dispatch: routes through the custom Function only
    when the underlying gemm is forward-only (FP8 / FP4) and an upstream
    tensor needs grad. The bf16/fp32 path stays on stock ``F.linear``,
    which is autograd-aware natively.
    """
    if (
        weight.dtype in (torch.float8_e4m3fn, torch.float4_e2m1fn_x2)
        and torch.is_grad_enabled()
        and x.requires_grad
    ):
        return _DSV4FrozenQuantizedLinearFn.apply(x, weight, scale)
    return _quantized_linear(x, weight, scale)


class DSV4Linear(nn.Module):
    """Mirror of `inference.model.Linear`: holds weight + optional scale Parameter.

    Quant-vs-not is determined by `weight.dtype`, not by `scale is None`. The
    `weight.scale` attribute alias mirrors the official inference layout so the
    native-quant safetensors loader can write to either name. That alias
    is a plain Python attribute on the Parameter object: it survives `.cuda()`,
    `.to()`, and ordinary `load_state_dict()`, but does NOT survive
    `load_state_dict(assign=True)` because that rebinds `self.weight` to a new
    Parameter. Forward consults `self.scale` (registered Parameter) so it is
    immune to that hazard.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        dtype = dtype or torch.bfloat16

        if dtype == torch.float4_e2m1fn_x2:
            self.weight = nn.Parameter(
                torch.empty(out_features, in_features // 2, dtype=torch.float4_e2m1fn_x2)
            )
            scale = nn.Parameter(
                torch.empty(out_features, in_features // _FP4_BLOCK_SIZE, dtype=_DEFAULT_SCALE_DTYPE)
            )
        elif dtype == torch.float8_e4m3fn:
            self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype))
            scale = nn.Parameter(
                torch.empty(
                    (out_features + _FP8_BLOCK_SIZE - 1) // _FP8_BLOCK_SIZE,
                    (in_features + _FP8_BLOCK_SIZE - 1) // _FP8_BLOCK_SIZE,
                    dtype=_DEFAULT_SCALE_DTYPE,
                )
            )
        else:
            self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype))
            scale = None

        if scale is not None:
            self.weight.scale = scale
            self.scale = scale
        else:
            self.register_parameter("scale", None)

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = _quantized_linear_with_grad(x, self.weight, self.scale)
        if self.bias is not None:
            y = y + self.bias
        return y

    def apply_input_rotation(self, x: torch.Tensor) -> torch.Tensor:
        """No-op identity for non-OFT-wrapped linears.

        V4 attention reads ``self.wo_a.weight.view(...)`` directly into
        an einsum (deepseek_v4.py:1048), bypassing ``self.wo_a.forward``.
        Without OFT, this is a benign optimisation. With OFT, the
        rotation would be silently skipped because it lives inside
        ``DSV4OFTLinear.forward``.

        The fix: V4 attention calls ``out = self.wo_a.apply_input_rotation(out)``
        before reading ``self.wo_a.weight``. The default impl on
        ``DSV4Linear`` is identity (no rotation when not OFT-wrapped);
        ``DSV4OFTLinear`` overrides this to apply its rotation. This
        keeps the einsum-bypass optimisation while preserving the OFT
        rotation under training.
        """
        return x


def _tp_world(tp_group: Optional["dist.ProcessGroup"]) -> int:
    if tp_group is None:
        return 1
    return tp_group.size()


class DSV4ColumnParallelLinear(DSV4Linear):
    """Shards `out_features` across TP. No reduction on output."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        dtype: Optional[torch.dtype] = None,
        tp_group: Optional["dist.ProcessGroup"] = None,
    ):
        world = _tp_world(tp_group)
        assert out_features % world == 0, (
            f"DSV4ColumnParallelLinear: out_features={out_features} "
            f"not divisible by tp_world={world}"
        )
        super().__init__(in_features, out_features // world, bias, dtype)
        self.tp_group = tp_group
        set_tensor_model_parallel_attributes(self.weight, True, 0, 1)
        if self.scale is not None:
            set_tensor_model_parallel_attributes(self.scale, True, 0, 1)
        if self.bias is not None:
            set_tensor_model_parallel_attributes(self.bias, True, 0, 1)

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        """TP shard along axis 0 for weight / scale / bias.

        Standard Megatron ``ColumnParallelLinear.sharded_state_dict`` only
        sharded ``weight`` + ``bias`` along axis 0; DSV4 additionally carries
        an FP8/FP4 block-quant ``scale`` (also TP-sharded along axis 0 — see
        ``__init__``). Without this override the default
        ``sharded_state_dict_default`` path falls back to flat
        ``state_dict()`` + ``make_sharded_tensors_for_checkpoint`` with no
        axis map, which emits the local per-rank shape as the *global*
        shape and breaks cross-TP checkpoint loads.
        """
        state_dict = self.state_dict(prefix="", keep_vars=True)
        axis_map = {"weight": 0}
        if self.scale is not None:
            axis_map["scale"] = 0
        if self.bias is not None:
            axis_map["bias"] = 0
        return make_sharded_tensors_for_checkpoint(
            state_dict,
            prefix,
            axis_map,
            sharded_offsets,
            tp_group=self.tp_group,
            dp_cp_group=metadata["dp_cp_group"],
        )


class DSV4RowParallelLinear(DSV4Linear):
    """Shards `in_features` across TP and reduces output across TP.

    Output reduction follows the standard Megatron ``RowParallelLinear`` pattern:

    * ``sequence_parallel=True``: ``reduce_scatter_to_sequence_parallel_region``
      replaces the all-reduce. Output is sharded along the leading sequence
      dimension, so the caller no longer needs an outer
      ``scatter_to_sequence_parallel_region``. Halves the TP bandwidth on
      the output side relative to the legacy "all_reduce + outer scatter".
    * ``expert_skip_comm=True``: skip the TP reduction entirely. Returns the
      partial (un-reduced) output. Mirrors stock RowParallel's
      ``explicit_expert_comm`` path — the caller (a token dispatcher or a
      per-expert wrapper) is responsible for combining partials across TP.
      Combined with an outer ``reduce_scatter`` this implements the standard
      Megatron MoE "token combine" pattern. Bias is forbidden in this mode
      because adding bias to a partial would double-count it after the
      outer reduce.
    * ``sequence_parallel=False`` and ``expert_skip_comm=False`` (default):
      ``reduce_from_tensor_model_parallel_region`` (standard all-reduce);
      every rank holds the full output. Existing call sites that don't set
      either flag are unaffected.

    Bias is allocated on the canonical `self.bias` attribute (NOT a separate
    `row_bias`) and is added AFTER the reduction + bf16 cast (matches sglang's
    DSV4RowParallelLinear); pre-reduce addition would multiply bias by
    `tp_world`. Standard idioms like `if layer.bias is not None` work
    uniformly across DSV4Linear / Column / Row variants.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        dtype: Optional[torch.dtype] = None,
        tp_group: Optional["dist.ProcessGroup"] = None,
        sequence_parallel: bool = False,
        expert_skip_comm: bool = False,
    ):
        world = _tp_world(tp_group)
        assert in_features % world == 0, (
            f"DSV4RowParallelLinear: in_features={in_features} "
            f"not divisible by tp_world={world}"
        )
        # Disable bias on base init: bias must be replicated and added post-reduce.
        super().__init__(in_features // world, out_features, bias=False, dtype=dtype)
        if expert_skip_comm and bias:
            raise ValueError(
                "DSV4RowParallelLinear: expert_skip_comm=True forbids bias — the "
                "caller (dispatcher / wrapper) is responsible for the cross-TP "
                "reduction, and adding bias before that reduce would scale bias "
                "by tp_world."
            )
        if expert_skip_comm and sequence_parallel:
            raise ValueError(
                "DSV4RowParallelLinear: expert_skip_comm and sequence_parallel are "
                "mutually exclusive — skip_comm returns partial output and leaves "
                "the reduce to the caller, while sequence_parallel does the reduce "
                "(via reduce_scatter) internally."
            )
        self.tp_group = tp_group
        self.sequence_parallel = sequence_parallel
        self.expert_skip_comm = expert_skip_comm
        set_tensor_model_parallel_attributes(self.weight, True, 1, 1)
        if self.scale is not None:
            set_tensor_model_parallel_attributes(self.scale, True, 1, 1)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        """TP shard along axis 1 for weight / scale; bias replicated.

        DSV4 row-parallel ``bias`` is added *after* the cross-TP reduction
        (see ``forward``), so it is replicated across TP ranks rather than
        sharded — same convention as standard
        ``RowParallelLinear.sharded_state_dict``.
        """
        state_dict = self.state_dict(prefix="", keep_vars=True)
        axis_map = {"weight": 1}
        if self.scale is not None:
            axis_map["scale"] = 1
        return make_sharded_tensors_for_checkpoint(
            state_dict,
            prefix,
            axis_map,
            sharded_offsets,
            tp_group=self.tp_group,
            dp_cp_group=metadata["dp_cp_group"],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = _quantized_linear_with_grad(x, self.weight, self.scale)
        if (
            self.tp_group is not None
            and self.tp_group.size() > 1
            and not self.expert_skip_comm
        ):
            # Promote to FP32 for the row-parallel reduction (without it,
            # summing BF16 partials drops precision relative to TP=1's single
            # FP32 accumulation in the FP8/FP4 GEMM kernel), then cast back
            # before bias add to match sglang's DSV4RowParallelLinear
            # (deepseek_v4.py:634-645).
            out_dtype = x.dtype
            y = y.float()
            if self.sequence_parallel:
                y = reduce_scatter_to_sequence_parallel_region(y, group=self.tp_group)
            else:
                y = reduce_from_tensor_model_parallel_region(y, group=self.tp_group)
            y = y.to(out_dtype)
        if self.bias is not None:
            y = y + self.bias
        return y
