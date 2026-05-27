# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""SGLang-style fused routing helpers for DeepSeek V4 grouped MoE.

These helpers keep Megatron's grouped expert storage, but use the same
token/top-k -> expert-major expanded layout that SGLang uses around its DSV4
CUDA-graph grouped FP4 GEMM path.
"""

import os
from functools import lru_cache
from typing import Tuple

import tilelang
import tilelang.language as T
import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

tilelang.set_log_level("WARNING")

_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
}

_FP32 = "float32"
_INT32 = "int32"


_CLAMP_SILU_MUL_CPP_SRC = r"""
#include <torch/extension.h>

void dsv4_clamp_silu_mul_topk_forward_cuda(
    torch::Tensor gate,
    torch::Tensor up,
    torch::Tensor topk_weights,
    torch::Tensor pos_to_token_topk,
    torch::Tensor out,
    torch::Tensor act,
    double swiglu_limit);
std::vector<torch::Tensor> dsv4_clamp_silu_mul_topk_backward_cuda(
    torch::Tensor gate,
    torch::Tensor up,
    torch::Tensor topk_weights,
    torch::Tensor pos_to_token_topk,
    torch::Tensor act,
    torch::Tensor grad_out,
    double swiglu_limit);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("topk_forward_out", &dsv4_clamp_silu_mul_topk_forward_cuda);
  m.def("topk_backward", &dsv4_clamp_silu_mul_topk_backward_cuda);
}
"""


_CLAMP_SILU_MUL_CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>

namespace {

__device__ __forceinline__ float load_bf16(const __nv_bfloat16* p, int64_t i) {
  return __bfloat162float(p[i]);
}

__device__ __forceinline__ float load_weight(const __nv_bfloat16* p, int64_t i) {
  return __bfloat162float(p[i]);
}

__device__ __forceinline__ float load_weight(const float* p, int64_t i) {
  return p[i];
}

__device__ __forceinline__ __nv_bfloat16 store_bf16(float v) {
  return __float2bfloat16_rn(v);
}

__device__ __forceinline__ float clamp_gate(float x, float limit) {
  return limit > 0.0f ? fminf(x, limit) : x;
}

__device__ __forceinline__ float clamp_up(float x, float limit) {
  return limit > 0.0f ? fminf(fmaxf(x, -limit), limit) : x;
}

__device__ __forceinline__ float signed_zero_for_unweighted_act(
    float g,
    float u,
    float limit) {
  float gc = clamp_gate(g, limit);
  float uc = clamp_up(u, limit);
  uint32_t sign = (__float_as_uint(gc) ^ __float_as_uint(uc)) & 0x80000000u;
  return __uint_as_float(sign);
}

__device__ __forceinline__ float silu_exact(float x) {
  return x / (1.0f + expf(-x));
}

__device__ __forceinline__ float sigmoid_for_silu_backward(float x) {
  float e = expf(-x);
  return 1.0f / (1.0f + e);
}

__device__ __forceinline__ bool gate_clamp_allows_grad(float x, float limit) {
  return limit <= 0.0f || x <= limit;
}

__device__ __forceinline__ bool up_clamp_allows_grad(float x, float limit) {
  return limit <= 0.0f || (x >= -limit && x <= limit);
}

__device__ __forceinline__ void store_weight_grad(__nv_bfloat16* p, int64_t i, float v) {
  p[i] = store_bf16(v);
}

__device__ __forceinline__ void store_weight_grad(float* p, int64_t i, float v) {
  p[i] = v;
}

template <typename weight_t>
__global__ void topk_forward_kernel(
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ up,
    const weight_t* __restrict__ topk_weights,
    const int32_t* __restrict__ pos_to_token_topk,
    __nv_bfloat16* __restrict__ out,
    float* __restrict__ act_out,
    int64_t total,
    int64_t hidden,
    int64_t num_topk_slots,
    float limit) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) {
    return;
  }
  int64_t row = idx / hidden;
  int32_t slot = pos_to_token_topk[row];
  float g = load_bf16(gate, idx);
  float u = load_bf16(up, idx);
  if (slot < 0 || slot >= num_topk_slots) {
    float zero = signed_zero_for_unweighted_act(g, u, limit);
    act_out[idx] = zero;
    out[idx] = store_bf16(zero);
    return;
  }
  float weight = load_weight(topk_weights, slot);
  float gc = clamp_gate(g, limit);
  float uc = clamp_up(u, limit);
  float act = silu_exact(gc) * uc;
  act_out[idx] = act;
  out[idx] = store_bf16(act * weight);
}

template <typename weight_t>
__global__ void topk_backward_kernel(
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ up,
    const weight_t* __restrict__ topk_weights,
    const int32_t* __restrict__ pos_to_token_topk,
    const float* __restrict__ act,
    const __nv_bfloat16* __restrict__ grad_out,
    __nv_bfloat16* __restrict__ grad_gate,
    __nv_bfloat16* __restrict__ grad_up,
    weight_t* __restrict__ grad_weights,
    int64_t rows,
    int64_t hidden,
    int64_t num_topk_slots,
    float limit) {
  using BlockReduce = cub::BlockReduce<float, 256>;
  __shared__ typename BlockReduce::TempStorage reduce_storage;
  int64_t row = blockIdx.x;
  int32_t slot = pos_to_token_topk[row];
  bool valid_slot = slot >= 0 && slot < num_topk_slots;
  float weight = valid_slot ? load_weight(topk_weights, slot) : 0.0f;
  float weight_grad = 0.0f;

  for (int64_t col = threadIdx.x; col < hidden; col += blockDim.x) {
    int64_t idx = row * hidden + col;
    float g = load_bf16(gate, idx);
    float u = load_bf16(up, idx);
    float go = load_bf16(grad_out, idx);
    float gc = clamp_gate(g, limit);
    float uc = clamp_up(u, limit);
    float silu = silu_exact(gc);
    float common = go * weight;
    float grad_silu = common * uc;
    float sigmoid = sigmoid_for_silu_backward(gc);
    float gg = (grad_silu * sigmoid) * (1.0f + gc * (1.0f - sigmoid));
    float gu = common * silu;
    if (!gate_clamp_allows_grad(g, limit)) {
      gg = 0.0f;
    }
    if (!up_clamp_allows_grad(u, limit)) {
      gu = 0.0f;
    }
    grad_gate[idx] = store_bf16(gg);
    grad_up[idx] = store_bf16(gu);
    weight_grad += go * act[idx];
  }

  float weight_sum = BlockReduce(reduce_storage).Sum(weight_grad);
  if (threadIdx.x == 0 && valid_slot) {
    store_weight_grad(grad_weights, slot, weight_sum);
  }
}

template <typename weight_t>
void launch_topk_forward(
    torch::Tensor gate,
    torch::Tensor up,
    torch::Tensor topk_weights,
    torch::Tensor pos_to_token_topk,
    torch::Tensor out,
    torch::Tensor act,
    double swiglu_limit) {
  int threads = 256;
  auto stream = at::cuda::getCurrentCUDAStream(gate.get_device());
  int64_t hidden = gate.size(1);
  int64_t total = gate.numel();
  int blocks = (total + threads - 1) / threads;
  topk_forward_kernel<<<blocks, threads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(gate.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(up.data_ptr<at::BFloat16>()),
      reinterpret_cast<const weight_t*>(topk_weights.data_ptr()),
      pos_to_token_topk.data_ptr<int32_t>(),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
      act.data_ptr<float>(),
      total,
      hidden,
      topk_weights.numel(),
      static_cast<float>(swiglu_limit));
}

}  // namespace

void dsv4_clamp_silu_mul_topk_forward_cuda(
    torch::Tensor gate,
    torch::Tensor up,
    torch::Tensor topk_weights,
    torch::Tensor pos_to_token_topk,
    torch::Tensor out,
    torch::Tensor act,
    double swiglu_limit) {
  if (topk_weights.scalar_type() == at::ScalarType::BFloat16) {
    launch_topk_forward<__nv_bfloat16>(
        gate, up, topk_weights, pos_to_token_topk, out, act, swiglu_limit);
    return;
  }
  if (topk_weights.scalar_type() == at::ScalarType::Float) {
    launch_topk_forward<float>(
        gate, up, topk_weights, pos_to_token_topk, out, act, swiglu_limit);
    return;
  }
  TORCH_CHECK(false, "topk_weights must be bfloat16 or float32");
}

template <typename weight_t>
std::vector<torch::Tensor> launch_topk_backward(
    torch::Tensor gate,
    torch::Tensor up,
    torch::Tensor topk_weights,
    torch::Tensor pos_to_token_topk,
    torch::Tensor act,
    torch::Tensor grad_out,
    double swiglu_limit) {
  auto grad_gate = torch::empty_like(gate);
  auto grad_up = torch::empty_like(up);
  auto grad_weights = torch::empty_like(topk_weights);
  int64_t rows = gate.size(0);
  int64_t hidden = gate.size(1);
  int threads = 256;
  auto stream = at::cuda::getCurrentCUDAStream(gate.get_device());
  C10_CUDA_CHECK(cudaMemsetAsync(
      grad_weights.data_ptr(),
      0,
      grad_weights.nbytes(),
      stream));
  topk_backward_kernel<<<rows, threads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(gate.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(up.data_ptr<at::BFloat16>()),
      reinterpret_cast<const weight_t*>(topk_weights.data_ptr()),
      pos_to_token_topk.data_ptr<int32_t>(),
      act.data_ptr<float>(),
      reinterpret_cast<const __nv_bfloat16*>(grad_out.data_ptr<at::BFloat16>()),
      reinterpret_cast<__nv_bfloat16*>(grad_gate.data_ptr<at::BFloat16>()),
      reinterpret_cast<__nv_bfloat16*>(grad_up.data_ptr<at::BFloat16>()),
      reinterpret_cast<weight_t*>(grad_weights.data_ptr()),
      rows,
      hidden,
      topk_weights.numel(),
      static_cast<float>(swiglu_limit));
  return {grad_gate, grad_up, grad_weights};
}

std::vector<torch::Tensor> dsv4_clamp_silu_mul_topk_backward_cuda(
    torch::Tensor gate,
    torch::Tensor up,
    torch::Tensor topk_weights,
    torch::Tensor pos_to_token_topk,
    torch::Tensor act,
    torch::Tensor grad_out,
    double swiglu_limit) {
  if (topk_weights.scalar_type() == at::ScalarType::BFloat16) {
    return launch_topk_backward<__nv_bfloat16>(
        gate, up, topk_weights, pos_to_token_topk, act, grad_out, swiglu_limit);
  }
  if (topk_weights.scalar_type() == at::ScalarType::Float) {
    return launch_topk_backward<float>(
        gate, up, topk_weights, pos_to_token_topk, act, grad_out, swiglu_limit);
  }
  TORCH_CHECK(false, "topk_weights must be bfloat16 or float32");
}
"""


@lru_cache(maxsize=1)
def _dsv4_clamp_silu_mul_ext():
    return load_inline(
        name="megatron_dsv4_clamp_silu_mul_topk_ext",
        cpp_sources=_CLAMP_SILU_MUL_CPP_SRC,
        cuda_sources=_CLAMP_SILU_MUL_CUDA_SRC,
        with_cuda=True,
        extra_cuda_cflags=[],
        verbose=False,
    )


def _torch_clamp_silu_mul_topk(
    gate: torch.Tensor,
    up: torch.Tensor,
    topk_weights: torch.Tensor,
    pos_to_token_topk: torch.Tensor,
    swiglu_limit: float,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    gate_f = gate.float()
    up_f = up.float()
    if swiglu_limit and swiglu_limit > 0:
        up_f = torch.clamp(up_f, min=-swiglu_limit, max=swiglu_limit)
        gate_f = torch.clamp(gate_f, max=swiglu_limit)
    y = F.silu(gate_f) * up_f
    flat_weights = F.pad(topk_weights.reshape(-1), (1, 0))
    slots = pos_to_token_topk.to(torch.int64)
    valid = (slots >= 0) & (slots < topk_weights.numel())
    gather_idx = torch.where(valid, slots + 1, torch.zeros_like(slots))
    y = y * flat_weights.gather(0, gather_idx).unsqueeze(-1)
    return y.to(out_dtype)


class _DSV4ClampSiluMulTopK(torch.autograd.Function):
    """Fused clamp -> silu -> mul -> route-weight for DSV4 expanded experts.

    When ``MEGATRON_DSV4_MOE_RECOMPUTE_ACT`` is set (default ON), the fp32
    ``act`` activation tensor (~half of the per-microbatch MoE memory) is
    recomputed during backward instead of being saved. The forward inputs
    (gate / up / weights / pos) are already saved, so we just re-call the
    fused forward kernel during backward to regenerate ``act``.
    """

    @staticmethod
    def forward(
        ctx,
        gate: torch.Tensor,
        up: torch.Tensor,
        topk_weights: torch.Tensor,
        pos_to_token_topk: torch.Tensor,
        swiglu_limit: float,
        out_dtype: torch.dtype,
    ) -> torch.Tensor:
        gate_c = gate.contiguous()
        up_c = up.contiguous()
        weights_c = topk_weights.contiguous()
        pos_c = pos_to_token_topk.contiguous().to(torch.int32)
        out = torch.empty_like(gate_c)
        act = torch.empty(gate_c.shape, device=gate_c.device, dtype=torch.float32)
        _dsv4_clamp_silu_mul_ext().topk_forward_out(
            gate_c,
            up_c,
            weights_c,
            pos_c,
            out,
            act,
            float(swiglu_limit),
        )
        ctx.recompute_act = (
            os.environ.get("MEGATRON_DSV4_MOE_RECOMPUTE_ACT", "1") == "1"
        )
        if ctx.recompute_act:
            ctx.save_for_backward(gate_c, up_c, weights_c, pos_c)
        else:
            ctx.save_for_backward(gate_c, up_c, weights_c, pos_c, act)
        ctx.swiglu_limit = float(swiglu_limit)
        return out.reshape_as(gate)

    @staticmethod
    def backward(ctx, grad_out):
        if ctx.recompute_act:
            gate, up, topk_weights, pos_to_token_topk = ctx.saved_tensors
            # Recompute fp32 act from the saved inputs. dummy_out is a scratch
            # required by the forward kernel signature (we don't use it).
            dummy_out = torch.empty_like(gate)
            act = torch.empty(gate.shape, device=gate.device, dtype=torch.float32)
            _dsv4_clamp_silu_mul_ext().topk_forward_out(
                gate,
                up,
                topk_weights,
                pos_to_token_topk,
                dummy_out,
                act,
                ctx.swiglu_limit,
            )
        else:
            gate, up, topk_weights, pos_to_token_topk, act = ctx.saved_tensors
        grad_gate, grad_up, grad_weights = _dsv4_clamp_silu_mul_ext().topk_backward(
            gate,
            up,
            topk_weights,
            pos_to_token_topk,
            act,
            grad_out.contiguous(),
            ctx.swiglu_limit,
        )
        return grad_gate, grad_up, grad_weights, None, None, None


def dsv4_clamp_silu_mul_topk(
    gate: torch.Tensor,
    up: torch.Tensor,
    topk_weights: torch.Tensor,
    pos_to_token_topk: torch.Tensor,
    swiglu_limit: float,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Fused `clamp -> silu -> mul -> route-weight` for DSV4 expanded routes."""

    assert pos_to_token_topk is not None, "pos_to_token_topk is required"
    if gate.numel() == 0:
        return _torch_clamp_silu_mul_topk(
            gate, up, topk_weights, pos_to_token_topk, swiglu_limit, out_dtype
        )
    if (
        not gate.is_cuda
        or gate.dtype != torch.bfloat16
        or up.dtype != torch.bfloat16
        or out_dtype != torch.bfloat16
        or topk_weights.dtype not in (torch.bfloat16, torch.float32)
    ):
        return _torch_clamp_silu_mul_topk(
            gate, up, topk_weights, pos_to_token_topk, swiglu_limit, out_dtype
        )
    return _DSV4ClampSiluMulTopK.apply(
        gate,
        up,
        topk_weights,
        pos_to_token_topk,
        swiglu_limit,
        out_dtype,
    )


def dsv4_clamp_silu_mul_preexpanded(
    gate: torch.Tensor,
    up: torch.Tensor,
    preexpanded_weights: torch.Tensor,
    swiglu_limit: float,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Use the top-k fused kernel when route weights are already row-expanded."""

    flat_weights = preexpanded_weights.reshape(-1)
    rows = gate.numel() // gate.shape[-1] if gate.dim() > 0 else 0
    if flat_weights.numel() != rows:
        raise ValueError(
            "DSV4 preexpanded activation weights must have one value per row: "
            f"got {flat_weights.numel()} weights for {rows} rows"
        )
    pos_to_token_topk = torch.arange(
        rows,
        device=gate.device,
        dtype=torch.int32,
    )
    return dsv4_clamp_silu_mul_topk(
        gate,
        up,
        flat_weights,
        pos_to_token_topk,
        swiglu_limit,
        out_dtype,
    )


def fused_route_num_expanded_tokens(
    num_routed_items: int,
    num_experts: int,
    alignment: int,
) -> int:
    """Return SGLang's conservative graph-friendly expanded-token bound."""

    max_active_experts = min(num_experts, num_routed_items)
    return (
        (
            num_routed_items
            + (alignment - 1) * max_active_experts
            + alignment - 1
        )
        // alignment
        * alignment
    )


def get_fused_route_mapping(
    topk_ids: torch.Tensor,
    num_experts: int,
    alignment: int = 32,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the TileKernels fused route mapping used by SGLang's graph path."""

    from tile_kernels.moe.get_fused_mapping_kernel import get_fused_mapping

    topk_ids = topk_ids.to(torch.int64).contiguous()
    num_expanded_tokens = fused_route_num_expanded_tokens(
        topk_ids.numel(),
        num_experts,
        alignment,
    )
    (
        pos_to_expert,
        _pos_to_token,
        pos_to_token_topk,
        token_topk_to_pos,
        _expert_start,
        _expert_end,
        _expert_count,
        _counts_list,
    ) = get_fused_mapping(
        topk_ids,
        num_experts,
        num_expanded_tokens,
        alignment,
        force_no_sync=True,
    )
    return (
        pos_to_expert.contiguous(),
        pos_to_token_topk.contiguous(),
        token_topk_to_pos.contiguous(),
    )


def _scatter_expanded_grad_to_tokens(
    grad_expanded: torch.Tensor,
    token_topk_to_pos: torch.Tensor,
    num_tokens: int,
) -> torch.Tensor:
    hidden = grad_expanded.shape[-1]
    grad_tokens = grad_expanded.new_zeros((num_tokens, hidden))
    flat_pos = token_topk_to_pos.reshape(-1).to(torch.int64)
    valid = flat_pos >= 0
    if valid.any():
        token_ids = torch.arange(
            num_tokens,
            device=grad_expanded.device,
            dtype=torch.int64,
        ).repeat_interleave(token_topk_to_pos.shape[1])
        grad_tokens.index_add_(
            0,
            token_ids[valid],
            grad_expanded.reshape(-1, hidden)[flat_pos[valid]],
        )
    return grad_tokens


def _scatter_token_grad_to_expanded(
    grad_tokens: torch.Tensor,
    token_topk_to_pos: torch.Tensor,
    num_expanded_tokens: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    hidden = grad_tokens.shape[-1]
    grad_expanded = torch.zeros(
        (num_expanded_tokens, hidden),
        device=grad_tokens.device,
        dtype=dtype,
    )
    flat_pos = token_topk_to_pos.reshape(-1).to(torch.int64)
    valid = flat_pos >= 0
    if valid.any():
        token_ids = torch.arange(
            grad_tokens.shape[0],
            device=grad_tokens.device,
            dtype=torch.int64,
        ).repeat_interleave(token_topk_to_pos.shape[1])
        grad_expanded.index_add_(
            0,
            flat_pos[valid],
            grad_tokens[token_ids[valid]].to(dtype),
        )
    return grad_expanded


class _ExpandToFusedFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        token_topk_to_pos: torch.Tensor,
        pos_to_expert: torch.Tensor,
    ) -> torch.Tensor:
        from tile_kernels.moe.expand_to_fused_kernel import expand_to_fused

        token_topk_to_pos = token_topk_to_pos.contiguous()
        pos_to_expert = pos_to_expert.contiguous()
        ctx.save_for_backward(token_topk_to_pos)
        ctx.num_tokens = x.shape[0]
        return expand_to_fused(x.contiguous(), token_topk_to_pos, pos_to_expert)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (token_topk_to_pos,) = ctx.saved_tensors
        grad_x = _scatter_expanded_grad_to_tokens(
            grad_output.contiguous(),
            token_topk_to_pos,
            ctx.num_tokens,
        )
        return grad_x, None, None


def expand_to_fused_route(
    x: torch.Tensor,
    token_topk_to_pos: torch.Tensor,
    pos_to_expert: torch.Tensor,
) -> torch.Tensor:
    """Autograd-aware wrapper around TileKernels' fused route expansion."""

    return _ExpandToFusedFn.apply(x, token_topk_to_pos, pos_to_expert)


@lru_cache(maxsize=None)
@tilelang.jit(pass_configs=_PASS_CONFIGS)
def _reduce_fused_topk_fp32_kernel(
    hidden: int,
    num_topk: int,
    in_dtype: str,
):
    num_expanded_tokens = T.dynamic("num_expanded_tokens")
    num_tokens = T.dynamic("num_tokens")

    @T.prim_func
    def kernel(
        x: T.Tensor[(num_expanded_tokens, hidden), in_dtype],
        token_topk_to_pos: T.Tensor[(num_tokens, num_topk), _INT32],
        out: T.Tensor[(num_tokens, hidden), _FP32],
    ):
        with T.Kernel(num_tokens, threads=128) as (pid_token,):
            reduced = T.alloc_fragment((hidden,), _FP32)
            topk_to_pos = T.alloc_fragment((num_topk,), _INT32)

            T.clear(reduced)
            T.copy(token_topk_to_pos[pid_token, :], topk_to_pos)
            for k in T.unroll(num_topk):
                pos = topk_to_pos[k]
                T.assume(pos < num_expanded_tokens)
                if pos >= 0:
                    for i in T.Parallel(hidden):
                        reduced[i] += T.Cast(_FP32, x[pos, i])

            for i in T.Parallel(hidden):
                out[pid_token, i] = reduced[i]

    return kernel


def _reduce_fused_topk_fp32_forward(
    x: torch.Tensor,
    token_topk_to_pos: torch.Tensor,
) -> torch.Tensor:
    assert x.is_contiguous(), "expanded expert outputs must be contiguous"
    assert token_topk_to_pos.is_contiguous(), "token_topk_to_pos must be contiguous"
    assert x.dim() == 2, "expanded expert outputs must be [expanded_tokens, hidden]"
    assert token_topk_to_pos.dim() == 2, "token_topk_to_pos must be [tokens, topk]"
    hidden = x.size(-1)
    num_tokens = token_topk_to_pos.size(0)
    out = torch.empty((num_tokens, hidden), dtype=torch.float32, device=x.device)
    if num_tokens > 0:
        kernel = _reduce_fused_topk_fp32_kernel(
            hidden,
            token_topk_to_pos.size(1),
            T.dtype(x.dtype),
        )
        kernel(x, token_topk_to_pos.to(torch.int32), out)
    return out


class _ReduceFusedTopkFP32Fn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        token_topk_to_pos: torch.Tensor,
    ) -> torch.Tensor:
        token_topk_to_pos = token_topk_to_pos.contiguous()
        ctx.save_for_backward(token_topk_to_pos)
        ctx.num_expanded_tokens = x.shape[0]
        ctx.x_dtype = x.dtype
        return _reduce_fused_topk_fp32_forward(x.contiguous(), token_topk_to_pos)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (token_topk_to_pos,) = ctx.saved_tensors
        grad_x = _scatter_token_grad_to_expanded(
            grad_output.contiguous(),
            token_topk_to_pos,
            ctx.num_expanded_tokens,
            ctx.x_dtype,
        )
        return grad_x, None


def reduce_fused_topk_fp32(
    x: torch.Tensor,
    token_topk_to_pos: torch.Tensor,
) -> torch.Tensor:
    """Autograd-aware SGLang-style FP32 top-k route reduction."""

    return _ReduceFusedTopkFP32Fn.apply(x, token_topk_to_pos)


__all__ = [
    "dsv4_clamp_silu_mul_preexpanded",
    "dsv4_clamp_silu_mul_topk",
    "expand_to_fused_route",
    "fused_route_num_expanded_tokens",
    "get_fused_route_mapping",
    "reduce_fused_topk_fp32",
]
