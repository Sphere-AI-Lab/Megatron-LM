# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Re-exports the official DeepSeek V4 inference kernels.

The kernels are vendored verbatim in ``dsv4_official_kernel`` (a byte-for-byte
copy of the upstream DeepSeek-V4 ``inference/kernel.py``) so Megatron's
native-quant V4 path is byte-identical with SGLang/inference. Do not
reimplement: import.
"""

import os

import torch

from megatron.core.transformer.experimental_attention_variant import (
    dsv4_official_kernel as _official_kernel,
)
from megatron.core.transformer.experimental_attention_variant.det_fp8_gemm import (
    det_act_quant,
    det_fp8_gemm,
)


def _use_det_fp8_gemm() -> bool:
    return os.environ.get("MEGATRON_DSV4_DET_FP8_GEMM", "0") == "1"


act_quant = _official_kernel.act_quant
fp4_act_quant = _official_kernel.fp4_act_quant


def fp8_gemm(
    a: torch.Tensor,
    a_s: torch.Tensor,
    b: torch.Tensor,
    b_s: torch.Tensor,
    scale_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if _use_det_fp8_gemm():
        return det_fp8_gemm(a, a_s, b, b_s, scale_dtype)
    return _official_kernel.fp8_gemm(a, a_s, b, b_s, scale_dtype)


fp4_gemm = _official_kernel.fp4_gemm
sparse_attn = _official_kernel.sparse_attn
hc_split_sinkhorn = _official_kernel.hc_split_sinkhorn

__all__ = [
    "act_quant",
    "det_act_quant",
    "det_fp8_gemm",
    "fp4_act_quant",
    "fp8_gemm",
    "fp4_gemm",
    "sparse_attn",
    "hc_split_sinkhorn",
]
