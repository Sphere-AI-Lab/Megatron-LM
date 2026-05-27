# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""CUDA-event latency benchmark for DSV4 grouped FP4 GEMM shapes.

This intentionally reports GPU elapsed time only. Setup, random tensor
initialization, and any host-side Python overhead are outside the measured
region except for the unavoidable event recording around each CUDA workload.
"""

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path

import torch

from megatron.core.transformer.experimental_attention_variant.dsv4_grouped_fp4 import (
    grouped_fp4_gemm,
    grouped_fp4_linear_backward_fast,
    grouped_fp4_linear_backward_torch_reference,
)


@dataclass(frozen=True)
class DSV4Shape:
    name: str
    hidden_size: int
    moe_intermediate_size: int
    n_routed_experts: int


def _load_shape(model_dir: Path) -> DSV4Shape:
    inference_config = model_dir / "inference" / "config.json"
    config_path = inference_config if inference_config.exists() else model_dir / "config.json"
    with config_path.open() as f:
        config = json.load(f)
    return DSV4Shape(
        name=model_dir.name,
        hidden_size=int(config.get("dim", config.get("hidden_size"))),
        moe_intermediate_size=int(
            config.get("moe_inter_dim", config.get("moe_intermediate_size"))
        ),
        n_routed_experts=int(config["n_routed_experts"]),
    )


def _make_pos_to_expert(
    expanded_rows: int,
    local_experts: int,
    device: torch.device,
) -> torch.Tensor:
    rows = ((expanded_rows + 31) // 32) * 32
    block_experts = torch.arange(rows // 32, device=device, dtype=torch.int32)
    block_experts = block_experts.remainder(local_experts)
    return block_experts.repeat_interleave(32).contiguous()


def _make_fp4_weight(
    local_experts: int,
    out_features: int,
    in_features: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    weight = torch.empty(
        local_experts,
        out_features,
        in_features // 2,
        device=device,
        dtype=torch.float4_e2m1fn_x2,
    )
    weight.view(torch.uint8).random_(0, 256)
    scale = torch.empty(
        local_experts,
        out_features,
        in_features // 32,
        device=device,
        dtype=torch.float8_e8m0fnu,
    )
    scale.fill_(1.0)
    return weight.contiguous(), scale.contiguous()


def _event_ms(fn, warmup: int, iters: int) -> tuple[float, float, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    samples = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    torch.cuda.synchronize()
    return sum(samples) / len(samples), min(samples), max(samples)


def _bench_linear(
    shape: DSV4Shape,
    linear_name: str,
    in_features: int,
    out_features: int,
    ep_size: int,
    expanded_rows: int,
    warmup: int,
    iters: int,
    device: torch.device,
) -> None:
    assert shape.n_routed_experts % ep_size == 0
    local_experts = shape.n_routed_experts // ep_size
    pos_to_expert = _make_pos_to_expert(expanded_rows, local_experts, device)
    rows = pos_to_expert.numel()

    weight, scale = _make_fp4_weight(local_experts, out_features, in_features, device)
    x_q = torch.empty(rows, in_features, device=device, dtype=torch.float8_e4m3fn)
    x_q.view(torch.uint8).random_(0, 256)
    x_s = torch.empty(
        rows,
        (in_features + 127) // 128,
        device=device,
        dtype=torch.float8_e8m0fnu,
    )
    x_s.fill_(1.0)
    grad_output = torch.randn(rows, out_features, device=device, dtype=torch.bfloat16)

    forward_avg, forward_min, forward_max = _event_ms(
        lambda: grouped_fp4_gemm(
            x_q, x_s, weight, scale, pos_to_expert, torch.float8_e8m0fnu
        ),
        warmup,
        iters,
    )
    backward_fast_avg, backward_fast_min, backward_fast_max = _event_ms(
        lambda: grouped_fp4_linear_backward_fast(
            grad_output, weight, scale, pos_to_expert, torch.bfloat16
        ),
        warmup,
        iters,
    )
    backward_torch_avg, backward_torch_min, backward_torch_max = _event_ms(
        lambda: grouped_fp4_linear_backward_torch_reference(
            grad_output, weight, scale, pos_to_expert, torch.bfloat16
        ),
        warmup,
        iters,
    )

    print(
        f"{shape.name},{linear_name},ep={ep_size},local_experts={local_experts},"
        f"rows={rows},in={in_features},out={out_features},"
        f"forward_gemm_ms(avg/min/max)={forward_avg:.3f}/{forward_min:.3f}/{forward_max:.3f},"
        "backward_fast_ms(avg/min/max)="
        f"{backward_fast_avg:.3f}/{backward_fast_min:.3f}/{backward_fast_max:.3f},"
        "backward_torch_ms(avg/min/max)="
        f"{backward_torch_avg:.3f}/{backward_torch_min:.3f}/{backward_torch_max:.3f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        action="append",
        type=Path,
        default=None,
    )
    parser.add_argument("--ep-size", type=int, default=4)
    parser.add_argument("--expanded-rows", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda:0" if args.device == "cuda" else args.device)
    torch.cuda.set_device(device)
    torch.set_default_dtype(torch.bfloat16)

    model_dirs = args.model_dir
    if not model_dirs:
        model_root = os.environ.get("DSV4_MODEL_ROOT")
        if not model_root:
            raise SystemExit("pass --model-dir or set DSV4_MODEL_ROOT")
        model_dirs = [
            Path(model_root) / "DeepSeek-V4-Flash",
            Path(model_root) / "DeepSeek-V4-Pro",
        ]

    print(
        "model,linear,ep,local_experts,rows,in,out,"
        "forward_gemm_ms,backward_fast_ms,backward_torch_ms"
    )
    for model_dir in model_dirs:
        shape = _load_shape(model_dir)
        _bench_linear(
            shape,
            "w1_w3",
            shape.hidden_size,
            shape.moe_intermediate_size,
            args.ep_size,
            args.expanded_rows,
            args.warmup,
            args.iters,
            device,
        )
        torch.cuda.empty_cache()
        _bench_linear(
            shape,
            "w2",
            shape.moe_intermediate_size,
            shape.hidden_size,
            args.ep_size,
            args.expanded_rows,
            args.warmup,
            args.iters,
            device,
        )
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
