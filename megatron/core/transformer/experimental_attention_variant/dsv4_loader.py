# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Load native-quant DeepSeek V4 safetensors checkpoints into Megatron modules.

Native checkpoint key layout (one shard ``model{rank}-mp{world}.safetensors``):

::

    embed.weight                                # [vocab, dim] bf16
    head.weight                                 # [vocab, dim] bf16
    norm.weight                                 # [dim] bf16
    hc_head_fn / hc_head_base / hc_head_scale   # fp32 (decoder-level mHC head)

    layers.{i}.attn_norm.weight                 # [dim] bf16
    layers.{i}.ffn_norm.weight                  # [dim] bf16

    layers.{i}.attn.wq_a.weight                 # [q_lora, dim] fp8_e4m3
    layers.{i}.attn.wq_a.scale                  # fp8_e8m0 (block scale)
    layers.{i}.attn.q_norm.weight               # [q_lora] bf16
    layers.{i}.attn.wq_b.weight                 # [n_heads*head_dim, q_lora] fp8
    layers.{i}.attn.wq_b.scale
    layers.{i}.attn.wkv.weight                  # [head_dim, dim] fp8
    layers.{i}.attn.wkv.scale
    layers.{i}.attn.kv_norm.weight              # [head_dim] bf16
    layers.{i}.attn.wo_a.weight                 # bf16 (no scale)
    layers.{i}.attn.wo_b.weight                 # fp8
    layers.{i}.attn.wo_b.scale
    layers.{i}.attn.attn_sink                   # [n_heads] fp32

    # Compress branches (only on layers with compress_ratio > 0):
    layers.{i}.attn.compressor.{ape, wkv.weight, wgate.weight, norm.weight}
    # Indexer branch (only on layers with compress_ratio == 4):
    layers.{i}.attn.indexer.wq_b.{weight, scale}
    layers.{i}.attn.indexer.weights_proj.weight
    layers.{i}.attn.indexer.compressor.{ape, wkv.weight, wgate.weight, norm.weight}

    layers.{i}.ffn.gate.weight                  # [n_experts, dim] bf16
    layers.{i}.ffn.gate.{bias|tid2eid}          # bias fp32 [n_experts] OR tid2eid int64 [vocab, topk]
    layers.{i}.ffn.experts.{e}.{w1,w2,w3}.weight    # FP4 (packed e2m1fn_x2)
    layers.{i}.ffn.experts.{e}.{w1,w2,w3}.scale     # fp8_e8m0
    layers.{i}.ffn.shared_experts.{w1,w2,w3}.weight # FP8
    layers.{i}.ffn.shared_experts.{w1,w2,w3}.scale  # fp8_e8m0

    layers.{i}.{hc_attn_fn, hc_attn_base, hc_attn_scale,
                hc_ffn_fn, hc_ffn_base, hc_ffn_scale}   # fp32 per-layer mHC

Note on scale-key suffix: native checkpoints use ``<lin>.scale`` (not
``<lin>.weight.scale``) as the suffix; the rewrite below maps that to the
parameter at ``<rewritten-parent>.scale`` on the matching ``DSV4Linear``.
"""

import os
from typing import List

import torch

from safetensors.torch import safe_open

from megatron.core.transformer.experimental_attention_variant.dsv4_linear import (
    DSV4Linear,
)


_NAME_REWRITES = {
    "hc_head_fn": "hc_head_params.hc_head_fn",
    "hc_head_base": "hc_head_params.hc_head_base",
    "hc_head_scale": "hc_head_params.hc_head_scale",
    "norm.weight": "final_layernorm.weight",
}


def _rewrite_block_key(k: str) -> str:
    """Rewrite a native-ckpt key to the corresponding DeepSeekV4TransformerBlock state-dict key.

    Returns the key unchanged when no rewrite applies (caller treats anything
    not in ``block.named_parameters()`` as leftover).
    """
    if k in _NAME_REWRITES:
        return _NAME_REWRITES[k]
    if k.startswith("layers."):
        parts = k.split(".")
        i = parts[1]
        sub_top = parts[2]
        if sub_top == "attn_norm":
            return f"layers.{i}.input_layernorm." + ".".join(parts[3:])
        if sub_top == "ffn_norm":
            return f"layers.{i}.pre_mlp_layernorm." + ".".join(parts[3:])
        if sub_top == "attn":
            sub = ".".join(parts[3:])
            # Indexer wraps wq_b / weights_proj as DSV4Linear instances named
            # ``linear_wq_b`` / ``linear_weights_proj`` on the Megatron side.
            sub = sub.replace("indexer.wq_b", "indexer.linear_wq_b")
            sub = sub.replace("indexer.weights_proj", "indexer.linear_weights_proj")
            return f"layers.{i}.self_attention.{sub}"
        if sub_top == "ffn":
            sub = ".".join(parts[3:])
            return f"layers.{i}.mlp.{sub}"
        if sub_top.startswith("hc_"):
            # Per-layer mHC parameters live directly on the layer.
            return k
    return k


def load_dsv4_native_safetensors(
    block,
    embed_weight: torch.Tensor,
    head_weight: torch.Tensor,
    ckpt_dir: str,
    mp_rank: int = 0,
    mp_world: int = 1,
) -> List[str]:
    """Load native quant tensors into ``block``, ``embed_weight``, ``head_weight``.

    Returns the list of safetensors keys that were not consumed (so callers
    can audit completeness; e.g. MTP layers and KV caches are expected to be
    left over because they are not part of the inference forward model).

    Idempotent: calling twice produces the same final state because every copy
    is a ``.copy_`` into the destination tensor (no allocation, no rebinding).
    """
    path = os.path.join(ckpt_dir, f"model{mp_rank}-mp{mp_world}.safetensors")
    block_state = dict(block.named_parameters(recurse=True))

    # Resolve the matching DSV4Linear modules so we can poke ``.scale`` for
    # native-ckpt keys ending in ``.scale`` (the official layout stores the
    # weight scale as a sibling tensor at ``<lin>.scale``).
    dsv4_linears = {
        n: m
        for n, m in block.named_modules()
        if isinstance(m, DSV4Linear) and m.scale is not None
    }

    leftover: List[str] = []
    with safe_open(path, framework="pt", device="cpu") as f:
        keys = list(f.keys())
        for k in keys:
            if k == "embed.weight":
                # Loader caller passes fp32 destination; the source is bf16.
                embed_weight.data.copy_(f.get_tensor(k).to(embed_weight.device).float())
                continue
            if k == "head.weight":
                head_weight.data.copy_(f.get_tensor(k).to(head_weight.device).float())
                continue

            tgt = _rewrite_block_key(k)
            tensor = f.get_tensor(k)

            # Native ckpt stores the weight scale as ``<lin>.scale`` (sibling of
            # ``<lin>.weight``). Route a ``.scale`` key to a DSV4Linear's
            # registered scale Parameter ONLY when the parent name resolves to a
            # known DSV4Linear; otherwise fall through to the regular parameter
            # path. This correctly handles fp32 mHC tensors named ``*.hc_*_scale``
            # (or any future fp32 Parameter ending in ``.scale``) without an
            # error-prone substring denylist.
            if tgt.endswith(".scale"):
                lin_name = tgt[: -len(".scale")]
                lin = dsv4_linears.get(lin_name)
                if lin is not None:
                    # Source dtype matches destination (fp8_e8m0fnu); copy
                    # without a cast to avoid corrupting non-IEEE quant dtypes.
                    lin.scale.data.copy_(tensor.to(lin.scale.device))
                    continue
                # else: not a DSV4Linear scale; fall through to block_state.

            if tgt in block_state:
                p = block_state[tgt]
                # FP4 (float4_e2m1fn_x2) and FP8 (float8_e4m3fn / e8m0fnu)
                # tensors are stored in the safetensors file in their native
                # dtype already; ``.to(p.dtype)`` is a no-op for matching dtypes
                # but would corrupt data if source were a packed proxy (e.g.,
                # uint8). Native safetensors store source dtypes matching the
                # destination parameters, so this path is safe.
                p.data.copy_(tensor.to(p.device).to(p.dtype))
                continue

            leftover.append(k)
    return leftover


def _make_dsv4_config_from_official(args):
    """Build a Megatron ``MLATransformerConfig`` from the official ``ModelArgs``.

    Mirrors ``DeepSeekV4TransformerLayer.from_official`` and
    ``DeepSeekV4TransformerBlock.from_official``: indexer fields and
    ``layernorm_epsilon`` are required for parity (V4 uses 1e-6, not Megatron's
    default 1e-5; the indexer fields fire whenever any ``compress_ratios``
    entry equals 4).
    """
    from megatron.core.transformer.transformer_config import MLATransformerConfig

    cfg = MLATransformerConfig(
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
    return cfg
