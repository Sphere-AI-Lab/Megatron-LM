from __future__ import annotations

import torch
import triton
import triton.language as tl


def _eager_q_rmsnorm(x: torch.Tensor, eps: float) -> torch.Tensor:
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)


def _block_size(n_cols: int) -> int:
    if n_cols > 1024:
        raise RuntimeError(
            f"dsv4_q_rmsnorm supports last dim <= 1024, got {n_cols}."
        )
    return max(1, triton.next_power_of_2(n_cols))


@triton.jit
def _q_rmsnorm_fwd_kernel(
    x_ptr,
    out_ptr,
    rsigma_ptr,
    n_cols: tl.constexpr,
    eps: tl.constexpr,
    block_n: tl.constexpr,
    store_rsigma: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, block_n)
    mask = cols < n_cols
    offsets = row * n_cols + cols
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    sum_sq = tl.sum(tl.where(mask, x * x, 0.0))
    rsigma = tl.rsqrt(sum_sq / n_cols + eps)
    y = x * rsigma
    tl.store(out_ptr + offsets, y, mask=mask)
    if store_rsigma:
        tl.store(rsigma_ptr + row, rsigma)


@triton.jit
def _q_rmsnorm_bwd_kernel(
    grad_out_ptr,
    x_ptr,
    rsigma_ptr,
    grad_x_ptr,
    n_cols: tl.constexpr,
    block_n: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, block_n)
    mask = cols < n_cols
    offsets = row * n_cols + cols
    grad_out = tl.load(grad_out_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    rsigma = tl.load(rsigma_ptr + row).to(tl.float32)
    row_dot = tl.sum(tl.where(mask, grad_out * x, 0.0))
    grad_x = grad_out * rsigma - x * (rsigma * rsigma * rsigma) * row_dot / n_cols
    tl.store(grad_x_ptr + offsets, grad_x, mask=mask)


def _q_rmsnorm_forward(
    x: torch.Tensor,
    eps: float,
    *,
    save_rsigma: bool,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    shape = tuple(x.shape)
    n_cols = shape[-1]
    x_2d = x.reshape(-1, n_cols).contiguous()
    n_rows = x_2d.shape[0]
    out = torch.empty_like(x_2d)
    if n_rows == 0:
        return out.reshape(shape), None, x_2d
    rsigma = (
        torch.empty((n_rows,), device=x.device, dtype=torch.float32)
        if save_rsigma
        else None
    )
    block_n = _block_size(n_cols)
    _q_rmsnorm_fwd_kernel[(n_rows,)](
        x_2d,
        out,
        rsigma if rsigma is not None else out,
        n_cols,
        float(eps),
        block_n,
        save_rsigma,
        num_warps=4 if block_n >= 128 else 1,
    )
    return out.reshape(shape), rsigma, x_2d


class _DSV4QRMSNormFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, eps: float) -> torch.Tensor:
        out, rsigma, x_2d = _q_rmsnorm_forward(x, eps, save_rsigma=True)
        ctx.shape = tuple(x.shape)
        ctx.n_cols = x.shape[-1]
        ctx.save_for_backward(x_2d, rsigma)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x_2d, rsigma = ctx.saved_tensors
        n_cols = ctx.n_cols
        grad_2d = grad_output.reshape(-1, n_cols).contiguous()
        grad_x = torch.empty_like(x_2d)
        n_rows = x_2d.shape[0]
        if n_rows > 0:
            block_n = _block_size(n_cols)
            _q_rmsnorm_bwd_kernel[(n_rows,)](
                grad_2d,
                x_2d,
                rsigma,
                grad_x,
                n_cols,
                block_n,
                num_warps=4 if block_n >= 128 else 1,
            )
        return grad_x.reshape(ctx.shape), None


def dsv4_q_rmsnorm(x: torch.Tensor, eps: float) -> torch.Tensor:
    # Compute the q-only norm in fp32 and let the store cast once to the
    # output dtype. This keeps the bf16 path simple and matches Megatron.
    if (
        not x.is_cuda
        or x.dtype not in (torch.bfloat16, torch.float32)
        or x.shape[-1] > 1024
    ):
        return _eager_q_rmsnorm(x, eps)
    if not torch.is_grad_enabled() or not x.requires_grad:
        out, _, _ = _q_rmsnorm_forward(x, eps, save_rsigma=False)
        return out
    return _DSV4QRMSNormFn.apply(x, eps)
