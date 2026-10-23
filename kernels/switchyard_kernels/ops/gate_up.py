"""Dense MLP gate/up + SwiGLU fusion for Qwen3.6-27B.

Batch 1 uses one program per output row; batches 2-16 reuse weight tiles across
token rows. Anything larger, or any layout the guards reject, falls back to
vLLM's ``fused_experts``/``silu_and_mul`` path in the adapter.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _gate_up(X, W, Y, H: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    row = tl.program_id(0)
    k = tl.arange(0, K)
    x = tl.load(X + k, k < H, 0).to(tl.float32)
    g = tl.load(W + row * H + k, k < H, 0).to(tl.float32)
    u = tl.load(W + (N + row) * H + k, k < H, 0).to(tl.float32)
    a = tl.sum(g * x, 0).to(tl.bfloat16).to(tl.float32)
    b = tl.sum(u * x, 0).to(tl.bfloat16).to(tl.float32)
    tl.store(Y + row, a * tl.sigmoid(a) * b)


@triton.jit
def _gate_up_batched(
    X,
    W,
    Y,
    B: tl.constexpr,
    N: tl.constexpr,
    H: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    pm = tl.program_id(1)
    pn = tl.program_id(0)
    rows = pm * BM + tl.arange(0, BM)
    cols = pn * BN + tl.arange(0, BN)
    ks = tl.arange(0, BK)
    acc_g = tl.zeros((BM, BN), tl.float32)
    acc_u = tl.zeros((BM, BN), tl.float32)
    for kt in range(tl.cdiv(H, BK)):
        k = kt * BK + ks
        x = tl.load(
            X + rows[:, None] * H + k[None, :],
            (rows[:, None] < B) & (k[None, :] < H),
            0,
        )
        wg = tl.load(
            W + cols[None, :] * H + k[:, None],
            (cols[None, :] < N) & (k[:, None] < H),
            0,
        )
        wu = tl.load(
            W + (N + cols[None, :]) * H + k[:, None],
            (cols[None, :] < N) & (k[:, None] < H),
            0,
        )
        acc_g = tl.dot(x, wg, acc_g)
        acc_u = tl.dot(x, wu, acc_u)
    gate = acc_g.to(tl.bfloat16).to(tl.float32)
    up = acc_u.to(tl.bfloat16).to(tl.float32)
    result = gate * tl.sigmoid(gate) * up
    tl.store(
        Y + rows[:, None] * N + cols[None, :],
        result,
        (rows[:, None] < B) & (cols[None, :] < N),
    )


@triton.jit
def _down_vec(
    X, W, Y, H: tl.constexpr, N: tl.constexpr, R: tl.constexpr, BK: tl.constexpr
):
    row = tl.program_id(0) * R + tl.arange(0, R)
    k = tl.arange(0, BK)
    acc = tl.full((R,), 0.0, tl.float32)
    for tile in range(tl.cdiv(N, BK)):
        col = tile * BK + k
        x = tl.load(X + col, col < N, 0).to(tl.float32)
        w = tl.load(
            W + row[:, None] * N + col[None, :],
            (row[:, None] < H) & (col[None, :] < N),
            0,
        ).to(tl.float32)
        acc += tl.sum(w * x[None, :], 1)
    tl.store(Y + row, acc, row < H)


BATCHED_LIMIT = 16
HIDDEN = 5120


@torch.library.custom_op("jarvis_dense::gate_up", mutates_args=())
def gate_up(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Fused gate/up projection with SwiGLU, BF16-rounded like the public path."""
    n = weight.shape[0] // 2
    batch = x.shape[0]
    if batch == 1:
        path = "triton_b1"
    elif batch <= BATCHED_LIMIT and x.is_contiguous():
        path = "triton_batched"
    else:
        path = "vllm_fallback"
    from ..announce import announce

    announce(
        "JARVIS_DENSE_DISPATCH",
        key=batch,
        batch=batch,
        custom=path.startswith("triton"),
        path=path,
    )
    out = x.new_empty((batch, n))
    if path == "triton_b1":
        _gate_up[(n,)](x, weight, out, HIDDEN, n, 8192, num_warps=4)
    elif path == "triton_batched":
        bm, bn, bk = 16, 32, 64
        _gate_up_batched[(triton.cdiv(n, bn), triton.cdiv(batch, bm))](
            x, weight, out, batch, n, HIDDEN, bm, bn, bk, num_warps=4
        )
    else:
        import torch.nn.functional as F

        raw = F.linear(x, weight)
        torch.ops._C.silu_and_mul(out, raw)
    return out


@gate_up.register_fake
def _fake_gate_up(x, weight):
    return x.new_empty((x.shape[0], weight.shape[0] // 2))


@torch.library.custom_op("jarvis_dense::down_projection", mutates_args=())
def down_projection(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Single-token down projection; larger batches use plain matmul."""
    out = x.new_empty((x.shape[0], weight.shape[0]))
    if x.shape[0] == 1:
        _down_vec[(triton.cdiv(weight.shape[0], 2),)](
            x, weight, out, weight.shape[0], weight.shape[1], 2, 2048, num_warps=4
        )
    else:
        torch.mm(x, weight.T, out=out)
    return out


@down_projection.register_fake
def _fake_down_projection(x, weight):
    return x.new_empty((x.shape[0], weight.shape[0]))
