"""Routed-expert gate/up + SwiGLU, then down projection + top-k reduction.

Two kernels instead of the public sort/pad/grouped-GEMM path, specialized for
single-token decode with eight experts out of 256. The two-kernel strategy has
public prior art; see ``kernels/README.md``.
"""

import triton
import triton.language as tl


@triton.jit
def up(
    X,
    W,
    IDS,
    Z,
    H: tl.constexpr,
    N: tl.constexpr,
    TOP: tl.constexpr,
    R: tl.constexpr,
    BH: tl.constexpr,
):
    row = tl.program_id(0) * R + tl.arange(0, R)
    token_slot = tl.program_id(1)
    token = token_slot // TOP
    k = tl.arange(0, BH)
    expert = tl.load(IDS + token_slot)
    x = tl.load(X + token * H + k, k < H, 0).to(tl.float32)
    g = tl.load(
        W + (expert * 2 * N + row[:, None]) * H + k[None, :],
        (row[:, None] < N) & (k[None, :] < H),
        0,
    ).to(tl.float32)
    u = tl.load(
        W + (expert * 2 * N + N + row[:, None]) * H + k[None, :],
        (row[:, None] < N) & (k[None, :] < H),
        0,
    ).to(tl.float32)
    a = tl.sum(g * x[None, :], 1).to(tl.bfloat16).to(tl.float32)
    b = tl.sum(u * x[None, :], 1).to(tl.bfloat16).to(tl.float32)
    z = a * tl.sigmoid(a) * b
    tl.store(Z + token_slot * N + row, z, row < N)


@triton.jit
def down(
    Z,
    W,
    IDS,
    TW,
    Y,
    H: tl.constexpr,
    N: tl.constexpr,
    TOP: tl.constexpr,
    R: tl.constexpr,
    BN: tl.constexpr,
):
    row = tl.program_id(0) * R + tl.arange(0, R)
    token = tl.program_id(1)
    s = tl.arange(0, TOP)
    k = tl.arange(0, BN)
    token_slots = token * TOP + s
    e = tl.load(IDS + token_slots)
    wt = tl.load(TW + token_slots)
    z = tl.load(Z + token_slots[:, None] * N + k[None, :], k[None, :] < N, 0).to(
        tl.float32
    )
    w = tl.load(
        W + (e[None, :, None] * H + row[:, None, None]) * N + k[None, None, :],
        (row[:, None, None] < H) & (k[None, None, :] < N),
        0,
    ).to(tl.float32)
    a = tl.sum(w * z[None, :, :], 2) * wt[None, :]
    a = a.to(tl.bfloat16).to(tl.float32)
    y = tl.sum(a, 1)
    tl.store(Y + token * H + row, y, row < H)
