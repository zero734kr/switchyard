"""FP32 GDN recurrent-state update for a single decode token.

This is the prior custom variant of the public packed-decode kernel; the
adapter currently launches the public kernel with a retuned configuration
instead, and keeps this one for the algebraic and fused-norm comparisons.
"""

import triton
import triton.language as tl
from vllm.third_party.flash_linear_attention.ops.op import exp as public_exp


@triton.jit
def step(
    QKV,
    A,
    B,
    AL,
    DT,
    STATE,
    IDS,
    Z,
    W,
    OUT,
    HK: tl.constexpr,
    HV: tl.constexpr,
    BV: tl.constexpr,
    FUSE_NORM: tl.constexpr,
    ALGEBRAIC: tl.constexpr,
    SS: tl.constexpr = 0,
):
    tile = tl.program_id(0)
    nh = tl.program_id(1)
    n, hv = nh // HV, nh % HV
    hk = hv // (HV // HK)
    slot = tl.load(IDS + n).to(tl.int64)
    v = tile * BV + tl.arange(0, BV)
    k = tl.arange(0, 128)
    if slot <= 0:
        tl.store(OUT + nh * 128 + v, 0.0, v < 128)
        return
    q = tl.load(QKV + n * (2 * HK + HV) * 128 + hk * 128 + k).to(tl.float32)
    key = tl.load(QKV + n * (2 * HK + HV) * 128 + (HK + hk) * 128 + k).to(tl.float32)
    value = tl.load(
        QKV + n * (2 * HK + HV) * 128 + 2 * HK * 128 + hv * 128 + v, v < 128, 0
    ).to(tl.float32)
    q = q / tl.sqrt(tl.sum(q * q, 0) + 1.0e-6) * (128.0**-0.5)
    key = key / tl.sqrt(tl.sum(key * key, 0) + 1.0e-6)
    a = tl.load(A + nh).to(tl.float32)
    b = tl.load(B + nh).to(tl.float32)
    al = tl.load(AL + hv).to(tl.float32)
    dt = tl.load(DT + hv).to(tl.float32)
    x = a + dt
    softplus = tl.where(x <= 20.0, tl.log(1.0 + tl.exp(x)), x)
    decay = public_exp(-tl.exp(al) * softplus)
    beta = tl.sigmoid(b).to(tl.bfloat16).to(tl.float32)
    state_stride: tl.constexpr = SS if SS else HV * 16384
    ptr = STATE + slot * state_stride + hv * 16384 + v[:, None] * 128 + k[None, :]
    old = tl.load(ptr, v[:, None] < 128, 0).to(tl.float32) * decay
    delta = (value - tl.sum(old * key[None, :], 1)) * beta
    updated = old + delta[:, None] * key[None, :]
    if ALGEBRAIC:
        raw = tl.sum(old * q[None, :], 1) + delta * tl.sum(key * q, 0)
    else:
        raw = tl.sum(updated * q[None, :], 1)
    tl.store(ptr, updated, v[:, None] < 128)
    raw = raw.to(tl.bfloat16).to(tl.float32)
    if FUSE_NORM:
        inv = tl.rsqrt(tl.sum(raw * raw, 0) / 128.0 + 1.0e-6)
        weight = tl.load(W + v).to(tl.float32)
        z = tl.load(Z + nh * 128 + v).to(tl.float32)
        raw = raw * inv * weight * (z * tl.sigmoid(z))
    tl.store(OUT + nh * 128 + v, raw, v < 128)
