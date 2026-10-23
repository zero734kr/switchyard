"""One launch for the GDN QKVZ and BA projections.

Measured operator; kept verbatim from the probe it was timed with.
"""

import triton
import triton.language as tl


@triton.jit
def project_pair(
    X,
    WQ,
    WB,
    OQ,
    OB,
    H: tl.constexpr,
    DQ: tl.constexpr,
    DB: tl.constexpr,
    BK: tl.constexpr,
    R: tl.constexpr,
):
    pid = tl.program_id(0)
    k = tl.arange(0, BK)
    x = tl.load(X + k, k < H, 0).to(tl.float32)

    if pid < tl.cdiv(DQ, R):
        row = pid * R + tl.arange(0, R)
        w = tl.load(
            WQ + row[:, None] * H + k[None, :],
            (row[:, None] < DQ) & (k[None, :] < H),
            0,
        ).to(tl.float32)
        y = tl.sum(w * x[None, :], 1)
        tl.store(OQ + row, y, row < DQ)
    else:
        row = (pid - tl.cdiv(DQ, R)) * R + tl.arange(0, R)
        w = tl.load(
            WB + row[:, None] * H + k[None, :],
            (row[:, None] < DB) & (k[None, :] < H),
            0,
        ).to(tl.float32)
        y = tl.sum(w * x[None, :], 1)
        tl.store(OB + row, y, row < DB)
