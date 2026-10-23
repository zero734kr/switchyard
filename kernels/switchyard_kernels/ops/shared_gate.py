"""Fused BF16 shared-expert gate, multiply and routed-output combine.

Not installed by any adapter yet: it is kept as a measured operator, not as a
serving path. The three entry points differ only in which inputs already exist,
so they share one kernel and one validator.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _gate_mul_add(
    gate_ptr,
    shared_ptr,
    routed_ptr,
    output_ptr,
    H: tl.constexpr,
    BLOCK_H: tl.constexpr,
    ADD_ROUTED: tl.constexpr,
    GATE_IS_SIGMOID: tl.constexpr,
):
    row = tl.program_id(1)
    cols = tl.program_id(0) * BLOCK_H + tl.arange(0, BLOCK_H)
    mask = cols < H

    if GATE_IS_SIGMOID:
        sigmoid_bf16 = tl.load(gate_ptr + row).to(tl.bfloat16)
    else:
        gate = tl.load(gate_ptr + row).to(tl.float32)
        sigmoid_bf16 = (1.0 / (1.0 + tl.exp(-gate))).to(tl.bfloat16)
    shared = tl.load(shared_ptr + row * H + cols, mask, other=0).to(tl.bfloat16)
    # Match the public BF16 sequence: sigmoid -> BF16 multiply -> BF16 add.
    gated_shared = (sigmoid_bf16.to(tl.float32) * shared.to(tl.float32)).to(tl.bfloat16)
    if ADD_ROUTED:
        routed = tl.load(routed_ptr + row * H + cols, mask, other=0).to(tl.bfloat16)
        result = (gated_shared.to(tl.float32) + routed.to(tl.float32)).to(tl.bfloat16)
    else:
        result = gated_shared
    tl.store(output_ptr + row * H + cols, result, mask)


SIGMOID_GATE = "sigmoid gate"
GATE_LOGITS = "gate logits"


def _launch(gate, shared, routed, output, block_h, *, add_routed, gate_is_sigmoid):
    gate_name = SIGMOID_GATE if gate_is_sigmoid else GATE_LOGITS
    if gate.dtype != torch.bfloat16 or not gate.is_contiguous():
        raise TypeError(f"{gate_name} must be contiguous BF16")
    if shared.dtype != torch.bfloat16 or not shared.is_contiguous():
        raise TypeError("shared output must be contiguous BF16")
    if shared.ndim != 2:
        raise ValueError("shared output must be [tokens, hidden]")
    if gate.numel() != shared.shape[0]:
        raise ValueError(
            f"expected one {gate_name} per [tokens, hidden] row"
            if not gate_is_sigmoid
            else "expected one sigmoid gate per token"
        )
    if add_routed:
        if routed.dtype != torch.bfloat16:
            raise TypeError("routed outputs must be BF16")
        if not routed.is_contiguous():
            raise ValueError("routed outputs must be contiguous")
        if routed.shape != shared.shape:
            raise ValueError(
                "shared and routed outputs must have the same [tokens, hidden] shape"
            )
    reference = routed if add_routed else shared
    if output is None:
        output = torch.empty_like(reference)
    elif (
        output.shape != reference.shape
        or output.dtype != torch.bfloat16
        or not output.is_contiguous()
    ):
        raise ValueError("output must be contiguous BF16 with the same shape")
    tokens, hidden = shared.shape
    _gate_mul_add[(triton.cdiv(hidden, block_h), tokens)](
        gate,
        shared,
        routed if add_routed else shared,
        output,
        hidden,
        block_h,
        add_routed,
        gate_is_sigmoid,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return output


def shared_gate_mul(gate_logits, shared_output, output=None, block_h=256):
    """Return sigmoid(gate_logits) * shared_output with explicit BF16 rounding."""
    return _launch(
        gate_logits,
        shared_output,
        shared_output,
        output,
        block_h,
        add_routed=False,
        gate_is_sigmoid=False,
    )


def shared_gate_mul_add(
    gate_logits, shared_output, routed_output, output=None, block_h=256
):
    """Return sigmoid(gate_logits) * shared_output + routed_output."""
    return _launch(
        gate_logits,
        shared_output,
        routed_output,
        output,
        block_h,
        add_routed=True,
        gate_is_sigmoid=False,
    )


def shared_mul_add(
    sigmoid_gate, shared_output, routed_output, output=None, block_h=256
):
    """Return sigmoid_gate * shared_output + routed_output.

    ``sigmoid_gate`` comes from the public PyTorch sigmoid so its BF16 result is
    preserved; the kernel fuses the following multiply and add.
    """
    return _launch(
        sigmoid_gate,
        shared_output,
        routed_output,
        output,
        block_h,
        add_routed=True,
        gate_is_sigmoid=True,
    )
