"""Fused BF16 shared-expert gate, multiply, and routed-output combine."""

import torch
import triton
import triton.language as tl


@triton.jit
def _shared_gate_mul(
    gate_ptr,
    shared_ptr,
    output_ptr,
    H: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    row = tl.program_id(1)
    cols = tl.program_id(0) * BLOCK_H + tl.arange(0, BLOCK_H)
    mask = cols < H

    gate = tl.load(gate_ptr + row).to(tl.float32)
    sigmoid_bf16 = (1.0 / (1.0 + tl.exp(-gate))).to(tl.bfloat16)
    shared = tl.load(shared_ptr + row * H + cols, mask, other=0).to(tl.bfloat16)
    result = (sigmoid_bf16.to(tl.float32) * shared.to(tl.float32)).to(tl.bfloat16)
    tl.store(output_ptr + row * H + cols, result, mask)


@triton.jit
def _shared_gate_mul_add(
    gate_ptr,
    shared_ptr,
    routed_ptr,
    output_ptr,
    H: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    row = tl.program_id(1)
    cols = tl.program_id(0) * BLOCK_H + tl.arange(0, BLOCK_H)
    mask = cols < H

    gate = tl.load(gate_ptr + row).to(tl.float32)
    sigmoid_bf16 = (1.0 / (1.0 + tl.exp(-gate))).to(tl.bfloat16)
    shared = tl.load(shared_ptr + row * H + cols, mask, other=0).to(tl.bfloat16)
    routed = tl.load(routed_ptr + row * H + cols, mask, other=0).to(tl.bfloat16)

    # Match the public BF16 sequence: sigmoid -> BF16 multiply -> BF16 add.
    gated_shared = (sigmoid_bf16.to(tl.float32) * shared.to(tl.float32)).to(tl.bfloat16)
    result = (gated_shared.to(tl.float32) + routed.to(tl.float32)).to(tl.bfloat16)
    tl.store(output_ptr + row * H + cols, result, mask)


def shared_gate_mul(gate_logits, shared_output, output=None, block_h=256):
    """Return sigmoid(gate_logits) * shared_output with explicit BF16 rounding."""
    if shared_output.dtype != torch.bfloat16 or not shared_output.is_contiguous():
        raise TypeError("shared output must be contiguous BF16")
    if gate_logits.dtype != torch.bfloat16 or not gate_logits.is_contiguous():
        raise TypeError("gate logits must be contiguous BF16")
    if shared_output.ndim != 2 or gate_logits.numel() != shared_output.shape[0]:
        raise ValueError("expected one scalar gate logit per [tokens, hidden] row")

    tokens, hidden = shared_output.shape
    if output is None:
        output = torch.empty_like(shared_output)
    elif (
        output.shape != shared_output.shape
        or output.dtype != torch.bfloat16
        or not output.is_contiguous()
    ):
        raise ValueError("output must be contiguous BF16 with the same shape")
    _shared_gate_mul[(triton.cdiv(hidden, block_h), tokens)](
        gate_logits,
        shared_output,
        output,
        hidden,
        block_h,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return output


@triton.jit
def _shared_mul_add(
    sigmoid_ptr,
    shared_ptr,
    routed_ptr,
    output_ptr,
    H: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    row = tl.program_id(1)
    cols = tl.program_id(0) * BLOCK_H + tl.arange(0, BLOCK_H)
    mask = cols < H

    sigmoid_bf16 = tl.load(sigmoid_ptr + row).to(tl.bfloat16)
    shared = tl.load(shared_ptr + row * H + cols, mask, other=0).to(tl.bfloat16)
    routed = tl.load(routed_ptr + row * H + cols, mask, other=0).to(tl.bfloat16)
    gated_shared = (sigmoid_bf16.to(tl.float32) * shared.to(tl.float32)).to(tl.bfloat16)
    result = (gated_shared.to(tl.float32) + routed.to(tl.float32)).to(tl.bfloat16)
    tl.store(output_ptr + row * H + cols, result, mask)


def shared_gate_mul_add(
    gate_logits, shared_output, routed_output, output=None, block_h=256
):
    """Return sigmoid(gate_logits) * shared_output + routed_output.

    The operation is specialized for contiguous BF16 tensors with gate shape
    [tokens, 1] (or [tokens]) and shared/routed/output shape [tokens, hidden].
    """
    if shared_output.dtype != torch.bfloat16 or routed_output.dtype != torch.bfloat16:
        raise TypeError("shared/routed outputs must be BF16")
    if not shared_output.is_contiguous() or not routed_output.is_contiguous():
        raise ValueError("shared/routed outputs must be contiguous")
    if gate_logits.dtype != torch.bfloat16 or not gate_logits.is_contiguous():
        raise TypeError("gate logits must be contiguous BF16")
    if gate_logits.numel() != shared_output.shape[0]:
        raise ValueError("expected one scalar gate logit per token")
    if shared_output.shape != routed_output.shape or shared_output.ndim != 2:
        raise ValueError(
            "shared and routed outputs must have the same [tokens, hidden] shape"
        )

    tokens, hidden = shared_output.shape
    if output is None:
        output = torch.empty_like(routed_output)
    elif (
        output.shape != routed_output.shape
        or output.dtype != torch.bfloat16
        or not output.is_contiguous()
    ):
        raise ValueError("output must be contiguous BF16 with the same shape")
    _shared_gate_mul_add[(triton.cdiv(hidden, block_h), tokens)](
        gate_logits,
        shared_output,
        routed_output,
        output,
        hidden,
        block_h,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return output


def shared_mul_add(
    sigmoid_gate, shared_output, routed_output, output=None, block_h=256
):
    """Return sigmoid_gate * shared_output + routed_output.

    sigmoid_gate is produced by the public PyTorch sigmoid to preserve its
    BF16 result. The Triton kernel fuses the following multiply and add.
    """
    if sigmoid_gate.dtype != torch.bfloat16 or not sigmoid_gate.is_contiguous():
        raise TypeError("sigmoid gate must be contiguous BF16")
    if shared_output.dtype != torch.bfloat16 or routed_output.dtype != torch.bfloat16:
        raise TypeError("shared/routed outputs must be BF16")
    if not shared_output.is_contiguous() or not routed_output.is_contiguous():
        raise ValueError("shared/routed outputs must be contiguous")
    if sigmoid_gate.numel() != shared_output.shape[0]:
        raise ValueError("expected one sigmoid gate per token")
    if shared_output.shape != routed_output.shape or shared_output.ndim != 2:
        raise ValueError(
            "shared and routed outputs must have the same [tokens, hidden] shape"
        )

    tokens, hidden = shared_output.shape
    if output is None:
        output = torch.empty_like(routed_output)
    elif (
        output.shape != routed_output.shape
        or output.dtype != torch.bfloat16
        or not output.is_contiguous()
    ):
        raise ValueError("output must be contiguous BF16 with the same shape")
    _shared_mul_add[(triton.cdiv(hidden, block_h), tokens)](
        sigmoid_gate,
        shared_output,
        routed_output,
        output,
        hidden,
        block_h,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return output
