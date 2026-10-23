"""Shape, dtype and layout predicates shared by the adapters.

These are deliberately explicit and cheap: every adapter must fall back to the
public vLLM path when a guard fails, and a failed guard must never be silent.
"""

import torch

BF16 = torch.bfloat16


def contiguous_bf16(*tensors):
    return all(tensor.dtype is BF16 and tensor.is_contiguous() for tensor in tensors)


def shapes(*pairs):
    """``shapes((tensor, (a, b)), ...)`` -> all shapes equal their expected tuples."""
    return all(tuple(tensor.shape) == tuple(expected) for tensor, expected in pairs)


def unquantized(layer):
    return type(layer.quant_method).__name__ == "UnquantizedLinearMethod"


def unbiased(*layers):
    return all(layer.bias is None for layer in layers)
