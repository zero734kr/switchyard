"""Qwen3.6-27B dense MLP adapter.

Wraps ``Qwen2MoeMLP.forward`` for the 27B shape only. The guard is extensive on
purpose: anything unexpected must reach the public vLLM path unmodified.
"""

import torch

from .. import guards
from ..ops.gate_up import gate_up

EXPECTED_GATE_UP = (34816, 5120)
ATTRIBUTE = "_switchyard_dense"


def install():
    from vllm.model_executor.models.qwen2_moe import Qwen2MoeMLP

    if getattr(Qwen2MoeMLP, ATTRIBUTE, False):
        return
    original = Qwen2MoeMLP.forward

    def forward(self, x):
        weight = self.gate_up_proj.weight
        supported = (
            guards.shapes((weight, EXPECTED_GATE_UP))
            and weight.dtype is torch.bfloat16
            and weight.is_contiguous()
            and self.expert_gate is None
            and guards.unquantized(self.gate_up_proj)
            and self.gate_up_proj.bias is None
        )
        if not supported:
            return original(self, x)
        return self.down_proj(gate_up(x, weight))[0]

    Qwen2MoeMLP.forward = forward
    setattr(Qwen2MoeMLP, ATTRIBUTE, True)
