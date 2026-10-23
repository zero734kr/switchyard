"""Qwen3.6-35B-A3B routed-expert adapter.

Adds a guarded decode path for ``TritonExperts.apply`` with the two-kernel
strategy described in ``kernels/README.md``. Every rejected shape falls back to
the installed implementation and reports why, once.
"""

import torch
import triton

from ..announce import announce
from ..ops.moe_experts import down, up

ATTRIBUTE = "_switchyard_moe"
UP_ROWS = 2
DOWN_ROWS = 2
MAX_TOKENS = 8
HIDDEN = 2048
INTERMEDIATE = 512
EXPERTS = 256
TOP_K = 8


def supported(self, hidden_states, w1, w2, topk_weights, topk_ids, output, **rest):
    """Every condition the fused path relies on, in one readable place."""
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation

    tokens = hidden_states.shape[0] if hidden_states.ndim == 2 else 0
    return (
        hidden_states.ndim == 2
        and 1 <= tokens <= MAX_TOKENS
        and hidden_states.shape[1] == HIDDEN
        and w1.shape == (EXPERTS, 2 * INTERMEDIATE, HIDDEN)
        and w2.shape == (EXPERTS, HIDDEN, INTERMEDIATE)
        and topk_ids.shape == (tokens, TOP_K)
        and topk_weights.shape == (tokens, TOP_K)
        and output.shape == (tokens, HIDDEN)
        and (
            hidden_states.dtype
            == w1.dtype
            == w2.dtype
            == output.dtype
            == torch.bfloat16
        )
        and hidden_states.is_contiguous()
        and w1.is_contiguous()
        and w2.is_contiguous()
        and topk_ids.is_contiguous()
        and topk_weights.is_contiguous()
        and output.is_contiguous()
        and rest["expert_map"] is None
        and rest["global_num_experts"] in (-1, EXPERTS)
        and rest["a1q_scale"] is None
        and rest["a2_scale"] is None
        and rest["activation"] == MoEActivation.SILU
        and not rest["apply_router_weight_on_input"]
        and self._lora_context is None
        and self.w1_bias is None
        and self.w2_bias is None
        and self.activation_config.clamp_limit is None
    )


def install():
    from vllm.model_executor.layers.fused_moe.experts.triton_moe import TritonExperts

    if getattr(TritonExperts, ATTRIBUTE, False):
        return
    original = TritonExperts.apply

    def apply(
        self,
        output,
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        activation,
        global_num_experts,
        expert_map,
        a1q_scale,
        a2_scale,
        workspace13,
        workspace2,
        expert_tokens_meta,
        apply_router_weight_on_input,
    ):
        arguments = {
            "activation": activation,
            "global_num_experts": global_num_experts,
            "expert_map": expert_map,
            "a1q_scale": a1q_scale,
            "a2_scale": a2_scale,
            "apply_router_weight_on_input": apply_router_weight_on_input,
        }
        if supported(
            self, hidden_states, w1, w2, topk_weights, topk_ids, output, **arguments
        ):
            tokens = hidden_states.shape[0]
            announce(
                "JARVIS_MOE_TRITON_DISPATCH",
                key=tokens,
                tokens=tokens,
                cuda_graph_capture=torch.cuda.is_current_stream_capturing(),
            )
            intermediate = torch.empty(
                (tokens, TOP_K, INTERMEDIATE),
                device=hidden_states.device,
                dtype=torch.bfloat16,
            )
            up[(triton.cdiv(INTERMEDIATE, UP_ROWS), tokens * TOP_K)](
                hidden_states,
                w1,
                topk_ids,
                intermediate,
                HIDDEN,
                INTERMEDIATE,
                TOP_K,
                UP_ROWS,
                HIDDEN,
                num_warps=4,
            )
            down[(triton.cdiv(HIDDEN, DOWN_ROWS), tokens)](
                intermediate,
                w2,
                topk_ids,
                topk_weights,
                output,
                HIDDEN,
                INTERMEDIATE,
                TOP_K,
                DOWN_ROWS,
                INTERMEDIATE,
                num_warps=4,
            )
            return None

        if hidden_states.ndim == 2:
            key = (
                hidden_states.shape[0],
                tuple(hidden_states.shape),
                tuple(w1.shape),
                tuple(w2.shape),
                str(hidden_states.dtype),
            )
            announce(
                "JARVIS_MOE_TRITON_FALLBACK",
                key=key,
                tokens=hidden_states.shape[0],
                hidden=hidden_states.shape[1],
                w1=tuple(w1.shape),
                w2=tuple(w2.shape),
                dtype=hidden_states.dtype,
                contiguous=(
                    f"{hidden_states.is_contiguous()}/{w1.is_contiguous()}/"
                    f"{w2.is_contiguous()}/{output.is_contiguous()}"
                ),
            )

        return original(
            self,
            output,
            hidden_states,
            w1,
            w2,
            topk_weights,
            topk_ids,
            activation,
            global_num_experts,
            expert_map,
            a1q_scale,
            a2_scale,
            workspace13,
            workspace2,
            expert_tokens_meta,
            apply_router_weight_on_input,
        )

    TritonExperts.apply = apply
    setattr(TritonExperts, ATTRIBUTE, True)
