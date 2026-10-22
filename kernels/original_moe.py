"""Experimental BF16 Qwen MoE decode fusion; validated with vLLM 0.28.0.

The two-kernel strategy has public prior art; see LITERATURE.md. ``install``
adds a guarded single-token path without modifying installed package files.
"""

import torch, triton, triton.language as tl, json, statistics, os
from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts


@triton.jit
def up(
    X, W, IDS, Z, H: tl.constexpr, N: tl.constexpr, R: tl.constexpr, BH: tl.constexpr
):
    slot = tl.program_id(0)
    row = tl.program_id(1) * R + tl.arange(0, R)
    k = tl.arange(0, BH)
    expert = tl.load(IDS + slot)
    x = tl.load(X + k, k < H, 0).to(tl.float32)
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
    tl.store(Z + slot * N + row, z, row < N)


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
    s = tl.arange(0, TOP)
    k = tl.arange(0, BN)
    e = tl.load(IDS + s)
    wt = tl.load(TW + s)
    z = tl.load(Z + s[:, None] * N + k[None, :], k[None, :] < N, 0).to(tl.float32)
    w = tl.load(
        W + (e[None, :, None] * H + row[:, None, None]) * N + k[None, None, :],
        (row[:, None, None] < H) & (k[None, None, :] < N),
        0,
    ).to(tl.float32)
    a = tl.sum(w * z[None, :, :], 2) * wt[None, :]
    a = a.to(tl.bfloat16).to(tl.float32)
    y = tl.sum(a, 1)
    tl.store(Y + row, y, row < H)


def install():
    """Install the opt-in, version-specific TritonExperts.apply wrapper."""
    from vllm.model_executor.layers.fused_moe.experts.triton_moe import TritonExperts
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation

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
        supported = (
            hidden_states.shape == (1, 2048)
            and w1.shape == (256, 1024, 2048)
            and (w2.shape == (256, 2048, 512))
            and (topk_ids.shape == (1, 8))
            and (output.shape == (1, 2048))
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
            and output.is_contiguous()
            and (expert_map is None)
            and (global_num_experts in (-1, 256))
            and (a1q_scale is None)
            and (a2_scale is None)
            and (activation == MoEActivation.SILU)
            and (not apply_router_weight_on_input)
            and (self._lora_context is None)
            and (self.w1_bias is None)
            and (self.w2_bias is None)
            and (self.activation_config.clamp_limit is None)
        )

        if supported:
            if not getattr(self, "_jarvis_announced", False):
                print("JARVIS_MOE_SINGLE_TOKEN_ENABLED", flush=True)
                self._jarvis_announced = True

            z = torch.empty((8, 512), device=hidden_states.device, dtype=torch.bfloat16)
            up[8, 256](hidden_states, w1, topk_ids, z, 2048, 512, 2, 2048, num_warps=4)
            down[1024,](
                z, w2, topk_ids, topk_weights, output, 2048, 512, 8, 2, 512, num_warps=4
            )

            return None

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
