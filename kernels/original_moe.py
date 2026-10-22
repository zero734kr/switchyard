"""BF16 Qwen MoE decode fusion for vLLM 0.28.0.

The two-kernel strategy has public prior art; see LITERATURE.md. ``install``
adds a guarded batched decode path without modifying installed package files.
"""

import torch
import triton
import triton.language as tl

_ANNOUNCED_FALLBACKS = set()


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
    z = tl.load(
        Z + token_slots[:, None] * N + k[None, :], k[None, :] < N, 0
    ).to(tl.float32)
    w = tl.load(
        W + (e[None, :, None] * H + row[:, None, None]) * N + k[None, None, :],
        (row[:, None, None] < H) & (k[None, None, :] < N),
        0,
    ).to(tl.float32)
    a = tl.sum(w * z[None, :, :], 2) * wt[None, :]
    a = a.to(tl.bfloat16).to(tl.float32)
    y = tl.sum(a, 1)
    tl.store(Y + token * H + row, y, row < H)


def install():
    """Install a guarded TritonExperts.apply wrapper for decode batches."""
    from vllm.model_executor.layers.fused_moe.experts.triton_moe import TritonExperts
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation

    original = TritonExperts.apply
    up_rows = 2
    down_rows = 2
    max_tokens = 8

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
        num_tokens = hidden_states.shape[0] if hidden_states.ndim == 2 else 0
        supported = (
            hidden_states.ndim == 2
            and 1 <= num_tokens <= max_tokens
            and hidden_states.shape[1] == 2048
            and w1.shape == (256, 1024, 2048)
            and (w2.shape == (256, 2048, 512))
            and (topk_ids.shape == (num_tokens, 8))
            and (topk_weights.shape == (num_tokens, 8))
            and (output.shape == (num_tokens, 2048))
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
            if num_tokens not in _ANNOUNCED_FALLBACKS:
                capturing = torch.cuda.is_current_stream_capturing()
                print(
                    f"JARVIS_MOE_TRITON_DISPATCH tokens={num_tokens} "
                    f"cuda_graph_capture={capturing}",
                    flush=True,
                )
                _ANNOUNCED_FALLBACKS.add(num_tokens)

            z = torch.empty(
                (num_tokens, 8, 512),
                device=hidden_states.device,
                dtype=torch.bfloat16,
            )
            up[(triton.cdiv(512, up_rows), num_tokens * 8)](
                hidden_states,
                w1,
                topk_ids,
                z,
                2048,
                512,
                8,
                up_rows,
                2048,
                num_warps=4,
            )
            down[(triton.cdiv(2048, down_rows), num_tokens)](
                z,
                w2,
                topk_ids,
                topk_weights,
                output,
                2048,
                512,
                8,
                down_rows,
                512,
                num_warps=4,
            )

            return None

        if hidden_states.ndim == 2:
            fallback_key = (
                hidden_states.shape[0],
                tuple(hidden_states.shape),
                tuple(w1.shape),
                tuple(w2.shape),
                str(hidden_states.dtype),
            )
            if fallback_key not in _ANNOUNCED_FALLBACKS:
                print(
                    "JARVIS_MOE_TRITON_FALLBACK "
                    f"tokens={hidden_states.shape[0]} hidden={hidden_states.shape[1]} "
                    f"w1={tuple(w1.shape)} w2={tuple(w2.shape)} "
                    f"dtype={hidden_states.dtype} contiguous="
                    f"{hidden_states.is_contiguous()}/{w1.is_contiguous()}/"
                    f"{w2.is_contiguous()}/{output.is_contiguous()}",
                    flush=True,
                )
                _ANNOUNCED_FALLBACKS.add(fallback_key)

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
