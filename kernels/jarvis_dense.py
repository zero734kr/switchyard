"""Opt-in experimental BF16 Qwen 27B dense single-token gate/up fusion."""
import os, sys, pathlib, json
import torch
import torch.nn.functional as F
import triton
import triton.language as tl

@triton.jit
def _gate_up(X, W, Y, H: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    row = tl.program_id(0)
    k = tl.arange(0, K)
    x = tl.load(X + k, k < H, 0).to(tl.float32)
    g = tl.load(W + row * H + k, k < H, 0).to(tl.float32)
    u = tl.load(W + (N + row) * H + k, k < H, 0).to(tl.float32)
    a = tl.sum(g * x, 0).to(tl.bfloat16).to(tl.float32)
    b = tl.sum(u * x, 0).to(tl.bfloat16).to(tl.float32)
    tl.store(Y + row, a * tl.sigmoid(a) * b)

@triton.jit
def _gate_up_batched(X, W, Y, B: tl.constexpr, N: tl.constexpr, H: tl.constexpr,
                     BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pm = tl.program_id(1)
    pn = tl.program_id(0)
    rows = pm * BM + tl.arange(0, BM)
    cols = pn * BN + tl.arange(0, BN)
    ks = tl.arange(0, BK)
    acc_g = tl.zeros((BM, BN), tl.float32)
    acc_u = tl.zeros((BM, BN), tl.float32)
    for kt in range(tl.cdiv(H, BK)):
        k = kt * BK + ks
        x = tl.load(X + rows[:, None] * H + k[None, :],
                    (rows[:, None] < B) & (k[None, :] < H), 0)
        wg = tl.load(W + cols[None, :] * H + k[:, None],
                     (cols[None, :] < N) & (k[:, None] < H), 0)
        wu = tl.load(W + (N + cols[None, :]) * H + k[:, None],
                     (cols[None, :] < N) & (k[:, None] < H), 0)
        acc_g = tl.dot(x, wg, acc_g)
        acc_u = tl.dot(x, wu, acc_u)
    gate = acc_g.to(tl.bfloat16).to(tl.float32)
    up = acc_u.to(tl.bfloat16).to(tl.float32)
    result = gate * tl.sigmoid(gate) * up
    tl.store(Y + rows[:, None] * N + cols[None, :], result,
             (rows[:, None] < B) & (cols[None, :] < N))

@triton.jit
def _down_vec(X, W, Y, H: tl.constexpr, N: tl.constexpr,
              R: tl.constexpr, BK: tl.constexpr):
    row = tl.program_id(0) * R + tl.arange(0, R)
    k = tl.arange(0, BK)
    acc = tl.full((R,), 0.0, tl.float32)
    for tile in range(tl.cdiv(N, BK)):
        col = tile * BK + k
        x = tl.load(X + col, col < N, 0).to(tl.float32)
        w = tl.load(W + row[:, None] * N + col[None, :],
                    (row[:, None] < H) & (col[None, :] < N), 0).to(tl.float32)
        acc += tl.sum(w * x[None, :], 1)
    tl.store(Y + row, acc, row < H)

_seen_batches = set()
_public_impl = os.environ.get('JARVIS_DENSE_IMPL') == 'public'
_down_impl = os.environ.get('JARVIS_DENSE_DOWN') == '1'
if _public_impl:
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
    from upstream_a2 import gatedmlp_separated_a2_vec_kernel as _upstream
    _configs=[]
    for _path in pathlib.Path(__file__).resolve().parent.parent.glob('public-probe*.log'):
        for _line in _path.read_text().splitlines():
            try:
                _r=json.loads(_line)
                if _r.get('source')=='DeepFusionKernel_A2' and 'median_us' in _r:
                    _configs.append(_r)
            except ValueError:
                pass
    _best=min(_configs,key=lambda r:r['median_us'])
    print(f'JARVIS_PUBLIC_CONFIG {_best}',flush=True)

@torch.library.custom_op('jarvis_dense::gate_up', mutates_args=())
def gate_up(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    n = weight.shape[0] // 2
    batch = x.shape[0]
    if batch == 1 and _public_impl:
        path = "public_a2"
    elif batch == 1:
        path = "triton_b1"
    elif not _public_impl and 1 < batch <= 16 and x.is_contiguous():
        path = "triton_batched"
    else:
        path = "vllm_fallback"
    if batch not in _seen_batches:
        print(f'JARVIS_DENSE_DISPATCH batch={batch} custom={path.startswith("triton")} path={path}', flush=True)
        _seen_batches.add(batch)
    out = x.new_empty((batch, n))
    if path == "public_a2":
        x3, y3 = x.view(1, 1, 5120), out.view(1, 1, n)
        g, u = weight[:n].T, weight[n:].T
        _upstream[(triton.cdiv(n, _best['BLOCK_K']), 1)](x3,u,g,y3,*x3.stride(),*u.stride(),*g.stride(),*y3.stride(),1,1,5120,n,BLOCK_N=_best['BLOCK_N'],BLOCK_K=_best['BLOCK_K'],num_warps=_best['warps'],num_stages=_best['stages'])
    elif path == "triton_b1":
        _gate_up[(n,)](x, weight, out, 5120, n, 8192, num_warps=4)
    elif path == "triton_batched":
        bm, bn, bk = 16, 32, 64
        _gate_up_batched[(triton.cdiv(n, bn), triton.cdiv(batch, bm))](
            x, weight, out, batch, n, 5120, bm, bn, bk, num_warps=4)
    else:
        raw = F.linear(x, weight)
        torch.ops._C.silu_and_mul(out, raw)
    return out

@gate_up.register_fake
def _fake(x, weight):
    return x.new_empty((x.shape[0], weight.shape[0] // 2))

@torch.library.custom_op('jarvis_dense::down_projection', mutates_args=())
def down_projection(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    out = x.new_empty((x.shape[0], weight.shape[0]))
    if x.shape[0] == 1:
        _down_vec[(triton.cdiv(weight.shape[0], 2),)](
            x, weight, out, weight.shape[0], weight.shape[1], 2, 2048,
            num_warps=4)
    else:
        torch.mm(x, weight.T, out=out)
    return out

@down_projection.register_fake
def _fake_down(x, weight):
    return x.new_empty((x.shape[0], weight.shape[0]))

def install():
    from vllm.model_executor.models.qwen2_moe import Qwen2MoeMLP
    if getattr(Qwen2MoeMLP, '_jarvis_dense', False):
        return
    original = Qwen2MoeMLP.forward
    if _down_impl:
        print('JARVIS_DENSE_DOWN enabled rows=2 bk=2048 split=1', flush=True)
    def forward(self, x):
        w = self.gate_up_proj.weight
        if (tuple(w.shape) == (34816, 5120)
                and w.dtype == torch.bfloat16 and w.is_contiguous()
                and self.expert_gate is None
                and type(self.gate_up_proj.quant_method).__name__ == 'UnquantizedLinearMethod'
                and self.gate_up_proj.bias is None):
            hidden = gate_up(x, w)
            dw = self.down_proj.weight
            if (_down_impl and hidden.shape[0] == 1
                    and tuple(dw.shape) == (5120, 17408)
                    and dw.dtype == torch.bfloat16 and dw.is_contiguous()
                    and type(self.down_proj.quant_method).__name__ == 'UnquantizedLinearMethod'
                    and self.down_proj.bias is None):
                return down_projection(hidden, dw)
            return self.down_proj(hidden)[0]
        return original(self, x)
    Qwen2MoeMLP.forward = forward
    Qwen2MoeMLP._jarvis_dense = True
