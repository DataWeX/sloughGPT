"""
Clean benchmark: SloNet (CPU backend) vs PyTorch on real workloads.
"""

import time, sys, os, math
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from domains.training.slonet import Tensor, no_grad, SloLSTM, SloLinear, SloEmbedding
from domains.slolib.gpu import _CPUBackend

cpu = _CPUBackend()

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("Torch not available")

WARMUP = 10
ITERS = 50

def bench(name, fn_sn, fn_th=None):
    for _ in range(WARMUP):
        fn_sn()
    t = []
    for _ in range(ITERS):
        t0 = time.perf_counter_ns()
        fn_sn()
        t.append(time.perf_counter_ns() - t0)
    sn_ms = np.median(t) / 1e6
    label = f"  {name:<45} SloNet: {sn_ms:>8.2f} ms"
    if fn_th:
        for _ in range(WARMUP):
            fn_th()
        t = []
        for _ in range(ITERS):
            t0 = time.perf_counter_ns()
            fn_th()
            t.append(time.perf_counter_ns() - t0)
        th_ms = np.median(t) / 1e6
        ratio = sn_ms / th_ms if th_ms > 0 else 0
        label += f"  Torch: {th_ms:>8.2f} ms  {'🏆 FASTER' if ratio < 1 else f'{ratio:.2f}x slower'}"
    print(label)
    return sn_ms

print("=" * 80)
print("SloNet vs PyTorch — CPU Performance Benchmark")
print("=" * 80)

# --- Matmul ---
print("\n--- Matmul ---")
for M, N, K in [(256, 256, 256), (512, 512, 512), (1024, 1024, 1024)]:
    A = np.random.randn(M, K).astype(np.float32)
    B = np.random.randn(K, N).astype(np.float32)
    def make_sn(a, b):
        def sn():
            with no_grad():
                (Tensor(a) @ Tensor(b)).data
        return sn
    def make_th(a, b):
        ta = torch.from_numpy(a)
        tb = torch.from_numpy(b)
        def th():
            with torch.no_grad():
                (ta @ tb).numpy()
        return th
    bench(f"matmul ({M}x{K}) @ ({K}x{N})",
          make_sn(A, B), make_th(A, B) if HAS_TORCH else None)

# --- LSTM ---
print("\n--- LSTM ---")
for vocab, seq, hidden in [(256, 16, 256), (1000, 32, 512)]:
    x_np = np.random.randint(0, vocab, (1, seq)).astype(np.int64)
    
    sn_lstm = SloLSTM(vocab, 256, hidden, 2, 0.0)
    def make_sn(m, x):
        def sn():
            with no_grad():
                m.forward(Tensor(x))[0].data
        return sn
    
    if HAS_TORCH:
        class T_LSTM(nn.Module):
            def __init__(self):
                super().__init__()
                self.emb = nn.Embedding(vocab, 256)
                self.lstm = nn.LSTM(256, hidden, 2, batch_first=True)
                self.fc = nn.Linear(hidden, vocab)
            def forward(self, x):
                x = self.emb(x)
                x, _ = self.lstm(x)
                return self.fc(x[:, -1, :])
        th_lstm = T_LSTM().eval()
        tx = torch.from_numpy(x_np).long()
        def make_th(m, x):
            def th():
                with torch.no_grad():
                    m(x).numpy()
            return th
        bench(f"LSTM 2L-{hidden} ({vocab}v/{seq}s)",
              make_sn(sn_lstm, x_np), make_th(th_lstm, tx))
    else:
        bench(f"LSTM 2L-{hidden} ({vocab}v/{seq}s)",
              make_sn(sn_lstm, x_np))

# --- Attention (CPU backend only) ---
print("\n--- Attention (CPU) ---")
for B, H, N, E in [(4, 4, 32, 16), (8, 4, 128, 64)]:
    S = N
    q = np.random.randn(B, H, N, E).astype(np.float32)
    k = np.random.randn(B, H, S, E).astype(np.float32)
    v = np.random.randn(B, H, S, E).astype(np.float32)
    scale = 1.0 / math.sqrt(E)
    
    def make_sn(qn, kn, vn):
        def sn():
            cpu.scaled_dot_attention(qn, kn, vn)
        return sn
    
    if HAS_TORCH:
        tq = torch.from_numpy(q)
        tk = torch.from_numpy(k)
        tv = torch.from_numpy(v)
        def make_th(qq, kk, vv):
            def th():
                with torch.no_grad():
                    s = torch.einsum("bhnk,bhsk->bhns", qq, kk) * scale
                    a = torch.softmax(s, dim=-1)
                    torch.einsum("bhns,bhsk->bhnk", a, vv).numpy()
            return th
        bench(f"attention {B}x{H}x{N}x{E}",
              make_sn(q, k, v), make_th(tq, tk, tv))
    else:
        bench(f"attention {B}x{H}x{N}x{E}",
              make_sn(q, k, v))

# --- Fused Ops ---
print("\n--- Fused Ops ---")
X = np.random.randn(128, 3072).astype(np.float32)
W = np.random.randn(3072).astype(np.float32)
B = np.random.randn(3072).astype(np.float32)

bench("layer_norm+gelu (fused)", lambda: cpu.fused_layer_norm_gelu(X, W, B))
bench("layer_norm+gelu (separate)", lambda: cpu.gelu(cpu.layer_norm(X, W, B)))

print("\n" + "=" * 80)
