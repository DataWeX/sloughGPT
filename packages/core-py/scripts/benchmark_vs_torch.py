"""
Benchmark SloNet vs PyTorch on real neural network workloads.
Tests: LSTM forward, matmul, attention, layer_norm, softmax, full model.
"""

import time
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from domains.training.slonet import (Tensor, no_grad, zeros, SloLSTM,
    SloEmbedding, SloLinear, softmax, cross_entropy, gelu, silu)

# Check if PyTorch is available
try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("Torch not available — benchmarking SloNet only\n")

print("=" * 70)
print("SloNet vs PyTorch Benchmark — Intel CPU")
print("=" * 70)

WARMUP = 3
ITERATIONS = 20


def benchmark(name, fn_slonet, fn_torch=None, iters=ITERATIONS):
    """Benchmark SloNet function against equivalent PyTorch."""
    # Warmup
    for _ in range(WARMUP):
        fn_slonet()

    # Benchmark SloNet
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn_slonet()
        times.append(time.perf_counter() - t0)
    sn_time = np.median(times) * 1000  # ms

    result = f"  {name:<40} SloNet: {sn_time:>7.2f} ms"

    if fn_torch and HAS_TORCH:
        for _ in range(WARMUP):
            fn_torch()
        times = []
        for _ in range(iters):
            t0 = time.perf_counter()
            fn_torch()
            times.append(time.perf_counter() - t0)
        th_time = np.median(times) * 1000
        ratio = th_time / sn_time if sn_time > 0 else float('inf')
        result += f"  Torch: {th_time:>7.2f} ms  SN/Torch: {ratio:.2f}x"
        if ratio > 1.0:
            result += "  🏆 FASTER"
        else:
            result += f"  ({1/ratio:.2f}x slower)"
    else:
        result += "  (no torch comparison)"

    print(result)
    return sn_time


def make_matmul(M, N, K):
    """Matmul benchmark: (M,K) @ (K,N)"""
    A = np.random.randn(M, K).astype(np.float32)
    B = np.random.randn(K, N).astype(np.float32)

    def sn():
        with no_grad():
            ta = Tensor(A, requires_grad=False)
            tb = Tensor(B, requires_grad=False)
            tc = ta @ tb
            _ = tc.data

    def th():
        ta = torch.from_numpy(A)
        tb = torch.from_numpy(B)
        tc = ta @ tb
        _ = tc.numpy()

    return sn, th


def make_lstm(batch=1, seq_len=32, vocab=256, embed=256, hidden=512, num_layers=2):
    """LSTM forward benchmark."""
    x_np = np.random.randint(0, vocab, (batch, seq_len)).astype(np.int64)

    sn_model = SloLSTM(vocab, embed, hidden, num_layers, dropout=0.0)

    def sn():
        with no_grad():
            tx = Tensor(x_np, requires_grad=False)
            logits, _ = sn_model.forward(tx)
            _ = logits.data
    sn()  # init hidden

    if HAS_TORCH:
        class TorchLSTM(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(vocab, embed)
                self.lstm = nn.LSTM(embed, hidden, num_layers, batch_first=True)
                self.fc = nn.Linear(hidden, vocab)
            def forward(self, x):
                x = self.embed(x)
                x, _ = self.lstm(x)
                x = x[:, -1, :]
                return self.fc(x)

        th_model = TorchLSTM().eval()
        def th():
            with torch.no_grad():
                tx = torch.from_numpy(x_np).long()
                logits = th_model(tx)
                _ = logits.numpy()

        return sn, th

    return sn, None


def make_softmax(B=8, N=64, S=64):
    """Softmax + attention benchmark."""
    q_np = np.random.randn(B, 4, N, 16).astype(np.float32)
    k_np = np.random.randn(B, 4, S, 16).astype(np.float32)
    v_np = np.random.randn(B, 4, S, 16).astype(np.float32)

    def sn():
        with no_grad():
            from domains.slolib.gpu import get_accelerator
            acc = get_accelerator()
            o = acc.scaled_dot_attention(q_np, k_np, v_np)
            _ = o

    if HAS_TORCH:
        def th():
            q = torch.from_numpy(q_np)
            k = torch.from_numpy(k_np)
            v = torch.from_numpy(v_np)
            scale = 1.0 / math.sqrt(16)
            scores = torch.einsum("bhnk,bhsk->bhns", q, k) * scale
            attn = torch.softmax(scores, dim=-1)
            o = torch.einsum("bhns,bhsk->bhnk", attn, v)
            _ = o.numpy()
        return sn, th
    return sn, None


print("\n--- Matmul Benchmarks ---")
import math

for M, N, K in [(256, 256, 256), (512, 512, 512), (128, 256, 768), (1024, 1024, 1024)]:
    sn, th = make_matmul(M, N, K)
    benchmark(f"matmul ({M}x{K} @ {K}x{N})", sn, th)

print("\n--- LSTM Forward Benchmarks ---")
for vocab, seq_len, hidden in [(256, 16, 256), (256, 32, 512), (1000, 64, 512)]:
    sn, th = make_lstm(1, seq_len, vocab, 256, hidden, 2)
    label = f"LSTM 2L-{hidden} ({vocab}v/{seq_len}s)"
    benchmark(label, sn, th)

print("\n--- Attention Benchmarks ---")
for B, N, S, E in [(4, 32, 32, 16), (8, 64, 64, 16)]:
    q = np.random.randn(B, 4, N, E).astype(np.float32)
    k = np.random.randn(B, 4, S, E).astype(np.float32)
    v = np.random.randn(B, 4, S, E).astype(np.float32)

    def make_fn(qn, kn, vn):
        def sn():
            with no_grad():
                from domains.slolib.gpu import get_accelerator
                acc = get_accelerator()
                acc.scaled_dot_attention(qn, kn, vn)
        return sn

    def make_th(qn, kn, vn):
        def th():
            import torch
            q = torch.from_numpy(qn)
            k = torch.from_numpy(kn)
            v = torch.from_numpy(vn)
            scale = 1.0 / math.sqrt(E)
            scores = torch.einsum("bhnk,bhsk->bhns", q, k) * scale
            attn = torch.softmax(scores, dim=-1)
            torch.einsum("bhns,bhsk->bhnk", attn, v).numpy()
        return th

    label = f"attention B{B}xH4x{N}x{E}"
    benchmark(label, make_fn(q, k, v), make_th(q, k, v) if HAS_TORCH else None)

print("\n--- Fused Ops Benchmarks ---")
# Fused layer_norm + gelu vs separate
X = np.random.randn(128, 768).astype(np.float32)
W = np.random.randn(768).astype(np.float32)
B = np.random.randn(768).astype(np.float32)

def sn_separate():
    acc = get_accelerator()
    n = acc.layer_norm(X, W, B)
    acc.gelu(n)

def sn_fused():
    from domains.slolib.gpu import get_accelerator
    acc = get_accelerator()
    acc.fused_layer_norm_gelu(X, W, B)

benchmark("layer_norm+gelu separate", sn_separate)
benchmark("layer_norm+gelu fused     ", sn_fused)

print("\n" + "=" * 70)
print("Done.")
