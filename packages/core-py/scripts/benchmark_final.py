"""Final benchmark: SloNet vs PyTorch CPU performance."""
import time, sys, os, math
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from domains.training.slonet import Tensor, no_grad, SloLSTM
from domains.slolib.gpu import _CPUBackend
cpu = _CPUBackend()

HAS_TORCH = True
try:
    import torch; import torch.nn as nn
except ImportError:
    HAS_TORCH = False

WARMUP = 10; ITERS = 50

def bench(name, fn_sn, fn_th=None):
    for _ in range(WARMUP): fn_sn()
    t = []
    for _ in range(ITERS):
        t0 = time.perf_counter_ns(); fn_sn(); t.append(time.perf_counter_ns() - t0)
    sn_ms = np.median(t) / 1e6
    label = f"  {name:<45} SloNet: {sn_ms:>8.2f} ms"
    if fn_th:
        for _ in range(WARMUP): fn_th()
        t = []
        for _ in range(ITERS):
            t0 = time.perf_counter_ns(); fn_th(); t.append(time.perf_counter_ns() - t0)
        th_ms = np.median(t) / 1e6
        r = sn_ms / th_ms
        if r < 1.0:
            label += f"  Torch: {th_ms:>8.2f} ms  🏆 {1/r:.1f}x FASTER"
        else:
            label += f"  Torch: {th_ms:>8.2f} ms  {r:.1f}x slower"
    print(label)

print("=" * 80)
print("FINAL: SloNet vs PyTorch CPU Benchmark")
print("=" * 80)

print("\n--- Matmul ---")
for M,N,K in [(256,256,256),(512,512,512),(1024,1024,1024)]:
    A = np.random.randn(M,K).astype(np.float32)
    B = np.random.randn(K,N).astype(np.float32)
    def ms(a,b):
        def sn(): (Tensor(a) @ Tensor(b)).data
        return sn
    def mt(a,b):
        ta,tb = torch.from_numpy(a),torch.from_numpy(b)
        def th(): (ta @ tb).numpy()
        return th
    bench(f"matmul {M}x{K} @ {K}x{N}", ms(A,B), mt(A,B) if HAS_TORCH else None)

print("\n--- LSTM ---")
for vocab,seq,hidden in [(256,16,256),(1000,32,512)]:
    x_np = np.random.randint(0,vocab,(1,seq)).astype(np.int64)
    m = SloLSTM(vocab,256,hidden,2,0.0)
    x_t = Tensor(x_np)
    if HAS_TORCH:
        class T(nn.Module):
            def __init__(s):
                super().__init__()
                s.emb = nn.Embedding(vocab,256)
                s.lstm = nn.LSTM(256,hidden,2,batch_first=True)
                s.fc = nn.Linear(hidden,vocab)
            def forward(s,x): x=s.emb(x);x,_=s.lstm(x);return s.fc(x[:,-1,:])
        tm = T().eval(); tx = torch.from_numpy(x_np).long()
        def th_fn():
            with torch.no_grad():
                return tm(tx).numpy()
    else:
        th_fn = None
    bench(f"LSTM Tensor {hidden}d {vocab}v/{seq}s", lambda: m.forward(x_t)[0].data, th_fn)
    bench(f"LSTM NumPy  {hidden}d {vocab}v/{seq}s", lambda: m.forward_numpy(x_np)[0], th_fn)

print("\n--- Attention (CPU) ---")
for B,H,N,E in [(4,4,32,16),(8,8,128,64)]:
    q = np.random.randn(B,H,N,E).astype(np.float32)
    k = np.random.randn(B,H,N,E).astype(np.float32)
    v = np.random.randn(B,H,N,E).astype(np.float32)
    sc = 1.0/math.sqrt(E)
    if HAS_TORCH:
        tq,tk,tv = torch.from_numpy(q),torch.from_numpy(k),torch.from_numpy(v)
        th_fn = lambda: (torch.softmax(torch.einsum("bhnk,bhsk->bhns",tq,tk)*sc,dim=-1)@tv).numpy()
    else:
        th_fn = None
    bench(f"attention {B}x{H}x{N}x{E}", lambda: cpu.scaled_dot_attention(q,k,v), th_fn)

print("\n--- Fused Ops ---")
X = np.random.randn(128,3072).astype(np.float32)
W = np.random.randn(3072).astype(np.float32)
B = np.random.randn(3072).astype(np.float32)
bench("layer_norm+gelu fused", lambda: cpu.fused_layer_norm_gelu(X,W,B))
bench("layer_norm+gelu separate", lambda: cpu.gelu(cpu.layer_norm(X,W,B)))

print("\n" + "=" * 80)
print("Done.")
