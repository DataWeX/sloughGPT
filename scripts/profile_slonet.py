"""
Profile SloNet hot paths: instruments every soul operation and SloTransformer
forward+backward to identify where time is spent.

Usage:
    python scripts/profile_slonet.py
"""

import sys
sys.path.insert(0, "packages/core-py")

import time
import numpy as np
from collections import defaultdict
from functools import wraps

# ---------------------------------------------------------------------------
# Instrumentation
# ---------------------------------------------------------------------------
TIMINGS: dict = defaultdict(float)
COUNTS: dict = defaultdict(int)
SIZES: dict = defaultdict(list)


def clear_stats():
    TIMINGS.clear()
    COUNTS.clear()
    SIZES.clear()


def print_stats(top_n=50):
    sorted_items = sorted(TIMINGS.items(), key=lambda x: -x[1])
    total_time = sum(TIMINGS.values())
    print(f"\n{'='*70}")
    print(f"  Operation Profile  (total: {total_time*1000:.1f} ms)")
    print(f"{'='*70}")
    print(f"  {'#':>3}  {'Op':<40} {'Calls':>6} {'Total ms':>9} {'Avg ms':>9} {'%':>6}")
    print(f"  {'-'*69}")
    for i, (name, t) in enumerate(sorted_items[:top_n]):
        cnt = COUNTS[name]
        pct = t / total_time * 100 if total_time > 0 else 0
        print(f"  {i+1:>3}  {name:<40} {cnt:>6} {t*1000:>9.2f} {t*1000/cnt if cnt else 0:>9.3f} {pct:>5.1f}%")
    print(f"  {'-'*69}")
    rest = sum(t for n, t in sorted_items[top_n:])
    if rest > 0:
        rest_pct = rest / total_time * 100
        print(f"  {'':>3}  {'<rest>':<40} {sum(COUNTS[n] for n, _ in sorted_items[top_n:]):>6} {rest*1000:>9.2f} {'':>9} {rest_pct:>5.1f}%")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    import numpy as np
    from domains.training import slonet as sn

    # Manually instrument all core ops by patching the module dict
    # We need to handle static/class methods properly, so we patch module-level
    # functions directly (they don't get 'self' injected).
    ops_single = [
        "_add", "_mul", "_neg", "_pow", "_sum", "_mean", "_max",
        "_matmul", "_transpose", "_reshape", "_slice",
        "sigmoid", "tanh", "relu", "gelu", "silu", "softmax",
        "_layernorm", "_rmsnorm", "cross_entropy",
    ]

    originals = {}
    for op in ops_single:
        if hasattr(sn, op):
            orig = getattr(sn, op)
            @wraps(orig)
            def make_wrapper(fn=orig, name=op):
                def wrapper(*args, **kwargs):
                    t0 = time.perf_counter()
                    result = fn(*args, **kwargs)
                    dt = time.perf_counter() - t0
                    TIMINGS[name] += dt
                    COUNTS[name] += 1
                    return result
                return wrapper
            originals[op] = orig
            setattr(sn, op, make_wrapper())

    # For class methods, we instrument them inline by patching the class
    # (but not staticmethods — skip staticmethod because Python injects `self`)
    class_methods = [
        ("SloLinear", "forward"),
        ("SloMultiHeadAttention", "forward"),
        ("SloMultiHeadAttention", "_attention_4d"),  # staticmethod — handled separately
        ("SloFeedForward", "forward"),
        ("SloRMSNorm", "forward") if hasattr(sn, 'SloRMSNorm') else None,
        ("SloTransformerBlock", "forward"),
        ("SloTransformer", "forward"),
        ("SloTransformer", "generate"),
    ]

    for pair in class_methods:
        if pair is None:
            continue
        cls_name, method_name = pair
        cls = getattr(sn, cls_name)
        if not hasattr(cls, method_name):
            continue
        # Check via __dict__ to detect staticmethod BEFORE descriptor unwrapping
        cls_dict_entry = cls.__dict__.get(method_name)
        is_static = isinstance(cls_dict_entry, staticmethod)
        orig_fn = cls_dict_entry.__func__ if is_static else getattr(cls, method_name)
        if is_static:
            @wraps(orig_fn)
            def make_static_wrapper(f=orig_fn, name=f"{cls_name}.{method_name}"):
                def wrapper(*args, **kwargs):
                    t0 = time.perf_counter()
                    result = f(*args, **kwargs)
                    dt = time.perf_counter() - t0
                    TIMINGS[name] += dt
                    COUNTS[name] += 1
                    return result
                return wrapper
            setattr(cls, method_name, staticmethod(make_static_wrapper()))
        else:
            @wraps(orig_fn)
            def make_method_wrapper(f=orig_fn, name=f"{cls_name}.{method_name}"):
                def wrapper(self, *args, **kwargs):
                    t0 = time.perf_counter()
                    result = f(self, *args, **kwargs)
                    dt = time.perf_counter() - t0
                    TIMINGS[name] += dt
                    COUNTS[name] += 1
                    return result
                return wrapper
            setattr(cls, method_name, make_method_wrapper())

    # =============================================
    # Profile 1: SloTransformer forward pass only
    # =============================================
    print("=" * 60)
    print("Profile 1: SloTransformer Forward Pass (seq_len=128)")
    print("=" * 60)

    vocab = 256
    n_embed = 128
    n_layer = 4
    n_head = 4
    block_size = 128

    tfm = sn.SloTransformer(
        vocab_size=vocab, n_embed=n_embed, n_layer=n_layer,
        n_head=n_head, block_size=block_size, dropout=0.0, tie_weights=False,
    )

    input_ids = np.random.randint(0, min(vocab, 50), size=(1, block_size)).astype(np.int64)

    clear_stats()
    # Warmup
    for _ in range(5):
        tfm.clear_kv_cache()
        logits, _ = tfm.forward(input_ids, use_cache=False)
    clear_stats()

    n_runs = 10
    t0 = time.perf_counter()
    for _ in range(n_runs):
        tfm.clear_kv_cache()
        logits, _ = tfm.forward(input_ids, use_cache=False)
    elapsed = time.perf_counter() - t0

    print(f"\n  Forward time: {elapsed*1000/n_runs:.2f} ms/run (avg of {n_runs})")
    print_stats(50)

    # =============================================
    # Profile 2: Full forward + backward pass
    # =============================================
    print("=" * 60)
    print("Profile 2: SloTransformer Forward + Backward (seq_len=32)")
    print("=" * 60)

    tfm2 = sn.SloTransformer(
        vocab_size=vocab, n_embed=n_embed, n_layer=n_layer,
        n_head=n_head, block_size=block_size, dropout=0.0, tie_weights=False,
    )
    adam = sn.SloAdam(lr=0.001)
    input_ids2 = np.random.randint(0, min(vocab, 50), size=(1, 32)).astype(np.int64)
    target_ids = np.random.randint(0, min(vocab, 50), size=(1, 32)).astype(np.int64)
    x = sn.tensor(input_ids2, requires_grad=True)
    y = sn.tensor(target_ids, requires_grad=False)

    clear_stats()
    for _ in range(3):
        tfm2.clear_kv_cache()
        logits, _ = tfm2.forward(x.data, use_cache=False)
        loss = sn.cross_entropy(logits, y)
        loss.backward()
        adam.step(tfm2.parameters())
        for p in tfm2.parameters():
            if p.grad is not None:
                p.grad = None
    clear_stats()

    n_runs = 5
    t0 = time.perf_counter()
    for _ in range(n_runs):
        tfm2.clear_kv_cache()
        logits, _ = tfm2.forward(x.data, use_cache=False)
        loss = sn.cross_entropy(logits, y)
        loss.backward()
        adam.step(tfm2.parameters())
        for p in tfm2.parameters():
            if p.grad is not None:
                p.grad = None
    elapsed = time.perf_counter() - t0

    print(f"\n  Forward+Backward time: {elapsed*1000/n_runs:.2f} ms/run (avg of {n_runs})")
    print_stats(50)

    # =============================================
    # Profile 3: LSTM forward + backward
    # =============================================
    print("=" * 60)
    print("Profile 3: LSTM Forward+Backward (vocab=256, emb=32, hid=64, layers=2)")
    print("=" * 60)

    net = sn.SloNet(
        layers=[
            sn.SloEmbedding(256, 32),
            sn.SloLSTM(256, 32, 64, num_layers=2, dropout=0.0),
        ],
        soul_name="profiler",
    )
    adam = sn.SloAdam(lr=0.01)

    clear_stats()
    for _ in range(3):
        x = sn.tensor([[5] * 16], requires_grad=True)
        y = sn.tensor([[6] * 16])
        h = net.layers[1].init_hidden()
        logits, _ = net.layers[1].forward(x, h)
        loss = sn.cross_entropy(logits, y.reshape(-1))
        loss.backward()
        adam.step(net.parameters())
        for p in net.parameters():
            if p.grad is not None:
                p.grad = None
    clear_stats()

    n_runs = 5
    t0 = time.perf_counter()
    for _ in range(n_runs):
        x = sn.tensor([[5] * 16], requires_grad=True)
        y = sn.tensor([[6] * 16])
        h = net.layers[1].init_hidden()
        logits, _ = net.layers[1].forward(x, h)
        loss = sn.cross_entropy(logits, y.reshape(-1))
        loss.backward()
        adam.step(net.parameters())
        net.layers[1].zero_grad()
    elapsed = time.perf_counter() - t0

    print(f"\n  LSTM time: {elapsed*1000/n_runs:.2f} ms/run (avg of {n_runs})")
    print_stats(50)


if __name__ == "__main__":
    main()
