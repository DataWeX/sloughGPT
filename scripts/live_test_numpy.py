#!/usr/bin/env python3
"""Live test of the full NumpyEngine pipeline — run directly, no opencode timeout."""

import sys
import time
sys.path.insert(0, "packages/core-py")

from domains.infrastructure.numpy_engine import NumpyEngine
from domains.infrastructure.morph_tokenizer import MorphTokenizer
from domains.infrastructure.model_server import NumpyBackend, ModelServer
from domains.infrastructure.model_compat import wrap_model, detect_model_type, ModelType

def main():
    print("=" * 60)
    print("LIVE TEST: NumpyEngine + MorphTokenizer + NumpyBackend")
    print("=" * 60)

    # 1. Load engine
    t0 = time.time()
    engine = NumpyEngine.from_pretrained("gpt2")
    print(f"\n[1] Loaded GPT-2 in {time.time()-t0:.1f}s")
    print(f"    arch={engine.arch}, vocab={engine.vocab_size}, ctx={engine.max_context}")

    # 2. Test generation (multiple prompts)
    prompts = [
        "The capital of France is",
        "Hello, how are you?",
        "Once upon a time",
        "The meaning of life is",
        "def fibonacci(n):",
    ]
    print("\n[2] Generation tests:")
    for p in prompts:
        t0 = time.time()
        result = engine.generate(p, max_new_tokens=30, temperature=0.0)
        elapsed = time.time() - t0
        print(f"    [{elapsed:.1f}s] {repr(result[:80])}")

    # 3. MorphTokenizer roundtrip
    print("\n[3] MorphTokenizer BPE:")
    tok = MorphTokenizer.from_pretrained("gpt2")
    for text in ["Hello world", "The quick brown fox", "import torch"]:
        ids = tok.encode(text)
        decoded = tok.decode(ids)
        ok = "✓" if decoded == text else "✗"
        print(f"    {ok} {repr(text)} → {ids} → {repr(decoded)}")

    # 4. NumpyBackend
    print("\n[4] NumpyBackend:")
    backend = NumpyBackend(engine)
    t0 = time.time()
    result = backend.generate("Tell me a joke", max_new_tokens=50, temperature=0.7, top_p=0.9, top_k=50, repetition_penalty=1.1)
    print(f"    [{time.time()-t0:.1f}s] tokens={result['tokens_generated']}")
    print(f"    {repr(result['text'][:100])}")

    # 5. Stream test
    print("\n[5] Stream test:")
    t0 = time.time()
    tokens = list(backend.generate_stream("Roses are red,", max_new_tokens=20, temperature=0.0, top_p=1.0, top_k=0, repetition_penalty=1.0))
    print(f"    [{time.time()-t0:.1f}s] {len(tokens)} tokens: {''.join(tokens)[:80]}")

    # 6. Cancel test
    print("\n[6] Cancel test:")
    from threading import Event
    cancel = Event()
    cancel.set()
    tokens = list(backend.generate_stream("Hello" * 100, max_new_tokens=100, temperature=0.0, top_p=1.0, top_k=0, repetition_penalty=1.0, cancel_event=cancel))
    print(f"    Cancelled immediately: {len(tokens)} tokens (should be 0)")

    # 7. model_compat
    print("\n[7] model_compat:")
    wrapped = wrap_model(engine, model_id="gpt2")
    print(f"    type={wrapped.model_type.value}")
    out = wrapped.generate_text("2+2=", max_new_tokens=10)
    print(f"    generate_text: {repr(out[:60])}")

    # 8. Determinism check
    print("\n[8] Determinism (temp=0):")
    r1 = engine.generate("Hello", max_new_tokens=10, temperature=0.0)
    r2 = engine.generate("Hello", max_new_tokens=10, temperature=0.0)
    ok = "✓" if r1 == r2 else "✗"
    print(f"    {ok} r1={repr(r1)}")
    print(f"    {ok} r2={repr(r2)}")

    # 9. Temperature sampling
    print("\n[9] Temperature sampling (temp=1.0):")
    results = set()
    for _ in range(5):
        r = engine.generate("The color of the sky is", max_new_tokens=10, temperature=1.0)
        results.add(r)
    print(f"    5 runs → {len(results)} unique outputs (should be >1)")

    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()
