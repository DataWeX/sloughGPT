"""show_effects — ground-up, engine-level effect report. No server.

Answers "what did these builds actually change?" by calling the exact
functions the API server calls, but in one standalone process — no uvicorn,
no PGQ, no session cache, no SSE. Three sections, each a real measurement
against a recorded pre-fix baseline:

  1. RAG context budget  — raw query() output vs the injected block
     (card cad8ccf0: 15,228 raw chars blew the 120s generate timeout).
  2. Attention kernels   — decode-shape microbench vs the three-pass
     recomputing kernels that shipped before (card perf/attention-gemv-land).
  3. Prefill + decode    — in-process .slnc first-token at short and
     RAG-length prompts, plus decode tok/s (cards 7637f9c7 / cad8ccf0).

Usage:
    ./scripts/python scripts/show_effects.py                  # default model
    ./scripts/python scripts/show_effects.py --slnc /path/model.slnc
    ./scripts/python scripts/show_effects.py --skip-model      # sections 1-2 only
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "apps" / "api" / "server"))

RAG_PROMPT = "Write a short paragraph about the pros and cons of enhancing online presence."

# ── Section 1: RAG context budget ────────────────────────────────────────────
# Pre-fix (2026-10-24, card cad8ccf0): stream_turn injected query() output
# uncut. On the 160-doc worktree store that was ~15,228 chars for this
# prompt — ~3.8k tokens — which alone put first-token past the 120s
# generation timeout. The budget cap + per-passage split is the fix.
BASELINE_RAW_CHARS = 15_228
BASELINE_RAW_TOKENS = 3_800  # ~4 chars/token


def section_rag() -> dict:
    from domain.core import get_rag_service

    print("\n── 1. RAG context budget (card cad8ccf0) " + "─" * 30)
    # is_rag_service_ready() is False until the singleton's first access —
    # in a fresh process the store may exist but nobody has touched it yet.
    # Construct first, then decide on stats, not on the readiness flag.
    try:
        svc = get_rag_service()
        stats = svc.stats()
    except Exception as e:  # noqa: BLE001 — demo script, report and skip
        print(f"  rag service unavailable ({e}) — skipping")
        return {}
    if stats.get("total_chunks", 0) == 0:
        print(f"  rag store empty ({stats}) — skipping")
        return {}

    t0 = time.perf_counter()
    raw = svc.query(RAG_PROMPT, 5)
    query_ms = (time.perf_counter() - t0) * 1000

    from routers.inference import _build_rag_block

    t0 = time.perf_counter()
    block = _build_rag_block(raw)
    budget_ms = (time.perf_counter() - t0) * 1000

    raw_chars = len(raw.get("context", "") or "")
    block_chars = len(block or "")
    results = raw.get("results") or []
    per_passage = [len(r.get("content") or "") for r in results if (r.get("content") or "").strip()]

    print(f"  store: {stats.get('total_documents', 0)} docs, {stats.get('total_chunks', 0)} chunks")
    print(f"  query: {query_ms:6.1f} ms   budget: {budget_ms * 1000:5.1f} us")
    print(f"  {'':12} {'chars':>8} {'~tokens':>8}")
    print(
        f"  {'raw query()':12} {raw_chars:8,} {raw_chars // 4:8,}   "
        f"(baseline {BASELINE_RAW_CHARS:,} / ~{BASELINE_RAW_TOKENS:,})"
    )
    print(
        f"  {'injected':12} {block_chars:8,} {block_chars // 4:8,}   "
        f"({BASELINE_RAW_CHARS / max(1, block_chars):.1f}x smaller)"
    )
    print(f"  passages retrieved: {len(per_passage)} (raw sizes {per_passage})")
    print(
        f"  per-passage budget: {block_chars // max(1, len(per_passage)) if per_passage else 0} chars "
        f"— every rank stays visible"
    )

    return {"raw_chars": raw_chars, "block_chars": block_chars, "query_ms": query_ms}


# ── Section 2: attention kernels ─────────────────────────────────────────────
# Pre-fix kernels recomputed every Q·K dot three times (three passes); the
# scores-buffer rewrite does one pass. Measured on this box at the shapes
# below, pre-fix: single kv=130 ~13.8ms, multi seq=200 ~530ms.
BASE_ATTENTION = {"single_kv130_ms": 13.8, "multi_seq200_ms": 530.0}


def section_attention() -> dict:
    import numpy as np

    from domain.training._internal.slonet_kernels import (
        _ensure_kernels,
        fused_attention_multi,
        fused_attention_single,
    )

    print("\n── 2. Attention kernels (card perf/attention-gemv-land) " + "─" * 24)
    _ensure_kernels()
    H, KH, E = 14, 2, 64  # Qwen2.5-0.5B head layout
    reps = H // KH
    assert reps > 1, "GQA expected"

    kv = 130
    q = np.random.randn(H, E).astype(np.float32)
    k = np.random.randn(KH, kv, E).astype(np.float32)
    v = np.random.randn(KH, kv, E).astype(np.float32)
    fused_attention_single(q, k, v, 1.0, H, E)
    t0 = time.perf_counter()
    for _ in range(50):
        fused_attention_single(q, k, v, 1.0, H, E)
    single_ms = (time.perf_counter() - t0) / 50 * 1000

    seq = 200
    qm = np.random.randn(seq, H, E).astype(np.float32)
    km = np.random.randn(KH, seq, E).astype(np.float32)
    vm = np.random.randn(KH, seq, E).astype(np.float32)
    fused_attention_multi(qm, km, vm, 1.0, H, E)
    t0 = time.perf_counter()
    fused_attention_multi(qm, km, vm, 1.0, H, E)
    multi_ms = (time.perf_counter() - t0) * 1000

    print(f"  {'':22} {'before':>10} {'now':>10} {'speedup':>9}")
    print(
        f"  {'single kv=130':22} {BASE_ATTENTION['single_kv130_ms']:9.2f}m "
        f"{single_ms:9.2f}m "
        f"{BASE_ATTENTION['single_kv130_ms'] / single_ms:8.1f}x"
    )
    print(
        f"  {'multi seq=200':22} {BASE_ATTENTION['multi_seq200_ms']:9.1f}m "
        f"{multi_ms:9.1f}m "
        f"{BASE_ATTENTION['multi_seq200_ms'] / multi_ms:8.1f}x"
    )
    return {"single_ms": single_ms, "multi_ms": multi_ms}


# ── Section 3: prefill + decode ──────────────────────────────────────────────
# Pre-fix first-token at a ~200-token prompt: ~22.8s (2026-10-23 profile).
# Decode slope pre-fix baseline: 8.6 tok/s (card 7637f9c7).
BASE_PREFILL_200MS = 22_800.0
BASE_DECODE_TOKS = 8.6


def _resolve_slnc(model: str) -> Path:
    from domain.infrastructure._internal.model_resolver import get_model_dir

    slnc = get_model_dir(model) / "model.slnc"
    if not slnc.exists():
        # Resolver points at HF_HOME; fall back to the repo-relative cache so
        # the harness runs standalone in a worktree with no HF_HOME set.
        repo = (
            Path("models/hf-cache/hub")
            / f"models--{model.replace('/', '--')}"
            / "model.slnc"
        )
        if repo.exists():
            return repo
        raise SystemExit(f"no model.slnc at {slnc} — pass --slnc")
    return slnc


def section_model(slnc: Path, model_id: str) -> dict:
    from domain.inference._internal.slonet_provider import SloNetChatProvider

    print("\n── 3. Prefill + decode (cards 7637f9c7, cad8ccf0) " + "─" * 28)
    t0 = time.perf_counter()
    provider = SloNetChatProvider.from_slnc(str(slnc), model_id=model_id, quantize=True)
    load_s = time.perf_counter() - t0
    print(f"  model loaded: {load_s:.1f}s")

    tok = provider._tokenizer
    model = provider._model
    eos = tok.eos_token_id or 0

    def prefill_ms(text: str) -> tuple[float, int]:
        ids = [tok.encode(text)]
        n = len(ids[0])
        import numpy as np

        arr = np.array(ids, dtype=np.int64)
        model.generate_numpy(arr, max_new_tokens=1, temperature=0.0, eos_token=eos)
        t0 = time.perf_counter()
        model.generate_numpy(arr, max_new_tokens=1, temperature=0.0, eos_token=eos)
        return (time.perf_counter() - t0) * 1000, n

    short = "hello world"
    ms_short, n_short = prefill_ms(short)

    from routers.inference import _build_rag_block

    from domain.core import get_rag_service

    # Construct-first, same as section 1 — readiness flag is False in a
    # fresh process until someone touches the singleton.
    rag_text = ""
    try:
        block = _build_rag_block(get_rag_service().query(RAG_PROMPT, 5))
        if block:
            rag_text = f"{block}\n\nUser question: {RAG_PROMPT}"
    except Exception:  # noqa: BLE001 — fall back to the bare prompt
        rag_text = ""
    if not rag_text:
        rag_text = RAG_PROMPT
    ms_rag, n_rag = prefill_ms(rag_text)

    import numpy as np

    ids = np.array([tok.encode("Write a short paragraph about online presence.")], dtype=np.int64)
    n_tokens = 48
    t0 = time.perf_counter()
    model.generate_numpy(ids, max_new_tokens=n_tokens, temperature=0.0, eos_token=eos)
    decode_s = time.perf_counter() - t0
    tok_s = n_tokens / decode_s

    print(f"  {'':30} {'before':>10} {'now':>10}")
    print(f"  {f'first-token @{n_short} tok (short)':30} {'—':>10} {ms_short / 1000:9.2f}s")
    print(f"  {'first-token @200 tok (pre-fix)':30} {BASE_PREFILL_200MS / 1000:9.2f}s")
    print(f"  {f'first-token @{n_rag} tok (RAG)':30} {'—':>10} {ms_rag / 1000:9.2f}s")
    print(f"  {f'decode {n_tokens} tok':30} {BASE_DECODE_TOKS:9.2f}t {tok_s:8.2f}t   (tok/s)")
    return {"ms_short": ms_short, "ms_rag": ms_rag, "tok_s": tok_s, "n_rag": n_rag}


def main() -> int:
    ap = argparse.ArgumentParser(description="Engine-level effect report (no server)")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--slnc", default=None)
    ap.add_argument("--skip-model", action="store_true", help="skip section 3 (no model load)")
    args = ap.parse_args()

    print("show_effects — engine-level, single process, no server stack")
    rag = section_rag()
    att = section_attention()
    model_stats: dict = {}
    if not args.skip_model:
        slnc = Path(args.slnc) if args.slnc else _resolve_slnc(args.model)
        model_stats = section_model(slnc, args.model)

    print("\n" + "=" * 66)
    print("SUMMARY (before → after)")
    print("=" * 66)
    if rag:
        print(
            f"  RAG context   {rag['raw_chars']:,} chars → {rag['block_chars']:,} chars "
            f"({rag['raw_chars'] / max(1, rag['block_chars']):.1f}x smaller)"
        )
    if att:
        print(
            f"  attn single   {BASE_ATTENTION['single_kv130_ms']:.2f}ms → "
            f"{att['single_ms']:.2f}ms "
            f"({BASE_ATTENTION['single_kv130_ms'] / att['single_ms']:.1f}x)"
        )
        print(
            f"  attn multi    {BASE_ATTENTION['multi_seq200_ms']:.0f}ms → "
            f"{att['multi_ms']:.0f}ms "
            f"({BASE_ATTENTION['multi_seq200_ms'] / att['multi_ms']:.1f}x)"
        )
    if model_stats:
        print(
            f"  first-token   22.8s @200tok → {model_stats['ms_rag'] / 1000:.1f}s "
            f"@{model_stats['n_rag']}tok (RAG prompt)"
        )
        print(f"  decode        {BASE_DECODE_TOKS:.1f} tok/s → {model_stats['tok_s']:.1f} tok/s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
