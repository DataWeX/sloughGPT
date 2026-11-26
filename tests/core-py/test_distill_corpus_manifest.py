"""Gates for the distill-corpus SPEC (card 20261125_203428): manifest + build-vs-spec drift.

A spec no gate can see is not a spec — this file is why the tracked manifest stays honest:

1. Always (CI-safe): the tracked manifest passes the SloughGPT v1 contract and its
   declared discipline fields (stability classes, tiers, split names, split_hash).
2. Always: the adapter never imports a training loop (known-debt rule, AST pin).
3. When local ``data/`` sources exist (they are gitignored — absent in CI): a full
   build into tmp must match the manifest's declared inventory — frozen counts/hashes
   exact, growing counts as a minimum, agent_runs still empty, contamination gone,
   dedup exact, chat_sft_seed's 12 heldout records forced, deterministic rebuild.
4. When in-place artifacts exist: what's on disk matches ``build_stats.json``
   (catches hand-edits of built files).
"""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import pytest

from domain.training._internal.dataset_manifest import ManifestError, load_manifest
from domain.training._internal.distill_corpus_adapter import build

_ROOT = Path(__file__).resolve().parents[2]
_MANIFEST = _ROOT / "data" / "distill_corpus" / "dataset_manifest.json"
_CORPUS_DIR = _MANIFEST.parent
_MARKERS = ("<|im_start|>", "<|im_end|>")
#: Loops/adapters the corpus builder must never import (one-trainer rule).
_FORBIDDEN_IMPORTS = ("train_pipeline", "chat_trainer", "comprehensive_trainer", "torch")

_HAS_LOCAL_SOURCES = (_ROOT / "data" / "api_conversations" / "corpus.jsonl").is_file()
needs_sources = pytest.mark.skipif(
    not _HAS_LOCAL_SOURCES, reason="gitignored data/ sources absent (CI)"
)
needs_artifacts = pytest.mark.skipif(
    not (_CORPUS_DIR / "input.txt").is_file(), reason="corpus not built here yet"
)


def _norm(messages: list[dict]) -> str:
    return "\n".join(f"{m.get('role', '')}:{str(m.get('content', '')).strip()}" for m in messages)


# ── always: the tracked spec itself ─────────────────────────────────────


def test_manifest_passes_v1_contract():
    m = load_manifest(_MANIFEST)
    assert m["dataset_id"] == "distill_corpus"
    assert m["schema_version"] == "1.0"


def test_manifest_declares_spec_discipline():
    m = load_manifest(_MANIFEST)
    assert m["splits"]["train"] == "input.txt"  # distill.py candidate #1
    assert m["splits"]["test"] == "heldout.jsonl"
    assert len(m["splits"]["split_hash"]) == 16
    for src in m["sources"]:
        assert src["stability"] in {"frozen", "growing", "empty_ok"}, src["uri"]
        assert isinstance(src["record_count"], int), src["uri"]
        assert src["quality_tier"] in {"A", "B", "C", "A_target"}, src["uri"]
        assert src["filters"], src["uri"]  # every source declares its filter chain
    names = {t["name"] for t in m["transformations"]}
    assert {
        "filter_degenerate",
        "drop_template_contamination",
        "dedup_exact",
        "tier_assign",
        "split_heldout",
    } <= names


def test_adapter_never_imports_a_training_loop():
    src = _ROOT / "domain" / "training" / "_internal" / "distill_corpus_adapter.py"
    tree = ast.parse(src.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            mods = [node.module or ""]
        else:
            continue
        for mod in mods:
            assert not any(bad in mod for bad in _FORBIDDEN_IMPORTS), (
                f"corpus builder imports {mod} — build must not couple to a training loop"
            )
            if mod.startswith("domain.training"):
                assert mod == "domain.training._internal.dataset_manifest", mod


# ── local data present: build must match the declared inventory ────────


@needs_sources
def test_build_matches_manifest_spec(tmp_path):
    m = load_manifest(_MANIFEST)
    stats = build(_MANIFEST, out_dir=tmp_path)

    declared = {s["uri"]: s for s in m["sources"]}
    observed = stats["observed_per_source"]
    for uri, src in declared.items():
        n = observed[uri]
        if src["stability"] == "frozen":
            assert n == src["record_count"], f"{uri}: {n} != {src['record_count']}"
        elif src["stability"] == "growing":
            assert n >= src["record_count"], f"{uri}: {n} < minimum {src['record_count']}"
        else:  # empty_ok
            assert n == 0, f"{uri} unexpectedly populated: {n} — recon + version-bump first"

    train = (tmp_path / "input.txt").read_text(encoding="utf-8")
    heldout = [json.loads(x) for x in (tmp_path / "heldout.jsonl").read_text().splitlines() if x]

    # contamination is gone from what we train on
    assert not any(mk in train for mk in _MARKERS)
    assert not any(mk in _norm(r["messages"]) for r in heldout for mk in _MARKERS)

    # split discipline: exact partition (each record lands on one side by construction),
    # no duplicate inside heldout
    assert stats["train_records"] + stats["heldout_records"] == stats["kept_unique"]
    held_keys = [_norm(r["messages"]) for r in heldout]
    assert len(held_keys) == len(set(held_keys)), "duplicate inside heldout"

    # chat_sft_seed's declared heldout (12) is forced, never trained on
    seed_held = sum(1 for r in heldout if r["source"].endswith("chat_sft_seed.jsonl"))
    assert seed_held == 12, f"chat_sft heldout forced = {seed_held}, must be 12"
    assert stats["dropped_duplicates"] > 0, "dedup ran on this corpus by construction"
    assert 0.05 <= len(heldout) / max(1, stats["kept_unique"]) <= 0.30

    # determinism: same manifest + same sources → byte-identical splits
    again = build(_MANIFEST, out_dir=tmp_path / "again")
    assert again["train_sha256"] == stats["train_sha256"]
    assert again["heldout_sha256"] == stats["heldout_sha256"]


@needs_sources
def test_growing_journal_collapse_is_caught(tmp_path):
    """A journal that collapsed below its declared minimum must fail the build."""
    broken = json.loads(json.dumps(load_manifest(_MANIFEST)))
    for s in broken["sources"]:
        if s["stability"] == "growing":
            s["record_count"] = 10_000_000  # impossible minimum
    # sources are relative to the manifest dir — temp manifest beside the real one
    real = _CORPUS_DIR / "manifest_broken_test.json"
    real.write_text(json.dumps(broken), encoding="utf-8")
    try:
        with pytest.raises(ManifestError, match="minimum"):
            build(real, out_dir=tmp_path)
    finally:
        real.unlink(missing_ok=True)


# ── local artifacts: disk matches its own stats ─────────────────────────


@needs_artifacts
def test_on_disk_artifacts_match_build_stats():
    stats = json.loads((_CORPUS_DIR / "build_stats.json").read_text(encoding="utf-8"))
    train = (_CORPUS_DIR / "input.txt").read_bytes()
    assert hashlib.sha256(train).hexdigest()[:16] == stats["train_sha256"]
    held = [
        json.loads(x)
        for x in (_CORPUS_DIR / "heldout.jsonl").read_text(encoding="utf-8").splitlines()
        if x
    ]
    joined = "".join(_norm(r["messages"]) for r in held)
    assert hashlib.sha256(joined.encode()).hexdigest()[:16] == stats["heldout_sha256"]
    assert len(held) == stats["heldout_records"]
