"""Sequential stage runner + probe standings (domain/training/_internal)."""

from __future__ import annotations

import importlib.util
import json
import math
import sys
import threading
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "build_corpus.py"
_spec = importlib.util.spec_from_file_location("build_corpus_seq", _SCRIPT)
assert _spec and _spec.loader
build_corpus_mod = importlib.util.module_from_spec(_spec)
sys.modules["build_corpus_seq"] = build_corpus_mod
_spec.loader.exec_module(build_corpus_mod)

from domain.training._internal.probes import (  # noqa: E402
    evaluate_probe,
    load_probe_texts,
    probe_path_for,
)
from domain.training._internal.sequential import (  # noqa: E402
    build_shared_tokenizer,
    load_trained_on,
    manifest_ref,
    run_sequential_stages,
)

# Distinct char sets: corpus A is code-flavored (#, @), corpus B carries ~ and |.
ROWS_A = [f"def fn_{i}(x): return x + {i}  # alpha-{i} marker@" for i in range(40)]
ROWS_B = [f"The quiet river ~winds~ slowly eastward | beta-{i} marker." for i in range(40)]


def _seal(tmp_path: Path, dataset_id: str, rows: list[str]) -> str:
    src = tmp_path / f"{dataset_id}.jsonl"
    src.write_text("".join(json.dumps({"text": r}) + "\n" for r in rows), encoding="utf-8")
    return build_corpus_mod.build_corpus(
        dataset_id,
        "local",
        str(src),
        store_root=tmp_path / "crunched",
        probe=3,
    )


def _trainer_kwargs(tmp_path: Path) -> dict:
    return {
        "n_embed": 32,
        "n_layer": 1,
        "n_head": 2,
        "block_size": 16,
        "batch_size": 4,
        "epochs": 5,
        "lr": 1e-3,
        "device": "cpu",
        "checkpoint_dir": str(tmp_path / "ckpt"),
        "eval_interval": 10_000,
        "log_interval": 10_000,
        "checkpoint_interval": 10_000,
        "warmup_steps": 2,
    }


class TestManifestRef:
    def test_dataset_id_and_version_from_manifest(self, tmp_path: Path) -> None:
        m = _seal(tmp_path, "ref_ds", ROWS_A)
        assert manifest_ref(m) == "ref_ds@v1"


class TestProbePathAndLoad:
    def test_sibling_probe_file_roundtrip(self, tmp_path: Path) -> None:
        m = _seal(tmp_path, "load_ds", ROWS_A)
        p = probe_path_for(m)
        assert p == Path(m).parent / "probe.jsonl"
        texts = load_probe_texts(p)
        assert len(texts) == 3
        assert all("marker@" in t for t in texts)


class TestSharedTokenizer:
    def test_union_vocab_covers_both_corpora(self, tmp_path: Path) -> None:
        m_a = _seal(tmp_path, "uni_a", ROWS_A)
        m_b = _seal(tmp_path, "uni_b", ROWS_B)
        union = build_shared_tokenizer([m_a, m_b])
        only_a = build_shared_tokenizer([m_a])
        only_b = build_shared_tokenizer([m_b])
        assert "#" in union.stoi and "@" in union.stoi
        assert "~" in union.stoi and "|" in union.stoi
        assert "~" not in only_a.stoi  # A alone never saw B's glyphs
        assert "|" not in only_a.stoi
        assert "#" not in only_b.stoi  # B alone never saw A's glyphs
        assert union.vocab_size == len(union.stoi)
        assert set(union.itos[int(i)] for i in range(union.vocab_size)) <= set(union.stoi)


class TestRunSequentialStages:
    def test_two_stage_chain_with_live_standings(self, tmp_path: Path) -> None:
        m_a = _seal(tmp_path, "seqa", ROWS_A)
        m_b = _seal(tmp_path, "seqb", ROWS_B)
        standings_seen: list[list] = []

        res = run_sequential_stages(
            [m_a, m_b],
            trainer_kwargs=_trainer_kwargs(tmp_path),
            steps_per_stage=4,
            probe_batches=2,
            on_standings=lambda standings, _chain: standings_seen.append(
                [s["dataset"] for s in standings]
            ),
        )

        standings = res["standings"]
        assert len(standings) == 2
        s0, s1 = standings
        assert s0["dataset"] == "seqa@v1" and s1["dataset"] == "seqb@v1"

        # Continuity: stage 2 picks up the restored global step (delta budget).
        assert s1["global_step"] == s0["global_step"] + 4
        assert s0["global_step"] == 4

        # Stage 1 probes itself only; stage 2 probes both (forgetting readout).
        assert set(s0["probes"]) == {"seqa@v1"}
        assert set(s1["probes"]) == {"seqa@v1", "seqb@v1"}
        for value in list(s0["probes"].values()) + list(s1["probes"].values()):
            assert isinstance(value, float) and math.isfinite(value)

        # Host callback saw standings update after each stage.
        assert standings_seen == [["seqa@v1"], ["seqa@v1", "seqb@v1"]]

        # Artifact-local chain: trained_on.json in the checkpoint dir + result copy.
        ckpt = tmp_path / "ckpt"
        chain = load_trained_on(ckpt)
        assert [e["ref"] for e in chain] == ["seqa@v1", "seqb@v1"]
        assert [e["ref"] for e in res["trained_on"]] == ["seqa@v1", "seqb@v1"]
        assert (ckpt / "trained_on.json").is_file()

        # Shared vocab actually reached the trainer (probe comparability).
        last = res["last_trainer"]
        assert last is not None
        shared = build_shared_tokenizer([m_a, m_b])
        assert last.stoi == shared.stoi
        assert "~" in last.stoi

    def test_evaluate_probe_returns_finite_loss(self, tmp_path: Path) -> None:
        m_a = _seal(tmp_path, "ev_a", ROWS_A)
        m_b = _seal(tmp_path, "ev_b", ROWS_B)
        res = run_sequential_stages(
            [m_a, m_b],
            trainer_kwargs=_trainer_kwargs(tmp_path),
            steps_per_stage=3,
            probe_batches=2,
        )
        measured = evaluate_probe(res["last_trainer"], probe_path_for(m_a), num_batches=2)
        assert measured is not None
        assert math.isfinite(measured["probe_loss"])
        assert measured["probe_ppl"] > 1.0
        assert measured["records"] == 3
        assert measured["tokens"] > 0

    def test_cancel_before_stage_zero_runs_nothing(self, tmp_path: Path) -> None:
        m_a = _seal(tmp_path, "cancel_ds", ROWS_A)
        cancel = threading.Event()
        cancel.set()
        res = run_sequential_stages(
            [m_a],
            trainer_kwargs=_trainer_kwargs(tmp_path),
            steps_per_stage=2,
            cancel_event=cancel,
        )
        assert res["standings"] == []
        assert res["last_trainer"] is None
        assert not (tmp_path / "ckpt" / "trained_on.json").exists()

    def test_empty_manifests_rejected(self) -> None:
        import pytest

        with pytest.raises(ValueError, match="no manifests"):
            run_sequential_stages([], trainer_kwargs={})
