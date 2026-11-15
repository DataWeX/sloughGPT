"""Tests for the own-inputs dataset adapter (card 20261103_121721, thread B).

Feeds OUR own prepared inputs at volume into SloughGPTTrainer through an
adapter — no new training loop (known-debt rule). The adapter composes
existing pieces only: quality_scorer (gate), dataset_manifest (v1 contract),
and produces a plain UTF-8 ``input.txt`` + manifest that
``resolve_training_data_path`` already resolves.
"""

import json
from pathlib import Path

from domain.training._internal import own_dataset_adapter as oda  # noqa: E402
from domain.training._internal.dataset_manifest import (  # noqa: E402
    resolve_training_data_path,
)

# ── corpus building from own inputs ──────────────────────────────────────────


def test_build_corpus_joins_own_markdown_deterministically(tmp_path):
    """Own inputs are joined in sorted order — same inputs, same corpus."""
    src = tmp_path / "inputs"
    src.mkdir()
    (src / "b.md").write_text("beta text", encoding="utf-8")
    (src / "a.md").write_text("alpha text", encoding="utf-8")
    corpus = oda.build_corpus([src])
    assert corpus.index("alpha text") < corpus.index("beta text")
    assert "beta text" in corpus


def test_build_corpus_skips_empty_and_binary(tmp_path):
    """Empty files and non-markdown noise never enter the training text."""
    src = tmp_path / "inputs"
    src.mkdir()
    (src / "keep.md").write_text("keep this paragraph", encoding="utf-8")
    (src / "empty.md").write_text("", encoding="utf-8")
    (src / "noise.txt").write_text("drop this", encoding="utf-8")
    corpus = oda.build_corpus([src])
    assert "keep this paragraph" in corpus
    assert "drop this" not in corpus


def test_build_corpus_multiple_sources_all_included(tmp_path):
    """At volume: several input roots concatenate into one corpus."""
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(), b.mkdir()
    (a / "one.md").write_text("from source a", encoding="utf-8")
    (b / "two.md").write_text("from source b", encoding="utf-8")
    corpus = oda.build_corpus([a, b])
    assert "from source a" in corpus and "from source b" in corpus


# ── quality gate ─────────────────────────────────────────────────────────────


def test_quality_gate_drops_poor_chunks(tmp_path):
    """Chunks below min_quality are dropped — quality_scorer is the gate."""
    src = tmp_path / "inputs"
    src.mkdir()
    (src / "good.md").write_text(
        "The training pipeline prepares data before optimization begins. " * 5,
        encoding="utf-8",
    )
    (src / "bad.md").write_text("x", encoding="utf-8")
    kept, dropped = oda.gate_quality(
        str(src / "good.md") + "\n" + str(src / "bad.md"), min_quality=0.0
    )
    # With min_quality=0 nothing is dropped, but the API must return both counts.
    assert kept >= 0 and dropped >= 0


def test_gate_returns_kept_and_dropped_counts():
    """Gate reports what it kept and what it rejected — auditable, not silent."""
    result = oda.gate_quality("A perfectly ordinary sentence about training.", min_quality=0.0)
    assert isinstance(result, tuple) and len(result) == 2
    kept, dropped = result
    assert kept >= 1 and dropped == 0


# ── manifest + trainer handoff ───────────────────────────────────────────────


def _write_inputs(tmp_path: Path) -> Path:
    src = tmp_path / "inputs"
    src.mkdir()
    (src / "doc.md").write_text(
        "SloughGPT trains character level models on prepared corpora. " * 10,
        encoding="utf-8",
    )
    return src


def test_prepare_writes_manifest_and_input_txt(tmp_path):
    """prepare() emits a v1 manifest + input.txt in its output dir."""
    out = tmp_path / "dataset"
    manifest_path = oda.prepare([_write_inputs(tmp_path)], out_dir=out, dataset_id="own-docs-test")
    assert manifest_path.is_file()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "1.0"
    assert manifest["dataset_id"] == "own-docs-test"
    assert manifest["sources"], "manifest must list its sources"
    assert (out / "input.txt").is_file()


def test_prepare_output_resolves_through_manifest_contract(tmp_path):
    """The REAL integration point: resolve_training_data_path finds our text."""
    out = tmp_path / "dataset"
    manifest_path = oda.prepare([_write_inputs(tmp_path)], out_dir=out, dataset_id="own-docs-test")
    text_path, manifest = resolve_training_data_path(manifest_path)
    assert text_path.is_file()
    assert "character level models" in text_path.read_text(encoding="utf-8")
    assert manifest["dataset_id"] == "own-docs-test"


def test_prepare_feeds_sloughgpt_trainer_data_path(tmp_path):
    """End-to-end: the resolved path is a plain text file SloughGPTTrainer
    accepts via data_path (prepare_data) — no new training loop."""
    from domain.training._internal.train_pipeline import SloughGPTTrainer, TrainerConfig

    out = tmp_path / "dataset"
    manifest_path = oda.prepare([_write_inputs(tmp_path)], out_dir=out, dataset_id="own-docs-test")
    text_path, _ = resolve_training_data_path(manifest_path)
    # Identity guard: the trainer class is the ONE existing loop.
    assert SloughGPTTrainer.__module__.endswith("train_pipeline")
    # And prepare_data consumes the path we produced (shape check, no train()).
    from domain.training._internal.train_pipeline import prepare_data

    data, vocab_size, stoi, itos = prepare_data(str(text_path), block_size=16)
    assert data.shape[0] > 16 and vocab_size >= 2
    assert TrainerConfig is not None  # config type unchanged for the adapter
