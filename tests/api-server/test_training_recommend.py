"""The recommend banner must size the dataset training will actually use.

The journey flow sends a dataset id or a just-cache path — sizing only repo
datasets/ and data/ made imported datasets report "0 examples".
"""

from __future__ import annotations

from pathlib import Path

from training.router import _count_dataset_examples


def test_counts_lines_of_a_repo_dataset_path(tmp_path: Path):
    fp = tmp_path / "datasets" / "demo" / "input.txt"
    fp.parent.mkdir(parents=True)
    fp.write_text("a\nb\nc\n", encoding="utf-8")
    assert _count_dataset_examples(str(fp), tmp_path) == 3


def test_counts_external_cache_corpus_for_dataset_id(tmp_path: Path, monkeypatch):
    cache = tmp_path / "cache"
    entry = cache / "journey_select"
    entry.mkdir(parents=True)
    (entry / "corpus.jsonl").write_text('{"x":1}\n{"x":2}\n', encoding="utf-8")

    from domain.training._internal import cache_tags

    monkeypatch.setattr(cache_tags, "get_cache_root", lambda: cache)

    assert _count_dataset_examples("journey_select", tmp_path / "repo") == 2


def test_unknown_id_counts_zero(tmp_path: Path):
    assert _count_dataset_examples("definitely-not-a-real-dataset", tmp_path / "repo") == 0


def test_paths_outside_allowed_roots_are_not_read(tmp_path: Path):
    secret = tmp_path / "outside" / "secret.txt"
    secret.parent.mkdir(parents=True)
    secret.write_text("one\ntwo\n", encoding="utf-8")
    repo = tmp_path / "repo"
    repo.mkdir()
    assert _count_dataset_examples(str(secret), repo) == 0
