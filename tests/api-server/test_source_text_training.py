"""Pasted source_text must train like any other dataset source.

The 3-click flow promises "paste text directly if no dataset exists" — the
text is materialized as a just-cache dataset, then everything downstream
(validation, resolution, pre-flight, preview, recovery) works unchanged.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from pydantic import ValidationError
from training.resolution import materialize_source_text, resolve_training_inputs
from training.schemas import TrainingRequest


def _base(**over):
    body = {"name": "paste-job", "model": "slonet-native"}
    body.update(over)
    return body


class TestExactlyOneSourceValidator:
    def test_source_text_alone_is_a_valid_source(self):
        TrainingRequest(**_base(source_text="x" * 120))

    def test_source_text_with_dataset_is_rejected(self):
        with pytest.raises(ValidationError):
            TrainingRequest(**_base(dataset="shakespeare", source_text="x" * 120))

    def test_no_source_at_all_is_rejected(self):
        with pytest.raises(ValidationError):
            TrainingRequest(**_base())

    def test_dataset_alone_still_validates(self):
        TrainingRequest(**_base(dataset="shakespeare"))


class TestMaterializeSourceText:
    def test_writes_cache_entry_that_resolves_like_any_dataset(self, tmp_path, monkeypatch):
        from domain.training._internal import cache_tags

        monkeypatch.setattr(cache_tags, "get_cache_root", lambda: tmp_path / "cache")

        text = "line one of pasted training data\n" + "line two continues the story here. " * 4
        ds_id = materialize_source_text(text, "native-training-12345")

        assert ds_id.startswith("paste-native-training-12345-")
        entry = tmp_path / "cache" / ds_id
        corpus = entry / "corpus.jsonl"
        assert corpus.is_file()
        rows = [json.loads(line) for line in corpus.read_text(encoding="utf-8").splitlines()]
        assert [r["text"] for r in rows] == [ln for ln in text.splitlines() if ln.strip()]

        path, stem, _meta, _kind = resolve_training_inputs(ds_id, None, None)
        assert Path(path).is_file()
        assert stem == ds_id

    def test_rejects_text_under_100_bytes(self):
        with pytest.raises(ValueError, match="too small"):
            materialize_source_text("short", "name")

    def test_generated_id_is_id_safe(self, tmp_path, monkeypatch):
        from domain.training._internal import cache_tags

        monkeypatch.setattr(cache_tags, "get_cache_root", lambda: tmp_path / "cache")
        ds_id = materialize_source_text("y" * 150, "Weird Name! (#) 2026")
        assert re.match(r"^[a-zA-Z0-9_\-]+$", ds_id)
