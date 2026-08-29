"""Tests for domains.training — DataPreprocessor and more DatasetManager."""

import json
import pytest
from domains.training import (
    DataPreprocessor, PreprocessingStepType,
    DatasetManager, DatasetConfig, DatasetType, DataFormat,
)


class TestPreprocessingStepType:
    def test_values(self):
        assert PreprocessingStepType.CLEAN.value == "clean"
        assert PreprocessingStepType.FILTER.value == "filter"


class TestDataPreprocessor:
    def test_no_steps_returns_record(self):
        p = DataPreprocessor()
        r = p.process_record({"text": "hello"})
        assert r == {"text": "hello"}

    def test_cleaning_lowercase(self):
        p = DataPreprocessor()
        p.add_cleaning(text_field="text", lowercase=True)
        r = p.process_record({"text": "HELLO WORLD"})
        assert r["text"] == "hello world"

    def test_cleaning_whitespace(self):
        p = DataPreprocessor()
        p.add_cleaning(text_field="text")
        r = p.process_record({"text": "  hello   world  "})
        assert r["text"] == "hello world"

    def test_cleaning_no_lower(self):
        p = DataPreprocessor()
        p.add_cleaning(text_field="text", lowercase=False)
        r = p.process_record({"text": "HELLO"})
        assert r["text"] == "HELLO"

    def test_filter_pass(self):
        p = DataPreprocessor()
        p.add_filter(text_field="text", min_length=5)
        r = p.process_record({"text": "hello world"})
        assert r is not None

    def test_filter_reject(self):
        p = DataPreprocessor()
        p.add_filter(text_field="text", min_length=20)
        r = p.process_record({"text": "hi"})
        assert r is None

    def test_filter_missing_field(self):
        p = DataPreprocessor()
        p.add_filter(text_field="text", min_length=10)
        r = p.process_record({"other": "data"})
        assert r is None

    def test_chaining(self):
        p = DataPreprocessor()
        result = p.add_cleaning().add_filter()
        assert result is p
        assert len(p.steps) == 2

    def test_process_batch(self):
        p = DataPreprocessor()
        p.add_filter(text_field="text", min_length=5)
        records = [
            {"text": "hello world"},
            {"text": "hi"},
            {"text": "another long text"},
        ]
        result = p.process_batch(records)
        assert len(result) == 2

    def test_clean_then_filter(self):
        p = DataPreprocessor()
        p.add_cleaning(text_field="text")
        p.add_filter(text_field="text", min_length=5)
        r = p.process_record({"text": "  HELLO  "})
        assert r is not None
        assert r["text"] == "hello"

    def test_custom_field(self):
        p = DataPreprocessor()
        p.add_cleaning(text_field="content")
        r = p.process_record({"content": "TEST"})
        assert r["content"] == "test"


class TestDatasetManagerStream:
    def test_stream_dataset(self, tmp_path):
        f = tmp_path / "data.jsonl"
        f.write_text('{"t": 1}\n{"t": 2}\n')
        mgr = DatasetManager()
        mgr.register_dataset(DatasetConfig("d1", DatasetType.TEXT, DataFormat.JSONL, str(f)))
        items = list(mgr.stream_dataset("d1"))
        assert len(items) == 2

    def test_stream_yields_dicts(self, tmp_path):
        f = tmp_path / "data.jsonl"
        f.write_text('{"a": 1}\n{"b": 2}\n')
        mgr = DatasetManager()
        mgr.register_dataset(DatasetConfig("d1", DatasetType.TEXT, DataFormat.JSONL, str(f)))
        items = list(mgr.stream_dataset("d1"))
        assert items[0] == {"a": 1}
        assert items[1] == {"b": 2}


class TestDatasetManagerSummarize:
    def test_summarize(self, tmp_path):
        f1 = tmp_path / "a.jsonl"
        f1.write_text('{"t": 1}\n')
        f2 = tmp_path / "b.jsonl"
        f2.write_text('{"t": 2}\n')
        mgr = DatasetManager()
        mgr.register_dataset(DatasetConfig("a", DatasetType.TEXT, DataFormat.JSONL, str(f1)))
        mgr.register_dataset(DatasetConfig("b", DatasetType.CODE, DataFormat.JSONL, str(f2)))
        summary = mgr.summarize()
        assert "text" in summary
        assert "code" in summary


