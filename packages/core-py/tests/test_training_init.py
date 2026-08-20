"""Tests for domains.training — detect_dataset_type, DatasetType, DatasetManager."""

import json
import pytest
from domains.training import (
    DatasetType, DataFormat, DatasetConfig, DatasetManager,
    detect_dataset_type,
)


class TestDatasetType:
    def test_all_members(self):
        assert len(DatasetType) == 8

    def test_text_value(self):
        assert DatasetType.TEXT.value == "text"


class TestDataFormat:
    def test_all_members(self):
        assert len(DataFormat) == 3


class TestDetectDatasetType:
    def test_text_file(self, tmp_path):
        f = tmp_path / "data.txt"
        f.write_text("hello world\n" * 10)
        assert detect_dataset_type(str(f)) == DatasetType.TEXT

    def test_code_file(self, tmp_path):
        f = tmp_path / "code.py"
        f.write_text("def foo():\nimport os\nclass Bar:\n    pass\n")
        assert detect_dataset_type(str(f)) == DatasetType.CODE

    def test_conversation_jsonl(self, tmp_path):
        f = tmp_path / "data.jsonl"
        f.write_text(json.dumps({"messages": [{"role": "user", "content": "hi"}]}) + "\n")
        assert detect_dataset_type(str(f)) == DatasetType.CONVERSATION

    def test_instruction_jsonl(self, tmp_path):
        f = tmp_path / "data.jsonl"
        f.write_text(json.dumps({"instruction": "do this", "response": "ok"}) + "\n")
        assert detect_dataset_type(str(f)) == DatasetType.INSTRUCTION

    def test_audio_jsonl(self, tmp_path):
        f = tmp_path / "data.jsonl"
        f.write_text(json.dumps({"audio": "file.wav", "text": "hello"}) + "\n")
        assert detect_dataset_type(str(f)) == DatasetType.AUDIO_TEXT

    def test_image_jsonl(self, tmp_path):
        f = tmp_path / "data.jsonl"
        f.write_text(json.dumps({"image": "photo.jpg", "caption": "a cat"}) + "\n")
        assert detect_dataset_type(str(f)) == DatasetType.IMAGE_TEXT

    def test_empty_dir(self, tmp_path):
        assert detect_dataset_type(str(tmp_path)) == DatasetType.TEXT

    def test_dir_with_txt(self, tmp_path):
        (tmp_path / "data.txt").write_text("hello\n")
        assert detect_dataset_type(str(tmp_path)) == DatasetType.TEXT


class TestDatasetConfig:
    def test_fields(self):
        cfg = DatasetConfig(
            name="shakespeare",
            dataset_type=DatasetType.TEXT,
            data_format=DataFormat.JSON,
            path="/tmp/data.jsonl",
        )
        assert cfg.name == "shakespeare"
        assert cfg.max_samples is None


class TestDatasetManager:
    def test_register_and_list(self):
        mgr = DatasetManager()
        cfg = DatasetConfig("d1", DatasetType.TEXT, DataFormat.JSON, "/tmp/d1.txt")
        mgr.register_dataset(cfg)
        assert "d1" in mgr.datasets

    def test_list_by_type(self):
        mgr = DatasetManager()
        mgr.register_dataset(DatasetConfig("t1", DatasetType.TEXT, DataFormat.JSON, "/tmp/t1.txt"))
        mgr.register_dataset(DatasetConfig("c1", DatasetType.CODE, DataFormat.JSON, "/tmp/c1.txt"))
        text = mgr.list_by_type(DatasetType.TEXT)
        assert len(text) == 1
        assert text[0].name == "t1"

    def test_load_dataset(self, tmp_path):
        f = tmp_path / "data.jsonl"
        f.write_text('{"text": "line1"}\n{"text": "line2"}\n')
        mgr = DatasetManager()
        mgr.register_dataset(DatasetConfig("d1", DatasetType.TEXT, DataFormat.JSONL, str(f)))
        records = mgr.load_dataset("d1")
        assert len(records) == 2

    def test_load_max_samples(self, tmp_path):
        f = tmp_path / "data.jsonl"
        f.write_text('{"text": "a"}\n{"text": "b"}\n{"text": "c"}\n')
        mgr = DatasetManager()
        mgr.register_dataset(DatasetConfig("d1", DatasetType.TEXT, DataFormat.JSONL, str(f), max_samples=2))
        records = mgr.load_dataset("d1")
        assert len(records) == 2

    def test_load_not_found(self):
        mgr = DatasetManager()
        with pytest.raises(ValueError, match="not found"):
            mgr.load_dataset("nonexistent")
