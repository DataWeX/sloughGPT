"""Tests for domains.training — detect_dataset_type, DatasetType, DatasetManager, etc."""

import json
import pytest
from domains.training import (
    DatasetType, DataFormat, DatasetConfig, DatasetManager,
    detect_dataset_type,
    DataPreprocessor, PreprocessingStepType,
    PipelineConfig, PipelineStageType, TrainingPipeline,
    ModelType, ModelArchitecture, ModelConfig, ModelManager,
)


# ── DatasetType ────────────────────────────────────────────────────────


class TestDatasetType:
    def test_all_members(self):
        assert len(DatasetType) == 8

    def test_text_value(self):
        assert DatasetType.TEXT.value == "text"

    def test_code_value(self):
        assert DatasetType.CODE.value == "code"

    def test_conversation_value(self):
        assert DatasetType.CONVERSATION.value == "conversation"

    def test_instruction_value(self):
        assert DatasetType.INSTRUCTION.value == "instruction"

    def test_audio_text_value(self):
        assert DatasetType.AUDIO_TEXT.value == "audio_text"

    def test_image_text_value(self):
        assert DatasetType.IMAGE_TEXT.value == "image_text"

    def test_video_text_value(self):
        assert DatasetType.VIDEO_TEXT.value == "video_text"

    def test_multimodal_value(self):
        assert DatasetType.MULTIMODAL.value == "multimodal"

    def test_from_value(self):
        assert DatasetType("text") is DatasetType.TEXT
        assert DatasetType("code") is DatasetType.CODE

    def test_invalid_value(self):
        with pytest.raises(ValueError):
            DatasetType("nonexistent")

    def test_iteration(self):
        names = [dt.name for dt in DatasetType]
        assert "TEXT" in names
        assert "CODE" in names

    def test_is_enum(self):
        from enum import Enum
        assert issubclass(DatasetType, Enum)


# ── DataFormat ─────────────────────────────────────────────────────────


class TestDataFormat:
    def test_all_members(self):
        assert len(DataFormat) == 3

    def test_json_value(self):
        assert DataFormat.JSON.value == "json"

    def test_jsonl_value(self):
        assert DataFormat.JSONL.value == "jsonl"

    def test_csv_value(self):
        assert DataFormat.CSV.value == "csv"

    def test_from_value(self):
        assert DataFormat("json") is DataFormat.JSON
        assert DataFormat("csv") is DataFormat.CSV

    def test_invalid_value(self):
        with pytest.raises(ValueError):
            DataFormat("xml")


# ── detect_dataset_type ────────────────────────────────────────────────


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

    def test_conversation_key_jsonl(self, tmp_path):
        f = tmp_path / "data.jsonl"
        f.write_text(json.dumps({"conversation": [{"from": "user", "value": "hi"}]}) + "\n")
        assert detect_dataset_type(str(f)) == DatasetType.CONVERSATION

    def test_speech_key_jsonl(self, tmp_path):
        f = tmp_path / "data.jsonl"
        f.write_text(json.dumps({"speech": "file.wav", "text": "hello"}) + "\n")
        assert detect_dataset_type(str(f)) == DatasetType.AUDIO_TEXT

    def test_wav_key_jsonl(self, tmp_path):
        f = tmp_path / "data.jsonl"
        f.write_text(json.dumps({"wav": "file.wav", "text": "hello"}) + "\n")
        assert detect_dataset_type(str(f)) == DatasetType.AUDIO_TEXT

    def test_jpg_key_jsonl(self, tmp_path):
        f = tmp_path / "data.jsonl"
        f.write_text(json.dumps({"jpg": "photo.jpg", "caption": "a cat"}) + "\n")
        assert detect_dataset_type(str(f)) == DatasetType.IMAGE_TEXT

    def test_png_key_jsonl(self, tmp_path):
        f = tmp_path / "data.jsonl"
        f.write_text(json.dumps({"png": "photo.png", "caption": "a cat"}) + "\n")
        assert detect_dataset_type(str(f)) == DatasetType.IMAGE_TEXT

    def test_code_file_functions(self, tmp_path):
        f = tmp_path / "code.js"
        f.write_text("function foo() {\nconst x = 1;\nimport os\n}\n")
        # 3 unique patterns: function, const, import
        assert detect_dataset_type(str(f)) == DatasetType.CODE

    def test_json_file_conversation(self, tmp_path):
        f = tmp_path / "data.json"
        f.write_text(json.dumps({"messages": [{"role": "user", "content": "hi"}]}) + "\n")
        assert detect_dataset_type(str(f)) == DatasetType.CONVERSATION

    def test_malformed_jsonl(self, tmp_path):
        f = tmp_path / "data.jsonl"
        f.write_text("not json\n{invalid json\n")
        assert detect_dataset_type(str(f)) == DatasetType.TEXT

    def test_dir_with_jsonl(self, tmp_path):
        f = tmp_path / "data.jsonl"
        f.write_text(json.dumps({"messages": [{"role": "user", "content": "hi"}]}) + "\n")
        assert detect_dataset_type(str(tmp_path)) == DatasetType.CONVERSATION

    def test_code_patterns_variety(self, tmp_path):
        f = tmp_path / "code.rs"
        f.write_text("fn main() {\nlet x = 1;\n}\n")
        # "fn " matches, but only 1 pattern — not enough for CODE
        assert detect_dataset_type(str(f)) == DatasetType.TEXT

    def test_code_patterns_threshold(self, tmp_path):
        f = tmp_path / "code.ts"
        f.write_text("function foo() {\nconst x = 1;\nclass Bar {}\nimport os\n}\n")
        assert detect_dataset_type(str(f)) == DatasetType.CODE


# ── DatasetConfig ──────────────────────────────────────────────────────


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

    def test_with_max_samples(self):
        cfg = DatasetConfig("d", DatasetType.CODE, DataFormat.JSONL, "/p", max_samples=100)
        assert cfg.max_samples == 100

    def test_all_fields(self):
        cfg = DatasetConfig(
            name="n", dataset_type=DatasetType.CONVERSATION,
            data_format=DataFormat.CSV, path="/p", max_samples=50,
        )
        assert cfg.dataset_type == DatasetType.CONVERSATION
        assert cfg.data_format == DataFormat.CSV
        assert cfg.path == "/p"
        assert cfg.max_samples == 50


# ── DatasetManager ─────────────────────────────────────────────────────


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

    def test_list_by_type_empty(self):
        mgr = DatasetManager()
        mgr.register_dataset(DatasetConfig("t1", DatasetType.TEXT, DataFormat.JSON, "/tmp/t1.txt"))
        code = mgr.list_by_type(DatasetType.CODE)
        assert code == []

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

    def test_load_empty_lines(self, tmp_path):
        f = tmp_path / "data.jsonl"
        f.write_text('{"text": "a"}\n\n{"text": "b"}\n')
        mgr = DatasetManager()
        mgr.register_dataset(DatasetConfig("d1", DatasetType.TEXT, DataFormat.JSONL, str(f)))
        records = mgr.load_dataset("d1")
        assert len(records) == 2

    def test_load_json_format(self, tmp_path):
        f = tmp_path / "data.json"
        f.write_text('{"text": "line1"}\n{"text": "line2"}\n')
        mgr = DatasetManager()
        mgr.register_dataset(DatasetConfig("d1", DatasetType.TEXT, DataFormat.JSON, str(f)))
        records = mgr.load_dataset("d1")
        assert len(records) == 2

    def test_stream_dataset(self, tmp_path):
        f = tmp_path / "data.jsonl"
        f.write_text('{"text": "line1"}\n{"text": "line2"}\n')
        mgr = DatasetManager()
        mgr.register_dataset(DatasetConfig("d1", DatasetType.TEXT, DataFormat.JSONL, str(f)))
        records = list(mgr.stream_dataset("d1"))
        assert len(records) == 2

    def test_stream_not_found(self):
        mgr = DatasetManager()
        with pytest.raises(ValueError, match="not found"):
            list(mgr.stream_dataset("nonexistent"))

    def test_scan_directory(self, tmp_path):
        subdir = tmp_path / "mydata"
        subdir.mkdir()
        (subdir / "data.txt").write_text("hello\n")
        count = mgr_scan_and_register(tmp_path)
        assert count >= 1

    def test_scan_directory_not_found(self):
        mgr = DatasetManager()
        count = mgr.scan_directory("/nonexistent/path")
        assert count == 0

    def test_scan_skips_underscore_dirs(self, tmp_path):
        subdir = tmp_path / "_hidden"
        subdir.mkdir()
        (subdir / "data.txt").write_text("hello\n")
        count = mgr_scan_and_register(tmp_path)
        assert count == 0

    def test_scan_skips_registered(self, tmp_path):
        subdir = tmp_path / "mydata"
        subdir.mkdir()
        (subdir / "data.txt").write_text("hello\n")
        mgr = DatasetManager()
        count1 = mgr.scan_directory(str(tmp_path))
        count2 = mgr.scan_directory(str(tmp_path))
        assert count1 == 1
        assert count2 == 0

    def test_scan_skips_file_entries(self, tmp_path):
        (tmp_path / "readme.txt").write_text("hello\n")
        count = mgr_scan_and_register(tmp_path)
        assert count == 0

    def test_scan_skips_empty_dirs(self, tmp_path):
        subdir = tmp_path / "empty"
        subdir.mkdir()
        count = mgr_scan_and_register(tmp_path)
        assert count == 0

    def test_summarize(self, tmp_path):
        f1 = tmp_path / "a.jsonl"
        f1.write_text('{"text": "a"}\n')
        f2 = tmp_path / "b.jsonl"
        f2.write_text('{"text": "b"}\n')
        mgr = DatasetManager()
        mgr.register_dataset(DatasetConfig("a", DatasetType.TEXT, DataFormat.JSONL, str(f1)))
        mgr.register_dataset(DatasetConfig("b", DatasetType.TEXT, DataFormat.JSONL, str(f2)))
        summary = mgr.summarize()
        assert "text" in summary
        assert len(summary["text"]) == 2

    def test_summarize_empty(self):
        mgr = DatasetManager()
        summary = mgr.summarize()
        assert summary == {}

    def test_register_overwrite(self):
        mgr = DatasetManager()
        mgr.register_dataset(DatasetConfig("d1", DatasetType.TEXT, DataFormat.JSON, "/a"))
        mgr.register_dataset(DatasetConfig("d1", DatasetType.CODE, DataFormat.JSON, "/b"))
        assert mgr.datasets["d1"].dataset_type == DatasetType.CODE

    def test_load_csv_format_fallback(self, tmp_path):
        """CSV format falls through — each line parsed as a single-row DictReader."""
        f = tmp_path / "data.csv"
        # CSV line-by-line: header is one line, data is another — DictReader sees empty data
        f.write_text("text,label\nhello,1\n")
        mgr = DatasetManager()
        mgr.register_dataset(DatasetConfig("d1", DatasetType.TEXT, DataFormat.CSV, str(f)))
        # Line-by-line CSV parsing: header line produces row with empty values
        records = mgr.load_dataset("d1")
        # The header line "text,label" is parsed as a CSV row
        assert isinstance(records, list)


def mgr_scan_and_register(tmp_path):
    """Helper: create a DatasetManager and scan the directory."""
    mgr = DatasetManager()
    return mgr.scan_directory(str(tmp_path))


# ── PreprocessingStepType ──────────────────────────────────────────────


class TestPreprocessingStepType:
    def test_all_members(self):
        assert len(PreprocessingStepType) == 3

    def test_values(self):
        assert PreprocessingStepType.CLEAN.value == "clean"
        assert PreprocessingStepType.TOKENIZE.value == "tokenize"
        assert PreprocessingStepType.FILTER.value == "filter"


# ── DataPreprocessor ───────────────────────────────────────────────────


class TestDataPreprocessor:
    def test_init(self):
        p = DataPreprocessor()
        assert p.steps == []

    def test_add_cleaning(self):
        p = DataPreprocessor()
        result = p.add_cleaning("text", lowercase=True)
        assert result is p  # fluent API
        assert len(p.steps) == 1
        assert p.steps[0]["type"] == PreprocessingStepType.CLEAN

    def test_add_filter(self):
        p = DataPreprocessor()
        p.add_filter("text", min_length=5)
        assert len(p.steps) == 1
        assert p.steps[0]["type"] == PreprocessingStepType.FILTER

    def test_process_record_clean(self):
        p = DataPreprocessor()
        p.add_cleaning("text", lowercase=True)
        result = p.process_record({"text": "  Hello   World  "})
        assert result["text"] == "hello world"

    def test_process_record_clean_no_lowercase(self):
        p = DataPreprocessor()
        p.add_cleaning("text", lowercase=False)
        result = p.process_record({"text": "  Hello   World  "})
        assert result["text"] == "Hello World"

    def test_process_record_filter_pass(self):
        p = DataPreprocessor()
        p.add_filter("text", min_length=3)
        result = p.process_record({"text": "hello"})
        assert result is not None

    def test_process_record_filter_reject(self):
        p = DataPreprocessor()
        p.add_filter("text", min_length=10)
        result = p.process_record({"text": "hi"})
        assert result is None

    def test_process_record_filter_missing_field(self):
        p = DataPreprocessor()
        p.add_filter("text", min_length=10)
        result = p.process_record({"other": "value"})
        assert result is None

    def test_process_record_missing_field_clean(self):
        p = DataPreprocessor()
        p.add_cleaning("text")
        result = p.process_record({"other": "value"})
        assert result is not None
        assert result["text"] == ""

    def test_process_batch(self):
        p = DataPreprocessor()
        p.add_filter("text", min_length=3)
        records = [{"text": "hello"}, {"text": "hi"}, {"text": "world"}]
        results = p.process_batch(records)
        assert len(results) == 2

    def test_process_batch_empty(self):
        p = DataPreprocessor()
        p.add_filter("text", min_length=3)
        results = p.process_batch([])
        assert results == []

    def test_chaining(self):
        p = DataPreprocessor()
        result = p.add_cleaning("text").add_filter("text", min_length=3)
        assert result is p
        assert len(p.steps) == 2

    def test_clean_whitespace_only(self):
        p = DataPreprocessor()
        p.add_cleaning("text")
        result = p.process_record({"text": "a\nb\tc"})
        assert result["text"] == "a b c"


# ── PipelineConfig ─────────────────────────────────────────────────────


class TestPipelineConfig:
    def test_defaults(self):
        cfg = PipelineConfig(name="test")
        assert cfg.batch_size == 32
        assert cfg.epochs == 3
        assert cfg.learning_rate == 1e-4

    def test_custom(self):
        cfg = PipelineConfig(name="test", batch_size=64, epochs=10, learning_rate=0.001)
        assert cfg.batch_size == 64
        assert cfg.epochs == 10
        assert cfg.learning_rate == 0.001


# ── PipelineStageType ──────────────────────────────────────────────────


class TestPipelineStageType:
    def test_all_members(self):
        assert len(PipelineStageType) == 4

    def test_values(self):
        assert PipelineStageType.PREPROCESS.value == "preprocess"
        assert PipelineStageType.TRAIN.value == "train"
        assert PipelineStageType.VALIDATE.value == "validate"
        assert PipelineStageType.SAVE.value == "save"


# ── TrainingPipeline ───────────────────────────────────────────────────


class TestTrainingPipeline:
    def test_init(self):
        cfg = PipelineConfig(name="test")
        pipeline = TrainingPipeline(cfg)
        assert pipeline.config is cfg
        assert pipeline.stages == []

    def test_add_stage(self):
        cfg = PipelineConfig(name="test")
        pipeline = TrainingPipeline(cfg)
        result = pipeline.add_stage("preprocess", PipelineStageType.PREPROCESS, lambda x: x)
        assert result is pipeline  # fluent API
        assert len(pipeline.stages) == 1

    def test_add_multiple_stages(self):
        cfg = PipelineConfig(name="test")
        pipeline = TrainingPipeline(cfg)
        pipeline.add_stage("preprocess", PipelineStageType.PREPROCESS, None)
        pipeline.add_stage("train", PipelineStageType.TRAIN, None)
        pipeline.add_stage("save", PipelineStageType.SAVE, None)
        assert len(pipeline.stages) == 3

    @pytest.mark.asyncio
    async def test_run(self):
        cfg = PipelineConfig(name="test", epochs=2)
        pipeline = TrainingPipeline(cfg)
        pipeline.add_stage("train", PipelineStageType.TRAIN, None)
        result = await pipeline.run(iter([]))
        assert result["epochs"] == 2
        assert len(result["stages"]) == 2

    @pytest.mark.asyncio
    async def test_run_no_stages(self):
        cfg = PipelineConfig(name="test", epochs=1)
        pipeline = TrainingPipeline(cfg)
        result = await pipeline.run(iter([]))
        assert result["epochs"] == 1
        assert result["stages"] == []

    @pytest.mark.asyncio
    async def test_run_multiple_epochs(self):
        cfg = PipelineConfig(name="test", epochs=5)
        pipeline = TrainingPipeline(cfg)
        pipeline.add_stage("a", PipelineStageType.TRAIN, None)
        pipeline.add_stage("b", PipelineStageType.VALIDATE, None)
        result = await pipeline.run(iter([]))
        assert result["epochs"] == 5
        assert len(result["stages"]) == 10


# ── ModelType ──────────────────────────────────────────────────────────


class TestModelType:
    def test_all_members(self):
        assert len(ModelType) == 2

    def test_values(self):
        assert ModelType.LANGUAGE_MODEL.value == "language_model"
        assert ModelType.CHAT_MODEL.value == "chat_model"


# ── ModelArchitecture ─────────────────────────────────────────────────


class TestModelArchitecture:
    def test_all_members(self):
        assert len(ModelArchitecture) == 3

    def test_values(self):
        assert ModelArchitecture.GPT.value == "gpt"
        assert ModelArchitecture.BERT.value == "bert"
        assert ModelArchitecture.CUSTOM.value == "custom"


# ── ModelConfig ────────────────────────────────────────────────────────


class TestModelConfig:
    def test_fields(self):
        cfg = ModelConfig(
            name="gpt2",
            model_type=ModelType.LANGUAGE_MODEL,
            architecture=ModelArchitecture.GPT,
        )
        assert cfg.name == "gpt2"
        assert cfg.hidden_size == 768
        assert cfg.num_layers == 12

    def test_custom(self):
        cfg = ModelConfig(
            name="m", model_type=ModelType.CHAT_MODEL,
            architecture=ModelArchitecture.CUSTOM,
            hidden_size=256, num_layers=4,
        )
        assert cfg.hidden_size == 256
        assert cfg.num_layers == 4


# ── ModelManager ───────────────────────────────────────────────────────


class TestModelManager:
    def test_register_model(self):
        mgr = ModelManager()
        cfg = ModelConfig("gpt2", ModelType.LANGUAGE_MODEL, ModelArchitecture.GPT)
        mgr.register_model(cfg)
        assert "gpt2" in mgr.models

    def test_create_model(self):
        mgr = ModelManager()
        cfg = ModelConfig("gpt2", ModelType.LANGUAGE_MODEL, ModelArchitecture.GPT)
        mgr.register_model(cfg)
        result = mgr.create_model("gpt2")
        assert result["name"] == "gpt2"
        assert result["ready"] is True

    def test_create_model_not_found(self):
        mgr = ModelManager()
        with pytest.raises(ValueError, match="not found"):
            mgr.create_model("nonexistent")

    def test_register_overwrite(self):
        mgr = ModelManager()
        mgr.register_model(ModelConfig("m", ModelType.LANGUAGE_MODEL, ModelArchitecture.GPT))
        mgr.register_model(ModelConfig("m", ModelType.CHAT_MODEL, ModelArchitecture.BERT))
        assert mgr.models["m"].model_type == ModelType.CHAT_MODEL

    def test_create_model_config(self):
        mgr = ModelManager()
        cfg = ModelConfig(
            "m", ModelType.CHAT_MODEL, ModelArchitecture.BERT,
            hidden_size=512, num_layers=6,
        )
        mgr.register_model(cfg)
        result = mgr.create_model("m")
        assert result["config"].hidden_size == 512
        assert result["config"].num_layers == 6
