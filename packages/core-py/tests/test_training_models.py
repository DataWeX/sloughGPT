"""Tests for domains.training — ModelManager, ModelConfig, ModelType, ModelArchitecture."""

import json
import os
import tempfile
import pytest
from domains.training import (
    ModelManager, ModelConfig, ModelType, ModelArchitecture,
    DatasetType, DatasetConfig, DatasetManager, DataFormat,
    DataPreprocessor, PreprocessingStepType,
    TrainingPipeline, PipelineConfig, PipelineStageType,
    detect_dataset_type,
)


# ---------------------------------------------------------------------------
# ModelType
# ---------------------------------------------------------------------------

class TestModelType:
    def test_all_members(self):
        assert len(ModelType) == 2
    def test_values(self):
        assert ModelType.LANGUAGE_MODEL.value == "language_model"
        assert ModelType.CHAT_MODEL.value == "chat_model"
    def test_names(self):
        assert ModelType.LANGUAGE_MODEL.name == "LANGUAGE_MODEL"
        assert ModelType.CHAT_MODEL.name == "CHAT_MODEL"
    def test_iteration(self):
        types = list(ModelType)
        assert len(types) == 2
    def test_from_value(self):
        assert ModelType("language_model") == ModelType.LANGUAGE_MODEL
        assert ModelType("chat_model") == ModelType.CHAT_MODEL


# ---------------------------------------------------------------------------
# ModelArchitecture
# ---------------------------------------------------------------------------

class TestModelArchitecture:
    def test_all_members(self):
        assert len(ModelArchitecture) == 3
    def test_values(self):
        assert ModelArchitecture.GPT.value == "gpt"
        assert ModelArchitecture.BERT.value == "bert"
        assert ModelArchitecture.CUSTOM.value == "custom"
    def test_names(self):
        assert ModelArchitecture.GPT.name == "GPT"
    def test_from_value(self):
        assert ModelArchitecture("gpt") == ModelArchitecture.GPT


# ---------------------------------------------------------------------------
# ModelConfig
# ---------------------------------------------------------------------------

class TestModelConfig:
    def test_defaults(self):
        cfg = ModelConfig(
            name="gpt2",
            model_type=ModelType.LANGUAGE_MODEL,
            architecture=ModelArchitecture.GPT,
        )
        assert cfg.hidden_size == 768
        assert cfg.num_layers == 12

    def test_custom(self):
        cfg = ModelConfig(
            name="bert",
            model_type=ModelType.CHAT_MODEL,
            architecture=ModelArchitecture.BERT,
            hidden_size=512,
            num_layers=6,
        )
        assert cfg.hidden_size == 512
        assert cfg.num_layers == 6

    def test_all_fields(self):
        cfg = ModelConfig(
            name="custom_model",
            model_type=ModelType.LANGUAGE_MODEL,
            architecture=ModelArchitecture.CUSTOM,
            hidden_size=256,
            num_layers=4,
        )
        assert cfg.name == "custom_model"
        assert cfg.model_type == ModelType.LANGUAGE_MODEL
        assert cfg.architecture == ModelArchitecture.CUSTOM

    def test_equality(self):
        cfg1 = ModelConfig(name="m", model_type=ModelType.LANGUAGE_MODEL, architecture=ModelArchitecture.GPT)
        cfg2 = ModelConfig(name="m", model_type=ModelType.LANGUAGE_MODEL, architecture=ModelArchitecture.GPT)
        assert cfg1 == cfg2


# ---------------------------------------------------------------------------
# ModelManager
# ---------------------------------------------------------------------------

class TestModelManager:
    def test_register_and_create(self):
        mgr = ModelManager()
        cfg = ModelConfig(
            name="gpt2",
            model_type=ModelType.LANGUAGE_MODEL,
            architecture=ModelArchitecture.GPT,
        )
        mgr.register_model(cfg)
        result = mgr.create_model("gpt2")
        assert result["name"] == "gpt2"
        assert result["ready"] is True

    def test_create_not_found(self):
        mgr = ModelManager()
        with pytest.raises(ValueError, match="not found"):
            mgr.create_model("nonexistent")

    def test_register_multiple(self):
        mgr = ModelManager()
        mgr.register_model(ModelConfig(name="m1", model_type=ModelType.LANGUAGE_MODEL, architecture=ModelArchitecture.GPT))
        mgr.register_model(ModelConfig(name="m2", model_type=ModelType.CHAT_MODEL, architecture=ModelArchitecture.BERT))
        assert len(mgr.models) == 2

    def test_overwrite_registration(self):
        mgr = ModelManager()
        mgr.register_model(ModelConfig(name="m", model_type=ModelType.LANGUAGE_MODEL, architecture=ModelArchitecture.GPT))
        mgr.register_model(ModelConfig(name="m", model_type=ModelType.CHAT_MODEL, architecture=ModelArchitecture.BERT))
        assert len(mgr.models) == 1

    def test_create_returns_config(self):
        mgr = ModelManager()
        cfg = ModelConfig(name="test", model_type=ModelType.LANGUAGE_MODEL, architecture=ModelArchitecture.GPT)
        mgr.register_model(cfg)
        result = mgr.create_model("test")
        assert result["config"] == cfg

    def test_init_empty(self):
        mgr = ModelManager()
        assert len(mgr.models) == 0


# ---------------------------------------------------------------------------
# DatasetType
# ---------------------------------------------------------------------------

class TestDatasetType:
    def test_all_members(self):
        assert len(DatasetType) == 8

    def test_values(self):
        assert DatasetType.TEXT.value == "text"
        assert DatasetType.CODE.value == "code"
        assert DatasetType.CONVERSATION.value == "conversation"
        assert DatasetType.INSTRUCTION.value == "instruction"
        assert DatasetType.AUDIO_TEXT.value == "audio_text"
        assert DatasetType.IMAGE_TEXT.value == "image_text"
        assert DatasetType.VIDEO_TEXT.value == "video_text"
        assert DatasetType.MULTIMODAL.value == "multimodal"

    def test_from_value(self):
        assert DatasetType("text") == DatasetType.TEXT
        assert DatasetType("code") == DatasetType.CODE


# ---------------------------------------------------------------------------
# DataFormat
# ---------------------------------------------------------------------------

class TestDataFormat:
    def test_all_members(self):
        assert len(DataFormat) == 3

    def test_values(self):
        assert DataFormat.JSON.value == "json"
        assert DataFormat.JSONL.value == "jsonl"
        assert DataFormat.CSV.value == "csv"


# ---------------------------------------------------------------------------
# DatasetConfig
# ---------------------------------------------------------------------------

class TestDatasetConfig:
    def test_construction(self):
        cfg = DatasetConfig(
            name="train_data",
            dataset_type=DatasetType.TEXT,
            data_format=DataFormat.JSONL,
            path="/data/train.jsonl",
        )
        assert cfg.name == "train_data"
        assert cfg.max_samples is None

    def test_with_max_samples(self):
        cfg = DatasetConfig(
            name="small", dataset_type=DatasetType.CODE,
            data_format=DataFormat.JSON, path="/data/code.json",
            max_samples=100,
        )
        assert cfg.max_samples == 100


# ---------------------------------------------------------------------------
# DatasetManager
# ---------------------------------------------------------------------------

class TestDatasetManager:
    def test_register_dataset(self):
        mgr = DatasetManager()
        cfg = DatasetConfig(name="d1", dataset_type=DatasetType.TEXT, data_format=DataFormat.JSONL, path="/tmp/d.jsonl")
        mgr.register_dataset(cfg)
        assert "d1" in mgr.datasets

    def test_list_by_type(self):
        mgr = DatasetManager()
        mgr.register_dataset(DatasetConfig(name="text1", dataset_type=DatasetType.TEXT, data_format=DataFormat.JSONL, path="/tmp/t.jsonl"))
        mgr.register_dataset(DatasetConfig(name="code1", dataset_type=DatasetType.CODE, data_format=DataFormat.JSONL, path="/tmp/c.jsonl"))
        text_datasets = mgr.list_by_type(DatasetType.TEXT)
        assert len(text_datasets) == 1
        assert text_datasets[0].name == "text1"

    def test_summarize(self):
        mgr = DatasetManager()
        mgr.register_dataset(DatasetConfig(name="t1", dataset_type=DatasetType.TEXT, data_format=DataFormat.JSONL, path="/tmp/t.jsonl"))
        mgr.register_dataset(DatasetConfig(name="c1", dataset_type=DatasetType.CODE, data_format=DataFormat.JSONL, path="/tmp/c.jsonl"))
        summary = mgr.summarize()
        assert "text" in summary
        assert "code" in summary

    def test_load_not_found(self):
        mgr = DatasetManager()
        with pytest.raises(ValueError, match="Dataset not found"):
            mgr.load_dataset("nonexistent")

    def test_stream_not_found(self):
        mgr = DatasetManager()
        gen = mgr.stream_dataset("nonexistent")
        with pytest.raises(ValueError, match="Dataset not found"):
            next(gen)

    def test_init_empty(self):
        mgr = DatasetManager()
        assert len(mgr.datasets) == 0


# ---------------------------------------------------------------------------
# PreprocessingStepType
# ---------------------------------------------------------------------------

class TestPreprocessingStepType:
    def test_all_members(self):
        assert len(PreprocessingStepType) == 3

    def test_values(self):
        assert PreprocessingStepType.CLEAN.value == "clean"
        assert PreprocessingStepType.TOKENIZE.value == "tokenize"
        assert PreprocessingStepType.FILTER.value == "filter"


# ---------------------------------------------------------------------------
# DataPreprocessor
# ---------------------------------------------------------------------------

class TestDataPreprocessor:
    def test_add_cleaning(self):
        pp = DataPreprocessor()
        pp.add_cleaning("text", lowercase=True)
        assert len(pp.steps) == 1

    def test_add_filter(self):
        pp = DataPreprocessor()
        pp.add_filter("text", min_length=5)
        assert len(pp.steps) == 1

    def test_process_record_clean(self):
        pp = DataPreprocessor()
        pp.add_cleaning("text", lowercase=True)
        result = pp.process_record({"text": "  HELLO  WORLD  "})
        assert result["text"] == "hello world"

    def test_process_record_filter_pass(self):
        pp = DataPreprocessor()
        pp.add_filter("text", min_length=3)
        result = pp.process_record({"text": "hello"})
        assert result is not None

    def test_process_record_filter_reject(self):
        pp = DataPreprocessor()
        pp.add_filter("text", min_length=10)
        result = pp.process_record({"text": "hi"})
        assert result is None

    def test_process_batch(self):
        pp = DataPreprocessor()
        pp.add_filter("text", min_length=3)
        records = [{"text": "hello"}, {"text": "hi"}, {"text": "world"}]
        results = pp.process_batch(records)
        assert len(results) == 2

    def test_chaining(self):
        pp = DataPreprocessor()
        result = pp.add_cleaning("text").add_filter("text", min_length=1)
        assert result is pp

    def test_no_steps(self):
        pp = DataPreprocessor()
        result = pp.process_record({"text": "unchanged"})
        assert result["text"] == "unchanged"


# ---------------------------------------------------------------------------
# PipelineStageType
# ---------------------------------------------------------------------------

class TestPipelineStageType:
    def test_all_members(self):
        assert len(PipelineStageType) == 4

    def test_values(self):
        assert PipelineStageType.PREPROCESS.value == "preprocess"
        assert PipelineStageType.TRAIN.value == "train"
        assert PipelineStageType.VALIDATE.value == "validate"
        assert PipelineStageType.SAVE.value == "save"


# ---------------------------------------------------------------------------
# PipelineConfig
# ---------------------------------------------------------------------------

class TestPipelineConfig:
    def test_defaults(self):
        cfg = PipelineConfig(name="test_pipeline")
        assert cfg.batch_size == 32
        assert cfg.epochs == 3
        assert cfg.learning_rate == 1e-4

    def test_custom(self):
        cfg = PipelineConfig(name="p", batch_size=64, epochs=10, learning_rate=1e-3)
        assert cfg.batch_size == 64
        assert cfg.epochs == 10
        assert cfg.learning_rate == 1e-3


# ---------------------------------------------------------------------------
# TrainingPipeline
# ---------------------------------------------------------------------------

class TestTrainingPipeline:
    def test_init(self):
        cfg = PipelineConfig(name="p")
        pipe = TrainingPipeline(cfg)
        assert pipe.config == cfg
        assert len(pipe.stages) == 0

    def test_add_stage(self):
        cfg = PipelineConfig(name="p")
        pipe = TrainingPipeline(cfg)
        result = pipe.add_stage("preprocess", PipelineStageType.PREPROCESS, lambda x: x)
        assert result is pipe
        assert len(pipe.stages) == 1

    def test_add_multiple_stages(self):
        cfg = PipelineConfig(name="p")
        pipe = TrainingPipeline(cfg)
        pipe.add_stage("s1", PipelineStageType.PREPROCESS, None)
        pipe.add_stage("s2", PipelineStageType.TRAIN, None)
        pipe.add_stage("s3", PipelineStageType.SAVE, None)
        assert len(pipe.stages) == 3

    def test_run(self):
        import asyncio
        cfg = PipelineConfig(name="p", epochs=2)
        pipe = TrainingPipeline(cfg)
        pipe.add_stage("train", PipelineStageType.TRAIN, None)
        result = asyncio.run(pipe.run(iter([])))
        assert result["epochs"] == 2
        assert len(result["stages"]) == 2


# ---------------------------------------------------------------------------
# detect_dataset_type
# ---------------------------------------------------------------------------

class TestDetectDatasetType:
    def test_text_file(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("hello world\n")
            f.write("this is plain text\n")
            f.flush()
            try:
                assert detect_dataset_type(f.name) == DatasetType.TEXT
            finally:
                os.unlink(f.name)

    def test_jsonl_instruction(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write(json.dumps({"instruction": "do something", "response": "ok"}) + "\n")
            f.flush()
            try:
                assert detect_dataset_type(f.name) == DatasetType.INSTRUCTION
            finally:
                os.unlink(f.name)

    def test_jsonl_conversation(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write(json.dumps({"messages": [{"role": "user", "content": "hi"}]}) + "\n")
            f.flush()
            try:
                assert detect_dataset_type(f.name) == DatasetType.CONVERSATION
            finally:
                os.unlink(f.name)

    def test_jsonl_audio(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write(json.dumps({"audio": "/path/to/audio.wav", "text": "hello"}) + "\n")
            f.flush()
            try:
                assert detect_dataset_type(f.name) == DatasetType.AUDIO_TEXT
            finally:
                os.unlink(f.name)

    def test_jsonl_image(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write(json.dumps({"image": "photo.jpg", "caption": "a cat"}) + "\n")
            f.flush()
            try:
                assert detect_dataset_type(f.name) == DatasetType.IMAGE_TEXT
            finally:
                os.unlink(f.name)

    def test_code_file(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("def hello():\n")
            f.write("class World:\n")
            f.write("import os\n")
            f.flush()
            try:
                assert detect_dataset_type(f.name) == DatasetType.CODE
            finally:
                os.unlink(f.name)

    def test_empty_directory(self):
        with tempfile.TemporaryDirectory() as d:
            assert detect_dataset_type(d) == DatasetType.TEXT

    def test_directory_with_files(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "data.jsonl")
            with open(path, 'w') as f:
                f.write(json.dumps({"instruction": "test", "response": "ok"}) + "\n")
            assert detect_dataset_type(d) == DatasetType.INSTRUCTION


# ---------------------------------------------------------------------------
# Additional ModelManager tests
# ---------------------------------------------------------------------------

class TestModelManagerExtra:
    def test_create_model_has_config_key(self):
        mgr = ModelManager()
        cfg = ModelConfig(name="m", model_type=ModelType.LANGUAGE_MODEL, architecture=ModelArchitecture.GPT)
        mgr.register_model(cfg)
        result = mgr.create_model("m")
        assert "config" in result

    def test_create_model_has_name_key(self):
        mgr = ModelManager()
        cfg = ModelConfig(name="m", model_type=ModelType.LANGUAGE_MODEL, architecture=ModelArchitecture.GPT)
        mgr.register_model(cfg)
        result = mgr.create_model("m")
        assert "name" in result

    def test_models_dict_isolation(self):
        mgr1 = ModelManager()
        mgr2 = ModelManager()
        mgr1.register_model(ModelConfig(name="m1", model_type=ModelType.LANGUAGE_MODEL, architecture=ModelArchitecture.GPT))
        assert len(mgr2.models) == 0


# ---------------------------------------------------------------------------
# Additional DatasetManager tests
# ---------------------------------------------------------------------------

class TestDatasetManagerExtra:
    def test_register_multiple_same_type(self):
        mgr = DatasetManager()
        mgr.register_dataset(DatasetConfig(name="d1", dataset_type=DatasetType.TEXT, data_format=DataFormat.JSONL, path="/tmp/d1.jsonl"))
        mgr.register_dataset(DatasetConfig(name="d2", dataset_type=DatasetType.TEXT, data_format=DataFormat.JSONL, path="/tmp/d2.jsonl"))
        assert len(mgr.list_by_type(DatasetType.TEXT)) == 2

    def test_list_by_type_empty(self):
        mgr = DatasetManager()
        assert mgr.list_by_type(DatasetType.CODE) == []

    def test_summarize_empty(self):
        mgr = DatasetManager()
        assert mgr.summarize() == {}

    def test_overwrite_registration(self):
        mgr = DatasetManager()
        mgr.register_dataset(DatasetConfig(name="d", dataset_type=DatasetType.TEXT, data_format=DataFormat.JSONL, path="/tmp/d.jsonl"))
        mgr.register_dataset(DatasetConfig(name="d", dataset_type=DatasetType.CODE, data_format=DataFormat.JSONL, path="/tmp/d.jsonl"))
        assert len(mgr.datasets) == 1
        assert mgr.datasets["d"].dataset_type == DatasetType.CODE

    def test_load_from_jsonl_file(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write('{"text": "hello"}\n')
            f.write('{"text": "world"}\n')
            f.flush()
            try:
                mgr = DatasetManager()
                mgr.register_dataset(DatasetConfig(name="test", dataset_type=DatasetType.TEXT, data_format=DataFormat.JSONL, path=f.name))
                records = mgr.load_dataset("test")
                assert len(records) == 2
                assert records[0]["text"] == "hello"
            finally:
                os.unlink(f.name)

    def test_stream_from_jsonl_file(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write('{"text": "a"}\n')
            f.write('{"text": "b"}\n')
            f.flush()
            try:
                mgr = DatasetManager()
                mgr.register_dataset(DatasetConfig(name="test", dataset_type=DatasetType.TEXT, data_format=DataFormat.JSONL, path=f.name))
                gen = mgr.stream_dataset("test")
                first = next(gen)
                assert first["text"] == "a"
            finally:
                os.unlink(f.name)

    def test_load_with_max_samples(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write('{"text": "a"}\n')
            f.write('{"text": "b"}\n')
            f.write('{"text": "c"}\n')
            f.flush()
            try:
                mgr = DatasetManager()
                mgr.register_dataset(DatasetConfig(name="test", dataset_type=DatasetType.TEXT, data_format=DataFormat.JSONL, path=f.name, max_samples=2))
                records = mgr.load_dataset("test")
                assert len(records) == 2
            finally:
                os.unlink(f.name)


# ---------------------------------------------------------------------------
# Additional DataPreprocessor tests
# ---------------------------------------------------------------------------

class TestDataPreprocessorExtra:
    def test_process_record_filter_exact_boundary(self):
        pp = DataPreprocessor()
        pp.add_filter("text", min_length=5)
        assert pp.process_record({"text": "hello"}) is not None
        assert pp.process_record({"text": "hi"}) is None

    def test_process_record_multiple_cleaning_steps(self):
        pp = DataPreprocessor()
        pp.add_cleaning("text", lowercase=True)
        pp.add_cleaning("text", lowercase=False)
        result = pp.process_record({"text": "  HELLO  "})
        assert result["text"] == "hello"

    def test_process_batch_empty(self):
        pp = DataPreprocessor()
        assert pp.process_batch([]) == []

    def test_chaining_multiple(self):
        pp = DataPreprocessor()
        result = pp.add_cleaning("text").add_filter("text", min_length=1).add_cleaning("text")
        assert result is pp
        assert len(pp.steps) == 3

    def test_default_field_name(self):
        pp = DataPreprocessor()
        pp.add_cleaning()
        result = pp.process_record({"text": "  test  "})
        assert result["text"] == "test"

    def test_missing_field_gets_empty_string(self):
        pp = DataPreprocessor()
        pp.add_cleaning("text")
        result = pp.process_record({"other": "value"})
        assert "text" in result
        assert result["text"] == ""


# ---------------------------------------------------------------------------
# Additional PipelineConfig tests
# ---------------------------------------------------------------------------

class TestPipelineConfigExtra:
    def test_equality(self):
        cfg1 = PipelineConfig(name="p", batch_size=32, epochs=3, learning_rate=1e-4)
        cfg2 = PipelineConfig(name="p", batch_size=32, epochs=3, learning_rate=1e-4)
        assert cfg1 == cfg2

    def test_inequality(self):
        cfg1 = PipelineConfig(name="p1")
        cfg2 = PipelineConfig(name="p2")
        assert cfg1 != cfg2

    def test_repr(self):
        cfg = PipelineConfig(name="test")
        assert "test" in repr(cfg)


# ---------------------------------------------------------------------------
# Additional TrainingPipeline tests
# ---------------------------------------------------------------------------

class TestTrainingPipelineExtra:
    def test_empty_stages(self):
        cfg = PipelineConfig(name="p")
        pipe = TrainingPipeline(cfg)
        assert len(pipe.stages) == 0

    def test_stage_stores_type(self):
        cfg = PipelineConfig(name="p")
        pipe = TrainingPipeline(cfg)
        pipe.add_stage("s", PipelineStageType.TRAIN, None)
        assert pipe.stages[0]["type"] == PipelineStageType.TRAIN

    def test_stage_stores_handler(self):
        cfg = PipelineConfig(name="p")
        pipe = TrainingPipeline(cfg)
        handler = lambda x: x
        pipe.add_stage("s", PipelineStageType.TRAIN, handler)
        assert pipe.stages[0]["handler"] is handler


# ---------------------------------------------------------------------------
# Additional DatasetConfig tests
# ---------------------------------------------------------------------------

class TestDatasetConfigExtra:
    def test_equality(self):
        cfg1 = DatasetConfig(name="d", dataset_type=DatasetType.TEXT, data_format=DataFormat.JSONL, path="/tmp/d.jsonl")
        cfg2 = DatasetConfig(name="d", dataset_type=DatasetType.TEXT, data_format=DataFormat.JSONL, path="/tmp/d.jsonl")
        assert cfg1 == cfg2

    def test_inequality_different_path(self):
        cfg1 = DatasetConfig(name="d", dataset_type=DatasetType.TEXT, data_format=DataFormat.JSONL, path="/tmp/1.jsonl")
        cfg2 = DatasetConfig(name="d", dataset_type=DatasetType.TEXT, data_format=DataFormat.JSONL, path="/tmp/2.jsonl")
        assert cfg1 != cfg2

    def test_repr(self):
        cfg = DatasetConfig(name="d", dataset_type=DatasetType.TEXT, data_format=DataFormat.JSONL, path="/tmp/d.jsonl")
        assert "d" in repr(cfg)


# ---------------------------------------------------------------------------
# Additional ModelConfig tests
# ---------------------------------------------------------------------------

class TestModelConfigExtra:
    def test_repr(self):
        cfg = ModelConfig(name="m", model_type=ModelType.LANGUAGE_MODEL, architecture=ModelArchitecture.GPT)
        assert "m" in repr(cfg)

    def test_hidden_size_variants(self):
        for size in [128, 256, 512, 1024, 2048, 4096]:
            cfg = ModelConfig(name="m", model_type=ModelType.LANGUAGE_MODEL, architecture=ModelArchitecture.GPT, hidden_size=size)
            assert cfg.hidden_size == size

    def test_num_layers_variants(self):
        for layers in [1, 2, 4, 6, 8, 12, 24]:
            cfg = ModelConfig(name="m", model_type=ModelType.LANGUAGE_MODEL, architecture=ModelArchitecture.GPT, num_layers=layers)
            assert cfg.num_layers == layers


# ---------------------------------------------------------------------------
# Enum iteration
# ---------------------------------------------------------------------------

class TestEnumIteration:
    def test_model_type_iteration(self):
        types = list(ModelType)
        assert ModelType.LANGUAGE_MODEL in types
        assert ModelType.CHAT_MODEL in types

    def test_model_architecture_iteration(self):
        archs = list(ModelArchitecture)
        assert ModelArchitecture.GPT in archs
        assert ModelArchitecture.BERT in archs
        assert ModelArchitecture.CUSTOM in archs

    def test_dataset_type_iteration(self):
        types = list(DatasetType)
        assert len(types) == 8

    def test_data_format_iteration(self):
        fmts = list(DataFormat)
        assert len(fmts) == 3

    def test_pipeline_stage_iteration(self):
        stages = list(PipelineStageType)
        assert len(stages) == 4

    def test_preprocessing_step_iteration(self):
        steps = list(PreprocessingStepType)
        assert len(steps) == 3
