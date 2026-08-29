"""Tests for domains.training — DatasetType, DataFormat, DatasetConfig, PreprocessingStepType, PipelineStageType, ModelType, ModelArchitecture, ModelConfig, PipelineConfig (training.__init__)."""

from domains.training import (
    DatasetType, DataFormat, DatasetConfig, PreprocessingStepType,
    PipelineStageType, ModelType, ModelArchitecture, ModelConfig, PipelineConfig,
)


class TestDatasetType:
    def test_all_members(self):
        assert len(DatasetType) == 8
    def test_values(self):
        assert DatasetType.TEXT.value == "text"
        assert DatasetType.CODE.value == "code"
        assert DatasetType.CONVERSATION.value == "conversation"
        assert DatasetType.MULTIMODAL.value == "multimodal"


class TestDataFormat:
    def test_all_members(self):
        assert len(DataFormat) == 3
    def test_values(self):
        assert DataFormat.JSON.value == "json"
        assert DataFormat.JSONL.value == "jsonl"
        assert DataFormat.CSV.value == "csv"


class TestDatasetConfig:
    def test_fields(self):
        dc = DatasetConfig(name="test", dataset_type=DatasetType.TEXT, data_format=DataFormat.JSONL, path="/tmp/test.jsonl")
        assert dc.name == "test"
        assert dc.dataset_type == DatasetType.TEXT
        assert dc.max_samples is None


class TestPreprocessingStepType:
    def test_all_members(self):
        assert len(PreprocessingStepType) == 3
    def test_values(self):
        assert PreprocessingStepType.CLEAN.value == "clean"
        assert PreprocessingStepType.TOKENIZE.value == "tokenize"
        assert PreprocessingStepType.FILTER.value == "filter"


class TestPipelineStageType:
    def test_all_members(self):
        assert len(PipelineStageType) == 4
    def test_values(self):
        assert PipelineStageType.PREPROCESS.value == "preprocess"
        assert PipelineStageType.TRAIN.value == "train"
        assert PipelineStageType.SAVE.value == "save"


class TestModelType:
    def test_all_members(self):
        assert len(ModelType) == 2
    def test_values(self):
        assert ModelType.LANGUAGE_MODEL.value == "language_model"
        assert ModelType.CHAT_MODEL.value == "chat_model"


class TestModelArchitecture:
    def test_all_members(self):
        assert len(ModelArchitecture) == 3
    def test_values(self):
        assert ModelArchitecture.GPT.value == "gpt"
        assert ModelArchitecture.BERT.value == "bert"
        assert ModelArchitecture.CUSTOM.value == "custom"


class TestModelConfig:
    def test_fields(self):
        mc = ModelConfig(name="m1", model_type=ModelType.LANGUAGE_MODEL, architecture=ModelArchitecture.GPT)
        assert mc.name == "m1"
        assert mc.hidden_size == 768
        assert mc.num_layers == 12


class TestTrainingPipelineConfig:
    def test_fields(self):
        pc = PipelineConfig(name="p1")
        assert pc.name == "p1"
        assert pc.batch_size == 32
        assert pc.epochs == 3
        assert pc.learning_rate == 1e-4
