"""training — Unified training capabilities (datasets, checkpoints, LoRA).

Public API:
    DatasetType, DataFormat, DatasetConfig, DatasetManager
    detect_dataset_type, PreprocessingStepType, PipelineStageType
    PipelineConfig, TrainingPipeline
    ModelType, ModelArchitecture, ModelConfig, ModelManager, DataPreprocessor
    find_checkpoint, load_soul, load_lora_soul
    LoRAType, LoRAConfig, LoRALinear, LoRAEmbedding
    apply_lora_to_model, get_lora_parameters
"""

from __future__ import annotations

from domain.training._internal.checkpoints import (
    find_checkpoint,
    load_lora_soul,
    load_soul,
)
from domain.training._internal.training import (
    DataFormat,
    DataPreprocessor,
    DatasetConfig,
    DatasetManager,
    DatasetType,
    ModelArchitecture,
    ModelConfig,
    ModelManager,
    ModelType,
    PipelineConfig,
    PipelineStageType,
    PreprocessingStepType,
    TrainingPipeline,
    detect_dataset_type,
)

_LAZY_IMPORTS = {
    "DataImporter": ("._internal.data_import", "DataImporter"),
    "RepoImporter": ("._internal.data_import", "RepoImporter"),
    "HuggingFaceImporter": ("._internal.data_import", "HuggingFaceImporter"),
    "URLImporter": ("._internal.data_import", "URLImporter"),
    "ISBNImporter": ("._internal.data_import", "ISBNImporter"),
    "BooksSearch": ("._internal.data_import", "BooksSearch"),
    "GitHubSearch": ("._internal.data_import", "GitHubSearch"),
    "LoRAType": ("._internal.lora", "LoRAType"),
    "LoRAConfig": ("._internal.lora", "LoRAConfig"),
    "LoRALinear": ("._internal.lora", "LoRALinear"),
    "LoRAEmbedding": ("._internal.lora", "LoRAEmbedding"),
    "apply_lora_to_model": ("._internal.lora", "apply_lora_to_model"),
    "get_lora_parameters": ("._internal.lora", "get_lora_parameters"),
    "VideoCaptionTrainer": ("._internal.video_trainer", "VideoCaptionTrainer"),
    "list_video_checkpoints": ("._internal.video_trainer", "list_video_checkpoints"),
    "get_training_executor": ("._internal.executor", "get_training_executor"),
    "export_to_sou": ("._internal.slonet", "export_to_sou"),
    "DistillConfig": ("._internal.distill_gpt2", "DistillConfig"),
    "SloughGPTTrainer": ("._internal.train_pipeline", "SloughGPTTrainer"),
    "TrainerConfig": ("._internal.train_pipeline", "TrainerConfig"),
    "TokenTree": ("._internal.token_tree", "TokenTree"),
    "TokenizerEngine": (".tokenizer_engine", "TokenizerEngine"),
    "get_tokenizer_engine": (".tokenizer_engine", "get_tokenizer_engine"),
    "get_tokenizer_manager": (".tokenizer_engine", "get_tokenizer_manager"),
    "get_token_tree_manager": (".tokenizer_engine", "get_token_tree_manager"),
    "TrainingEngine": (".engine", "TrainingEngine"),
    "get_training_engine": (".engine", "get_training_engine"),
    "TrainingFeedClient": ("._internal.training_feed", "TrainingFeedClient"),
    "FeedBatchSampler": ("._internal.training_feed", "FeedBatchSampler"),
    "TrainingOutcomeTracker": ("._internal.outcome_tracker", "TrainingOutcomeTracker"),
    "AdaptiveConfigEngine": ("._internal.adaptive_config", "AdaptiveConfigEngine"),
    "get_auto_trainer": ("._internal.auto_trainer", "get_auto_trainer"),
    "generate_model_card": ("._internal.model_card", "generate_model_card"),
    "list_presets": ("._internal.presets", "list_presets"),
    "get_preset": ("._internal.presets", "get_preset"),
    "apply_preset": ("._internal.presets", "apply_preset"),
    "RepoImporter": ("._internal.data_import", "RepoImporter"),
    "HuggingFaceImporter": ("._internal.data_import", "HuggingFaceImporter"),
    "URLImporter": ("._internal.data_import", "URLImporter"),
    "ISBNImporter": ("._internal.data_import", "ISBNImporter"),
    "DataImporter": ("._internal.data_import", "DataImporter"),
    "BooksSearch": ("._internal.data_import", "BooksSearch"),
    "GitHubSearch": ("._internal.data_import", "GitHubSearch"),
    "import_from_sou": ("._internal.slonet", "import_from_sou"),
    # cache_tags / executor / service / slonet names consumed by the training
    # and datasets controllers (b22e878a facade port).
    "classify_kind": ("._internal.cache_tags", "classify_kind"),
    "entry_tags": ("._internal.cache_tags", "entry_tags"),
    "find_corpus_file": ("._internal.cache_tags", "find_corpus_file"),
    "get_cache_root": ("._internal.cache_tags", "get_cache_root"),
    "guess_mime": ("._internal.cache_tags", "guess_mime"),
    "write_entry_meta": ("._internal.cache_tags", "write_entry_meta"),
    "_instance": ("._internal.executor", "_instance"),
    "get_state": ("._internal.service", "get_state"),
    "_get_accelerator": ("._internal.slonet", "_get_accelerator"),
}


def __getattr__(name):
    if name in _LAZY_IMPORTS:
        module_path, attr = _LAZY_IMPORTS[name]
        import importlib

        mod = importlib.import_module(module_path, package=__name__)
        return getattr(mod, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "DatasetType",
    "DataFormat",
    "DatasetConfig",
    "DatasetManager",
    "detect_dataset_type",
    "PreprocessingStepType",
    "PipelineStageType",
    "PipelineConfig",
    "TrainingPipeline",
    "ModelType",
    "ModelArchitecture",
    "ModelConfig",
    "ModelManager",
    "DataPreprocessor",
    "find_checkpoint",
    "load_soul",
    "load_lora_soul",
    "DataImporter",
    "RepoImporter",
    "HuggingFaceImporter",
    "URLImporter",
    "ISBNImporter",
    "BooksSearch",
    "GitHubSearch",
    "LoRAType",
    "LoRAConfig",
    "LoRALinear",
    "LoRAEmbedding",
    "apply_lora_to_model",
    "get_lora_parameters",
    "VideoCaptionTrainer",
    "list_video_checkpoints",
    "get_training_executor",
    "classify_kind",
    "entry_tags",
    "find_corpus_file",
    "get_cache_root",
    "guess_mime",
    "write_entry_meta",
    "get_state",
    "export_to_sou",
    "DistillConfig",
    "SloughGPTTrainer",
    "TrainerConfig",
    "TokenTree",
    "TokenizerEngine",
    "get_tokenizer_engine",
    "get_tokenizer_manager",
    "get_token_tree_manager",
    "TrainingEngine",
    "get_training_engine",
    "TrainingFeedClient",
    "FeedBatchSampler",
    "TrainingOutcomeTracker",
    "AdaptiveConfigEngine",
    "get_auto_trainer",
    "generate_model_card",
    "list_presets",
    "get_preset",
    "apply_preset",
    "RepoImporter",
    "HuggingFaceImporter",
    "URLImporter",
    "ISBNImporter",
    "DataImporter",
    "BooksSearch",
    "GitHubSearch",
    "import_from_sou",
]
