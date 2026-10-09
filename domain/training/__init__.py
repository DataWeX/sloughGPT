"""training — Unified training capabilities (datasets, checkpoints, LoRA).

Public API:
    DatasetType, DataFormat, DatasetConfig, DatasetManager
    detect_dataset_type, PreprocessingStepType, PipelineStageType
    PipelineConfig, TrainingPipeline
    ModelType, ModelArchitecture, ModelConfig, ModelManager, DataPreprocessor
    find_checkpoint, load_soul, load_lora_soul
    LoRAType, LoRAConfig, LoRALinear, LoRAEmbedding
    apply_lora_to_model, get_lora_parameters
    LongRunRecorder (durable JSONL metrics for long unattended runs)
"""

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
    "extract_pairs_from_sessions": ("._internal.pair_extractor", "extract_pairs_from_sessions"),
    "extract_pairs_from_logs": ("._internal.pair_extractor", "extract_pairs_from_logs"),
    "write_training_text": ("._internal.pair_extractor", "write_training_text"),
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
    "LongRunRecorder": ("._internal.long_run", "LongRunRecorder"),
    "TokenTree": ("._internal.token_tree", "TokenTree"),
    "TokenizerEngine": (".tokenizer_engine", "TokenizerEngine"),
    "get_tokenizer_engine": (".tokenizer_engine", "get_tokenizer_engine"),
    "get_tokenizer_manager": (".tokenizer_engine", "get_tokenizer_manager"),
    "get_token_tree_manager": (".tokenizer_engine", "get_token_tree_manager"),
    "TrainingEngine": (".engine", "TrainingEngine"),
    "get_training_engine": (".engine", "get_training_engine"),
    "TrainingFeedClient": ("._internal.training_feed", "TrainingFeedClient"),
    "TrainingOutcomeTracker": ("._internal.outcome_tracker", "TrainingOutcomeTracker"),
    "AdaptiveConfigEngine": ("._internal.adaptive_config", "AdaptiveConfigEngine"),
    "import_from_sou": ("._internal.slonet", "import_from_sou"),
    "ExportConfig": ("._internal.export", "ExportConfig"),
    "export_model": ("._internal.export", "export_model"),
    "list_export_formats": ("._internal.export", "list_export_formats"),
    "get_cache_root": ("._internal.cache_tags", "get_cache_root"),
    "classify_kind": ("._internal.cache_tags", "classify_kind"),
    "entry_tags": ("._internal.cache_tags", "entry_tags"),
    "find_corpus_file": ("._internal.cache_tags", "find_corpus_file"),
    "guess_mime": ("._internal.cache_tags", "guess_mime"),
    "write_entry_meta": ("._internal.cache_tags", "write_entry_meta"),
    "get_state": ("._internal.service", "get_state"),
    "_get_accelerator": ("._internal.slonet", "_get_accelerator"),
    "list_presets": ("._internal.presets", "list_presets"),
    "get_preset": ("._internal.presets", "get_preset"),
    "apply_preset": ("._internal.presets", "apply_preset"),
    "get_auto_trainer": ("._internal.auto_trainer", "get_auto_trainer"),
    "generate_model_card": ("._internal.model_card", "generate_model_card"),
    "get_turbo_lock": ("._internal.state", "get_turbo_lock"),
    "get_turbo_state": ("._internal.state", "get_turbo_state"),
    "peek_training_executor": ("._internal.executor", "peek_training_executor"),
    "FeedBatchSampler": ("._internal.training_feed", "FeedBatchSampler"),
    "executor": ("._internal.executor", None),
}


def __getattr__(name):
    if name in _LAZY_IMPORTS:
        module_path, attr = _LAZY_IMPORTS[name]
        import importlib

        mod = importlib.import_module(module_path, package=__name__)
        return mod if attr is None else getattr(mod, attr)
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
    "extract_pairs_from_sessions",
    "extract_pairs_from_logs",
    "write_training_text",
    "LoRAType",
    "LoRAConfig",
    "LoRALinear",
    "LoRAEmbedding",
    "apply_lora_to_model",
    "get_lora_parameters",
    "VideoCaptionTrainer",
    "list_video_checkpoints",
    "get_training_executor",
    "export_to_sou",
    "DistillConfig",
    "SloughGPTTrainer",
    "TrainerConfig",
    "LongRunRecorder",
    "TokenTree",
    "TokenizerEngine",
    "get_tokenizer_engine",
    "get_tokenizer_manager",
    "get_token_tree_manager",
    "TrainingEngine",
    "get_training_engine",
    "TrainingFeedClient",
    "FeedBatchSampler",
    "get_turbo_lock",
    "get_turbo_state",
    "TrainingOutcomeTracker",
    "get_cache_root",
    "classify_kind",
    "entry_tags",
    "find_corpus_file",
    "guess_mime",
    "write_entry_meta",
    "get_state",
    "ExportConfig",
    "export_model",
    "list_export_formats",
    "AdaptiveConfigEngine",
    "generate_model_card",
    "import_from_sou",
    "list_presets",
    "get_preset",
    "apply_preset",
    "get_auto_trainer",
    "executor",
    "peek_training_executor",
]


# Pin BLAS threads on import of the training stack: OpenBLAS defaults to one
# thread per core, which sync-thrashes the many tiny matmuls of SloNet
# training on a shared box. Runs here because the pin only works via the
# post-import ctypes path (see resource_manager._pin_openblas_threads).
try:
    from domain.infrastructure._internal.resource_manager import (
        get_resource_manager,
    )

    _rm = get_resource_manager()
    _rm.apply_blas_env()
    _rm.apply_compute_limits()
    del _rm
except Exception:  # pragma: no cover - perf nicety must never block training
    pass
