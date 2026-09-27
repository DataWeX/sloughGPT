"""feedback — Feedback storage, meta-weight learning, response tracking.

Public API:
    FeedbackDB, get_feedback_db, Message, Feedback, SimilarPattern
    MetaWeightManager, MetaWeights, get_meta_weight_manager
    ResponseTracker, get_response_tracker
    WorkflowConfig, get_feedback_workflow
    get_per_user_lora, create_training_pipeline
    get_health_monitor, get_message_feedback, HFDPOTrainer
"""

from domain.feedback._internal.database import (
    Feedback,
    FeedbackDB,
    Message,
    SimilarPattern,
    get_feedback_db,
)
from domain.feedback._internal.hf_dpo import HFDPOTrainer
from domain.feedback._internal.message_feedback import get_message_feedback
from domain.feedback._internal.meta_weights import (
    MetaWeightManager,
    MetaWeights,
    get_meta_weight_manager,
)
from domain.feedback._internal.model_health import get_health_monitor
from domain.feedback._internal.per_user_lora import get_per_user_lora
from domain.feedback._internal.response_tracker import (
    ResponseTracker,
    get_response_tracker,
)
from domain.feedback._internal.training import create_training_pipeline
from domain.feedback._internal.workflow import WorkflowConfig, get_feedback_workflow

__all__ = [
    "FeedbackDB",
    "get_feedback_db",
    "Message",
    "Feedback",
    "SimilarPattern",
    "MetaWeightManager",
    "MetaWeights",
    "get_meta_weight_manager",
    "ResponseTracker",
    "get_response_tracker",
    "WorkflowConfig",
    "get_feedback_workflow",
    "get_per_user_lora",
    "create_training_pipeline",
    "get_health_monitor",
    "get_message_feedback",
    "HFDPOTrainer",
]

_LAZY_IMPORTS = {
    "get_lora_evaluator": ("._internal.lora_eval", "get_lora_evaluator"),
}


def __getattr__(name):
    if name in _LAZY_IMPORTS:
        module_path, attr = _LAZY_IMPORTS[name]
        import importlib

        mod = importlib.import_module(module_path, package=__name__)
        return getattr(mod, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
