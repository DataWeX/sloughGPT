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
