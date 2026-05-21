"""
Feedback and Meta-Weight Learning Module.

Stores user feedback, conversations, and uses vector search
to retrieve similar good responses for live generation adjustment.
"""

from .database import FeedbackDB, get_feedback_db, Message, Feedback, SimilarPattern
from .meta_weights import MetaWeightManager, MetaWeights, get_meta_weight_manager
from .training import FeedbackTrainer, TrainingExample, DPOPair, create_training_pipeline
from .online_train import OnlineLoRAUpdater, LoRAConfig, get_online_lora_updater
from .per_user_lora import PerUserLoRAStore, UserAdapter, get_per_user_lora
from .workflow import FeedbackWorkflowManager, WorkflowConfig, get_feedback_workflow

__all__ = [
    "FeedbackDB",
    "get_feedback_db",
    "Message",
    "Feedback",
    "SimilarPattern",
    "MetaWeightManager",
    "MetaWeights",
    "get_meta_weight_manager",
    "FeedbackTrainer",
    "TrainingExample",
    "DPOPair",
    "create_training_pipeline",
    "OnlineLoRAUpdater",
    "LoRAConfig",
    "get_online_lora_updater",
    "PerUserLoRAStore",
    "UserAdapter",
    "get_per_user_lora",
    "FeedbackWorkflowManager",
    "WorkflowConfig",
    "get_feedback_workflow",
]
