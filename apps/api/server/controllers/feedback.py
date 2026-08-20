"""
Feedback Controller - Business logic for user feedback

Storage backed by MogDB (the project's embedded document DB).
Conversations and feedback records are stored in indexed collections
instead of raw JSON/JSONL files.
"""
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone
from pathlib import Path
import os
import uuid
import logging

logger = logging.getLogger("slo.controllers.feedback")


def _trigger_hf_dpo():
    """Run HF DPO in background thread using the active model."""
    try:
        import state as server_state
        model = getattr(server_state, "model", None)
        tokenizer = getattr(server_state, "tokenizer", None)
        if model is None or tokenizer is None:
            return
        from domains.feedback.hf_dpo import HFDPOTrainer
        trainer = HFDPOTrainer(model=model, tokenizer=tokenizer)
        pairs = trainer.prepare_dpo_pairs()
        if len(pairs) >= 2:
            result = trainer.train(pairs=pairs)
            logger.info("HF DPO background: %s (pairs=%d)", result.get("status"), len(pairs), extra={"tag": "INFRA"})
    except Exception as e:
        logger.debug("HF DPO background skipped: %s", e)


def _get_mogdb(db_path: str):
    """Create a MogDB instance at the given path."""
    from mogdb import MogDB
    return MogDB(db_path)


class FeedbackController:
    """Controller for user feedback management.

    Uses MogDB collections for feedback records and conversations
    instead of raw JSON/JSONL file I/O.
    """

    def __init__(self, repo_root: Path, db_path: Optional[str] = None):
        self.repo_root = repo_root
        if db_path is None:
            db_path = str(repo_root / "data" / "feedback_mogdb")
        self._db = _get_mogdb(db_path)
        self._feedback = self._db.collection("feedback_records")
        self._conversations = self._db.collection("feedback_conversations")
        self._feedback.create_index("message_id")
        self._feedback.create_index("rating")
        self._feedback.create_sorted_index("timestamp")
        self._conversations.create_sorted_index("updated_at")
        self._workflow = None
        self._lora_updater = None

    @property
    def feedback_collection(self):
        """Expose the feedback collection for testing."""
        return self._feedback

    @property
    def conversations_collection(self):
        """Expose the conversations collection for testing."""
        return self._conversations

    def _get_workflow(self):
        """Lazy-load feedback workflow and wire the current model."""
        if self._workflow is None:
            try:
                from domains.feedback.workflow import get_feedback_workflow
                self._workflow = get_feedback_workflow()
                self._wire_model()
            except Exception as e:
                logger.debug("Feedback workflow init failed: %s", e)
        return self._workflow

    def _wire_model(self):
        """Set the current auto-train model on the workflow for background training."""
        try:
            from routers.auto_train import state as at_state
            if self._workflow and at_state.student_net is not None:
                self._workflow.set_model(at_state.student_net, at_state.student_tokenizer)
                return
            if self._workflow:
                import state as server_state
                model = getattr(server_state, "model", None)
                tokenizer = getattr(server_state, "tokenizer", None)
                if model is not None and tokenizer is not None:
                    self._workflow.set_model(model, tokenizer)
        except Exception as e:
            logger.debug("Failed to wire auto-train model to workflow: %s", e)

    def _get_lora_updater(self):
        """Lazy-load online LoRA updater."""
        if self._lora_updater is None:
            try:
                from domains.feedback.online_train import get_online_lora_updater
                self._lora_updater = get_online_lora_updater()
            except Exception as e:
                logger.debug("Online LoRA updater init failed: %s", e)
        return self._lora_updater

    def record_feedback(
        self,
        message_id: str,
        rating: str,
        session_id: Optional[str] = None,
        message_content: Optional[str] = None,
        user_message: Optional[str] = None,
        assistant_response: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Record user feedback and pipe into learning systems.

        Args:
            message_id: The ID of the message being rated.
            rating: 'thumbs_up', 'thumbs_down', or 'neutral'.
            session_id: Optional session ID.
            message_content: The assistant message content.
            user_message: The user's message.
            assistant_response: The assistant's response.

        Returns:
            Dict with status, feedback_id, message_id, rating, timestamp.

        Side effects:
            - Inserts a feedback record into MogDB.
            - Pipes into learning pipeline (workflow + LoRA updater).
            - Triggers HF DPO on thumbs-down.
        """
        feedback_id = f"fb_{datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:6]}"

        feedback = {
            "id": feedback_id,
            "message_id": message_id,
            "rating": rating,
            "session_id": session_id,
            "message_content": message_content,
            "user_message": user_message,
            "assistant_response": assistant_response,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        self._feedback.insert_one(feedback)

        # Pipe into learning pipeline
        if user_message and assistant_response:
            workflow = self._get_workflow()
            if workflow:
                workflow.record_feedback(
                    user_message=user_message,
                    assistant_response=assistant_response,
                    rating=rating,
                    conversation_id=session_id,
                    user_id="default",
                )
            lora = self._get_lora_updater()
            if lora:
                lora.add_feedback(
                    prompt=user_message,
                    response=assistant_response,
                    rating=rating,
                    quality_score=1.0 if rating == "thumbs_up" else 0.0,
                )

        # Trigger HF DPO in background on thumbs-down
        if rating == "thumbs_down":
            from domains.training.executor import get_training_executor
            executor = get_training_executor()
            executor.submit(_trigger_hf_dpo, f"dpo_{feedback_id}")

        return {
            "status": "recorded",
            "feedback_id": feedback_id,
            "message_id": message_id,
            "rating": rating,
            "timestamp": feedback["timestamp"],
        }

    def get_feedback(self, message_id: str) -> Optional[Dict[str, Any]]:
        """Get feedback for a message.

        Args:
            message_id: The message ID to look up.

        Returns:
            The feedback record dict, or None if not found.
        """
        return self._feedback.find_one({"message_id": message_id})

    def get_stats(self) -> Dict[str, Any]:
        """Get feedback statistics.

        Returns:
            Dict with thumbs_up, thumbs_down, total, up_ratio.
        """
        total = self._feedback.count()
        if total == 0:
            return {"thumbs_up": 0, "thumbs_down": 0, "total": 0, "up_ratio": 0}

        thumbs_up = self._feedback.count({"rating": "thumbs_up"})
        thumbs_down = self._feedback.count({"rating": "thumbs_down"})

        return {
            "thumbs_up": thumbs_up,
            "thumbs_down": thumbs_down,
            "total": total,
            "up_ratio": thumbs_up / total if total > 0 else 0,
        }

    def create_conversation(self, name: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Create a new conversation.

        Args:
            name: Conversation name.
            session_id: Optional session ID to associate.

        Returns:
            Dict with id, name, session_id, created_at, updated_at, pinned, starred, message_count.
        """
        conv_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        conv = {
            "id": conv_id,
            "name": name,
            "session_id": session_id or str(uuid.uuid4()),
            "created_at": now,
            "updated_at": now,
            "pinned": False,
            "starred": False,
            "message_count": 0,
        }
        self._conversations.insert_one(conv)
        return conv

    def list_conversations(self, limit: int = 50) -> List[Dict[str, Any]]:
        """List conversations sorted by most recently updated.

        Args:
            limit: Maximum number of conversations to return.

        Returns:
            List of conversation dicts, newest first.
        """
        return self._conversations.find(
            sort=[("updated_at", -1)],
            limit=limit,
        )

    def get_conversation(self, conv_id: str) -> Optional[Dict[str, Any]]:
        """Get a conversation by ID.

        Args:
            conv_id: The conversation UUID.

        Returns:
            Conversation dict or None if not found.
        """
        return self._conversations.find_one({"id": conv_id})

    def update_conversation(self, conv_id: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Update a conversation's fields.

        Args:
            conv_id: The conversation UUID.
            updates: Dict of fields to update.

        Returns:
            Updated conversation dict, or None if not found.
        """
        updates["updated_at"] = datetime.now(timezone.utc).isoformat()
        return self._conversations.find_one_and_update(
            {"id": conv_id},
            {"$set": updates},
            return_document="after",
        )

    def delete_conversation(self, conv_id: str) -> None:
        """Delete a conversation by ID.

        Args:
            conv_id: The conversation UUID.
        """
        self._conversations.delete_one({"id": conv_id})


_feedback_controller: Optional[FeedbackController] = None


def get_feedback_controller() -> FeedbackController:
    """Get or create the global FeedbackController singleton.

    The database path defaults to ``<repo>/data/feedback_mogdb`` and
    can be overridden via ``MOGDB_FEEDBACK_PATH`` environment variable.
    """
    global _feedback_controller
    if _feedback_controller is None:
        repo_root = Path(__file__).parent.parent.parent.parent
        db_path = os.environ.get("MOGDB_FEEDBACK_PATH")
        _feedback_controller = FeedbackController(repo_root, db_path=db_path)
    return _feedback_controller


def set_feedback_controller(controller: FeedbackController) -> None:
    """Replace the global FeedbackController singleton.

    Used by tests to inject a controller backed by a temporary database.
    """
    global _feedback_controller
    _feedback_controller = controller


def reset_feedback_controller() -> None:
    """Reset the global singleton so the next call creates a fresh instance."""
    global _feedback_controller
    _feedback_controller = None
