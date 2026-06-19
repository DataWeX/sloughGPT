"""
Feedback Controller - Business logic for user feedback
"""
from typing import Optional, List, Dict, Any
from datetime import datetime
from pathlib import Path
import json
import uuid
import logging

logger = logging.getLogger("man.controllers.feedback")


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
            logger.info("HF DPO background: %s (pairs=%d)", result.get("status"), len(pairs))
    except Exception as e:
        logger.debug("HF DPO background skipped: %s", e)


class FeedbackController:
    """Controller for user feedback management"""

    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self.feedback_dir = repo_root / "data" / "training_exports"
        self.conversations_dir = repo_root / "data" / "conversations"
        self.conversations_dir.mkdir(parents=True, exist_ok=True)
        self.feedback_dir.mkdir(parents=True, exist_ok=True)
        self._workflow = None
        self._lora_updater = None

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
        """Record user feedback and pipe into learning systems."""
        feedback_id = f"fb_{datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:6]}"

        feedback = {
            "id": feedback_id,
            "message_id": message_id,
            "rating": rating,
            "session_id": session_id,
            "message_content": message_content,
            "user_message": user_message,
            "assistant_response": assistant_response,
            "timestamp": datetime.now().isoformat(),
        }

        self.feedback_dir.mkdir(parents=True, exist_ok=True)
        feedback_file = self.feedback_dir / "feedback.jsonl"
        with open(feedback_file, "a") as f:
            f.write(json.dumps(feedback) + "\n")

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
            import threading
            t = threading.Thread(target=_trigger_hf_dpo, daemon=True)
            t.start()

        return {
            "status": "recorded",
            "feedback_id": feedback_id,
            "message_id": message_id,
            "rating": rating,
            "timestamp": feedback["timestamp"],
        }
    
    def get_feedback(self, message_id: str) -> Optional[Dict[str, Any]]:
        """Get feedback for a message"""
        feedback_file = self.feedback_dir / "feedback.jsonl"
        if not feedback_file.exists():
            return None
        
        with open(feedback_file) as f:
            for line in f:
                fb = json.loads(line)
                if fb.get("message_id") == message_id:
                    return fb
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get feedback statistics"""
        feedback_file = self.feedback_dir / "feedback.jsonl"
        if not feedback_file.exists():
            return {"thumbs_up": 0, "thumbs_down": 0, "total": 0}
        
        thumbs_up = 0
        thumbs_down = 0
        total = 0
        
        with open(feedback_file) as f:
            for line in f:
                fb = json.loads(line)
                total += 1
                if fb.get("rating") == "thumbs_up":
                    thumbs_up += 1
                elif fb.get("rating") == "thumbs_down":
                    thumbs_down += 1
        
        return {
            "thumbs_up": thumbs_up,
            "thumbs_down": thumbs_down,
            "total": total,
            "up_ratio": thumbs_up / total if total > 0 else 0,
        }

    def _load_conversations(self) -> Dict[str, Any]:
        """Load all conversations from disk"""
        conv_file = self.conversations_dir / "conversations.json"
        if conv_file.exists():
            with open(conv_file) as f:
                return json.load(f)
        return {}

    def _save_conversations(self, data: Dict[str, Any]) -> None:
        """Save conversations to disk"""
        conv_file = self.conversations_dir / "conversations.json"
        with open(conv_file, "w") as f:
            json.dump(data, f, indent=2)

    def create_conversation(self, name: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Create a new conversation"""
        conv_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
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
        data = self._load_conversations()
        data[conv_id] = conv
        self._save_conversations(data)
        return conv

    def list_conversations(self, limit: int = 50) -> List[Dict[str, Any]]:
        """List all conversations, sorted by updated_at"""
        data = self._load_conversations()
        convs = sorted(data.values(), key=lambda x: x.get("updated_at", ""), reverse=True)
        return convs[:limit]

    def get_conversation(self, conv_id: str) -> Optional[Dict[str, Any]]:
        """Get a conversation by ID"""
        data = self._load_conversations()
        return data.get(conv_id)

    def update_conversation(self, conv_id: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Update a conversation"""
        data = self._load_conversations()
        if conv_id not in data:
            return None
        data[conv_id].update(updates)
        data[conv_id]["updated_at"] = datetime.now().isoformat()
        self._save_conversations(data)
        return data[conv_id]

    def delete_conversation(self, conv_id: str) -> None:
        """Delete a conversation"""
        data = self._load_conversations()
        if conv_id in data:
            del data[conv_id]
            self._save_conversations(data)


_feedback_controller: Optional[FeedbackController] = None


def get_feedback_controller() -> FeedbackController:
    global _feedback_controller
    if _feedback_controller is None:
        repo_root = Path(__file__).parent.parent.parent.parent
        _feedback_controller = FeedbackController(repo_root)
    return _feedback_controller