"""
Automated Feedback Workflow Manager.

Orchestrates the complete feedback → training pipeline:
- Records feedback and updates adapters automatically
- Scheduled aggregation and pruning
- Periodic training data export
- Health monitoring and stats
"""

import copy
import threading
import time
from typing import Optional, Dict, Any
from dataclasses import dataclass

from .database import FeedbackDB, get_feedback_db
from .meta_weights import MetaWeightManager, get_meta_weight_manager
from .online_train import OnlineLoRAUpdater, get_online_lora_updater
from .per_user_lora import PerUserLoRAStore, get_per_user_lora


@dataclass
class WorkflowConfig:
    """Configuration for automated feedback workflow."""

    aggregate_interval_minutes: int = 60
    prune_interval_minutes: int = 120
    auto_dpo_interval_minutes: int = 120
    export_interval_hours: int = 24

    health_check_interval_seconds: int = 30

    auto_aggregate_threshold: int = 50
    auto_prune_threshold: int = 100
    min_feedback_for_aggregation: int = 3

    export_format: str = "dpo"
    export_path: str = "data/training_exports"


class FeedbackWorkflowManager:
    """
    Manages the complete automated feedback workflow.

    Runs scheduled tasks for:
    - Periodic aggregation of user adapters
    - Pruning of low-quality adapters
    - Exporting training data
    - Health monitoring
    """

    def __init__(
        self,
        config: WorkflowConfig = None,
        feedback_db: FeedbackDB = None,
        meta_manager: MetaWeightManager = None,
        lora_store: PerUserLoRAStore = None,
        lora_updater: OnlineLoRAUpdater = None,
    ):
        self.config = config or WorkflowConfig()

        self.db = feedback_db or get_feedback_db()
        self.meta_manager = meta_manager or get_meta_weight_manager()

        if lora_store:
            self.lora_store = lora_store
        else:
            self.lora_store = get_per_user_lora()
            self.lora_store.auto_aggregate_threshold = self.config.auto_aggregate_threshold
            self.lora_store.auto_prune_threshold = self.config.auto_prune_threshold
            self.lora_store.min_feedback_for_aggregation = self.config.min_feedback_for_aggregation

        self.lora_updater = lora_updater or get_online_lora_updater()

        self._running = False
        self._scheduler_thread: Optional[threading.Thread] = None
        self._health_thread: Optional[threading.Thread] = None
        self._last_aggregate_time: float = 0
        self._last_prune_time: float = 0
        self._last_export_time: float = 0
        self._last_dpo_time: float = 0
        self._last_rollback_time: float = 0.0
        self._last_health_check: float = 0

        self._new_thumbs_up = 0
        self._auto_train_threshold = 3

        self._stats = {
            "workflow_runs": 0,
            "aggregations_performed": 0,
            "prunes_performed": 0,
            "exports_performed": 0,
            "feedback_recorded": 0,
            "auto_train_steps": 0,
            "dpo_train_steps": 0,
            "dpo_train_rejected": 0,
            "user_adapter_trained": 0,
            "user_adapter_rejected": 0,
            "start_time": None,
        }

    def set_model(self, model, tokenizer):
        """Set the model and tokenizer for auto-training on feedback."""
        self._model = model
        self._tokenizer = tokenizer

    def record_feedback(
        self,
        user_message: str,
        assistant_response: str,
        rating: str,
        conversation_id: str = None,
        quality_score: float = None,
        user_id: str = "default",
    ) -> str:
        """
        Record feedback and trigger automatic updates.

        This is the main entry point - records feedback and
        automatically updates all learning systems.
        """
        feedback_id = self.meta_manager.record_feedback(
            user_message=user_message,
            assistant_response=assistant_response,
            rating=rating,
            conversation_id=conversation_id,
            quality_score=quality_score,
            user_id=user_id,
        )

        self.lora_updater.add_feedback(
            prompt=user_message,
            response=assistant_response,
            rating=rating,
            quality_score=quality_score,
        )

        self.lora_store.update_adapter(
            user_id=user_id,
            feedback_signal=1.0 if rating == "thumbs_up" else -1.0,
        )

        self._stats["feedback_recorded"] += 1

        if rating == "thumbs_up":
            self._new_thumbs_up += 1
            if self._new_thumbs_up >= self._auto_train_threshold:
                self._new_thumbs_up = 0
                self._maybe_auto_train()
            self._maybe_train_user_adapter(user_id)

        if rating == "thumbs_down":
            self._maybe_dpo_train()
            self._maybe_train_user_adapter(user_id)

        return feedback_id

    def _benchmark_ppl(self, net, tok) -> Optional[float]:
        """Evaluate perplexity across multiple benchmark phrases."""
        try:
            import numpy as np
            from domains.training.slonet import tensor, SloLSTM

            benchmarks = [
                "the quick brown fox jumps over the lazy dog",
                "hello how are you doing today",
                "artificial intelligence is transforming the world",
                "once upon a time in a faraway land",
                "the future of technology depends on innovation",
                "machine learning models learn from data",
                "natural language processing enables computers to understand text",
                "deep neural networks have many layers of computation",
            ]

            lstm_layers = [l for l in net.layers if isinstance(l, SloLSTM)]
            if not lstm_layers:
                return None
            lstm = lstm_layers[0]

            all_ppls = []
            for benchmark in benchmarks:
                input_ids = tok.encode(benchmark)
                if len(input_ids) < 2:
                    continue

                log_probs = []
                for i in range(len(input_ids) - 1):
                    ctx = np.array([input_ids[: i + 1][-64:]], dtype=np.int64)
                    x = tensor(ctx, requires_grad=False)
                    h = lstm.init_hidden()
                    logits_t, _ = lstm.forward(x, h)
                    l = logits_t.data[0, -1]
                    l = np.where(np.isfinite(l), l, -1e9)
                    ps = np.exp(l - l.max())
                    ps = ps / (ps.sum() + 1e-10)
                    actual = input_ids[i + 1]
                    if actual < len(ps):
                        log_probs.append(np.log(ps[actual] + 1e-10))

                if log_probs:
                    avg_nll = -sum(log_probs) / len(log_probs)
                    all_ppls.append(float(np.exp(avg_nll)))

            if not all_ppls:
                return None
            return float(np.mean(all_ppls))
        except Exception:
            return None

    def _snapshot_weights(self, net):
        """Deep copy all model weights for rollback."""
        return {
            str(i): copy.deepcopy(layer.weight.data)
            for i, layer in enumerate(net.layers)
            if hasattr(layer, 'weight') and layer.weight is not None
        }

    def _restore_weights(self, net, snapshot):
        """Restore model weights from a snapshot."""
        for i, layer in enumerate(net.layers):
            key = str(i)
            if key in snapshot and hasattr(layer, 'weight'):
                layer.weight.data = snapshot[key]

    def _maybe_auto_train(self):
        """Run a mini training step with a quality guard.

        Before training, snapshots model weights and measures perplexity.
        After training, re-measures perplexity. If quality degraded (>5%),
        rolls back the weights and logs the rejection.
        Uses gradient clipping and cosine LR scheduling for stable updates.
        """
        import logging
        import numpy as np
        logger = logging.getLogger("sloughgpt.feedback")

        net = getattr(self, '_model', None)
        tok = getattr(self, '_tokenizer', None)
        if net is None or tok is None:
            logger.debug("Auto-train skipped: no model set")
            return

        try:
            from .training import FeedbackTrainer
            from domains.training.slonet import SloAdam, cross_entropy, tensor, create_scheduler
            from .model_health import get_health_monitor

            if not hasattr(tok, 'encode'):
                return

            trainer = FeedbackTrainer()
            sft_data = trainer.prepare_sft_data(min_quality=0.5)
            if not sft_data:
                return

            texts = []
            for item in sft_data:
                prompt = (item.get("prompt") or "").strip()
                response = (item.get("response") or "").strip()
                if prompt and response:
                    texts.append(f"user: {prompt}\nassistant: {response}")
            if not texts:
                return

            before_ppl = self._benchmark_ppl(net, tok)
            snapshot = self._snapshot_weights(net)
            texts = texts[:8]
            optimizer = SloAdam(lr=0.0005, max_grad_norm=1.0)
            steps = 0
            total_loss = 0.0
            total_steps = len(texts) * 2 * 4
            scheduler = create_scheduler(optimizer, "cosine", total_steps=max(total_steps, 1), warmup_steps=4, min_lr=1e-5)

            for epoch in range(2):
                for text in texts:
                    input_ids = tok.encode(text[:256])
                    if len(input_ids) < 2:
                        continue
                    seq_len = min(len(input_ids) - 1, 64)
                    chunk_size = 16
                    for i in range(0, seq_len, chunk_size):
                        x_chunk = input_ids[i:i+chunk_size]
                        y_chunk = input_ids[i+1:i+chunk_size+1]
                        while len(x_chunk) < chunk_size:
                            x_chunk.append(tok.pad_id)
                        while len(y_chunk) < chunk_size:
                            y_chunk.append(tok.pad_id)
                        x = tensor([[x_chunk]], requires_grad=True)
                        y = tensor([[y_chunk]])
                        lstm = net.layers[1] if len(net.layers) > 1 else net.layers[0]
                        if not hasattr(lstm, 'init_hidden'):
                            continue
                        hidden = lstm.init_hidden()
                        logits, _ = lstm.forward(x, hidden)
                        loss = cross_entropy(logits, y.reshape(-1))
                        loss.backward()
                        optimizer.step(net.parameters())
                        scheduler.step()
                        steps += 1
                        total_loss += loss.data[()]

            after_ppl = self._benchmark_ppl(net, tok)
            avg_loss = total_loss / max(steps, 1)
            ppl_delta = None
            rejected = False

            if before_ppl is not None and after_ppl is not None:
                ppl_delta = ((after_ppl - before_ppl) / before_ppl) * 100
                if ppl_delta > 5.0:
                    self._restore_weights(net, snapshot)
                    rejected = True
                    self._last_rollback_time = time.time()
                    logger.warning(
                        f"Auto-train rejected: PPL increased {ppl_delta:+.1f}% "
                        f"({before_ppl:.1f} → {after_ppl:.1f})"
                    )

            self._stats["auto_train_steps"] += 1
            if rejected:
                self._stats.setdefault("auto_train_rejected", 0)
                self._stats["auto_train_rejected"] += 1
                logger.info(
                    f"Auto-train rolled back (quality guard): loss={avg_loss:.4f}, "
                    f"ppl_delta={ppl_delta:+.1f}%"
                )
            else:
                logger.info(
                    f"Auto-trained on feedback: {steps} steps, loss={avg_loss:.4f}, "
                    f"ppl={after_ppl:.1f if after_ppl else '?'} "
                    f"({ppl_delta:+.1f}% vs before)" if ppl_delta else ""
                )

            try:
                monitor = get_health_monitor()
                monitor.set_model(net, tok)
                monitor.run_benchmark()
            except Exception:
                pass
        except Exception as e:
            logger.warning(f"Auto-train skipped: {e}")

    def _maybe_dpo_train(self):
        """Run DPO preference learning on mixed-feedback pairs.

        When both thumbs-up and thumbs-down exist for the same prompt,
        trains the model to prefer the liked response and avoid the disliked one.

        Uses gradient descent on chosen responses + gradient ascent on rejected
        responses, producing the same effect as DPO: chosen probability up,
        rejected probability down. Quality guard prevents degradation.
        Uses gradient clipping and cosine LR scheduling for stable updates.
        """
        import logging
        import numpy as np
        logger = logging.getLogger("sloughgpt.feedback")

        net = getattr(self, '_model', None)
        tok = getattr(self, '_tokenizer', None)
        if net is None or tok is None:
            return

        try:
            from .training import FeedbackTrainer
            from domains.training.slonet import SloAdam, cross_entropy, tensor, create_scheduler

            trainer = FeedbackTrainer()
            pairs = trainer.prepare_dpo_pairs()
            if len(pairs) < 2:
                return

            before_ppl = self._benchmark_ppl(net, tok)
            snapshot = self._snapshot_weights(net)
            pairs = pairs[:6]
            optimizer = SloAdam(lr=0.0003, max_grad_norm=1.0)
            steps = 0
            total_loss = 0.0
            total_steps = len(pairs) * 2 * 4
            scheduler = create_scheduler(optimizer, "cosine", total_steps=max(total_steps, 1), warmup_steps=3, min_lr=1e-5)

            for pair in pairs:
                chosen_text = f"user: {pair.prompt}\nassistant: {pair.chosen}"
                rejected_text = f"user: {pair.prompt}\nassistant: {pair.rejected}"

                chosen_ids = tok.encode(chosen_text[:256])
                rejected_ids = tok.encode(rejected_text[:256])

                if len(chosen_ids) < 2 or len(rejected_ids) < 2:
                    continue

                def _train_sequence(ids, do_ascent=False):
                    nonlocal steps, total_loss
                    seq_len = min(len(ids) - 1, 64)
                    chunk_size = 16
                    for i in range(0, seq_len, chunk_size):
                        x_chunk = ids[i:i+chunk_size]
                        y_chunk = ids[i+1:i+chunk_size+1]
                        while len(x_chunk) < chunk_size:
                            x_chunk.append(tok.pad_id)
                        while len(y_chunk) < chunk_size:
                            y_chunk.append(tok.pad_id)
                        x = tensor([[x_chunk]], requires_grad=True)
                        y = tensor([[y_chunk]])
                        lstm = net.layers[1] if len(net.layers) > 1 else net.layers[0]
                        if not hasattr(lstm, 'init_hidden'):
                            continue
                        hidden = lstm.init_hidden()
                        logits, _ = lstm.forward(x, hidden)
                        loss = cross_entropy(logits, y.reshape(-1))
                        loss.backward()
                        if do_ascent:
                            for p in net.parameters():
                                if p.grad is not None:
                                    p.grad.data = -p.grad.data
                        optimizer.step(net.parameters())
                        scheduler.step()
                        steps += 1
                        total_loss += loss.data[()]

                _train_sequence(chosen_ids, do_ascent=False)
                _train_sequence(rejected_ids, do_ascent=True)

            after_ppl = self._benchmark_ppl(net, tok)
            avg_loss = total_loss / max(steps, 1)
            ppl_delta = None
            rejected = False

            if before_ppl is not None and after_ppl is not None:
                ppl_delta = ((after_ppl - before_ppl) / before_ppl) * 100
                if ppl_delta > 5.0:
                    self._restore_weights(net, snapshot)
                    rejected = True
                    self._last_rollback_time = time.time()
                    logger.warning(
                        f"DPO train rejected: PPL increased {ppl_delta:+.1f}% "
                        f"({before_ppl:.1f} → {after_ppl:.1f})"
                    )

            self._stats["dpo_train_steps"] = self._stats.get("dpo_train_steps", 0) + 1
            if rejected:
                self._stats["dpo_train_rejected"] = self._stats.get("dpo_train_rejected", 0) + 1
                logger.info(
                    f"DPO train rolled back: loss={avg_loss:.4f}, ppl_delta={ppl_delta:+.1f}%"
                )
            else:
                logger.info(
                    f"DPO trained on {len(pairs)} pairs: {steps} steps, loss={avg_loss:.4f}, "
                    f"ppl={after_ppl:.1f if after_ppl else '?'}"
                )
        except Exception as e:
            logger.warning(f"DPO train skipped: {e}")

    def _do_aggregate(self) -> None:
        """Run adapter aggregation and log the result.

        Calls the per‑user LoRA store to merge the top‑k adapters into a
        single `.soul` checkpoint, runs the evaluation pipeline, and updates the
        workflow stats. Errors are caught and logged so the scheduler never
        crashes.
        """
        import logging
        logger = logging.getLogger("sloughgpt.feedback")
        try:
            # Use defaults: top‑k 10, min_feedback_count 5, run_eval=True
            result = self.lora_store.aggregate_best_adapters()
            self._stats.setdefault("aggregations_performed", 0)
            self._stats["aggregations_performed"] += 1
            logger.info("[Workflow] Adapter aggregation completed: %s", result)
        except Exception as e:
            logger.error("[Workflow] Adapter aggregation failed: %s", e)

    def _maybe_train_user_adapter(self, user_id: str):
        """Train a per-user adapter using accumulated feedback."""
        import logging
        import numpy as np
        from pathlib import Path
        logger = logging.getLogger("sloughgpt.feedback")

        net = getattr(self, '_model', None)
        tok = getattr(self, '_tokenizer', None)
        if net is None or tok is None:
            return

        try:
            from .training import FeedbackTrainer
            from domains.training.slonet import SloAdam, cross_entropy, tensor, SloLSTM, SloAdapterLayer

            trainer = FeedbackTrainer()
            sft_data = trainer.prepare_sft_data(min_quality=0.3)
            user_examples = [d for d in sft_data if d.get("user_id", "default") == user_id]
            if not user_examples:
                user_examples = sft_data[:4]
            if not user_examples:
                return

            texts = []
            for item in user_examples:
                prompt = (item.get("prompt") or "").strip()
                response = (item.get("response") or "").strip()
                if prompt and response:
                    texts.append(f"user: {prompt}\nassistant: {response}")
            if not texts:
                return

            dim = getattr(net, 'hidden_dim', 768)
            adapter = SloAdapterLayer(dim=dim, rank=8, name=f"adapter_{user_id}")

            lstm_layers = [l for l in net.layers if isinstance(l, SloLSTM)]
            if not lstm_layers:
                return
            lstm = lstm_layers[0]

            before_ppl = self._benchmark_ppl(net, tok)

            texts = texts[:6]
            optimizer = SloAdam(lr=0.001)
            steps = 0
            total_loss = 0.0

            for epoch in range(3):
                for text in texts:
                    input_ids = tok.encode(text[:192])
                    if len(input_ids) < 2:
                        continue
                    seq_len = min(len(input_ids) - 1, 32)
                    chunk_size = 8
                    for i in range(0, seq_len, chunk_size):
                        x_chunk = input_ids[i:i+chunk_size]
                        y_chunk = input_ids[i+1:i+chunk_size+1]
                        while len(x_chunk) < chunk_size:
                            x_chunk.append(tok.pad_id)
                        while len(y_chunk) < chunk_size:
                            y_chunk.append(tok.pad_id)
                        x = tensor([[x_chunk]], requires_grad=True)
                        y = tensor([[y_chunk]])
                        hidden = lstm.init_hidden()
                        logits, _ = lstm.forward(x, hidden, adapter=adapter)
                        loss = cross_entropy(logits, y.reshape(-1))
                        loss.backward()
                        optimizer.step(adapter.parameters())
                        for p in net.parameters():
                            if p.grad is not None:
                                p.grad = None
                        steps += 1
                        total_loss += loss.data[()]

            after_ppl = self._benchmark_ppl(net, tok)
            avg_loss = total_loss / max(steps, 1)
            ppl_delta = None
            rejected = False

            if before_ppl is not None and after_ppl is not None:
                ppl_delta = ((after_ppl - before_ppl) / before_ppl) * 100
                if ppl_delta > 5.0:
                    rejected = True
                    self._last_rollback_time = time.time()
                    logger.warning(
                        f"User adapter rejected: PPL increased {ppl_delta:+.1f}%"
                    )

            adapter_path = Path("data/user_adapters") / f"{user_id}_adapter.npz"
            adapter_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                str(adapter_path),
                down_weight=adapter.down_proj.weight.data,
                up_weight=adapter.up_proj.weight.data,
                dim=dim,
                rank=8,
                user_id=user_id,
                steps=steps,
                loss=avg_loss,
                ppl_delta=ppl_delta or 0.0,
                rejected=rejected,
            )

            if not rejected and hasattr(net, 'get_user_adapter'):
                model_adapter = net.get_user_adapter(user_id, dim=dim, rank=8)
                model_adapter.down_proj.weight.data = adapter.down_proj.weight.data.copy()
                model_adapter.up_proj.weight.data = adapter.up_proj.weight.data.copy()
                net._active_user_id = user_id

            self._stats["user_adapter_trained"] = self._stats.get("user_adapter_trained", 0) + 1
            if rejected:
                self._stats["user_adapter_rejected"] = self._stats.get("user_adapter_rejected", 0) + 1

            logger.info(
                f"User adapter {'rejected' if rejected else 'trained'} for {user_id}: "
                f"{steps} steps, loss={avg_loss:.4f}"
            )
        except Exception as e:
            logger.warning(f"User adapter training skipped: {e}")

    def run_scheduled_tasks(self):
        """Execute periodic tasks based on configured intervals.

        - Aggregate adapters every ``aggregate_interval_minutes``.
        - Prune low‑quality adapters every ``prune_interval_minutes``.
        - Export training data every ``export_interval_hours``.

        Updates the ``_last_*`` timestamps so tasks are not run too
        frequently. Errors are caught and logged – the scheduler must never
        abort the workflow.
        """
        import logging
        logger = logging.getLogger("sloughgpt.feedback")
        now = time.time()
        # Aggregation
        if now - self._last_aggregate_time >= self.config.aggregate_interval_minutes * 60:
            try:
                self._do_aggregate()
                self._last_aggregate_time = now
            except Exception as e:
                logger.error("[Workflow] Scheduled aggregation failed: %s", e)
        # Pruning
        if now - self._last_prune_time >= self.config.prune_interval_minutes * 60:
            try:
                self._do_prune()
                self._last_prune_time = now
            except Exception as e:
                logger.error("[Workflow] Scheduled pruning failed: %s", e)
        # Export
        if now - self._last_export_time >= self.config.export_interval_hours * 3600:
            try:
                self._do_export()
                self._last_export_time = now
            except Exception as e:
                logger.error("[Workflow] Scheduled export failed: %s", e)
        # DPO training
        if now - self._last_dpo_time >= self.config.auto_dpo_interval_minutes * 60:
            try:
                self._do_dpo()
                self._last_dpo_time = now
            except Exception as e:
                logger.error("[Workflow] Scheduled DPO failed: %s", e)

    def _do_prune(self):
        """Perform pruning task."""
        try:
            deleted = self.lora_store.prune_low_quality(
                min_feedback_count=1,
                max_age_days=7,
            )
            if deleted:
                self._stats["prunes_performed"] += 1
                print(f"[Workflow] Pruned {len(deleted)} adapters")
        except Exception as e:
            print(f"[Workflow] Prune error: {e}")


    def _do_export(self):
        """Perform training data export task."""
        try:
            from pathlib import Path

            export_path = Path(self.config.export_path)
            export_path.mkdir(parents=True, exist_ok=True)

            timestamp = int(time.time())
            filepath = export_path / f"feedback_export_{timestamp}.jsonl"

            self.db.export_feedback_jsonl(str(filepath))
            self._stats["exports_performed"] += 1
            print(f"[Workflow] Exported training data to {filepath}")
        except Exception as e:
            print(f"[Workflow] Export error: {e}")

    def _do_dpo(self):
        """Run scheduled DPO preference learning on mixed-feedback pairs.

        Invoked by the scheduler when enough thumbs‑down feedback has been
        accumulated. Delegates to the same `_maybe_dpo_train` logic that is
        also triggered on individual thumbs‑down events.
        """
        import logging
        logger = logging.getLogger("sloughgpt.feedback")
        try:
            self._maybe_dpo_train()
            self._stats["dpo_train_steps"] = self._stats.get("dpo_train_steps", 0) + 1
        except Exception as e:
            logger.error("[Workflow] Scheduled DPO failed: %s", e)

    def _health_check(self):
        """Perform health check and run scheduled tasks."""
        try:
            self.run_scheduled_tasks()

            self._stats["workflow_runs"] += 1
            self._last_health_check = time.time()
        except Exception as e:
            print(f"[Workflow] Health check error: {e}")

    def start(self):
        """Start the automated workflow in background threads."""
        if self._running:
            return

        self._running = True
        self._stats["start_time"] = time.time()

        def scheduler_loop():
            while self._running:
                self._health_check()
                time.sleep(self.config.health_check_interval_seconds)

        self._scheduler_thread = threading.Thread(target=scheduler_loop, daemon=True)
        self._scheduler_thread.start()

        print("[Workflow] Started automated feedback workflow")

    def stop(self):
        """Stop the automated workflow."""
        self._running = False
        print("[Workflow] Stopped automated feedback workflow")

    def get_status(self) -> Dict[str, Any]:
        """Get current workflow status and statistics."""
        return {
            "running": self._running,
            "stats": self._stats.copy(),
            "pending_thumbs_up": self._new_thumbs_up,
            "auto_train_threshold": self._auto_train_threshold,
            "config": {
                "aggregate_interval_minutes": self.config.aggregate_interval_minutes,
                "prune_interval_minutes": self.config.prune_interval_minutes,
                "export_interval_hours": self.config.export_interval_hours,
                "auto_dpo_interval_minutes": self.config.auto_dpo_interval_minutes,
                "health_check_interval_seconds": self.config.health_check_interval_seconds,
            },
            "last_runs": {
                "aggregate": self._last_aggregate_time,
                "prune": self._last_prune_time,
                "export": self._last_export_time,
                "dpo": self._last_dpo_time,
                "health_check": self._last_health_check,
                "last_rollback": self._last_rollback_time,
            },
            "systems": {
                "feedback_db": self.db.get_stats() if hasattr(self.db, "get_stats") else {},
                "meta_weights": self.meta_manager.get_stats()
                if hasattr(self.meta_manager, "get_stats")
                else {},
                "lora_store": self.lora_store.get_stats(),
                "lora_updater": self.lora_updater.get_stats()
                if hasattr(self.lora_updater, "get_stats")
                else {},
            },
        }

    def trigger_aggregate(self) -> Dict[str, Any]:
        """Manually trigger aggregation."""
        self._do_aggregate()
        return {"status": "aggregated", "timestamp": time.time()}

    def trigger_prune(self) -> Dict[str, Any]:
        """Manually trigger pruning."""
        self._do_prune()
        return {"status": "pruned", "timestamp": time.time()}

    def trigger_export(self) -> Dict[str, Any]:
        """Manually trigger export."""
        self._do_export()
        return {"status": "exported", "timestamp": time.time()}


_workflow_manager: Optional[FeedbackWorkflowManager] = None


def get_feedback_workflow(
    config: WorkflowConfig = None,
) -> FeedbackWorkflowManager:
    """Get or create the global feedback workflow manager."""
    global _workflow_manager
    if _workflow_manager is None:
        _workflow_manager = FeedbackWorkflowManager(config=config)
    return _workflow_manager
