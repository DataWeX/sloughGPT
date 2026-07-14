"""
HF DPO Trainer — Direct Preference Optimization for HuggingFace models.

Uses the same logic as the SloNet DPO in workflow.py but targets
HF Transformers models (Qwen, GPT-2, etc.) with optional LoRA adapters.

Architecture:
  Feedback DB → prepare_dpo_pairs()
  For each (chosen, rejected, prompt):
    chosen:  forward → cross-entropy → backward (descent)
    rejected: forward → cross-entropy → backward → negate grads (ascent)
  Quality guard: PPL benchmark → rollback if >5% degradation
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    from domains.training.slonet_compat import torch
    nn = torch.nn
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        PreTrainedTokenizer,
        PreTrainedTokenizerFast,
    )
    _HF_DPO_AVAILABLE = True
except ImportError:
    _HF_DPO_AVAILABLE = False
from typing import Union

logger = logging.getLogger("man.hf_dpo")


@dataclass
class DPOPair:
    chosen: str
    rejected: str
    prompt: str


_BENCHMARK_PROMPTS = [
    "What is the capital of France?",
    "Explain gravity in simple terms.",
    "Write a short poem about a cat.",
    "What is 2 + 2?",
    "Describe the color blue.",
    "How do you make a sandwich?",
    "What is the meaning of life?",
    "Tell me a joke.",
]


class HFDPOTrainer:
    """DPO training for HuggingFace causal LM models.

    Args:
        model: A HuggingFace ``AutoModelForCausalLM`` instance (with or without LoRA).
        tokenizer: Corresponding tokenizer.
        device: Torch device (default: auto-detect)
        learning_rate: Optimizer learning rate (default: 5e-6 for LoRA)
        max_grad_norm: Gradient clipping norm (default: 1.0)
        weight_decay: AdamW weight decay (default: 0.01)
        db_path: Path to feedback SQLite DB
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        device: Optional[str] = None,
        learning_rate: float = 5e-6,
        max_grad_norm: float = 1.0,
        weight_decay: float = 0.01,
        db_path: str = "data/feedback.db",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.lr = learning_rate
        self.max_grad_norm = max_grad_norm
        self.weight_decay = weight_decay
        self.db_path = db_path
        self.model.to(self.device)

        # Stats
        self.steps = 0
        self.total_loss = 0.0
        self.accepted_count = 0
        self.rejected_count = 0

    # ── Data preparation ──────────────────────────────────────────

    def prepare_dpo_pairs(self, min_pairs: int = 2) -> list[DPOPair]:
        """Get (chosen, rejected, prompt) pairs from feedback DB.

        Same query logic as ``FeedbackTrainer.prepare_dpo_pairs()``.
        """
        import sqlite3

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT DISTINCT m.conversation_id
            FROM feedback f
            JOIN messages m ON f.message_id = m.id
            GROUP BY m.conversation_id
            HAVING COUNT(DISTINCT f.rating) > 1
            LIMIT 1000
        """)
        conv_ids = [row[0] for row in cursor.fetchall()]
        conn.close()

        pairs = []
        for conv_id in conv_ids:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT m.content, f.rating,
                       (SELECT mm.content FROM messages mm
                        WHERE mm.conversation_id = m.conversation_id
                        AND mm.role = 'user'
                        AND mm.created_at < m.created_at
                        ORDER BY mm.created_at DESC LIMIT 1) as prompt
                FROM messages m
                JOIN feedback f ON m.id = f.message_id
                WHERE m.conversation_id = ? AND m.role = 'assistant'
                """,
                (conv_id,),
            )
            rows = cursor.fetchall()
            conn.close()

            chosen = rejected = None
            prompt = ""
            for content, rating, pr in rows:
                if rating == "thumbs_up" and chosen is None:
                    chosen = content
                    prompt = pr or ""
                elif rating == "thumbs_down" and rejected is None:
                    rejected = content

            if chosen and rejected:
                pairs.append(DPOPair(chosen=chosen, rejected=rejected, prompt=prompt))

        return pairs

    # ── Quality guard ──────────────────────────────────────────────

    def _compute_ppl(self, texts: list[str]) -> float:
        """Compute average perplexity on a list of text prompts.

        Lower is better. Uses the model's own generations to measure
        how confidently it predicts the benchmark responses.
        """
        self.model.eval()
        total_nll = 0.0
        total_tokens = 0

        with torch.no_grad():
            for text in texts:
                messages = [{"role": "user", "content": text}]
                chat_text = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                )
                # Append a plausible continuation marker
                chat_text += " I think"

                ids = self.tokenizer(
                    chat_text, return_tensors="pt", truncation=True, max_length=128,
                ).input_ids.to(self.device)

                outputs = self.model(ids, labels=ids)
                loss = outputs.loss
                total_nll += loss.item() * ids.shape[1]
                total_tokens += ids.shape[1]

        if total_tokens == 0:
            return 100.0
        avg_nll = total_nll / total_tokens
        return float(torch.exp(torch.tensor(avg_nll)).item())

    def _take_snapshot(self) -> dict:
        """Snapshot LoRA / trainable parameter weights."""
        return {
            name: p.data.clone()
            for name, p in self.model.named_parameters()
            if p.requires_grad
        }

    def _restore_snapshot(self, snapshot: dict):
        """Restore parameter weights from a snapshot."""
        with torch.no_grad():
            for name, p in self.model.named_parameters():
                if name in snapshot:
                    p.data.copy_(snapshot[name])

    # ── Training ───────────────────────────────────────────────────

    def train(
        self,
        pairs: Optional[list[DPOPair]] = None,
        max_pairs: int = 6,
        chunk_size: int = 32,
        max_seq_len: int = 128,
        benchmark_interval: int = 3,
    ) -> dict:
        """Run DPO training on feedback pairs.

        Args:
            pairs: DPO pairs (auto-fetched from DB if None)
            max_pairs: Maximum number of pairs to train on
            chunk_size: Tokens per gradient step chunk
            max_seq_len: Maximum sequence length
            benchmark_interval: Check PPL every N pairs

        Returns:
            Dict with status, steps, loss, ppl_before, ppl_after, accepted
        """
        if pairs is None:
            pairs = self.prepare_dpo_pairs()

        if len(pairs) < 2:
            return {"status": "skipped", "reason": f"Only {len(pairs)} pair(s), need >=2"}

        pairs = pairs[:max_pairs]
        logger.info("DPO: training on %d pairs", len(pairs))

        # Benchmark before
        before_ppl = self._compute_ppl(_BENCHMARK_PROMPTS)
        snapshot = self._take_snapshot()
        logger.info("DPO: before PPL = %.2f", before_ppl)

        # Optimizer — only step on trainable (LoRA) params
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        if not trainable_params:
            return {"status": "skipped", "reason": "No trainable parameters (is LoRA enabled?)"}

        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=len(pairs) * 2 * 4, eta_min=1e-6,
        )

        self.model.train()
        self.steps = 0
        self.total_loss = 0.0

        for pair_idx, pair in enumerate(pairs):
            for label, text, do_ascent in [
                ("chosen", pair.chosen, False),
                ("rejected", pair.rejected, True),
            ]:
                full_text = (
                    f"<|user|>\n{pair.prompt}\n<|assistant|>\n{text}"
                )
                ids = self.tokenizer(
                    full_text, truncation=True, max_length=max_seq_len,
                    return_tensors="pt",
                ).input_ids.to(self.device)

                if ids.shape[1] < 4:
                    logger.debug("  Skipping %s: too short (%d tokens)", label, ids.shape[1])
                    continue

                seq_len = ids.shape[1] - 1

                # Process in chunks
                for start in range(0, seq_len, chunk_size):
                    end = min(start + chunk_size, seq_len)
                    x = ids[:, start:end]
                    y = ids[:, start + 1:end + 1]

                    # Pad if needed
                    if x.shape[1] < chunk_size:
                        pad_len = chunk_size - x.shape[1]
                        x = torch.nn.functional.pad(x, (0, pad_len), value=self.tokenizer.pad_token_id)
                        y = torch.nn.functional.pad(y, (0, pad_len), value=-100)

                    outputs = self.model(x, labels=y)
                    loss = outputs.loss
                    loss.backward()

                    if do_ascent:
                        # Negate gradients for rejected (gradient ascent)
                        for p in trainable_params:
                            if p.grad is not None:
                                p.grad.data.neg_()

                    # Clip and step
                    torch.nn.utils.clip_grad_norm_(trainable_params, self.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                    self.steps += 1
                    self.total_loss += loss.item()

            # Quality check at benchmark_interval
            if (pair_idx + 1) % benchmark_interval == 0:
                current_ppl = self._compute_ppl(_BENCHMARK_PROMPTS)
                ppl_delta = ((current_ppl - before_ppl) / before_ppl) * 100
                logger.info(
                    "  DPO checkpoint %d/%d: PPL=%.2f (delta=%+.1f%%)",
                    pair_idx + 1, len(pairs), current_ppl, ppl_delta,
                )
                if ppl_delta > 5.0:
                    self._restore_snapshot(snapshot)
                    self.rejected_count += 1
                    logger.warning("DPO rejected at pair %d: PPL +%.1f%%", pair_idx + 1, ppl_delta)
                    return {
                        "status": "rejected",
                        "steps": self.steps,
                        "avg_loss": self.total_loss / max(self.steps, 1),
                        "ppl_before": before_ppl,
                        "ppl_after": current_ppl,
                        "ppl_delta_pct": round(ppl_delta, 2),
                        "pairs_trained": pair_idx + 1,
                    }

        # Final benchmark
        after_ppl = self._compute_ppl(_BENCHMARK_PROMPTS)
        ppl_delta = ((after_ppl - before_ppl) / before_ppl) * 100

        self.accepted_count += 1

        if ppl_delta > 5.0:
            self._restore_snapshot(snapshot)
            self.rejected_count += 1
            logger.warning("DPO final check rejected: PPL +%.1f%%", ppl_delta)
            return {
                "status": "rejected",
                "steps": self.steps,
                "avg_loss": self.total_loss / max(self.steps, 1),
                "ppl_before": before_ppl,
                "ppl_after": after_ppl,
                "ppl_delta_pct": round(ppl_delta, 2),
                "pairs_trained": len(pairs),
            }

        result = {
            "status": "accepted",
            "steps": self.steps,
            "avg_loss": self.total_loss / max(self.steps, 1),
            "ppl_before": before_ppl,
            "ppl_after": after_ppl,
            "ppl_delta_pct": round(ppl_delta, 2),
            "pairs_trained": len(pairs),
        }
        logger.info(
            "DPO accepted: %d steps, loss=%.4f, PPL %.2f → %.2f (%.1f%%)",
            result["steps"], result["avg_loss"],
            result["ppl_before"], result["ppl_after"], result["ppl_delta_pct"],
        )
        return result

    def export_pairs(self, output_path: str = "data/dpo_pairs.jsonl") -> int:
        """Export DPO pairs to JSONL. Returns count."""
        pairs = self.prepare_dpo_pairs()
        count = 0
        with open(output_path, "w") as f:
            for p in pairs:
                f.write(json.dumps({
                    "chosen": p.chosen,
                    "rejected": p.rejected,
                    "prompt": p.prompt,
                }) + "\n")
                count += 1
        return count
