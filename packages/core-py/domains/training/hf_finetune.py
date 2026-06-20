"""HuggingFace model fine-tuning with optional LoRA.

Supports any AutoModelForCausalLM from HuggingFace Hub.
Uses transformers.Trainer for the training loop and peft for LoRA adapters.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

logger = logging.getLogger("man.hf_finetune")


class TextFileDataset(Dataset):
    """Dataset from a text file, tokenized with a HuggingFace tokenizer."""

    def __init__(
        self,
        file_path: str,
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
        max_length: int = 512,
        stride: int = 256,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0

        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        tokens = tokenizer.encode(text, add_special_tokens=False)
        self.examples: List[Dict[str, torch.Tensor]] = []

        for i in range(0, len(tokens), stride):
            chunk = tokens[i : i + max_length]
            if len(chunk) < 64:
                continue
            input_ids = torch.tensor(chunk, dtype=torch.long)
            labels = input_ids.clone()
            self.examples.append({"input_ids": input_ids, "labels": labels})

        self._pad_examples()

    def _pad_examples(self) -> None:
        """Pad all examples to the same length (max_length) so the batch collator can stack them."""
        max_len = self.max_length
        for ex in self.examples:
            cur = ex["input_ids"].size(0)
            if cur < max_len:
                pad_len = max_len - cur
                ex["input_ids"] = torch.cat([
                    ex["input_ids"],
                    torch.full((pad_len,), self.pad_token_id, dtype=torch.long),
                ])
                ex["labels"] = torch.cat([
                    ex["labels"],
                    torch.full((pad_len,), -100, dtype=torch.long),
                ])

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.examples[idx]


class HFFineTuner:
    """Fine-tune a HuggingFace causal LM on text data with optional LoRA.

    Usage:
        tuner = HFFineTuner(
            model_name="Qwen/Qwen2.5-0.5B-Instruct",
            data_path="datasets/shakespeare/input.txt",
            output_dir="models/hf-finetuned/qwen-shakespeare",
            use_lora=True,
            lora_rank=8,
            epochs=3,
            batch_size=4,
            learning_rate=2e-4,
        )
        result = tuner.train(on_progress=print)
    """

    def __init__(
        self,
        model_name: str,
        data_path: str,
        output_dir: str = "models/hf-finetuned",
        use_lora: bool = False,
        lora_rank: int = 8,
        lora_alpha: int = 16,
        lora_target_modules: Optional[List[str]] = None,
        epochs: int = 3,
        batch_size: int = 4,
        learning_rate: float = 2e-4,
        max_seq_length: int = 512,
        warmup_steps: int = 100,
        weight_decay: float = 0.01,
        gradient_accumulation_steps: int = 1,
        save_steps: int = 500,
        logging_steps: int = 10,
        device: Optional[str] = None,
    ):
        self.model_name = model_name
        self.data_path = data_path
        self.output_dir = output_dir
        self.use_lora = use_lora
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_target_modules = lora_target_modules or ["q_proj", "v_proj"]
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_seq_length = max_seq_length
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.save_steps = save_steps
        self.logging_steps = logging_steps
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def train(
        self,
        on_progress: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """Run the fine-tuning loop.

        Args:
            on_progress: Callback with progress info dict (epoch, loss, step, etc.)

        Returns:
            Dict with keys: status, model_path, final_loss, total_steps
        """
        logger.info(
            "Loading model %s on %s (LoRA=%s, rank=%d)",
            self.model_name,
            self.device,
            self.use_lora,
            self.lora_rank,
        )

        tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            dtype=torch.float32,
        )

        if self.use_lora:
            from peft import LoraConfig, get_peft_model, TaskType

            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=self.lora_rank,
                lora_alpha=self.lora_alpha,
                target_modules=self.lora_target_modules,
                lora_dropout=0.1,
                bias="none",
            )
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()

        model.to(self.device)

        logger.info("Loading dataset from %s", self.data_path)
        dataset = TextFileDataset(
            file_path=self.data_path,
            tokenizer=tokenizer,
            max_length=self.max_seq_length,
        )
        logger.info("Dataset has %d examples", len(dataset))

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
            pad_to_multiple_of=None,
        )

        output_path = Path(self.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        training_args = TrainingArguments(
            output_dir=str(output_path),
            overwrite_output_dir=True,
            num_train_epochs=self.epochs,
            per_device_train_batch_size=self.batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            warmup_steps=self.warmup_steps,
            logging_steps=self.logging_steps,
            save_steps=self.save_steps,
            save_total_limit=2,
            prediction_loss_only=True,
            remove_unused_columns=False,
            dataloader_drop_last=False,
            report_to="none",
            disable_tqdm=True,
            fp16=False,
            bf16=False,
            use_cpu=True,
        )

        total_steps = len(dataset) // self.batch_size * self.epochs

        class ProgressCallback:
            def __init__(self, fn, total):
                self.fn = fn
                self.total = total
                self.last_log = 0.0

            def on_log(self, args, state, control, logs=None, **kwargs):
                if self.fn is None:
                    return
                now = time.time()
                if now - self.last_log < 0.5:
                    return
                self.last_log = now
                loss = (logs or {}).get("loss")
                self.fn({
                    "epoch": state.epoch or 0,
                    "step": state.global_step,
                    "loss": loss,
                    "progress_pct": min(100, int(state.global_step / max(1, self.total) * 100)),
                    "total_steps": self.total,
                })

        from transformers.trainer_callback import TrainerCallback

        class _ProgressTrainerCallback(TrainerCallback):
            def __init__(self, cb):
                super().__init__()
                self._cb = cb
            def on_log(self, args, state, control, logs=None, **kwargs):
                self._cb.on_log(args, state, control, logs, **kwargs)

        progress_cb = _ProgressTrainerCallback(ProgressCallback(on_progress, total_steps))

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator,
            callbacks=[progress_cb],
        )

        logger.info("Starting training for %d epochs", self.epochs)
        train_result = trainer.train()

        final_loss = train_result.training_loss if hasattr(train_result, "training_loss") else None
        final_step = train_result.global_step if hasattr(train_result, "global_step") else total_steps

        save_path = str(output_path / "final")
        trainer.save_model(save_path)
        tokenizer.save_pretrained(save_path)

        config_path = output_path / "training_config.json"
        with open(config_path, "w") as f:
            json.dump({
                "model_name": self.model_name,
                "use_lora": self.use_lora,
                "lora_rank": self.lora_rank,
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "learning_rate": self.learning_rate,
                "final_loss": final_loss,
                "total_steps": final_step,
            }, f, indent=2)

        logger.info(
            "Training complete. Loss=%s, steps=%d, saved to %s",
            final_loss, final_step, save_path,
        )

        return {
            "status": "completed",
            "model_path": save_path,
            "final_loss": final_loss,
            "total_steps": final_step,
        }


class RewardFn:
    """Scoring function for generated completions.

    Supports three built-in modes:
    - ``length``: rewards medium-length responses (3-200 words), penalises
      empty or very long outputs.
    - ``keyword``: checks that the response contains ``keywords`` (case-
      insensitive substring match).
    - ``custom``: caller-supplied ``fn(completion: str) -> float``.
    """

    def __init__(
        self,
        mode: str = "length",
        keywords: Optional[List[str]] = None,
        fn: Optional[Callable[[str], float]] = None,
    ):
        if mode not in ("length", "keyword", "custom"):
            raise ValueError(f"Unknown reward mode: {mode!r}")
        if mode == "custom" and fn is None:
            raise ValueError("mode='custom' requires a fn(completion) -> float")
        self.mode = mode
        self.keywords = [k.lower() for k in (keywords or [])]
        self._custom_fn = fn

    def __call__(self, completion: str) -> float:
        if self.mode == "length":
            return self._length_reward(completion)
        if self.mode == "keyword":
            return self._keyword_reward(completion)
        return self._custom_fn(completion)  # type: ignore[misc]

    @staticmethod
    def _length_reward(text: str) -> float:
        words = text.split()
        n = len(words)
        if n == 0:
            return -1.0
        if n < 3:
            return -0.5
        if n <= 200:
            return 1.0
        return max(0.0, 1.0 - (n - 200) / 500)

    def _keyword_reward(self, text: str) -> float:
        text_lower = text.lower()
        hits = sum(1 for kw in self.keywords if kw in text_lower)
        if not self.keywords:
            return 1.0
        return hits / len(self.keywords)


class GRPOTrainer:
    """Group Relative Policy Optimization trainer.

    Simplified GRPO: for each prompt, generate G completions, score them,
    compute group-relative advantages, then update with a clipped policy
    gradient.  A KL penalty against a frozen reference model prevents
    reward hacking.

    Usage::

        trainer = GRPOTrainer(
            model_name="Qwen/Qwen2.5-0.5B-Instruct",
            prompts_path="datasets/shakespeare/input.txt",
            output_dir="models/grpo-qwen",
            num_generations=4,
            reward_fn=RewardFn(mode="keyword", keywords=["the", "and"]),
        )
        result = trainer.train(on_progress=print)
    """

    def __init__(
        self,
        model_name: str,
        prompts_path: str,
        output_dir: str = "models/grpo",
        num_generations: int = 4,
        learning_rate: float = 1e-6,
        kl_coef: float = 0.1,
        clip_range: float = 0.2,
        epochs: int = 1,
        max_new_tokens: int = 128,
        batch_size: int = 4,
        device: Optional[str] = None,
        reward_fn: Optional[RewardFn] = None,
        use_lora: bool = False,
        lora_rank: int = 8,
        lora_alpha: int = 16,
    ):
        self.model_name = model_name
        self.prompts_path = prompts_path
        self.output_dir = output_dir
        self.num_generations = num_generations
        self.learning_rate = learning_rate
        self.kl_coef = kl_coef
        self.clip_range = clip_range
        self.epochs = epochs
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.reward_fn = reward_fn or RewardFn(mode="length")
        self.use_lora = use_lora
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha

    def _load_prompts(self) -> List[str]:
        """Load prompts from a text file (one per line or paragraph)."""
        with open(self.prompts_path, "r", encoding="utf-8") as f:
            text = f.read()
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        if len(paragraphs) < 2:
            paragraphs = [line.strip() for line in text.split("\n") if line.strip()]
        return paragraphs

    def _generate_group(
        self,
        model: Any,
        tokenizer: Any,
        prompt: str,
        n: int,
    ) -> List[str]:
        """Generate N completions for a single prompt."""
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids)).to(self.device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.max_new_tokens,
                num_return_sequences=n,
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )

        completions = []
        prompt_len = input_ids.shape[1]
        for i in range(n):
            gen_ids = outputs[i, prompt_len:]
            text = tokenizer.decode(gen_ids, skip_special_tokens=True)
            completions.append(text)
        return completions

    def _compute_advantages(self, rewards: torch.Tensor) -> torch.Tensor:
        """Group-relative advantage: normalise rewards within each group."""
        mean = rewards.mean()
        std = rewards.std()
        if std < 1e-8:
            return torch.zeros_like(rewards)
        return (rewards - mean) / std

    def _policy_loss(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
    ) -> torch.Tensor:
        """Clipped policy gradient loss (PPO-style)."""
        ratio = torch.exp(log_probs - old_log_probs)
        clipped = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range)
        loss1 = ratio * advantages
        loss2 = clipped * advantages
        return -torch.min(loss1, loss2).mean()

    def _kl_penalty(self, log_probs: torch.Tensor, ref_log_probs: torch.Tensor) -> torch.Tensor:
        """KL divergence approximation: KL(pi || ref) ≈ ref_log - log_pi."""
        return (ref_log_probs - log_probs).mean()

    def train(
        self,
        on_progress: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """Run GRPO training.

        Args:
            on_progress: Callback with progress info dict.

        Returns:
            Dict with status, model_path, final_loss, reward_history, total_steps.
        """
        logger.info(
            "GRPO: loading model %s on %s (generations=%d, lr=%.2e, kl=%.2f)",
            self.model_name, self.device, self.num_generations,
            self.learning_rate, self.kl_coef,
        )

        tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            dtype=torch.float32,
        )
        model.to(self.device)

        if self.use_lora:
            from peft import LoraConfig, get_peft_model, TaskType
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=self.lora_rank,
                lora_alpha=self.lora_alpha,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.1,
                bias="none",
            )
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()

        # Frozen reference model for KL penalty
        ref_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            dtype=torch.float32,
        )
        ref_model.to(self.device)
        ref_model.eval()

        prompts = self._load_prompts()
        logger.info("Loaded %d prompts from %s", len(prompts), self.prompts_path)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)

        reward_history: List[Dict[str, Any]] = []
        all_rewards: List[float] = []
        total_steps = 0

        output_path = Path(self.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for epoch in range(self.epochs):
            model.train()
            epoch_rewards: List[float] = []
            epoch_losses: List[float] = []

            for batch_start in range(0, len(prompts), self.batch_size):
                batch_prompts = prompts[batch_start:batch_start + self.batch_size]

                for prompt in batch_prompts:
                    completions = self._generate_group(model, tokenizer, prompt, self.num_generations)

                    scores = []
                    for c in completions:
                        scores.append(self.reward_fn(c))
                    rewards_tensor = torch.tensor(scores, dtype=torch.float32, device=self.device)
                    advantages = self._compute_advantages(rewards_tensor)

                    # Compute log-probs for each completion
                    model_log_probs_list = []
                    ref_log_probs_list = []

                    for c in completions:
                        full_text = prompt + c
                        inputs = tokenizer(
                            full_text, return_tensors="pt", truncation=True,
                            max_length=256 + self.max_new_tokens,
                        )
                        input_ids = inputs["input_ids"].to(self.device)
                        attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids)).to(self.device)

                        prompt_inputs = tokenizer(
                            prompt, return_tensors="pt", truncation=True, max_length=256,
                        )
                        prompt_len = prompt_inputs["input_ids"].shape[1]

                        with torch.no_grad():
                            ref_out = ref_model(input_ids, attention_mask=attention_mask)
                            ref_logits = ref_out.logits[:, prompt_len - 1:-1, :]
                            ref_log_probs = torch.log_softmax(ref_logits, dim=-1)
                            ref_token_ids = input_ids[:, prompt_len:]
                            ref_selected = ref_log_probs.gather(2, ref_token_ids.unsqueeze(-1)).squeeze(-1)
                            ref_log_probs_list.append(ref_selected.sum())

                        model_out = model(input_ids, attention_mask=attention_mask)
                        model_logits = model_out.logits[:, prompt_len - 1:-1, :]
                        model_log_probs = torch.log_softmax(model_logits, dim=-1)
                        model_selected = model_log_probs.gather(2, ref_token_ids.unsqueeze(-1)).squeeze(-1)
                        model_log_probs_list.append(model_selected.sum())

                    model_log_probs_t = torch.stack(model_log_probs_list)
                    ref_log_probs_t = torch.stack(ref_log_probs_list)

                    policy_loss = self._policy_loss(model_log_probs_t, ref_log_probs_t.detach(), advantages)
                    kl = self._kl_penalty(model_log_probs_t, ref_log_probs_t.detach())
                    loss = policy_loss + self.kl_coef * kl

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                    step_reward = rewards_tensor.mean().item()
                    epoch_rewards.append(step_reward)
                    all_rewards.append(step_reward)
                    epoch_losses.append(loss.item())
                    total_steps += 1

                    if on_progress is not None:
                        on_progress({
                            "step": total_steps,
                            "epoch": epoch + 1,
                            "loss": loss.item(),
                            "reward": step_reward,
                            "kl": kl.item(),
                            "progress_pct": min(100, int(((epoch * len(prompts) + batch_start) / (self.epochs * len(prompts))) * 100)),
                            "total_steps": self.epochs * len(prompts),
                        })

            avg_reward = sum(epoch_rewards) / max(len(epoch_rewards), 1)
            avg_loss = sum(epoch_losses) / max(len(epoch_losses), 1)
            reward_history.append({
                "epoch": epoch + 1,
                "avg_reward": avg_reward,
                "avg_loss": avg_loss,
            })
            logger.info(
                "GRPO epoch %d/%d: avg_reward=%.4f, avg_loss=%.4f",
                epoch + 1, self.epochs, avg_reward, avg_loss,
            )

        # Save model
        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in Path(self.prompts_path).stem)[:60]
        save_path = str(output_path / f"{safe_name}_final")
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)

        final_reward = sum(all_rewards[-len(prompts):]) / max(len(prompts), 1) if all_rewards else 0.0
        final_loss = epoch_losses[-1] if epoch_losses else 0.0

        logger.info(
            "GRPO complete. final_reward=%.4f, final_loss=%.4f, saved to %s",
            final_reward, final_loss, save_path,
        )

        return {
            "status": "completed",
            "model_path": save_path,
            "final_reward": final_reward,
            "final_loss": final_loss,
            "total_steps": total_steps,
            "reward_history": reward_history,
        }


def create_hf_finetuner(
    model_name: str,
    data_path: str,
    output_dir: str = "models/hf-finetuned",
    use_lora: bool = False,
    lora_rank: int = 8,
    epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2e-4,
    **kwargs: Any,
) -> HFFineTuner:
    """Factory function to create an HFFineTuner instance."""
    return HFFineTuner(
        model_name=model_name,
        data_path=data_path,
        output_dir=output_dir,
        use_lora=use_lora,
        lora_rank=lora_rank,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        **kwargs,
    )
