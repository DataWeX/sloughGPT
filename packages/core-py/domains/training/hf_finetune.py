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

        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        tokens = tokenizer.encode(text, add_special_tokens=False)
        self.examples: List[Dict[str, torch.Tensor]] = []

        for i in range(0, len(tokens), stride):
            chunk = tokens[i : i + max_length]
            if len(chunk) < 64:
                continue
            input_ids = torch.tensor(chunk, dtype=torch.long)
            self.examples.append({"input_ids": input_ids, "labels": input_ids.clone()})

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
            torch_dtype=torch.float32,
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
