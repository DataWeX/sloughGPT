#!/usr/bin/env python3
"""Standalone HF fine-tune script — called by venv Python via subprocess.

Usage:
    python3 scripts/hf_train.py --data <text_file> --output <dir> --epochs 1 --lr 5e-5

Stream mode (--stream):
    Emits one JSON object per line to stdout for each training step.
    Last line is the final result JSON with "success" field.

Non-stream mode (default):
    Writes a single JSON result to stdout on completion.
"""

import argparse
import json
import sys
import time
from pathlib import Path


def _emit(obj, stream_mode):
    """Write a JSON line to stdout if stream mode is on."""
    if stream_mode:
        sys.stdout.write(json.dumps(obj) + "\n")
        sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to training text file")
    parser.add_argument("--output", required=True, help="Output directory for checkpoint")
    parser.add_argument("--model", default="gpt2", help="HuggingFace model name")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max-seq-length", type=int, default=256)
    parser.add_argument("--use-lora", action="store_true", default=True)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--stream", action="store_true", default=False,
                        help="Emit JSON progress lines per training step")
    args = parser.parse_args()

    t0 = time.time()
    stream = args.stream

    _emit({"phase": "TRAIN", "status": "loading", "message": "Loading model and tokenizer"}, stream)

    try:
        import torch
        from torch.utils.data import Dataset
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
            DataCollatorForLanguageModeling,
            TrainerCallback,
        )
    except ImportError as e:
        sys.stdout.write(json.dumps({"success": False, "error": f"Missing dependency: {e}"}) + "\n")
        sys.stdout.flush()
        return

    # ── Dataset ──────────────────────────────────────────────────────────
    class TextFileDataset(Dataset):
        def __init__(self, file_path, tokenizer, max_length):
            self.examples = []
            text = Path(file_path).read_text()
            chunks = [c.strip() for c in text.split("\n\n") if c.strip()]
            for chunk in chunks:
                encoded = tokenizer(
                    chunk,
                    truncation=True,
                    max_length=max_length,
                    padding="max_length",
                    return_tensors="pt",
                )
                input_ids = encoded["input_ids"].squeeze(0)
                attention_mask = encoded["attention_mask"].squeeze(0)
                labels = input_ids.clone()
                labels[attention_mask == 0] = -100
                self.examples.append({
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                })

        def __len__(self):
            return len(self.examples)

        def __getitem__(self, idx):
            return self.examples[idx]

    # ── Progress callback (stream mode) ──────────────────────────────────
    class _StreamCallback(TrainerCallback):
        """Emit JSON progress lines on each training log step."""

        def __init__(self, total_steps):
            self.total_steps = total_steps
            self.last_emit = 0.0

        def on_log(self, args, state, control, logs=None, **kwargs):
            now = time.time()
            if now - self.last_emit < 0.3:
                return
            self.last_emit = now
            loss = (logs or {}).get("loss")
            if loss is None:
                return
            _emit({
                "phase": "TRAIN",
                "status": "working",
                "step": state.global_step,
                "loss": round(loss, 4),
                "epoch": round(state.epoch or 0, 2),
                "progress_pct": min(100, int(state.global_step / max(1, self.total_steps) * 100)),
                "total_steps": self.total_steps,
            }, stream)

    # ── Load ─────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    _emit({"phase": "TRAIN", "status": "loading", "message": "Loading model weights"}, stream)

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=torch.float32,
    )

    if args.use_lora:
        from peft import LoraConfig, get_peft_model, TaskType
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.1,
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        # Redirect trainable params print to stderr so stdout stays clean for JSON parsing
        _real_stdout = sys.stdout
        sys.stdout = sys.stderr
        try:
            model.print_trainable_parameters()
        finally:
            sys.stdout = _real_stdout

    model.to("cpu")

    dataset = TextFileDataset(args.data, tokenizer, args.max_seq_length)

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    total_steps = max(1, (len(dataset) // args.batch_size) * args.epochs)

    _emit({
        "phase": "TRAIN",
        "status": "working",
        "message": f"Starting training: {len(dataset)} samples, {args.epochs} epochs, ~{total_steps} steps",
        "total_steps": total_steps,
        "samples": len(dataset),
        "epochs": args.epochs,
    }, stream)

    training_args = TrainingArguments(
        output_dir=str(output_path),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_steps=min(5, len(dataset)),
        weight_decay=0.01,
        logging_steps=1,
        save_strategy="epoch",
        report_to="none",
        fp16=False,
        dataloader_num_workers=0,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    callback = _StreamCallback(total_steps) if stream else None

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        callbacks=[callback] if callback else [],
    )

    # ── Train ────────────────────────────────────────────────────────────
    result = trainer.train()

    # Save
    _emit({"phase": "TRAIN", "status": "saving", "message": "Saving model"}, stream)
    model.save_pretrained(str(output_path / "final"))
    tokenizer.save_pretrained(str(output_path / "final"))

    elapsed = time.time() - t0

    final = {
        "success": True,
        "loss": float(result.training_loss),
        "steps": result.global_step,
        "elapsed_s": round(elapsed, 1),
        "model_path": str(output_path / "final"),
        "phase": "COMPLETE",
        "status": "complete",
    }

    # Always write final line (stream or not)
    sys.stdout.write(json.dumps(final) + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
