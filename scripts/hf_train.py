#!/usr/bin/env python3
"""Standalone HF fine-tune script — called by venv Python via subprocess.

Usage:
    python3 scripts/hf_train.py --data <text_file> --output <dir> --epochs 1 --lr 5e-5

Writes a JSON result to stdout on completion.
"""

import argparse
import json
import sys
import time
from pathlib import Path

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
    args = parser.parse_args()

    t0 = time.time()

    try:
        import torch
        from torch.utils.data import Dataset
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
            DataCollatorForLanguageModeling,
        )
    except ImportError as e:
        json.dump({"success": False, "error": f"Missing dependency: {e}"}, sys.stdout)
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

    # ── Load ─────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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
        model.print_trainable_parameters()

    model.to("cpu")

    dataset = TextFileDataset(args.data, tokenizer, args.max_seq_length)

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

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

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    # ── Train ────────────────────────────────────────────────────────────
    result = trainer.train()

    # Save
    model.save_pretrained(str(output_path / "final"))
    tokenizer.save_pretrained(str(output_path / "final"))

    elapsed = time.time() - t0

    json.dump({
        "success": True,
        "loss": float(result.training_loss),
        "steps": result.global_step,
        "elapsed_s": round(elapsed, 1),
        "model_path": str(output_path / "final"),
    }, sys.stdout)


if __name__ == "__main__":
    main()
