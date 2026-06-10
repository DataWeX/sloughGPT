"""
VLM-style multimodal training pipeline.

Trains a vision-language model using a vision encoder + projection connector + LLM.
Two-stage training:
  Stage 1: freeze LLM, train connector (vision-language alignment)
  Stage 2: full fine-tune with LoRA on LLM

Architecture:
  Image → SigLIP encoder → MLP connector → LLM (auto-regressive loss on text)
"""

from __future__ import annotations

import json
import logging
import os
import time
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoModelForCausalLM, AutoProcessor, AutoTokenizer

logger = logging.getLogger("man.vlm_trainer")

REPO_ROOT = Path(__file__).resolve().parents[4]


@dataclass
class VLMConfig:
    vision_encoder: str = "google/siglip-base-patch16-224"
    llm: str = "Qwen/Qwen2.5-0.5B-Instruct"
    connector_hidden_dim: int = 1024
    max_seq_length: int = 512
    stage1_epochs: int = 1
    stage2_epochs: int = 2
    stage1_lr: float = 1e-3
    stage2_lr: float = 2e-5
    batch_size: int = 4
    use_lora: bool = True
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"])
    freeze_vision: bool = True
    gradient_accumulation_steps: int = 1
    warmup_steps: int = 100
    weight_decay: float = 0.01
    output_dir: str = "models/vlm-finetuned"


class MLPConnector(nn.Module):
    """Projects vision encoder outputs into LLM embedding space."""

    def __init__(self, vision_dim: int, llm_dim: int, hidden_dim: int = 1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(vision_dim),
            nn.Linear(vision_dim, hidden_dim, bias=False),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.GELU(),
            nn.Linear(hidden_dim, llm_dim, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class VLMDataset(Dataset):
    """Image-text pair dataset for VLM training.

    Expects a JSONL file with:
      {"image_path": "datasets/coco/train/0001.jpg", "conversations": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]}

    Or simplified:
      {"image_path": "datasets/coco/train/0001.jpg", "caption": "A cat sitting on a chair"}
    """

    def __init__(
        self,
        data_path: str,
        processor,
        tokenizer: Any,
        max_seq_length: int = 512,
        image_size: int = 224,
    ):
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.image_size = image_size

        self.entries: List[Dict[str, Any]] = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.entries.append(json.loads(line))

        logger.info("VLMDataset: loaded %d entries from %s", len(self.entries), data_path)

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        entry = self.entries[idx]
        image_path = entry.get("image_path", "")
        conversations = entry.get("conversations", [])
        caption = entry.get("caption", "")

        # Resolve relative path against repo root
        if image_path and not Path(image_path).is_absolute():
            abs_path = REPO_ROOT / image_path
        else:
            abs_path = Path(image_path)

        # Load and process image
        try:
            from PIL import Image
            img = Image.open(abs_path).convert("RGB")
        except Exception as e:
            logger.warning("Failed to load image %s: %s", abs_path, e)
            img = Image.new("RGB", (self.image_size, self.image_size), color=0)

        pixel_values = self.processor(
            images=img,
            return_tensors="pt",
            size={"height": self.image_size, "width": self.image_size},
        ).pixel_values.squeeze(0)

        # Build text
        if conversations:
            parts = []
            for turn in conversations:
                role = turn.get("from", "")
                value = turn.get("value", "")
                if role == "human":
                    parts.append(f"<|user|>\n{value}")
                elif role == "gpt":
                    parts.append(f"<|assistant|>\n{value}")
            text = "\n".join(parts) + "\n<|endoftext|>"
        else:
            text = f"<|user|>\nDescribe this image in detail.\n<|assistant|>\n{caption}\n<|endoftext|>"

        tokenized = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_seq_length,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "pixel_values": pixel_values,
            "input_ids": tokenized["input_ids"].squeeze(0),
            "attention_mask": tokenized["attention_mask"].squeeze(0),
            "labels": tokenized["input_ids"].squeeze(0).clone(),
        }


def collate_vlm(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Collate function for VLM dataloader."""
    pixel_values = torch.stack([b["pixel_values"] for b in batch])
    input_ids = torch.stack([b["input_ids"] for b in batch])
    attention_mask = torch.stack([b["attention_mask"] for b in batch])
    labels = torch.stack([b["labels"] for b in batch])
    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


class VLMModel(nn.Module):
    """Vision-Language Model with trainable connector."""

    def __init__(self, config: VLMConfig):
        super().__init__()
        self.config = config

        logger.info("Loading vision encoder: %s", config.vision_encoder)
        raw_vision = AutoModel.from_pretrained(
            config.vision_encoder,
            trust_remote_code=True,
        )

        logger.info("Loading LLM: %s", config.llm)
        self.lm = AutoModelForCausalLM.from_pretrained(
            config.llm,
            trust_remote_code=True,
            dtype=torch.float32,
        )

        # Extract vision tower from dual-encoder models (SigLIP, CLIP, etc.)
        # Falls back to the raw model for single-encoder models
        if hasattr(raw_vision, "vision_model"):
            self.vision_model = raw_vision.vision_model
        else:
            self.vision_model = raw_vision

        # Detect vision hidden dimension robustly
        vcfg = self.vision_model.config
        vision_dim = getattr(vcfg, "hidden_size", None)
        if vision_dim is None:
            vision_dim = getattr(vcfg, "d_model", 768)
        llm_dim = self.lm.config.hidden_size

        logger.info("Vision dim: %d, LLM dim: %d (model: %s)", vision_dim, llm_dim, config.vision_encoder)
        self.connector = MLPConnector(vision_dim, llm_dim, config.connector_hidden_dim)

        if config.freeze_vision:
            for p in self.vision_model.parameters():
                p.requires_grad = False
            logger.info("Vision encoder frozen")

        # Apply LoRA at init if configured
        if config.use_lora:
            from peft import LoraConfig, get_peft_model, TaskType

            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=config.lora_rank,
                lora_alpha=config.lora_alpha,
                target_modules=config.lora_target_modules,
                lora_dropout=0.1,
                bias="none",
            )
            self.lm = get_peft_model(self.lm, lora_config)
            self.lm.print_trainable_parameters()
            # Freeze LLM by default (unfrozen for stage 2)
            for p in self.lm.parameters():
                p.requires_grad = False
            logger.info("LLM frozen (unfrozen for stage 2)")

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass: image → vision encoder → connector → LLM.

        Vision embeddings are placed at the beginning of the input sequence.
        """
        device = pixel_values.device
        batch_size = pixel_values.shape[0]

        # Get vision features — vision_model is the vision tower directly
        vision_outputs = self.vision_model(pixel_values=pixel_values)
        vision_embeds = vision_outputs.last_hidden_state  # (B, num_patches+1, dim)

        # Project vision features to LLM dimension
        vision_embeds = self.connector(vision_embeds)  # (B, num_patches+1, llm_dim)

        # Get LLM input embeddings for text tokens
        inputs_embeds = self.lm.get_input_embeddings()(input_ids)  # (B, seq_len, llm_dim)

        # Prepend vision embeddings to text embeddings
        # Vision tokens replace the first N positions of input
        n_vision = vision_embeds.shape[1]
        combined_embeds = torch.cat([vision_embeds, inputs_embeds[:, n_vision:]], dim=1)

        # Adjust attention mask if provided
        if attention_mask is not None:
            vision_mask = torch.ones(batch_size, n_vision, device=device)
            combined_attention_mask = torch.cat([vision_mask, attention_mask[:, n_vision:]], dim=1)
        else:
            combined_attention_mask = None

        # Adjust labels: set vision token positions to -100 (ignore in loss)
        if labels is not None:
            vision_labels = torch.full((batch_size, n_vision), -100, device=device)
            combined_labels = torch.cat([vision_labels, labels[:, n_vision:]], dim=1)
        else:
            combined_labels = None

        # Forward through LLM
        outputs = self.lm(
            inputs_embeds=combined_embeds,
            attention_mask=combined_attention_mask,
            labels=combined_labels,
            return_dict=True,
        )

        return {
            "loss": outputs.loss,
            "logits": outputs.logits,
        }


class VLMTrainer:
    """VLM multimodal trainer with two-stage training.

    Stage 1: Freeze LLM, train connector only (vision-language alignment)
    Stage 2: Full fine-tune with LoRA on LLM
    """

    def __init__(self, config: VLMConfig):
        self.config = config
        self.model: Optional[VLMModel] = None
        self.processor = None
        self.tokenizer = None
        self._stop_requested = False

    def load(self):
        """Load models and processors."""
        self.processor = AutoProcessor.from_pretrained(
            self.config.vision_encoder,
            trust_remote_code=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.llm,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = VLMModel(self.config)

    def _run_epoch(
        self,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        stage: str,
        epoch: int,
        total_epochs: int,
        total_steps: int,
        on_progress: Optional[Callable[[Dict[str, Any]], None]] = None,
        step_offset: int = 0,
    ) -> float:
        """Run a single training epoch.

        Returns average loss for the epoch.
        """
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        step = 0
        last_log = 0.0

        for batch in dataloader:
            if self._stop_requested:
                logger.info("Stop requested, breaking epoch")
                break

            device = next(self.model.parameters()).device
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = self.model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs["loss"]
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1
            step += 1

            # Progress callback
            if on_progress is not None:
                now = time.time()
                if now - last_log >= 0.5:
                    last_log = now
                    global_step = step_offset + (epoch * len(dataloader)) + step
                    on_progress({
                        "stage": stage,
                        "epoch": epoch + 1,
                        "step": global_step,
                        "loss": round(loss.item(), 4),
                        "progress_pct": min(100, int(global_step / max(1, total_steps) * 100)),
                        "total_steps": total_steps,
                    })

        return total_loss / max(1, n_batches)

    def train(
        self,
        data_path: str,
        on_progress: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """Run two-stage VLM training using plain PyTorch (no HF Trainer).

        Args:
            data_path: Path to JSONL with image-text pairs
            on_progress: Progress callback

        Returns:
            Dict with status, model_path, final_loss
        """
        if self.model is None:
            self.load()

        self.model.train()

        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create dataset
        dataset = VLMDataset(
            data_path=data_path,
            processor=self.processor,
            tokenizer=self.tokenizer,
            max_seq_length=self.config.max_seq_length,
        )

        if len(dataset) == 0:
            return {
                "status": "error",
                "error": f"No valid entries in {data_path}",
            }

        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=collate_vlm,
            num_workers=0,
            pin_memory=False,
        )

        steps_per_epoch = max(1, len(dataset) // self.config.batch_size)
        steps_s1 = steps_per_epoch * self.config.stage1_epochs
        steps_s2 = steps_per_epoch * self.config.stage2_epochs
        total_steps = steps_s1 + steps_s2

        ############################
        # Stage 1: Train Connector #
        ############################
        logger.info("=== Stage 1: Training connector (LLM frozen) ===")

        # Unfreeze connector
        for p in self.model.connector.parameters():
            p.requires_grad = True

        # Ensure LLM and vision are frozen
        for p in self.model.lm.parameters():
            p.requires_grad = False
        if self.config.freeze_vision:
            for p in self.model.vision_model.parameters():
                p.requires_grad = False

        optimizer_s1 = torch.optim.AdamW(
            [p for p in self.model.connector.parameters() if p.requires_grad],
            lr=self.config.stage1_lr,
            weight_decay=self.config.weight_decay,
        )

        final_loss = 0.0
        for epoch in range(self.config.stage1_epochs):
            if self._stop_requested:
                break
            epoch_loss = self._run_epoch(
                dataloader, optimizer_s1,
                stage="connector",
                epoch=epoch,
                total_epochs=self.config.stage1_epochs,
                total_steps=total_steps,
                on_progress=on_progress,
            )
            logger.info("Stage 1 epoch %d/%d — loss: %.4f",
                        epoch + 1, self.config.stage1_epochs, epoch_loss)
            final_loss = epoch_loss

        #############################
        # Stage 2: LoRA Fine-tune  #
        #############################
        logger.info("=== Stage 2: LoRA fine-tune ===")

        # Unfreeze only LoRA parameters (not all 494M LLM params)
        for name, p in self.model.lm.named_parameters():
            if "lora" in name.lower():
                p.requires_grad = True

        lora_params = [p for name, p in self.model.lm.named_parameters() if p.requires_grad]
        optimizer_s2 = torch.optim.AdamW(
            lora_params,
            lr=self.config.stage2_lr,
            weight_decay=self.config.weight_decay,
        )

        for epoch in range(self.config.stage2_epochs):
            if self._stop_requested:
                break
            epoch_loss = self._run_epoch(
                dataloader, optimizer_s2,
                stage="full_finetune",
                epoch=epoch,
                total_epochs=self.config.stage2_epochs,
                total_steps=total_steps,
                on_progress=on_progress,
                step_offset=steps_s1,
            )
            logger.info("Stage 2 epoch %d/%d — loss: %.4f",
                        epoch + 1, self.config.stage2_epochs, epoch_loss)
            final_loss = epoch_loss

        # Save final model — break weight tying first (Qwen ties embed_tokens ↔ lm_head)
        save_path = str(output_dir / "final")
        lm = self.model.lm
        try:
            if hasattr(lm, "base_model") and hasattr(lm.base_model, "lm_head"):
                lm.base_model.lm_head.weight.data = lm.base_model.lm_head.weight.data.clone()
            elif hasattr(lm, "lm_head"):
                lm.lm_head.weight.data = lm.lm_head.weight.data.clone()
        except Exception:
            pass
        lm.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)

        # Save connector weights separately
        connector_path = output_dir / "connector.pt"
        torch.save(self.model.connector.state_dict(), str(connector_path))

        # Save config
        config_save = output_dir / "vlm_config.json"
        with open(str(config_save), "w") as f:
            json.dump({
                "vision_encoder": self.config.vision_encoder,
                "llm": self.config.llm,
                "final_loss": final_loss,
                "total_steps": total_steps,
                "stage1_epochs": self.config.stage1_epochs,
                "stage2_epochs": self.config.stage2_epochs,
            }, f, indent=2)

        logger.info("VLM training complete. Loss=%s, steps=%d, saved to %s",
                      final_loss, total_steps, save_path)

        sou_path = str(output_dir / f"vlm_{Path(self.config.llm).name}.sou")
        try:
            self.export_to_sou(sou_path)
        except Exception as e:
            logger.warning("VLM .sou export failed: %s", e)
            sou_path = None

        return {
            "status": "completed",
            "model_path": save_path,
            "sou_path": sou_path,
            "final_loss": final_loss,
            "total_steps": total_steps,
        }

    def stop(self):
        """Request training stop."""
        self._stop_requested = True

    def export_to_sou(self, output_path: str) -> str:
        """Export VLM model to .sou binary format for the model catalog.

        Writes the connector weights and VLM config as a standard .sou file
        so it appears in the checkpoint catalog alongside SloNet models.

        Args:
            output_path: Destination path (should end in .sou)

        Returns:
            The output path as a string
        """
        import struct

        SOU_MAGIC = b"SOUL"

        if self.model is None:
            raise RuntimeError("No trained model to export — call train() first")

        output_dir = Path(self.config.output_dir)
        config_path = output_dir / "vlm_config.json"
        vlm_config = {}
        if config_path.is_file():
            with open(config_path) as f:
                vlm_config = json.load(f)

        metadata = {
            "version": 3,
            "model_type": "vlm",
            "soul_name": f"vlm-{self.config.llm.split('/')[-1]}",
            "soul_traits": {"warmth": 0.5, "creativity": 0.5, "curiosity": 0.7, "confidence": 0.5},
            "lineage": self.config.llm,
            "system_prompt": "You are a vision-language assistant. Describe images accurately and helpfully.",
            "vision_encoder": self.config.vision_encoder,
            "llm": self.config.llm,
            "vlm_config": vlm_config,
            "created_at": time.time(),
        }

        json_bytes = json.dumps(metadata, allow_nan=False, default=str).encode()
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "wb") as f:
            f.write(SOU_MAGIC)
            f.write(struct.pack("<I", 3))
            f.write(struct.pack("<I", len(json_bytes)))
            f.write(json_bytes)

            connector_state = self.model.connector.state_dict()
            params = [(k, v.detach().cpu().numpy().astype(np.float32)) for k, v in connector_state.items()]
            f.write(struct.pack("<I", len(params)))
            for key, arr in params:
                name_bytes = key.encode()
                f.write(struct.pack("<I", len(name_bytes)))
                f.write(name_bytes)
                f.write(struct.pack("<I", arr.ndim))
                for dim in arr.shape:
                    f.write(struct.pack("<I", dim))
                f.write(arr.tobytes())

        meta_path = path.with_suffix(".sou.meta.json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        logger.info("VLM exported to %s (%d connector weights)", path, len(params))
        return str(path)
