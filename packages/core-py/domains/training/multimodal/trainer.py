"""VLMTrainer — two-stage vision-language model trainer using HuggingFace."""

import json
import logging
import time
import os
from pathlib import Path
from typing import Any, Callable, Optional

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from transformers import (
        AutoModel,
        AutoModelForCausalLM,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )
    from peft import LoraConfig, get_peft_model, TaskType
    _VLM_TRAINER_AVAILABLE = True
except ImportError:
    _VLM_TRAINER_AVAILABLE = False

from .config import VLMConfig

logger = logging.getLogger("man.vlm_trainer")


class VisionConnector(nn.Module):
    """Linear projection from vision encoder hidden dim → LLM hidden dim."""

    def __init__(self, vision_dim: int, llm_dim: int):
        super().__init__()
        self.proj = nn.Linear(vision_dim, llm_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class VLMDataset(Dataset):
    """Dataset of image-caption pairs for VLM training.

    Expects a JSONL file with rows containing ``image_path``
    and ``caption`` fields (or ``conversations`` for multi-turn).
    """

    def __init__(self, data_path: str, image_transform: Optional[Callable] = None):
        self.samples: list[dict[str, Any]] = []
        self.image_transform = image_transform

        with open(data_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    self.samples.append(json.loads(line))

        logger.info("VLMDataset loaded %d samples from %s", len(self.samples), data_path)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.samples[idx]


class VLMTrainer:
    """Two-stage VLM trainer: connector pre-training → full LoRA fine-tuning.

    Stage 1 — Train only the vision→LLM connector projection
    Stage 2 — Train connector + LoRA adapters on the LLM

    All training runs on CPU with mixed precision disabled for stability.
    """

    def __init__(self, config: VLMConfig):
        self.config = config
        self.vision_encoder: Optional[nn.Module] = None
        self.llm: Optional[nn.Module] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self.connector: Optional[VisionConnector] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler: Optional[Any] = None
        self.device = torch.device(config.device)

    def _load_models(self):
        """Load vision encoder and LLM from HuggingFace."""
        logger.info("Loading vision encoder: %s", self.config.vision_encoder)
        self.vision_encoder = AutoModel.from_pretrained(
            self.config.vision_encoder,
            trust_remote_code=True,
        ).to(self.device).eval()

        if self.config.freeze_vision:
            for param in self.vision_encoder.parameters():
                param.requires_grad = False

        logger.info("Loading LLM: %s", self.config.llm)
        self.llm = AutoModelForCausalLM.from_pretrained(
            self.config.llm,
            trust_remote_code=True,
            torch_dtype=torch.float32,
        ).to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.llm,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Determine dimensions
        vision_dim = self.vision_encoder.config.hidden_size
        llm_dim = self.llm.config.hidden_size

        self.connector = VisionConnector(vision_dim, llm_dim).to(self.device)
        logger.info("Connector: %d → %d", vision_dim, llm_dim)

    def _get_vision_embedding(self, image_path: str) -> torch.Tensor:
        """Extract vision embedding from an image file."""
        from PIL import Image
        from transformers import AutoImageProcessor

        processor = AutoImageProcessor.from_pretrained(self.config.vision_encoder)
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.vision_encoder(**inputs)
            # Use CLS token or pooler output
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                embedding = outputs.pooler_output
            else:
                embedding = outputs.last_hidden_state[:, 0, :]
        return embedding

    def _stage1_step(self, image_path: str, caption: str) -> torch.Tensor:
        """Single training step for stage 1 (connector only)."""
        vision_emb = self._get_vision_embedding(image_path)
        projected = self.connector(vision_emb)

        # Prepend vision embedding as a pseudo-token to the text
        inputs = self.tokenizer(
            caption,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.config.max_seq_length,
        ).to(self.device)

        # Create embedding with vision token prepended
        embed_dim = self.llm.get_input_embeddings().weight.shape[1]
        projected = projected.to(torch.float32)

        # Expand to match sequence
        input_embeds = self.llm.get_input_embeddings()(inputs["input_ids"])
        vision_token = projected.unsqueeze(1)
        combined = torch.cat([vision_token, input_embeds], dim=1)

        # Adjust attention mask and labels
        attention_mask = torch.cat([
            torch.ones((1, 1), device=self.device),
            inputs["attention_mask"],
        ], dim=1)

        labels = torch.cat([
            torch.full((1, 1), -100, device=self.device),
            inputs["input_ids"],
        ], dim=1)

        outputs = self.llm(
            inputs_embeds=combined,
            attention_mask=attention_mask,
            labels=labels,
        )
        return outputs.loss

    def train(
        self,
        data_path: str,
        progress_callback: Optional[Callable[[dict[str, Any]], None]] = None,
    ) -> dict[str, Any]:
        """Run two-stage VLM training.

        Args:
            data_path: Path to JSONL with image_path/caption rows.
            progress_callback: Optional callback with progress info dict.

        Returns:
            dict with keys: status, final_loss, model_path, etc.
        """
        t0 = time.time()
        self._load_models()
        dataset = VLMDataset(data_path)

        if len(dataset) == 0:
            return {"status": "error", "error": "Empty dataset"}

        # ── Stage 1: Train connector only ──────────────────────────
        logger.info("Stage 1: training connector (%d epochs)", self.config.stage1_epochs)

        stage1_params = list(self.connector.parameters())
        self.optimizer = torch.optim.AdamW(stage1_params, lr=self.config.stage1_lr)
        total_steps = len(dataset) * self.config.stage1_epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=0, num_training_steps=total_steps
        )

        self.connector.train()
        global_step = 0
        stage1_losses: list[float] = []

        for epoch in range(self.config.stage1_epochs):
            epoch_loss = 0.0
            for idx in range(len(dataset)):
                sample = dataset[idx]
                image_path = sample.get("image_path", "")
                caption = sample.get("caption", "") or sample.get("conversations", [{}])[-1].get("value", "")

                if not image_path or not os.path.isfile(image_path):
                    continue
                if not caption:
                    continue

                loss = self._stage1_step(image_path, caption)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(stage1_params, 1.0)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

                epoch_loss += loss.item()
                global_step += 1

                if progress_callback:
                    progress_callback({
                        "stage": "stage1",
                        "epoch": epoch + 1,
                        "loss": loss.item(),
                        "progress_pct": int((global_step / total_steps) * 50),
                        "step": global_step,
                    })

            avg_epoch_loss = epoch_loss / max(len(dataset), 1)
            stage1_losses.append(avg_epoch_loss)
            logger.info("Stage 1 epoch %d/%d — loss: %.4f", epoch + 1, self.config.stage1_epochs, avg_epoch_loss)

        # ── Stage 2: Train connector + LoRA on LLM ─────────────────
        logger.info("Stage 2: training connector + LoRA (%d epochs)", self.config.stage2_epochs)

        # Apply LoRA to LLM
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=0.1,
        )
        self.llm = get_peft_model(self.llm, lora_config)
        self.llm.train()

        # Optimizer includes connector + LoRA params
        stage2_params = list(self.connector.parameters()) + list(self.llm.parameters())
        self.optimizer = torch.optim.AdamW(stage2_params, lr=self.config.stage2_lr)
        total_steps2 = len(dataset) * self.config.stage2_epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=0, num_training_steps=total_steps2
        )

        stage2_losses: list[float] = []

        for epoch in range(self.config.stage2_epochs):
            epoch_loss = 0.0
            for idx in range(len(dataset)):
                sample = dataset[idx]
                image_path = sample.get("image_path", "")
                caption = sample.get("caption", "") or sample.get("conversations", [{}])[-1].get("value", "")

                if not image_path or not os.path.isfile(image_path):
                    continue
                if not caption:
                    continue

                loss = self._stage1_step(image_path, caption)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(stage2_params, 1.0)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

                epoch_loss += loss.item()
                global_step += 1

                if progress_callback:
                    progress_callback({
                        "stage": "stage2",
                        "epoch": epoch + 1,
                        "loss": loss.item(),
                        "progress_pct": 50 + int((global_step / (total_steps + total_steps2)) * 50),
                        "step": global_step,
                    })

            avg_epoch_loss = epoch_loss / max(len(dataset), 1)
            stage2_losses.append(avg_epoch_loss)
            logger.info("Stage 2 epoch %d/%d — loss: %.4f", epoch + 1, self.config.stage2_epochs, avg_epoch_loss)

        # ── Save ───────────────────────────────────────────────────
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        torch.save(self.connector.state_dict(), output_dir / "connector.pt")

        # Save the LoRA adapter
        self.llm.save_pretrained(str(output_dir / "lora"))
        self.tokenizer.save_pretrained(str(output_dir / "lora"))

        # Save config
        config_dict = {
            "vision_encoder": self.config.vision_encoder,
            "llm": self.config.llm,
            "stage1_epochs": self.config.stage1_epochs,
            "stage2_epochs": self.config.stage2_epochs,
            "final_loss": stage2_losses[-1] if stage2_losses else (stage1_losses[-1] if stage1_losses else None),
            "stage1_losses": stage1_losses,
            "stage2_losses": stage2_losses,
            "total_steps": global_step,
            "elapsed_seconds": round(time.time() - t0, 1),
        }
        with open(output_dir / "vlm_config.json", "w") as f:
            json.dump(config_dict, f, indent=2)

        logger.info("VLM training complete in %.1fs — saved to %s", time.time() - t0, output_dir)

        final_loss = stage2_losses[-1] if stage2_losses else (stage1_losses[-1] if stage1_losses else 0.0)
        return {
            "status": "completed",
            "final_loss": final_loss,
            "model_path": str(output_dir),
            "total_steps": global_step,
            "elapsed_seconds": round(time.time() - t0, 1),
        }
