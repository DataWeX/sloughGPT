"""LoRA fine-tuning for .slnc models without torch.

Trains LoRA adapters on top of any model loaded via SloNetChatProvider.
Uses numpy autograd (SloNet) for gradient computation — no PyTorch required.

Usage::

    from domains.training.hf_lora_finetune import HFLoraTrainer, HFLoraConfig

    config = HFLoraConfig(
        model_path="models/gpt2.slnc",
        data_path="datasets/shakespeare/input.txt",
        rank=8,
        alpha=16.0,
        epochs=3,
        learning_rate=1e-4,
    )
    trainer = HFLoraTrainer(config)
    result = trainer.train()
    # Adapter saved to models/gpt2_lora_r8.npz
"""

from __future__ import annotations

import logging
import time
import threading
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from domains.training.lora import (
    LoRAConfig, LoRALinear, apply_lora_to_model, get_lora_parameters,
    count_lora_parameters,
)
from domains.training.slonet import (
    SloTransformer, Tensor, cross_entropy,
)
from domains.training.trainer_protocol import TrainResult

logger = logging.getLogger("slo.hf_lora")


__all__ = ["HFLoraConfig", "HFLoraTrainer"]


@dataclass
class HFLoraConfig:
    """Configuration for LoRA fine-tuning on .slnc models."""

    model_path: str = ""
    data_path: str = ""

    # LoRA hyperparameters
    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.0
    target_modules: Optional[List[str]] = field(default_factory=lambda: [
        "W_q", "W_k", "W_v", "W_o",
    ])

    # Training hyperparameters
    epochs: int = 3
    batch_size: int = 8
    block_size: int = 128
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 0
    grad_clip: float = 1.0
    grad_accumulation_steps: int = 1

    # Output
    output_dir: str = "models"
    adapter_name: Optional[str] = None  # auto-generated if None

    # Progress
    log_interval: int = 10
    progress_callback: Optional[Callable] = None

    # Cancellation
    _cancel_event: Optional[threading.Event] = field(default=None, repr=False)

    def __post_init__(self):
        if self.adapter_name is None:
            model_stem = Path(self.model_path).stem
            self.adapter_name = f"{model_stem}_lora_r{self.rank}"


class HFLoraTrainer:
    """LoRA fine-tuner for .slnc models — pure numpy, no torch.

    Loads a model from .slnc via SloNetChatProvider, applies LoRA adapters,
    trains on text data, and saves the adapter weights as .npz.
    """

    def __init__(self, config: HFLoraConfig):
        self.config = config
        self.model: Optional[SloTransformer] = None
        self.lora_params: Dict[str, Any] = {}
        self._is_training = False
        self._training_thread: Optional[threading.Thread] = None

    def load_model(self) -> SloTransformer:
        """Load model from .slnc file."""
        from domains.inference.slonet_provider import SloNetChatProvider

        model_path = Path(self.config.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        logger.info("Loading model from %s", model_path)
        provider = SloNetChatProvider.from_slnc(str(model_path))
        self.model = provider._model
        logger.info("Model loaded: %s vocab, %s embed, %s layers",
                    self.model.vocab_size, self.model.n_embed, self.model.n_layer)
        return self.model

    def apply_lora(self) -> Dict[str, Any]:
        """Apply LoRA adapters to the model."""
        if self.model is None:
            raise RuntimeError("Model not loaded — call load_model() first")

        config = LoRAConfig(
            rank=self.config.rank,
            alpha=self.config.alpha,
            dropout=self.config.dropout,
            target_modules=self.config.target_modules,
        )
        self.model = apply_lora_to_model(self.model, config)
        self.lora_params = get_lora_parameters(self.model)
        n_params = count_lora_parameters(self.model)
        logger.info("Applied LoRA: %d tensors, %s trainable parameters",
                    len(self.lora_params), f"{n_params:,}")
        return self.lora_params

    def _prepare_data(self):
        """Load and tokenize training data."""
        from domains.training.train_pipeline import prepare_data

        data_path = self.config.data_path
        if not Path(data_path).exists():
            raise FileNotFoundError(f"Data not found: {data_path}")

        # Use the model's tokenizer if available
        tokenizer = getattr(self.model, '_tokenizer', None)
        data, self._vocab_size, self._stoi, self._itos = prepare_data(
            data_path, self.config.block_size, tokenizer,
        )
        return data

    def train(self) -> TrainResult:
        """Run LoRA fine-tuning.

        Returns:
            TrainResult with adapter path and training metrics.
        """
        self._is_training = True
        start_time = time.time()

        try:
            # Load model
            self.load_model()

            # Apply LoRA
            self.apply_lora()

            # Prepare data
            data = self._prepare_data()
            dataset = _LoRADataset(data, self.config.block_size)
            if len(dataset) == 0:
                self._is_training = False
                return TrainResult(
                    success=False,
                    status="failed",
                    error=f"Data too short for block_size={self.config.block_size} "
                          f"(data={len(data)} chars, need >={self.config.block_size})",
                )

            # Create optimizer (only LoRA params)
            from domains.training.slonet import SloAdam
            lora_tensors = [p for p in self.lora_params.values()
                           if hasattr(p, 'data') and hasattr(p, 'requires_grad')]
            optimizer = SloAdam(lr=self.config.learning_rate)

            # Training loop
            best_loss = float('inf')
            total_steps = 0
            loss_history = []

            for epoch in range(self.config.epochs):
                if self._cancel_event and self._cancel_event.is_set():
                    break

                epoch_loss = 0.0
                n_batches = 0

                for i in range(0, len(dataset), self.config.batch_size):
                    if self._cancel_event and self._cancel_event.is_set():
                        break

                    # Get batch
                    batch_x = []
                    batch_y = []
                    for j in range(min(self.config.batch_size, len(dataset) - i)):
                        x, y = dataset[i + j]
                        batch_x.append(x)
                        batch_y.append(y)

                    if not batch_x:
                        break

                    x_batch = np.array(batch_x, dtype=np.int64)
                    y_batch = np.array(batch_y, dtype=np.int64)

                    # Forward pass
                    logits, _ = self.model.forward(Tensor(x_batch))
                    loss = cross_entropy(
                        logits.reshape(-1, self.model.vocab_size),
                        Tensor(y_batch.reshape(-1)),
                    )

                    # Backward pass
                    loss.backward()

                    # Gradient clipping
                    if self.config.grad_clip > 0:
                        total_norm = 0.0
                        for p in lora_tensors:
                            if hasattr(p, 'grad') and p.grad is not None:
                                total_norm += float(np.sum(p.grad.data ** 2))
                        total_norm = np.sqrt(total_norm)
                        if total_norm > self.config.grad_clip:
                            scale = self.config.grad_clip / total_norm
                            for p in lora_tensors:
                                if hasattr(p, 'grad') and p.grad is not None:
                                    p.grad.data *= scale

                    # Optimizer step
                    if (total_steps + 1) % self.config.grad_accumulation_steps == 0:
                        optimizer.step(lora_tensors)
                        for p in lora_tensors:
                            if hasattr(p, 'grad') and p.grad is not None:
                                p.grad.data[:] = 0.0

                    epoch_loss += float(loss.data)
                    n_batches += 1
                    total_steps += 1

                    # Log
                    if total_steps % self.config.log_interval == 0:
                        avg_loss = epoch_loss / max(n_batches, 1)
                        loss_history.append(avg_loss)
                        if self.config.progress_callback:
                            self.config.progress_callback({
                                "step": total_steps,
                                "epoch": epoch + 1,
                                "loss": avg_loss,
                                "lr": self.config.learning_rate,
                            })
                        logger.info(
                            "step=%d epoch=%d loss=%.4f lr=%.2e",
                            total_steps, epoch+1, avg_loss, self.config.learning_rate,
                        )

                # Epoch complete
                avg_epoch_loss = epoch_loss / max(n_batches, 1)
                if avg_epoch_loss < best_loss:
                    best_loss = avg_epoch_loss

                logger.info("Epoch %d/%d loss=%.4f",
                            epoch+1, self.config.epochs, avg_epoch_loss)

            # Save adapter
            adapter_path = self._save_adapter()

            elapsed = time.time() - start_time
            self._is_training = False

            return TrainResult(
                success=True,
                status="completed",
                final_loss=best_loss,
                total_steps=total_steps,
                model_path=str(adapter_path),
                checkpoint_name=self.config.adapter_name,
                epochs_completed=self.config.epochs,
                global_step=total_steps,
                method="hf_lora",
                metrics={
                    "rank": self.config.rank,
                    "alpha": self.config.alpha,
                    "n_lora_params": count_lora_parameters(self.model),
                    "elapsed_s": elapsed,
                    "loss_history": loss_history[-20:],  # last 20 entries
                },
            )

        except Exception as e:
            self._is_training = False
            logger.error("Training failed: %s", e)
            return TrainResult(
                success=False,
                status="failed",
                error=str(e),
            )

    def _save_adapter(self) -> Path:
        """Save LoRA adapter weights as .npz."""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        adapter_path = output_dir / f"{self.config.adapter_name}.npz"

        # Collect all LoRA weights
        adapter_dict = {}
        for name, param in self.lora_params.items():
            if hasattr(param, 'data'):
                adapter_dict[name] = param.data

        # Save metadata
        adapter_dict["_config/rank"] = np.array([self.config.rank])
        adapter_dict["_config/alpha"] = np.array([self.config.alpha])
        adapter_dict["_config/target_modules"] = np.array(
            [len(self.config.target_modules)], dtype=np.int64
        )
        for i, m in enumerate(self.config.target_modules):
            adapter_dict[f"_config/target_module_{i}"] = np.array([ord(c) for c in m], dtype=np.int64)

        np.savez_compressed(str(adapter_path), **adapter_dict)
        logger.info("Saved LoRA adapter to %s (%.1f KB)",
                    adapter_path, adapter_path.stat().st_size / 1024)
        return adapter_path

    @property
    def is_training(self) -> bool:
        return self._is_training

    def stop(self):
        """Signal training to stop."""
        if self.config._cancel_event:
            self.config._cancel_event.set()
        self._is_training = False


class _LoRADataset:
    """Simple dataset for LoRA fine-tuning."""

    def __init__(self, data: np.ndarray, block_size: int):
        if not isinstance(data, np.ndarray):
            data = np.asarray(data, dtype=np.int64)
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return max(0, len(self.data) - self.block_size)

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.block_size]
        y = self.data[idx + 1:idx + self.block_size + 1]
        return x, y


def load_lora_adapter(model: SloTransformer, adapter_path: str) -> SloTransformer:
    """Load a saved LoRA adapter into a model.

    Args:
        model: SloTransformer with LoRA layers already applied
        adapter_path: Path to .npz adapter file

    Returns:
        Model with loaded adapter weights
    """
    adapter = np.load(adapter_path)

    for name, param in get_lora_parameters(model).items():
        if name in adapter and hasattr(param, 'data'):
            param.data[:] = adapter[name]

    logger.info("Loaded LoRA adapter from %s", adapter_path)
    return model


def merge_lora_adapter(model: SloTransformer) -> SloTransformer:
    """Merge LoRA weights into base weights and replace LoRA layers with plain layers.

    After merging, the model runs at full inference speed with no LoRA overhead.
    The inlined fast path in generate_numpy requires SloLinear (with
    _get_weight_T_contig), so merged layers must be replaced.
    """
    from domains.training.lora import _walk_slo_tree, _set_nested, LoRAEmbedding
    from domains.training.slonet import SloLinear

    for path, module in _walk_slo_tree(model, []):
        if isinstance(module, LoRALinear):
            # Fold LoRA delta into base weight
            module.merge_weights()

            # Build a plain SloLinear with the merged weight
            new_linear = SloLinear(
                module.in_features, module.out_features,
                bias=module.use_bias, name=f"merged_{path.replace('.', '_')}",
                _lazy=True,
            )
            new_linear.weight.data[:] = module.weight.data
            if module.bias is not None and new_linear.bias is not None:
                new_linear.bias.data[:] = module.bias.data

            # Replace in the tree
            _set_nested(model, path.split("."), new_linear)
            logger.info("Merged LoRA into %s", path)

        elif isinstance(module, LoRAEmbedding):
            # Fold LoRA delta into base embedding weight
            module.merge_weights()

            # The SloEmbedding inside LoRAEmbedding now holds the merged weight
            base_embedding = module.weight
            base_embedding.name = f"merged_{path.replace('.', '_')}"

            # Replace in the tree
            _set_nested(model, path.split("."), base_embedding)
            logger.info("Merged LoRA embedding into %s", path)

    # Clear the LoRA flag
    model._has_lora = False
    return model
