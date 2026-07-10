"""
Distill GPT-2 teacher → smaller SloTransformer student.

Pure NumPy pipeline — no PyTorch dependency. Uses GPT-2's numpy forward
pass as the teacher and SloTransformer's autograd for the student.
"""

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from domains.infrastructure.arch_config import ArchConfig, build_arch
from domains.infrastructure.numpy_forward import forward_fast, pre_extract_weights
from domains.training.slonet import (
    SloAdam, SloTransformer, Tensor, export_to_sou, no_grad, tensor,
)

logger = logging.getLogger(__name__)


@dataclass
class DistillConfig:
    """Configuration for GPT-2 → SloTransformer distillation."""
    # Student architecture
    n_embed: int = 128
    n_layer: int = 4
    n_head: int = 4
    block_size: int = 128
    dropout: float = 0.1

    # Training
    epochs: int = 10
    lr: float = 3e-4
    batch_size: int = 8
    grad_clip: float = 1.0
    warmup_steps: int = 100

    # Distillation
    temperature: float = 4.0
    alpha: float = 0.5  # weight for hard CE loss
    beta: float = 0.5   # weight for soft KL loss

    # Teacher
    teacher_model: str = "gpt2"

    # Checkpointing
    checkpoint_dir: str = "models/auto-training"
    eval_interval: int = 50
    log_interval: int = 10


class TextDataset:
    """Simple character-level text dataset for distillation."""

    def __init__(self, text: str, block_size: int, stoi: Dict[str, int]):
        self.text = text
        self.block_size = block_size
        self.stoi = stoi
        self.ids = [stoi.get(c, 0) for c in text]
        self.n_samples = max(1, len(self.ids) - block_size - 1)

    def __len__(self):
        return self.n_samples

    def get_batch(self, batch_size: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
        """Get random batch of (x, y) pairs."""
        indices = rng.integers(0, self.n_samples, size=batch_size)
        x = np.zeros((batch_size, self.block_size), dtype=np.int32)
        y = np.zeros((batch_size, self.block_size), dtype=np.int32)
        for i, idx in enumerate(indices):
            chunk = self.ids[idx:idx + self.block_size + 1]
            x[i, :len(chunk) - 1] = chunk[:-1]
            y[i, :len(chunk) - 1] = chunk[1:]
        return x, y


def _load_gpt2_numpy() -> Tuple[dict, ArchConfig, dict]:
    """Load GPT-2 weights as numpy arrays + arch config + tokenizer vocab."""
    from safetensors import safe_open

    hf_path = Path.home() / ".cache/huggingface/hub/models--gpt2"
    snapshots = sorted((hf_path / "snapshots").glob("*"))
    if not snapshots:
        raise RuntimeError("GPT-2 not found in HuggingFace cache. Download first.")
    snap = snapshots[0]

    weights = {}
    for f in sorted(snap.glob("*.safetensors")):
        with safe_open(str(f), framework="numpy") as sf:
            for key in sf.keys():
                weights[key] = sf.get_tensor(key)

    arch = build_arch("gpt2", {}, set(weights.keys()))
    rw = pre_extract_weights(arch, weights)

    # Load GPT-2 tokenizer vocab
    tokenizer_path = snap / "tokenizer.json"
    if tokenizer_path.exists():
        tok_data = json.loads(tokenizer_path.read_text())
        vocab = tok_data["model"]["vocab"]  # dict: string → int
        itos = {i: s for s, i in vocab.items()}
        stoi = {s: i for i, s in vocab.items()}
    else:
        raise RuntimeError(f"tokenizer.json not found in {snap}")

    return rw, arch, {"stoi": stoi, "itos": itos, "vocab_size": len(stoi)}


def _teacher_forward(rw: dict, arch: ArchConfig, token_ids: List[int]) -> np.ndarray:
    """Run GPT-2 teacher forward pass, return logits as numpy."""
    return forward_fast(rw, arch, token_ids)


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x_max = x.max(axis=axis, keepdims=True)
    e_x = np.exp(x - x_max)
    return e_x / e_x.sum(axis=axis, keepdims=True)


def _kl_div_loss(student_log_probs: np.ndarray, teacher_probs: np.ndarray) -> float:
    """KL divergence: sum(teacher * (log(teacher) - student_log_probs))."""
    teacher_safe = np.where(teacher_probs < 1e-15, 1e-15, teacher_probs)
    log_teacher = np.log(teacher_safe)
    loss = teacher_probs * (log_teacher - student_log_probs)
    return float(loss.sum(axis=-1).mean())


def _cross_entropy_loss(logits: np.ndarray, targets: np.ndarray) -> float:
    """Cross-entropy loss."""
    # logits: (batch, vocab), targets: (batch,)
    log_probs = logits - logits.max(axis=-1, keepdims=True)
    log_probs = log_probs - np.log(np.exp(log_probs).sum(axis=-1, keepdims=True))
    batch_size = targets.shape[0]
    loss = -log_probs[np.arange(batch_size), targets].mean()
    return float(loss)


def distill_gpt2_to_slo(
    text: str,
    config: Optional[DistillConfig] = None,
    on_step: Optional[Callable[[int, float, int], None]] = None,
    cancel_event=None,
) -> Tuple[SloTransformer, Dict[str, str]]:
    """
    Distill GPT-2 into a smaller SloTransformer.

    Args:
        text: Training text corpus.
        config: Distillation config.
        on_step: Callback(step, loss, epoch).
        cancel_event: Threading event to cancel training.

    Returns:
        (trained_model, metadata_dict)
    """
    config = config or DistillConfig()
    logger.info("Loading GPT-2 teacher...")
    rw, teacher_arch, tok_vocab = _load_gpt2_numpy()
    stoi = tok_vocab["stoi"]
    itos = tok_vocab["itos"]
    vocab_size = tok_vocab["vocab_size"]
    logger.info("Teacher loaded: %d vocab, %d layers", vocab_size, teacher_arch.n_layers)

    # Create student
    logger.info("Creating student: n_embed=%d, n_layer=%d, n_head=%d",
                config.n_embed, config.n_layer, config.n_head)
    student = SloTransformer(
        vocab_size=vocab_size,
        n_embed=config.n_embed,
        n_layer=config.n_layer,
        n_head=config.n_head,
        block_size=config.block_size,
        max_seq_len=config.block_size * 2,
        dropout=config.dropout,
        use_rope=False,  # GPT-2 uses absolute pos emb
        use_abs_pos_emb=True,
        norm_type="layer_norm",
        tie_weights=True,
        soul_name="distilled-gpt2",
    )

    # Dataset
    dataset = TextDataset(text, config.block_size, stoi)
    logger.info("Dataset: %d chars, %d samples", len(text), len(dataset))
    optimizer = SloAdam(lr=config.lr)
    rng = np.random.default_rng(42)

    # Training loop
    total_steps = config.epochs * (len(dataset) // config.batch_size)
    step = 0
    best_loss = float("inf")

    # Build teacher token map for fast lookup
    # Teacher: forward_fast needs token_ids as a list
    logger.info("Starting distillation: %d epochs, %d total steps", config.epochs, total_steps)

    for epoch in range(config.epochs):
        if cancel_event and cancel_event.is_set():
            logger.info("Training cancelled")
            break

        epoch_loss = 0.0
        epoch_steps = 0

        for batch_idx in range(len(dataset) // config.batch_size):
            if cancel_event and cancel_event.is_set():
                break

            x_np, y_np = dataset.get_batch(config.batch_size, rng)

            # --- Teacher forward (frozen, no grad) ---
            teacher_logits_list = []
            for i in range(config.batch_size):
                token_ids = x_np[i].tolist()
                t_logits = _teacher_forward(rw, teacher_arch, token_ids)
                teacher_logits_list.append(t_logits)
            teacher_logits = np.stack(teacher_logits_list)  # (batch, seq, vocab)

            # --- Student forward ---
            x_tensor = tensor(x_np.tolist(), requires_grad=True)
            y_tensor = tensor(y_np.tolist())
            s_logits, _ = student.forward(x_tensor, y_tensor)

            # Convert to numpy
            if hasattr(s_logits, 'data'):
                s_np = s_logits.data
            else:
                s_np = np.array(s_logits)
            if s_np.ndim == 3:
                s_np = s_np.reshape(-1, s_np.shape[-1])

            # Truncate to min length
            t_flat = teacher_logits.reshape(-1, teacher_logits.shape[-1])
            min_len = min(s_np.shape[0], t_flat.shape[0])
            s_np = s_np[:min_len]
            t_flat = t_flat[:min_len]
            y_flat = y_np.reshape(-1)[:min_len]

            # --- Distillation loss ---
            # Soft loss: KL(student/T || teacher/T)
            s_soft = _softmax(s_np / config.temperature)
            t_soft = _softmax(t_flat / config.temperature)
            soft_loss = _kl_div_loss(
                np.log(np.where(s_soft < 1e-15, 1e-15, s_soft)),
                t_soft,
            ) * (config.temperature ** 2)

            # Hard loss: CE(student, ground truth)
            hard_loss = _cross_entropy_loss(s_np, y_flat)

            # Combined
            total_loss = config.alpha * hard_loss + config.beta * soft_loss

            # --- Backward ---
            loss_tensor = tensor([total_loss], requires_grad=True)
            loss_tensor.backward()
            optimizer.step(student.parameters())

            # Zero grads
            for p in student.parameters():
                if hasattr(p, 'grad'):
                    p.grad = None

            epoch_loss += total_loss
            epoch_steps += 1
            step += 1

            if step % config.log_interval == 0:
                avg = epoch_loss / epoch_steps
                logger.info("step %d/%d loss=%.4f (hard=%.4f soft=%.4f)",
                            step, total_steps, total_loss, hard_loss, soft_loss)
                if on_step:
                    on_step(step, total_loss, epoch)

            # Eval
            if step % config.eval_interval == 0 and epoch_steps > 0:
                avg_loss = epoch_loss / epoch_steps
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    logger.info("New best loss: %.4f", best_loss)

        if cancel_event and cancel_event.is_set():
            break

        avg_epoch = epoch_loss / max(epoch_steps, 1)
        logger.info("Epoch %d/%d avg_loss=%.4f", epoch + 1, config.epochs, avg_epoch)

    # Save checkpoint
    ckpt_dir = Path(config.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_name = f"distilled_gpt2_{int(time.time())}.soul"
    ckpt_path = ckpt_dir / ckpt_name

    export_to_sou(student, str(ckpt_path), soul_profile=None)

    metadata = {
        "checkpoint": str(ckpt_path),
        "final_loss": str(epoch_loss / max(epoch_steps, 1)),
        "best_loss": str(best_loss),
        "epochs": str(epoch + 1),
        "steps": str(step),
        "teacher": config.teacher_model,
        "student_config": json.dumps({
            "n_embed": config.n_embed,
            "n_layer": config.n_layer,
            "n_head": config.n_head,
            "block_size": config.block_size,
        }),
        "vocab_size": str(vocab_size),
    }

    logger.info("Distillation complete. Checkpoint: %s", ckpt_path)
    return student, metadata


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    # Quick test with small text
    text = "The quick brown fox jumps over the lazy dog. " * 100
    config = DistillConfig(
        n_embed=64, n_layer=2, n_head=4,
        epochs=3, batch_size=4, block_size=64,
        log_interval=5, eval_interval=25,
    )
    model, meta = distill_gpt2_to_slo(text, config)
    print(json.dumps(meta, indent=2))
