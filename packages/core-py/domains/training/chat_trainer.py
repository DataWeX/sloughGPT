"""
SloChatTrainer — Train a small SloTransformer directly from chat data.

Pure NumPy pipeline — zero PyTorch dependency. Extracts (user, assistant)
pairs from chat sessions, builds a character-level tokenizer, and trains
a small decoder-only transformer via next-token prediction.

Output: .soul checkpoint that can be loaded into the inference pipeline.
"""

import gc
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from domains.training.slonet import (
    SloAdam, SloTransformer, export_to_sou, tensor,
)
from domains.training.quality_scorer import score_batch
from domains.training.pair_extractor import extract_pairs_from_sessions

logger = logging.getLogger("slo.training.chat_trainer")


@dataclass
class ChatTrainConfig:
    """Configuration for on-device chat training."""
    n_embed: int = 128
    n_layer: int = 4
    n_head: int = 4
    block_size: int = 128
    dropout: float = 0.1

    epochs: int = 10
    lr: float = 3e-4
    batch_size: int = 8
    grad_clip: float = 1.0
    warmup_steps: int = 50
    eval_interval: int = 50
    log_interval: int = 10

    min_pair_quality: float = 2.0
    max_pairs: int = 500
    val_split: float = 0.1

    checkpoint_dir: str = "models/auto-training"
    soul_name: str = "chat-trained"

    session_ids: Optional[List[str]] = None
    resume_checkpoint: Optional[str] = None
    resume_epoch: int = 0
    resume_step: int = 0


class ChatTextDataset:
    """Character-level text dataset from formatted chat pairs."""

    def __init__(self, text: str, block_size: int, stoi: Dict[str, int]):
        self.text = text
        self.block_size = block_size
        self.stoi = stoi
        self.ids = [stoi.get(c, 0) for c in text]
        self.n_samples = max(1, len(self.ids) - block_size - 1)

    def __len__(self) -> int:
        return self.n_samples

    def get_batch(self, batch_size: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
        """Get random batch of (x, y) pairs for next-token prediction."""
        indices = rng.integers(0, self.n_samples, size=batch_size)
        x = np.zeros((batch_size, self.block_size), dtype=np.int32)
        y = np.zeros((batch_size, self.block_size), dtype=np.int32)
        for i, idx in enumerate(indices):
            chunk = self.ids[idx:idx + self.block_size + 1]
            x[i, :len(chunk) - 1] = chunk[:-1]
            y[i, :len(chunk) - 1] = chunk[1:]
        return x, y


def _cross_entropy_loss(logits: np.ndarray, targets: np.ndarray) -> float:
    """Cross-entropy loss on (batch, vocab) logits and (batch,) targets."""
    log_probs = logits - logits.max(axis=-1, keepdims=True)
    log_probs = log_probs - np.log(np.exp(log_probs).sum(axis=-1, keepdims=True))
    batch_size = targets.shape[0]
    return float(-log_probs[np.arange(batch_size), targets].mean())


def _build_vocab(pairs: List[Dict[str, str]]) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Build character-level vocabulary from chat pairs."""
    chars = set()
    for pair in pairs:
        chars.update(pair["user_msg"])
        chars.update(pair["assistant_msg"])

    sorted_chars = sorted(chars)
    stoi = {c: i + 1 for i, c in enumerate(sorted_chars)}
    stoi["\x00"] = 0
    itos = {i: c for c, i in stoi.items()}
    return stoi, itos


def _format_pairs_text(pairs: List[Dict[str, str]]) -> str:
    """Format pairs as training text."""
    parts = []
    for pair in pairs:
        parts.append(f"User: {pair['user_msg']}\nAssistant: {pair['assistant_msg']}\n\n")
    return "".join(parts)


def train_chat_model(
    pairs: List[Dict[str, str]],
    config: Optional[ChatTrainConfig] = None,
    on_step: Optional[Callable[[int, float, int], None]] = None,
    cancel_event=None,
) -> Tuple[SloTransformer, Dict[str, Any]]:
    """Train a SloTransformer on chat pairs.

    Args:
        pairs: List of {"user_msg": "...", "assistant_msg": "..."} dicts.
        config: Training configuration.
        on_step: Callback(step, loss, epoch).
        cancel_event: threading.Event() to cancel.

    Returns:
        (trained_model, metadata_dict)
    """
    config = config or ChatTrainConfig()

    if not pairs:
        raise ValueError("No training pairs provided")

    logger.info("SloChatTrainer: %d pairs, config=%s", len(pairs), config,
        extra={"tag": "TRAIN"})

    # Filter low-quality pairs
    scored = score_batch(pairs)
    good_pairs = [
        p for p, s in zip(pairs, scored)
        if s >= config.min_pair_quality
    ]
    logger.info("Quality filter: %d/%d pairs passed (threshold=%.1f)",
                len(good_pairs), len(pairs), config.min_pair_quality,
                extra={"tag": "TRAIN"})

    if len(good_pairs) < 5:
        logger.warning("Too few quality pairs (%d), using all", len(good_pairs),
            extra={"tag": "TRAIN"})
        good_pairs = pairs[:max(5, len(pairs))]

    if len(good_pairs) > config.max_pairs:
        good_pairs = good_pairs[:config.max_pairs]

    # Build vocab and text
    stoi, itos = _build_vocab(good_pairs)
    vocab_size = len(stoi)
    text = _format_pairs_text(good_pairs)
    logger.info("Vocab: %d chars, text: %d chars", vocab_size, len(text),
        extra={"tag": "TRAIN"})

    # Split train/val
    val_size = max(1, int(len(good_pairs) * config.val_split))
    val_pairs = good_pairs[:val_size]
    train_pairs = good_pairs[val_size:]
    val_text = _format_pairs_text(val_pairs)
    train_text = _format_pairs_text(train_pairs) if train_pairs else text

    # Create or resume model
    start_epoch = config.resume_epoch
    start_step = config.resume_step
    best_loss = float("inf")

    if config.resume_checkpoint and Path(config.resume_checkpoint).exists():
        logger.info("Resuming from: %s", config.resume_checkpoint,
            extra={"tag": "TRAIN"})
        from domains.training.slonet import import_from_sou
        model = import_from_sou(config.resume_checkpoint)
        if hasattr(model, 'metadata') and model.metadata:
            meta = model.metadata
            start_epoch = meta.get("epoch", config.resume_epoch)
            start_step = meta.get("step", config.resume_step)
            best_loss = meta.get("best_loss", float("inf"))
    else:
        logger.info("Creating model: embed=%d, layers=%d, heads=%d, block=%d",
                    config.n_embed, config.n_layer, config.n_head, config.block_size,
                    extra={"tag": "TRAIN"})
        model = SloTransformer(
            vocab_size=vocab_size,
            n_embed=config.n_embed,
            n_layer=config.n_layer,
            n_head=config.n_head,
            block_size=config.block_size,
            max_seq_len=config.block_size * 2,
            dropout=config.dropout,
            use_rope=True,
            norm_type="rms_norm",
            tie_weights=True,
            soul_name=config.soul_name,
        )

    # Datasets
    train_ds = ChatTextDataset(train_text, config.block_size, stoi)
    val_ds = ChatTextDataset(val_text, config.block_size, stoi) if val_text else None
    logger.info("Dataset: train=%d samples, val=%d samples",
                len(train_ds), len(val_ds) if val_ds else 0,
                extra={"tag": "TRAIN"})

    optimizer = SloAdam(lr=config.lr)
    rng = np.random.default_rng(42)

    # Training loop
    steps_per_epoch = max(1, len(train_ds) // config.batch_size)
    total_steps = config.epochs * steps_per_epoch
    step = start_step
    train_losses = []
    val_losses = []

    logger.info("Starting training: %d epochs, %d steps/epoch, %d total",
                config.epochs, steps_per_epoch, total_steps,
                extra={"tag": "TRAIN"})

    for epoch in range(start_epoch, config.epochs):
        if cancel_event and cancel_event.is_set():
            logger.info("Training cancelled at epoch %d", epoch,
                extra={"tag": "TRAIN"})
            break

        epoch_loss = 0.0
        epoch_tokens = 0

        for _step in range(steps_per_epoch):
            if cancel_event and cancel_event.is_set():
                break

            x_batch, y_batch = train_ds.get_batch(config.batch_size, rng)
            x_t = tensor(x_batch, requires_grad=False)
            y_t = tensor(y_batch, requires_grad=False)

            logits, _ = model.forward(x_t)

            if hasattr(logits, 'data'):
                logits_np = logits.data
            else:
                logits_np = np.array(logits)

            # Reshape for cross-entropy: (batch * seq, vocab) vs (batch * seq,)
            batch_size, seq_len, vocab = logits_np.shape
            flat_logits = logits_np.reshape(-1, vocab)
            flat_targets = y_batch.reshape(-1)

            # Compute loss
            loss_val = _cross_entropy_loss(flat_logits, flat_targets)

            # Backward
            loss_tensor = tensor([loss_val], requires_grad=True)
            loss_tensor.backward()

            # Clip gradients
            params = model.parameters()
            total_norm = 0.0
            for p in params:
                if hasattr(p, 'grad') and p.grad is not None:
                    g = p.grad.data if hasattr(p, 'grad') and hasattr(p.grad, 'data') else np.array(p.grad)
                    total_norm += float(np.sum(g ** 2))
            total_norm = total_norm ** 0.5
            if total_norm > config.grad_clip:
                scale = config.grad_clip / total_norm
                for p in params:
                    if hasattr(p, 'grad') and p.grad is not None:
                        p.grad.data *= scale

            # Step
            optimizer.step(params)
            for p in params:
                if hasattr(p, 'grad') and p.grad is not None:
                    p.grad.data = np.zeros_like(p.grad.data)

            step += 1
            epoch_loss += loss_val * batch_size * seq_len
            epoch_tokens += batch_size * seq_len
            train_losses.append(loss_val)

            if step % config.log_interval == 0:
                avg = epoch_loss / max(1, epoch_tokens)
                logger.info("Step %d/%d epoch=%d loss=%.4f",
                           step, total_steps, epoch, avg,
                           extra={"tag": "TRAIN"})
                if on_step:
                    on_step(step, avg, epoch)

            # Periodic eval
            if val_ds and step % config.eval_interval == 0:
                val_loss = _eval_loss(model, val_ds, config.batch_size, rng)
                val_losses.append(val_loss)
                logger.info("  val_loss=%.4f", val_loss,
                    extra={"tag": "TRAIN"})

            gc.collect()

        # End of epoch
        avg_epoch_loss = epoch_loss / max(1, epoch_tokens)
        logger.info("Epoch %d complete: avg_loss=%.4f", epoch, avg_epoch_loss,
            extra={"tag": "TRAIN"})

        # Checkpoint on best loss
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss

    # Final eval
    final_val = _eval_loss(model, val_ds, config.batch_size, rng) if val_ds else float("nan")
    if val_losses:
        val_losses.append(final_val)

    # Save checkpoint
    ckpt_dir = Path(config.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"{config.soul_name}.soul"

    export_to_sou(
        model, str(ckpt_path),
        metadata={
            "soul_name": config.soul_name,
            "vocab_size": vocab_size,
            "n_embed": config.n_embed,
            "n_layer": config.n_layer,
            "n_head": config.n_head,
            "block_size": config.block_size,
            "epoch": epoch + 1,
            "step": step,
            "epochs": config.epochs,
            "final_loss": float(avg_epoch_loss) if 'avg_epoch_loss' in dir() else 0.0,
            "best_loss": best_loss,
            "num_pairs": len(good_pairs),
            "num_chars": len(text),
            "stoi": stoi,
            "itos": itos,
        },
    )
    logger.info("Checkpoint saved: %s", ckpt_path,
        extra={"tag": "TRAIN"})

    metadata = {
        "checkpoint": str(ckpt_path),
        "final_loss": float(avg_epoch_loss) if 'avg_epoch_loss' in dir() else 0.0,
        "best_loss": best_loss,
        "val_loss": final_val if not np.isnan(final_val) else None,
        "num_pairs": len(good_pairs),
        "total_pairs": len(pairs),
        "vocab_size": vocab_size,
        "epochs_completed": epoch + 1 if 'epoch' in dir() else config.epochs,
        "total_steps": step,
        "train_losses": [float(l) for l in train_losses[-20:]],
        "val_losses": [float(l) for l in val_losses[-20:]],
        "stoi": stoi,
        "itos": itos,
    }

    return model, metadata


def _eval_loss(
    model: SloTransformer,
    val_ds: ChatTextDataset,
    batch_size: int,
    rng: np.random.Generator,
) -> float:
    """Compute average cross-entropy loss on validation set."""
    total_loss = 0.0
    total_tokens = 0
    n_batches = min(50, len(val_ds) // batch_size)

    for _ in range(n_batches):
        x_batch, y_batch = val_ds.get_batch(batch_size, rng)
        x_t = tensor(x_batch, requires_grad=False)
        logits, _ = model.forward(x_t)
        if hasattr(logits, 'data'):
            logits_np = logits.data
        else:
            logits_np = np.array(logits)

        bs, sl, v = logits_np.shape
        flat_logits = logits_np.reshape(-1, v)
        flat_targets = y_batch.reshape(-1)
        loss = _cross_entropy_loss(flat_logits, flat_targets)
        total_loss += loss * bs * sl
        total_tokens += bs * sl

    return total_loss / max(1, total_tokens)


def generate_from_chat_model(
    model: SloTransformer,
    stoi: Dict[str, int],
    itos: Dict[int, str],
    prompt: str,
    max_tokens: int = 100,
    temperature: float = 0.8,
) -> str:
    """Generate text from a trained chat model.

    Args:
        model: Trained SloTransformer.
        stoi: String-to-index mapping.
        itos: Index-to-string mapping.
        prompt: Input prompt.
        max_tokens: Max tokens to generate.
        temperature: Sampling temperature.

    Returns:
        Generated text (excluding prompt).
    """
    tokens = [stoi.get(c, 0) for c in prompt]
    if not tokens:
        tokens = [0]

    for _ in range(max_tokens):
        # Truncate to block_size
        ctx = tokens[-model.block_size:] if hasattr(model, 'block_size') else tokens[-128:]
        x = tensor([ctx], requires_grad=False)
        logits, _ = model.forward(x)
        if hasattr(logits, 'data'):
            logits_np = logits.data
        else:
            logits_np = np.array(logits)

        next_logits = logits_np[0, -1]

        if temperature > 0:
            next_logits = next_logits / temperature
            exp_logits = np.exp(next_logits - next_logits.max())
            probs = exp_logits / exp_logits.sum()
            rng = np.random.default_rng()
            next_id = int(rng.choice(len(probs), p=probs))
        else:
            next_id = int(np.argmax(next_logits))

        if next_id == 0:
            break
        tokens.append(next_id)

    return "".join(itos.get(t, "") for t in tokens[len(tokens) - (max_tokens):])


def evaluate_chat_model(
    model: SloTransformer,
    stoi: Dict[str, int],
    itos: Dict[int, str],
    pairs: List[Dict[str, str]],
    max_samples: int = 5,
) -> Dict[str, Any]:
    """Evaluate a trained chat model.

    Computes perplexity on pair text and generates sample responses.

    Args:
        model: Trained SloTransformer.
        stoi: String-to-index mapping.
        itos: Index-to-string mapping.
        pairs: Chat pairs to evaluate on.
        max_samples: Number of samples to generate.

    Returns:
        Dict with perplexity, samples, avg_response_len.
    """
    # Compute perplexity on all pair text
    text = _format_pairs_text(pairs)
    ds = ChatTextDataset(text, model.block_size, stoi)
    rng = np.random.default_rng(42)
    perplexity = _eval_loss(model, ds, 8, rng)
    # eval_loss returns cross-entropy; perplexity = exp(CE)
    perplexity_val = float(np.exp(min(perplexity, 20.0)))

    # Generate samples from prompts
    samples = []
    eval_prompts = [
        "User: Hello",
        "User: What",
        "User: How",
        "User: Can you",
        "User: Tell me",
    ]
    for prompt in eval_prompts[:max_samples]:
        response = generate_from_chat_model(model, stoi, itos, prompt, max_tokens=50, temperature=0.7)
        samples.append({"prompt": prompt, "response": response})

    # Average response length
    avg_len = 0.0
    if samples:
        avg_len = float(np.mean([len(s["response"].split()) for s in samples]))

    return {
        "perplexity": perplexity_val,
        "samples": samples,
        "avg_response_len": avg_len,
    }


def train_from_sessions(
    config: Optional[ChatTrainConfig] = None,
    on_step: Optional[Callable[[int, float, int], None]] = None,
    cancel_event=None,
) -> Tuple[SloTransformer, Dict[str, Any]]:
    """High-level: extract pairs from sessions and train.

    Args:
        config: Training config.
        on_step: Progress callback.
        cancel_event: Cancellation event.

    Returns:
        (trained_model, metadata_dict) with evaluation results.
    """
    config = config or ChatTrainConfig()
    pairs = extract_pairs_from_sessions(limit=config.max_pairs, session_ids=config.session_ids)
    if not pairs:
        raise ValueError("No chat sessions found to train on")

    logger.info("Extracted %d pairs from sessions", len(pairs),
        extra={"tag": "TRAIN"})

    model, metadata = train_chat_model(pairs, config, on_step, cancel_event)

    # Run evaluation using the same vocab as training
    try:
        stoi = metadata.get("stoi", {})
        itos = metadata.get("itos", {})
        eval_result = evaluate_chat_model(model, stoi, itos, pairs)
        metadata["perplexity"] = eval_result["perplexity"]
        metadata["samples"] = eval_result["samples"]
        metadata["avg_response_len"] = eval_result["avg_response_len"]
    except Exception as e:
        logger.warning("Evaluation failed: %s", e, extra={"tag": "TRAIN"})

    return model, metadata
