"""
SloChatTrainer — Train a small SloTransformer directly from chat data.

Pure NumPy pipeline. Extracts (user, assistant)
pairs from chat sessions, builds a character-level tokenizer, and trains
a small decoder-only transformer via next-token prediction.

Output: .soul checkpoint that can be loaded into the inference pipeline.
"""

from __future__ import annotations

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
        """Get random batch of (x, y) pairs for next-token prediction.

        Uses vectorized advanced indexing instead of Python-level loops.
        """
        indices = rng.integers(0, self.n_samples, size=batch_size)
        offsets = np.arange(self.block_size)
        ids = np.asarray(self.ids, dtype=np.int32)
        pos = indices[:, None] + offsets
        x = ids[pos]
        y = ids[pos + 1]
        return x.astype(np.int32), y.astype(np.int32)


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


def _cross_entropy_loss(logits: np.ndarray, targets: np.ndarray) -> float:
    """Cross-entropy loss on (batch, vocab) logits and (batch,) targets."""
    log_probs = logits - logits.max(axis=-1, keepdims=True)
    log_probs = log_probs - np.log(np.exp(log_probs).sum(axis=-1, keepdims=True))
    batch_size = targets.shape[0]
    return float(-log_probs[np.arange(batch_size), targets].mean())


@dataclass
class _ResumeState:
    """Validated state restored from a training checkpoint."""
    model: SloTransformer = None
    epoch: int = 0
    step: int = 0
    best_loss: float = float("inf")
    stoi: Optional[Dict[str, int]] = None
    itos: Optional[Dict[int, str]] = None
    vocab_size: int = 0
    optimizer_state: Optional[dict] = None
    found: bool = False


def _validate_int(value, lo: int, hi: int, name: str, default: int) -> int:
    """Validate an integer metadata field is within bounds."""
    if isinstance(value, (int, float)) and lo <= value < hi:
        return int(value)
    logger.warning("Checkpoint %s=%s invalid (expected [%s,%s)), using %s",
        name, value, lo, hi, default, extra={"tag": "TRAIN"})
    return default


def _validate_float(value, name: str, default: float) -> float:
    """Validate a float metadata field is finite and positive."""
    if isinstance(value, (int, float)) and np.isfinite(value) and value > 0:
        return float(value)
    return default


def _repair_itos(raw: dict) -> Dict[int, str]:
    """Convert JSON-deserialized itos (string keys) back to int keys.

    Skips corrupt entries that cannot be cast to int.
    """
    repaired = {}
    for k, v in raw.items():
        try:
            repaired[int(k)] = v
        except (ValueError, TypeError):
            logger.warning("Skipping corrupt itos entry: %s=%s", k, v,
                extra={"tag": "TRAIN"})
    return repaired


def _load_resume_state(
    config: ChatTrainConfig,
    data_stoi: Dict[str, int],
    data_itos: Dict[int, str],
    data_vocab_size: int,
) -> _ResumeState:
    """Load and validate a training checkpoint for resume.

    Returns a _ResumeState with validated fields. If the checkpoint is
    missing or corrupt, falls back to data-derived values.
    """
    result = _ResumeState()
    path = config.resume_checkpoint

    if not path or not Path(path).exists():
        return result

    logger.info("Resuming from: %s", path, extra={"tag": "TRAIN"})
    from domains.training.slonet import import_from_sou
    model = import_from_sou(path)
    raw = model.metadata if hasattr(model, 'metadata') and model.metadata else {}

    result.model = model
    result.found = True
    result.epoch = _validate_int(
        raw.get("epoch", config.resume_epoch), 0, config.epochs, "epoch", 0)
    result.step = _validate_int(
        raw.get("step", config.resume_step), 0, 1_000_000, "step", 0)
    result.best_loss = _validate_float(
        raw.get("best_loss", float("inf")), "best_loss", float("inf"))

    # Vocab: prefer checkpoint (prevents mismatch when data changes)
    raw_stoi = raw.get("stoi")
    raw_itos = raw.get("itos")
    if (raw_stoi and isinstance(raw_stoi, dict)
            and raw_itos and isinstance(raw_itos, dict)):
        result.stoi = raw_stoi
        result.itos = _repair_itos(raw_itos)
        if result.itos:
            result.vocab_size = len(result.stoi)
            logger.info("Using checkpoint vocab: %d chars", result.vocab_size,
                extra={"tag": "TRAIN"})
        else:
            logger.warning("itos empty after repair, using vocab from data",
                extra={"tag": "TRAIN"})
            result.stoi, result.itos = None, None
    else:
        logger.warning("Checkpoint missing stoi/itos, using vocab from data",
            extra={"tag": "TRAIN"})

    result.optimizer_state = raw.get("optimizer_state")
    return result


def train_chat_model(
    pairs: List[Dict[str, str]],
    config: Optional[ChatTrainConfig] = None,
    on_step: Optional[Callable] = None,
    cancel_event=None,
) -> Tuple[SloTransformer, Dict[str, Any]]:
    """Train a SloTransformer on chat pairs.

    Args:
        pairs: List of {"user_msg": "...", "assistant_msg": "..."} dicts.
        config: Training configuration.
        on_step: Callback(step, loss, epoch, total_steps=total_steps).
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

    # Split train/val — pairs are newest-first from extractor, so validate
    # on the oldest (tail) and train on the newer (head).
    val_size = max(1, int(len(good_pairs) * config.val_split))
    val_pairs = good_pairs[-val_size:]
    train_pairs = good_pairs[:-val_size] if val_size < len(good_pairs) else good_pairs
    val_text = _format_pairs_text(val_pairs)
    train_text = _format_pairs_text(train_pairs) if train_pairs else text

    # Create or resume model
    start_epoch = config.resume_epoch
    start_step = config.resume_step
    best_loss = float("inf")
    resume = _load_resume_state(config, stoi, itos, vocab_size)
    if resume.found:
        model = resume.model
        stoi = resume.stoi or stoi
        itos = resume.itos or itos
        vocab_size = resume.vocab_size or vocab_size
        start_epoch, start_step = resume.epoch, resume.step
        best_loss = resume.best_loss
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

    # Restore optimizer state from checkpoint
    if resume.optimizer_state:
        try:
            optimizer.load_state_dict(resume.optimizer_state, params=list(model.parameters()))
            logger.info("Restored optimizer state: t=%s",
                resume.optimizer_state.get("t", "?"), extra={"tag": "TRAIN"})
        except Exception as e:
            logger.warning("Could not restore optimizer state (%s), starting fresh",
                type(e).__name__, extra={"tag": "TRAIN"})

    # Training loop
    steps_per_epoch = max(1, len(train_ds) // config.batch_size)
    remaining_epochs = max(0, config.epochs - start_epoch)
    remaining_steps = remaining_epochs * steps_per_epoch
    total_steps = start_step + remaining_steps
    step = start_step
    train_losses: List[float] = []
    val_losses: List[float] = []
    avg_epoch_loss = 0.0
    last_epoch = start_epoch

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

            # Forward + loss in one pass — loss Tensor is connected to the
            # model's autograd graph, so .backward() produces real gradients.
            _, loss = model.forward(x_t, targets=y_t)
            loss_val = loss.item()
            loss.backward()

            # Clip gradients
            params = model.parameters()
            total_norm = 0.0
            for p in params:
                if p.grad is not None:
                    g = p.grad.data if hasattr(p.grad, 'data') else p.grad
                    total_norm += float(np.sum(g ** 2))
            total_norm = total_norm ** 0.5
            if total_norm > config.grad_clip:
                scale = config.grad_clip / total_norm
                for p in params:
                    if p.grad is not None:
                        p.grad.data *= scale

            # Step
            optimizer.step(params)
            for p in params:
                if p.grad is not None:
                    p.grad.data = np.zeros_like(p.grad.data)

            step += 1
            bs, sl = x_batch.shape
            epoch_loss += loss_val * bs * sl
            epoch_tokens += bs * sl
            train_losses.append(loss_val)

            if step % config.log_interval == 0:
                avg = epoch_loss / max(1, epoch_tokens)
                logger.info("Step %d/%d epoch=%d loss=%.4f",
                           step, total_steps, epoch, avg,
                           extra={"tag": "TRAIN"})
                if on_step:
                    on_step(step, avg, epoch, total_steps=total_steps)

            # Periodic eval
            if val_ds and step % config.eval_interval == 0:
                val_loss = _eval_loss(model, val_ds, config.batch_size, rng)
                val_losses.append(val_loss)
                logger.info("  val_loss=%.4f", val_loss,
                    extra={"tag": "TRAIN"})

            # Periodic GC — every 100 steps, not every step
            if step % 100 == 0:
                gc.collect()

        # End of epoch
        avg_epoch_loss = epoch_loss / max(1, epoch_tokens)
        last_epoch = epoch + 1
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
            "epoch": last_epoch,
            "step": step,
            "epochs": config.epochs,
            "final_loss": float(avg_epoch_loss),
            "best_loss": best_loss,
            "num_pairs": len(good_pairs),
            "num_chars": len(text),
            "stoi": stoi,
            "itos": itos,
            "optimizer_state": optimizer.state_dict(params=list(model.parameters())),
        },
    )
    logger.info("Checkpoint saved: %s", ckpt_path,
        extra={"tag": "TRAIN"})

    metadata = {
        "checkpoint": str(ckpt_path),
        "final_loss": float(avg_epoch_loss),
        "best_loss": best_loss,
        "val_loss": final_val if not np.isnan(final_val) else None,
        "num_pairs": len(good_pairs),
        "total_pairs": len(pairs),
        "vocab_size": vocab_size,
        "epochs_completed": last_epoch,
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
        y_t = tensor(y_batch, requires_grad=False)
        _, loss = model.forward(x_t, targets=y_t)
        bs, sl = x_batch.shape
        total_loss += loss.item() * bs * sl
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

    eos = 0  # index 0 is "\x00" (null char) — our padding/EOS token
    inp = np.array([tokens[-model.block_size:]], dtype=np.int64)
    out = model.generate(
        inp,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_k=40,
        top_p=0.95,
        repetition_penalty=1.1,
        eos_token=eos,
    )
    out_ids = out.data.flatten().tolist()
    if eos in out_ids:
        out_ids = out_ids[:out_ids.index(eos)]

    return "".join(itos.get(t, "") for t in out_ids[len(tokens):])


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
        from domains.training.pair_extractor import extract_pairs_from_corpus
        pairs = extract_pairs_from_corpus(limit=config.max_pairs)
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
