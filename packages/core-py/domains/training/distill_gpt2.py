"""
Distill GPT-2 teacher → smaller SloTransformer student.

Pure NumPy pipeline. Uses GPT-2's numpy forward
pass as the teacher and SloTransformer's autograd for the student.

Includes DistillEvaluator for post-training quality metrics:
- Perplexity (exp of avg CE loss)
- BLEU score (teacher vs student outputs)
- Sample generation quality comparison
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Any

import numpy as np

from domains.infrastructure.arch_config import ArchConfig, build_arch
from domains.infrastructure.numpy_forward import forward_fast, pre_extract_weights
from domains.training.slonet import (
    SloAdam, SloTransformer, export_to_sou, tensor,
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

    # Resume from checkpoint
    resume_checkpoint: Optional[str] = None  # path to .soul checkpoint to resume from
    resume_epoch: int = 0  # starting epoch (overridden by checkpoint metadata)
    resume_step: int = 0   # starting step (overridden by checkpoint metadata)


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
        """Get random batch of (x, y) pairs.

        Uses vectorized advanced indexing instead of Python-level loops.
        """
        indices = rng.integers(0, self.n_samples, size=batch_size)
        offsets = np.arange(self.block_size)
        ids = np.asarray(self.ids, dtype=np.int32)
        pos = indices[:, None] + offsets
        x = ids[pos]
        y = ids[pos + 1]
        return x.astype(np.int32), y.astype(np.int32)


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
        model = tok_data.get("model")
        vocab = model.get("vocab") if isinstance(model, dict) else None
        if not isinstance(vocab, dict):
            raise RuntimeError(
                f"tokenizer.json missing 'model.vocab' mapping in {tokenizer_path}"
            )
        itos = {i: s for s, i in vocab.items()}
        stoi = {s: i for s, i in vocab.items()}
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


def _compute_perplexity(loss: float) -> float:
    """Compute perplexity from cross-entropy loss: ppl = exp(loss)."""
    return float(np.exp(loss))


def _bleu_score(candidate: str, reference: str, max_n: int = 4) -> float:
    """Compute BLEU score between candidate and reference strings.

    Returns score as percentage (0-100).
    """
    cand_tokens = candidate.strip().split()
    ref_tokens = reference.strip().split()

    if not cand_tokens or not ref_tokens:
        return 0.0

    scores = []
    for n in range(1, min(max_n + 1, len(cand_tokens) + 1, len(ref_tokens) + 1)):
        cand_ngrams = {}
        for i in range(len(cand_tokens) - n + 1):
            ng = tuple(cand_tokens[i:i + n])
            cand_ngrams[ng] = cand_ngrams.get(ng, 0) + 1

        ref_ngrams = {}
        for i in range(len(ref_tokens) - n + 1):
            ng = tuple(ref_tokens[i:i + n])
            ref_ngrams[ng] = ref_ngrams.get(ng, 0) + 1

        matches = sum(min(cand_ngrams.get(ng, 0), ref_ngrams.get(ng, 0))
                      for ng in cand_ngrams)
        total = sum(cand_ngrams.values())
        precision = matches / total if total > 0 else 0
        if precision > 0:
            scores.append(precision)

    if not scores:
        return 0.0

    # Brevity penalty
    bp = min(1.0, np.exp(1 - len(ref_tokens) / max(len(cand_tokens), 1)))

    # Geometric mean of precisions
    geo_mean = np.exp(np.mean([np.log(s) for s in scores]))

    return bp * geo_mean * 100  # as percentage


@dataclass
class DistillEvalResult:
    """Evaluation results after distillation."""
    perplexity: float
    bleu_vs_teacher: float
    avg_response_len: float
    teacher_samples: List[str]
    student_samples: List[str]
    eval_prompts: List[str]
    inference_time_sec: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "perplexity": round(self.perplexity, 4),
            "bleu_vs_teacher": round(self.bleu_vs_teacher, 2),
            "avg_response_len": round(self.avg_response_len, 1),
            "inference_time_sec": round(self.inference_time_sec, 3),
            "num_samples": len(self.teacher_samples),
            "samples": [
                {
                    "prompt": p,
                    "teacher": t,
                    "student": s,
                }
                for p, t, s in zip(self.eval_prompts, self.teacher_samples, self.student_samples)
            ],
        }


class DistillEvaluator:
    """Evaluates distillation quality by comparing teacher vs student outputs.

    Metrics:
    - Perplexity: exp(avg CE loss) on eval prompts
    - BLEU: n-gram overlap between teacher and student generated text
    - Response quality: length, coherence comparison
    """

    EVAL_PROMPTS = [
        "The quick brown fox",
        "Hello, how are",
        "Once upon a time",
        "The meaning of life is",
        "def fibonacci(n):",
    ]

    def __init__(
        self,
        teacher_rw: dict,
        teacher_arch: ArchConfig,
        itos: Dict[int, str],
        stoi: Dict[str, int],
        eval_prompts: Optional[List[str]] = None,
        max_tokens: int = 50,
    ):
        """Initialize evaluator with teacher weights.

        Args:
            teacher_rw: Pre-extracted teacher weights.
            teacher_arch: Teacher architecture config.
            itos: Index-to-string tokenizer mapping.
            stoi: String-to-index tokenizer mapping.
            eval_prompts: Prompts to evaluate on (uses defaults if None).
            max_tokens: Max tokens to generate per prompt.
        """
        self.teacher_rw = teacher_rw
        self.teacher_arch = teacher_arch
        self.itos = itos
        self.stoi = stoi
        self.eval_prompts = eval_prompts or self.EVAL_PROMPTS
        self.max_tokens = max_tokens
        self._vocab_size = len(itos)

    def _generate_greedy(
        self,
        prompt: str,
        get_weight_fn,
        arch: ArchConfig,
        max_tokens: int,
    ) -> str:
        """Generate text greedily using a forward function.

        Args:
            prompt: Input prompt string.
            get_weight_fn: Function returning weight array by name.
            arch: Architecture config.
            max_tokens: Max tokens to generate.

        Returns:
            Generated text string.
        """
        tokens = [self.stoi.get(c, 0) for c in prompt]
        if not tokens:
            tokens = [0]

        for _ in range(max_tokens):
            logits = forward_fast(get_weight_fn, arch, tokens)
            next_token = int(np.argmax(logits[-1]))
            if next_token == 0:  # EOS
                break
            tokens.append(next_token)

        return "".join(self.itos.get(t, "") for t in tokens[len(prompt):])

    def _generate_student(
        self,
        prompt: str,
        student: SloTransformer,
        max_tokens: int,
    ) -> str:
        """Generate text from student model.

        Args:
            prompt: Input prompt string.
            student: Trained SloTransformer model.
            max_tokens: Max tokens to generate.

        Returns:
            Generated text string.
        """
        tokens = [self.stoi.get(c, 0) for c in prompt]
        if not tokens:
            tokens = [0]

        for _ in range(max_tokens):
            x = tensor([tokens], requires_grad=False)
            logits, _ = student.forward(x)
            if hasattr(logits, 'data'):
                logits_np = logits.data
            else:
                logits_np = np.array(logits)
            next_token = int(np.argmax(logits_np[0, -1]))
            if next_token == 0:
                break
            tokens.append(next_token)

        return "".join(self.itos.get(t, "") for t in tokens[len(prompt):])

    def _compute_perplexity_from_model(
        self,
        student: SloTransformer,
        text: str,
    ) -> float:
        """Compute perplexity on text using student model.

        Args:
            student: SloTransformer model.
            text: Text to evaluate.

        Returns:
            Perplexity value (exp of avg CE loss).
        """
        tokens = [self.stoi.get(c, 0) for c in text]
        if len(tokens) < 2:
            return 1.0

        # Process in chunks to avoid memory issues
        chunk_size = 128
        total_loss = 0.0
        total_tokens = 0

        for start in range(0, len(tokens) - 1, chunk_size):
            end = min(start + chunk_size, len(tokens) - 1)
            chunk = tokens[start:end]

            x = tensor([chunk], requires_grad=False)
            y = tensor([chunk[1:]], requires_grad=False)

            logits, _ = student.forward(x, y)
            if hasattr(logits, 'data'):
                logits_np = logits.data
            else:
                logits_np = np.array(logits)

            # Compute CE loss for this chunk
            if logits_np.ndim == 3:
                logits_np = logits_np.reshape(-1, logits_np.shape[-1])
            chunk_targets = chunk[1:]
            min_len = min(logits_np.shape[0], len(chunk_targets))
            if min_len > 0:
                loss = _cross_entropy_loss(logits_np[:min_len], np.array(chunk_targets[:min_len]))
                total_loss += loss * min_len
                total_tokens += min_len

        if total_tokens == 0:
            return 1.0

        avg_loss = total_loss / total_tokens
        return _compute_perplexity(avg_loss)

    def run(self, student: SloTransformer) -> DistillEvalResult:
        """Run full evaluation of student vs teacher.

        Args:
            student: Trained SloTransformer model.

        Returns:
            DistillEvalResult with all metrics.
        """
        import time as _time

        start_time = _time.time()
        teacher_samples = []
        student_samples = []

        # Generate from both models
        teacher_get_weight = lambda name: self.teacher_rw[name]

        for prompt in self.eval_prompts:
            t_text = self._generate_greedy(
                prompt, teacher_get_weight, self.teacher_arch, self.max_tokens
            )
            s_text = self._generate_student(prompt, student, self.max_tokens)
            teacher_samples.append(t_text)
            student_samples.append(s_text)

        # Compute BLEU (teacher vs student)
        bleu_scores = []
        for t_out, s_out in zip(teacher_samples, student_samples):
            if t_out and s_out:
                bleu_scores.append(_bleu_score(s_out, t_out))
        avg_bleu = float(np.mean(bleu_scores)) if bleu_scores else 0.0

        # Compute perplexity on combined text
        eval_text = " ".join(self.eval_prompts)
        perplexity = self._compute_perplexity_from_model(student, eval_text)

        # Average response length
        avg_len = float(np.mean([len(s.split()) for s in student_samples])) if student_samples else 0.0

        elapsed = _time.time() - start_time

        return DistillEvalResult(
            perplexity=perplexity,
            bleu_vs_teacher=avg_bleu,
            avg_response_len=avg_len,
            teacher_samples=teacher_samples,
            student_samples=student_samples,
            eval_prompts=self.eval_prompts,
            inference_time_sec=elapsed,
        )


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
    logger.info("Loading GPT-2 teacher...",
        extra={"tag": "TRAIN"},)
    rw, teacher_arch, tok_vocab = _load_gpt2_numpy()
    stoi = tok_vocab["stoi"]
    itos = tok_vocab["itos"]
    vocab_size = tok_vocab["vocab_size"]
    logger.info("Teacher loaded: %d vocab, %d layers", vocab_size, teacher_arch.n_layers,
        extra={"tag": "TRAIN"},)

    # Create or resume student
    start_epoch = 0
    start_step = 0
    best_loss = float("inf")

    if config.resume_checkpoint and Path(config.resume_checkpoint).exists():
        logger.info("Resuming from checkpoint: %s", config.resume_checkpoint,
            extra={"tag": "TRAIN"},)
        from domains.training.slonet import import_from_sou
        student = import_from_sou(config.resume_checkpoint)

        # Extract training state from checkpoint metadata
        if hasattr(student, 'metadata') and student.metadata:
            meta = student.metadata
            raw_epoch = meta.get("epoch", 0)
            raw_step = meta.get("step", 0)
            raw_best = meta.get("best_loss", float("inf"))
            # Validate epoch: must be int in [0, epochs)
            if isinstance(raw_epoch, (int, float)) and 0 <= raw_epoch < config.epochs:
                start_epoch = int(raw_epoch)
            else:
                logger.warning("Checkpoint epoch=%s invalid for %d epochs, starting from 0",
                    raw_epoch, config.epochs, extra={"tag": "TRAIN"})
            # Validate step: must be non-negative int
            if isinstance(raw_step, (int, float)) and raw_step >= 0:
                start_step = int(raw_step)
            else:
                logger.warning("Checkpoint step=%s invalid, starting from 0", raw_step,
                    extra={"tag": "TRAIN"})
            # Validate best_loss: must be finite positive
            if isinstance(raw_best, (int, float)) and np.isfinite(raw_best) and raw_best > 0:
                best_loss = float(raw_best)
            logger.info("Resumed at epoch %d, step %d, best_loss=%.4f",
                        start_epoch, start_step, best_loss,
                        extra={"tag": "TRAIN"})
    else:
        logger.info("Creating student: n_embed=%d, n_layer=%d, n_head=%d",
                    config.n_embed, config.n_layer, config.n_head,
                    extra={"tag": "TRAIN"})
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
    logger.info("Dataset: %d chars, %d samples", len(text), len(dataset),
        extra={"tag": "TRAIN"},)
    optimizer = SloAdam(lr=config.lr)
    rng = np.random.default_rng(42)

    # Training loop
    total_steps = config.epochs * (len(dataset) // config.batch_size)
    step = start_step
    epoch_loss = 0.0
    epoch_steps = 0
    epoch = start_epoch

    # Build teacher token map for fast lookup
    # Teacher: forward_fast needs token_ids as a list
    logger.info("Starting distillation: %d epochs, %d total steps (starting at epoch %d, step %d)",
                config.epochs, total_steps, start_epoch, start_step,
                extra={"tag": "TRAIN"})

    for epoch in range(start_epoch, config.epochs):
        if cancel_event and cancel_event.is_set():
            logger.info("Training cancelled",
                extra={"tag": "TRAIN"},)
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
                            step, total_steps, total_loss, hard_loss, soft_loss,
                            extra={"tag": "TRAIN"})
                if on_step:
                    on_step(step, total_loss, epoch)

            # Eval
            if step % config.eval_interval == 0 and epoch_steps > 0:
                avg_loss = epoch_loss / epoch_steps
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    logger.info("New best loss: %.4f", best_loss,
                        extra={"tag": "TRAIN"},)

        if cancel_event and cancel_event.is_set():
            break

        avg_epoch = epoch_loss / max(epoch_steps, 1)
        logger.info("Epoch %d/%d avg_loss=%.4f", epoch + 1, config.epochs, avg_epoch,
            extra={"tag": "TRAIN"},)

    # Save checkpoint
    ckpt_dir = Path(config.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_name = f"distilled_gpt2_{int(time.time())}.soul"
    ckpt_path = ckpt_dir / ckpt_name

    # Export with training state in metadata for resuming
    export_to_sou(
        student,
        str(ckpt_path),
        soul_profile=None,
        metadata={
            "epoch": epoch + 1,
            "step": step,
            "best_loss": best_loss,
            "final_loss": epoch_loss / max(epoch_steps, 1),
            "vocab_size": vocab_size,
        },
    )

    # Run evaluation
    logger.info("Running evaluation...",
        extra={"tag": "TRAIN"},)
    evaluator = DistillEvaluator(
        teacher_rw=rw,
        teacher_arch=teacher_arch,
        itos=itos,
        stoi=stoi,
    )
    eval_result = evaluator.run(student)

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
        "eval": json.dumps(eval_result.to_dict()),
        "perplexity": str(eval_result.perplexity),
        "bleu_vs_teacher": str(eval_result.bleu_vs_teacher),
    }

    logger.info("Distillation complete. Checkpoint: %s", ckpt_path,
        extra={"tag": "TRAIN"},)
    logger.info("Eval: perplexity=%.2f, bleu=%.1f%%", eval_result.perplexity, eval_result.bleu_vs_teacher,
        extra={"tag": "TRAIN"},)
    return student, metadata


if __name__ == "__main__":  # pragma: no cover (requires GPT-2 download)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    # Quick test with small text
    text = "The quick brown fox jumps over the lazy dog. " * 100
    config = DistillConfig(
        n_embed=64, n_layer=2, n_head=4,
        epochs=3, batch_size=4, block_size=64,
        log_interval=5, eval_interval=25,
    )
    model, meta = distill_gpt2_to_slo(text, config)
    logger.info("%s", json.dumps(meta, indent=2))
