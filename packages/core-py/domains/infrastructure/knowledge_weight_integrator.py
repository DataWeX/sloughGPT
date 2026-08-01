"""
Knowledge Weight Integrator - bakes knowledge facts into model weights.

Trains the native SloNet model on knowledge facts (character-level) and
stores the weight delta as a compact adapter. The adapter is small (MBs)
and can be re-applied to the base model during inference.

Runtime: pure NumPy / SloNet. No PyTorch, peft, or HuggingFace required.
"""

import json
import time
import logging
import threading
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

import numpy as np

logger = logging.getLogger("slo.infrastructure.knowledge_weight_integrator")

_ADAPTER_DIR = Path("data/knowledge_adapter")
_ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
_ADAPTER_PATH = _ADAPTER_DIR / "knowledge_lora"
_DELTA_PATH = _ADAPTER_DIR / "knowledge_delta.npz"
_MANIFEST_PATH = _ADAPTER_DIR / "manifest.json"

_lock = threading.Lock()
_trained_fact_count: int = 0
_last_trained: Optional[float] = None

_BLOCK_SIZE = 128


def _format_facts_as_text(facts: List[Dict[str, Any]], max_facts: int = 500) -> List[str]:
    """Format knowledge facts as plain text lines for LM training."""
    texts = []
    for f in facts:
        content = (f.get("content") or "").strip()
        topic = (f.get("topic") or "general").strip()
        if len(content) < 5:
            continue
        texts.append(f"<|knowledge|> [{topic}] {content}")
        if len(texts) >= max_facts:
            break
    return texts


def _build_char_vocab(texts: List[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Build a character-level vocab from the training texts."""
    chars = sorted(set("".join(texts)))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for i, c in enumerate(chars)}
    return stoi, itos


def _encode(texts: List[str], stoi: Dict[str, int], block_size: int) -> List[np.ndarray]:
    """Tokenize texts into 2D (1, seq) int arrays clipped to block size."""
    encoded = []
    for t in texts:
        arr = np.array([stoi.get(c, 0) for c in t][:block_size], dtype=np.int64)
        if arr.size == 0:
            continue
        encoded.append(arr.reshape(1, -1))
    return encoded


def _resolve_model(model, vocab_size: int):
    """Return a native SloNet model to train, or None if unavailable."""
    if model is not None:
        return model
    try:
        from domains.models import SloughGPTModel
        return SloughGPTModel(
            vocab_size=vocab_size, n_embed=128, n_layer=4, n_head=4,
            block_size=_BLOCK_SIZE, max_seq_len=_BLOCK_SIZE,
        )
    except Exception as e:
        logger.warning("Could not construct a native model for training: %s", e,
            extra={"tag": "INFRA"})
        return None


def _train_native(
    model,
    batches: List[np.ndarray],
    num_epochs: int,
    learning_rate: float,
) -> float:
    """Run gradient-based character LM training on the native model.

    Returns the mean loss over the last epoch.
    """
    from domains.training.slonet import SloAdam

    optimizer = SloAdam(
        lr=learning_rate, b1=0.9, b2=0.999, eps=1e-8,
        weight_decay=0.0, max_grad_norm=1.0,
    )
    params = model.parameters()
    model.train()

    epoch_loss = 0.0
    for _ in range(num_epochs):
        epoch_loss = 0.0
        for arr in batches:
            logits, loss = model.forward(arr, arr)
            loss.backward()
            optimizer.step(params)
            for p in params:
                p.grad = None
            epoch_loss += loss.item()
    return epoch_loss / max(len(batches), 1)


def _save_delta(model, stoi: Dict[str, int], itos: Dict[int, str], rank: int) -> None:
    """Snapshot the model weights as a delta adapter file."""
    arrays = {}
    for name, param in model.named_parameters():
        arrays[f"{name}.weight"] = np.array(param.data, dtype=np.float32, copy=True)
    arrays["_stoi"] = np.frombuffer(
        json.dumps(stoi).encode("utf-8"), dtype=np.uint8
    )
    arrays["_itos"] = np.frombuffer(
        json.dumps({int(k): v for k, v in itos.items()}).encode("utf-8"), dtype=np.uint8
    )
    arrays["_rank"] = np.array([rank], dtype=np.int32)
    np.savez_compressed(str(_DELTA_PATH), **arrays)


def train_knowledge_adapter(
    knowledge_facts: List[Dict[str, Any]],
    model=None,
    tokenizer=None,
    device: str = "cpu",
    num_epochs: int = 3,
    learning_rate: float = 5e-4,
    lora_rank: int = 8,
    max_facts: int = 500,
    save_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Train the native model on knowledge facts and save a weight-delta adapter.

    Args:
        knowledge_facts: List of fact dicts with content/topic/source/importance
        model: native SloNet model (if None, a default SloughGPTModel is built)
        tokenizer: kept for API compatibility; a char vocab is derived from facts
        device: kept for API compatibility; training always runs on CPU
        num_epochs: training epochs
        learning_rate: learning rate
        lora_rank: adapter rank (recorded in manifest)
        max_facts: max facts to use
        save_path: override save path (manifest is still written to the default dir)

    Returns:
        dict with status, fact_count, loss, elapsed
    """
    global _trained_fact_count, _last_trained

    texts = _format_facts_as_text(knowledge_facts, max_facts=max_facts)
    if not texts:
        return {"status": "no_facts", "fact_count": 0}

    stoi, itos = _build_char_vocab(texts)
    batches = _encode(texts, stoi, _BLOCK_SIZE)
    if not batches:
        return {"status": "no_facts", "fact_count": 0}

    native_model = _resolve_model(model, vocab_size=len(stoi))
    if native_model is None:
        return {"status": "model_unavailable", "fact_count": len(texts)}

    start = time.time()
    try:
        final_loss = _train_native(native_model, batches, num_epochs, learning_rate)
    except Exception as e:
        logger.error("Knowledge adapter training failed: %s", e, extra={"tag": "INFRA"})
        return {"status": f"training_failed: {e}", "fact_count": len(texts)}

    _save_delta(native_model, stoi, itos, rank=lora_rank)

    manifest = {
        "trained_at": time.time(),
        "fact_count": len(texts),
        "total_facts_available": len(knowledge_facts),
        "epochs": num_epochs,
        "learning_rate": learning_rate,
        "lora_rank": lora_rank,
        "post_training_loss": round(final_loss, 4),
        "device": "cpu",
        "vocab_size": len(stoi),
    }
    _MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))

    elapsed = time.time() - start
    with _lock:
        _trained_fact_count = len(texts)
        _last_trained = time.time()

    logger.info("Knowledge adapter trained in %.1fs on %d facts (loss %.4f)",
        elapsed, len(texts), final_loss, extra={"tag": "INFRA"})

    return {
        "status": "trained",
        "fact_count": len(texts),
        "post_training_loss": round(final_loss, 4),
        "elapsed": round(elapsed, 1),
    }


def load_knowledge_adapter(
    model,
    adapter_path: Optional[str] = None,
    device: str = "cpu",
    merge: bool = True,
):
    """
    Load a trained knowledge adapter and apply the weight delta to the model.

    Args:
        model: base native SloNet model
        adapter_path: path to saved adapter (default: data/knowledge_adapter/knowledge_delta.npz)
        device: kept for API compatibility
        merge: if True, apply delta into model weights in place

    Returns:
        model with adapter applied (or the unmodified model if none found)
    """
    path = Path(adapter_path or str(_DELTA_PATH))
    if not path.exists():
        logger.info("No knowledge adapter found at %s", path, extra={"tag": "INFRA"})
        return model

    try:
        data = np.load(str(path), allow_pickle=False)
    except Exception as e:
        logger.warning("Failed to read knowledge adapter: %s", e, extra={"tag": "INFRA"})
        return model

    if not merge:
        logger.info("Knowledge adapter loaded (unmerged; use merge=True to apply)",
            extra={"tag": "INFRA"})
        return model

    applied = 0
    for name, param in model.named_parameters():
        key = f"{name}.weight"
        if key not in data:
            continue
        current = np.asarray(param.data, dtype=np.float32)
        delta = np.asarray(data[key], dtype=np.float32)
        if current.shape == delta.shape:
            param.data[:] = current + delta
            applied += 1
    data.close()
    logger.info("Knowledge adapter merged into model (%d tensors)", applied,
        extra={"tag": "INFRA"})
    return model


def get_adapter_status() -> Dict[str, Any]:
    """Return status of the trained knowledge adapter."""
    manifest = {}
    if _MANIFEST_PATH.exists():
        try:
            manifest = json.loads(_MANIFEST_PATH.read_text())
        except Exception:
            pass

    adapter_exists = _DELTA_PATH.exists() or (
        _ADAPTER_PATH / "adapter_config.json"
    ).exists()
    return {
        "adapter_exists": adapter_exists,
        "fact_count": manifest.get("fact_count", 0),
        "total_facts_available": manifest.get("total_facts_available", 0),
        "trained_at": manifest.get("trained_at"),
        "post_training_loss": manifest.get("post_training_loss"),
        "epochs": manifest.get("epochs"),
        "lora_rank": manifest.get("lora_rank", 8),
    }
