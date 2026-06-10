"""
Knowledge Weight Integrator - bakes knowledge facts into model weights via LoRA.

Trains a lightweight LoRA adapter on knowledge facts so the model
internalizes knowledge without prompt injection. The adapter is small
(MBs) and auto-loaded during inference.
"""

import json
import time
import logging
import threading
from pathlib import Path
from typing import Optional, List, Dict, Any

import torch
from peft import LoraConfig, get_peft_model, PeftModel, TaskType

logger = logging.getLogger(__name__)

_ADAPTER_DIR = Path("data/knowledge_adapter")
_ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
_ADAPTER_PATH = _ADAPTER_DIR / "knowledge_lora"
_MANIFEST_PATH = _ADAPTER_DIR / "manifest.json"

_lock = threading.Lock()
_trained_fact_count: int = 0
_last_trained: Optional[float] = None


def _format_facts_as_text(facts: List[Dict[str, Any]], max_items: int = 500) -> List[str]:
    """Format knowledge facts as plain text lines for LM training."""
    texts = []
    for f in facts:
        content = (f.get("content") or "").strip()
        topic = (f.get("topic") or "general").strip()
        if len(content) < 5:
            continue
        texts.append(f"<|knowledge|> [{topic}] {content}")
        if len(texts) >= max_items:
            break
    return texts


def _compute_training_loss(model, tokenizer, texts: List[str], device: str) -> float:
    """Run a single forward pass and return perplexity on knowledge texts."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for text in texts:
            encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
            input_ids = encoded["input_ids"].to(device)
            labels = input_ids.clone()
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            total_loss += loss.item() * input_ids.shape[1]
            total_tokens += input_ids.shape[1]
    return total_loss / max(total_tokens, 1)


def train_knowledge_adapter(
    knowledge_facts: List[Dict[str, Any]],
    model: Optional[torch.nn.Module] = None,
    tokenizer=None,
    device: str = "cpu",
    num_epochs: int = 3,
    learning_rate: float = 5e-4,
    lora_rank: int = 8,
    max_facts: int = 500,
    save_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Train a LoRA adapter on knowledge facts.

    Args:
        knowledge_facts: List of fact dicts with content/topic/source/importance
        model: PyTorch model (if None, tries to load from provider)
        tokenizer: matching tokenizer
        device: training device
        num_epochs: training epochs
        learning_rate: learning rate for LoRA params
        lora_rank: LoRA rank
        max_facts: max facts to use
        save_path: override save path

    Returns:
        dict with status, fact_count, loss, elapsed
    """
    global _trained_fact_count, _last_trained

    texts = _format_facts_as_text(knowledge_facts, max_facts=max_facts)
    if not texts:
        return {"status": "no_facts", "fact_count": 0}

    if model is None:
        try:
            from domains.models.provider import get_provider_manager
            pm = get_provider_manager()
            provider = pm.get_text_provider()
            model = provider.get_model()
            tokenizer = provider.get_tokenizer()
        except Exception as e:
            return {"status": f"model_unavailable: {e}", "fact_count": len(texts)}

    if tokenizer is None:
        return {"status": "tokenizer_missing", "fact_count": len(texts)}

    model.to(device)
    model.train()

    # Configure LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_rank,
        lora_alpha=lora_rank * 2,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
    )

    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()

    optimizer = torch.optim.AdamW(peft_model.parameters(), lr=learning_rate)
    start = time.time()

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for text in texts:
            encoded = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=256,
                padding=True,
            )
            input_ids = encoded["input_ids"].to(device)
            labels = input_ids.clone()
            optimizer.zero_grad()
            outputs = peft_model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(peft_model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
        logger.info(f"Knowledge adapter epoch {epoch + 1}/{num_epochs}: loss={epoch_loss / max(len(texts), 1):.4f}")

    # Save adapter
    final_path = Path(save_path or str(_ADAPTER_PATH))
    peft_model.save_pretrained(str(final_path))
    tokenizer.save_pretrained(str(final_path))

    # Compute post-training loss
    post_loss = _compute_training_loss(peft_model, tokenizer, texts, device)

    # Manifest
    manifest = {
        "trained_at": time.time(),
        "fact_count": len(texts),
        "total_facts_available": len(knowledge_facts),
        "epochs": num_epochs,
        "learning_rate": learning_rate,
        "lora_rank": lora_rank,
        "post_training_loss": round(post_loss, 4),
        "device": device,
    }
    _MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))

    elapsed = time.time() - start
    with _lock:
        _trained_fact_count = len(texts)
        _last_trained = time.time()

    logger.info(f"Knowledge adapter trained in {elapsed:.1f}s on {len(texts)} facts")

    peft_model.to("cpu")
    model.to(device)

    return {
        "status": "trained",
        "fact_count": len(texts),
        "post_training_loss": round(post_loss, 4),
        "elapsed": round(elapsed, 1),
    }


def load_knowledge_adapter(
    model: torch.nn.Module,
    adapter_path: Optional[str] = None,
    device: str = "cpu",
    merge: bool = True,
) -> torch.nn.Module:
    """
    Load a trained knowledge adapter and optionally merge into model.

    Args:
        model: base PyTorch model
        adapter_path: path to saved adapter (default: data/knowledge_adapter/knowledge_lora)
        device: target device
        merge: if True, merge adapter into model weights (no inference overhead)

    Returns:
        model with adapter applied (PeftModel or regular nn.Module if merged)
    """
    path = Path(adapter_path or str(_ADAPTER_PATH))
    if not (path / "adapter_config.json").exists():
        logger.info("No knowledge adapter found at %s", path)
        return model

    try:
        peft_model = PeftModel.from_pretrained(model, str(path))
        peft_model.to(device)
        if merge:
            merged = peft_model.merge_and_unload()
            logger.info("Knowledge adapter merged into model")
            return merged
        logger.info("Knowledge adapter loaded (unmerged)")
        return peft_model
    except Exception as e:
        logger.warning("Failed to load knowledge adapter: %s", e)
        return model


def get_adapter_status() -> Dict[str, Any]:
    """Return status of the trained knowledge adapter."""
    manifest = {}
    if _MANIFEST_PATH.exists():
        try:
            manifest = json.loads(_MANIFEST_PATH.read_text())
        except Exception:
            pass

    adapter_exists = (_ADAPTER_PATH / "adapter_config.json").exists()
    return {
        "adapter_exists": adapter_exists,
        "fact_count": manifest.get("fact_count", 0),
        "total_facts_available": manifest.get("total_facts_available", 0),
        "trained_at": manifest.get("trained_at"),
        "post_training_loss": manifest.get("post_training_loss"),
        "epochs": manifest.get("epochs"),
        "lora_rank": manifest.get("lora_rank", 8),
    }
