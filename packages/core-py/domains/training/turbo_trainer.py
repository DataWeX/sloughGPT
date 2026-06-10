"""
TurboTrainer — Unified high-performance training pipeline.

Default architecture is NanoGPT (decoder-only transformer).
Also supports HF models and legacy SloNet LSTM.

Auto-detects MPS/GPU, uses mixed precision, gradient accumulation.
"""

from __future__ import annotations

import json
import logging
import math
import os
import sys
import time
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

logger = logging.getLogger("man.turbo_trainer")

# Ensure the torch shim is importable
_TRAINING_DIR = str(Path(__file__).parent)
if _TRAINING_DIR not in sys.path:
    sys.path.insert(0, _TRAINING_DIR)

TORCH_AVAILABLE = False
torch = None
nn = None
F = None
Dataset = object
DataLoader = None
try:
    import domains.training.torch as _torch_shim
    import domains.training.torch.nn as _nn_shim
    import domains.training.torch.nn.functional as _F_shim
    # Register as "torch" so internal sys.modules lookups work
    sys.modules["torch"] = _torch_shim
    sys.modules["torch.nn"] = _nn_shim
    sys.modules["torch.nn.functional"] = _F_shim
    torch = _torch_shim
    nn = _nn_shim
    F = _F_shim
    TORCH_AVAILABLE = True
except ImportError:
    pass

# Try optional torch.utils.data
try:
    from domains.training.torch.utils.data import Dataset, DataLoader
except ImportError:
    pass


@dataclass
class TurboConfig:
    model_spec: str = "transformer"
    data_path: str = ""
    output_dir: str = "models/turbo-trained"
    method: str = ""  # auto-detect: "transformer", "hf", "slonet"

    # Architecture (encoder-decoder Transformer defaults)
    vocab_size: int = 1000
    n_embed: int = 128
    n_head: int = 4
    n_encoder_layers: int = 3
    n_decoder_layers: int = 3
    dim_feedforward: int = 256
    dropout: float = 0.1
    max_src_len: int = 128
    max_tgt_len: int = 128

    # Training
    epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 0
    gradient_accumulation_steps: int = 1
    max_seq_length: int = 256
    save_steps: int = 500
    logging_steps: int = 10
    max_train_steps: Optional[int] = None

    # LoRA (HF only)
    use_lora: bool = False
    lora_rank: int = 8
    lora_alpha: int = 16

    # Device
    device: str = ""
    mixed_precision: str = ""  # "fp16", "bf16", or "" for auto

    # Early stopping
    early_stop_patience: int = 0
    early_stop_min_delta: float = 0.001


def _resolve_device() -> str:
    if torch is None:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _resolve_precision(device: str) -> str:
    if device == "cuda":
        cap = torch.cuda.get_device_capability() if torch.cuda.is_available() else (0, 0)
        return "bf16" if cap >= (8, 0) else "fp16"
    return "fp16" if device == "mps" else ""


class TextFileDataset(Dataset):
    def __init__(self, file_path: str, tokenizer, max_length: int, stride: Optional[int] = None):
        self.max_length = max_length
        self.stride = stride or max_length // 2
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        tokens = tokenizer.encode(text, add_special_tokens=False) if hasattr(tokenizer, 'encode') else list(text.encode("utf-8"))
        self.examples: List[Dict[str, torch.Tensor]] = []
        for i in range(0, len(tokens), self.stride):
            chunk = tokens[i: i + max_length]
            if len(chunk) < 64:
                continue
            t = torch.tensor(chunk, dtype=torch.long)
            self.examples.append({"input_ids": t, "labels": t.clone()})

    def __len__(self) -> int:
        return len(self.examples)
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.examples[idx]


class _ProgressCallback:
    def __init__(self, fn: Optional[Callable], total_steps: int):
        self.fn = fn
        self.total = total_steps
        self.last_log = 0.0
    def __call__(self, epoch: float, step: int, loss: Optional[float], phase: str = "train"):
        if self.fn is None:
            return
        now = time.time()
        if now - self.last_log < 0.3:
            return
        self.last_log = now
        self.fn({
            "epoch": epoch,
            "step": step,
            "loss": loss,
            "progress_pct": min(100, int(step / max(1, self.total) * 100)),
            "total_steps": self.total,
            "phase": phase,
        })


class TurboTrainer:
    def __init__(self, config: Optional[Union[TurboConfig, Dict[str, Any]]] = None):
        if config is None:
            self.config = TurboConfig()
        elif isinstance(config, dict):
            self.config = TurboConfig(**config)
        else:
            self.config = config

        if not self.config.device:
            self.config.device = _resolve_device()
        if not self.config.mixed_precision:
            self.config.mixed_precision = _resolve_precision(self.config.device)
        if not self.config.method:
            self.config.method = self._detect_method()

        logger.info(
            "TurboTrainer initialized: model=%s method=%s device=%s precision=%s",
            self.config.model_spec, self.config.method, self.config.device, self.config.mixed_precision,
        )

    def _detect_method(self) -> str:
        spec = self.config.model_spec
        if spec.startswith("hf:") or spec.startswith("huggingface:"):
            return "hf"
        if spec == "slonet" or spec.startswith("slonet:"):
            return "slonet"
        if spec in ("nanogpt", "gpt2"):
            return "nanogpt"
        if spec == "transformer" or spec == "encdec":
            return "transformer"
        if "/" in spec:
            return "hf"
        return "transformer"

    def train(self, on_progress: Optional[Callable[[Dict[str, Any]], None]] = None, **overrides) -> Dict[str, Any]:
        for k, v in overrides.items():
            if hasattr(self.config, k):
                setattr(self.config, k, v)

        method = self.config.method
        if method == "transformer":
            return self._train_transformer(on_progress)
        elif method == "nanogpt":
            return self._train_nanogpt(on_progress)
        elif method == "hf":
            return self._train_hf(on_progress)
        elif method == "slonet":
            return self._train_slonet(on_progress)
        else:
            raise ValueError(f"Unknown training method: {method}")

    def _build_transformer(self) -> Any:
        """Build encoder-decoder Transformer from torch shim."""

        tok_vocab = self.config.vocab_size

        class EncDecModel(nn.Module):
            def __init__(self, cfg):
                super().__init__()
                self.embed = nn.Embedding(tok_vocab, cfg.n_embed)
                self.pos_enc = nn.Embedding(cfg.max_src_len + cfg.max_tgt_len, cfg.n_embed)
                self.transformer = nn.Transformer(
                    d_model=cfg.n_embed,
                    nhead=cfg.n_head,
                    num_encoder_layers=cfg.n_encoder_layers,
                    num_decoder_layers=cfg.n_decoder_layers,
                    dim_feedforward=cfg.dim_feedforward,
                    dropout=cfg.dropout,
                    batch_first=False,
                )
                self.out_proj = nn.Linear(cfg.n_embed, tok_vocab)
                self.dropout = nn.Dropout(cfg.dropout)
                self.cfg = cfg

            def forward(self, src_ids, tgt_ids):
                N, S, Td = src_ids.shape[0], src_ids.shape[1], tgt_ids.shape[1]
                # src_ids, tgt_ids: (N, S), (N, Td)
                T_src = torch.Tensor(np.arange(S).reshape(1, S).repeat(N, axis=0).astype(np.int64), dtype=np.int64)
                T_tgt = torch.Tensor(np.arange(Td).reshape(1, Td).repeat(N, axis=0).astype(np.int64), dtype=np.int64)
                src_emb = self.dropout(self.embed(src_ids) + self.pos_enc(T_src))  # (N, S, E)
                tgt_emb = self.dropout(self.embed(tgt_ids) + self.pos_enc(T_tgt))  # (N, Td, E)
                # Transpose to (T, N, E) for transformer
                src_emb = src_emb.transpose(0, 1)  # (S, N, E)
                tgt_emb = tgt_emb.transpose(0, 1)  # (Td, N, E)
                memory, out = self.transformer(src_emb, tgt_emb)  # (Td, N, E)
                logits = self.out_proj(out)  # (Td, N, vocab)
                logits = logits.transpose(0, 1)  # (N, Td, vocab)
                return logits

            def generate(self, src_ids, max_len=64, bos_id=1, eos_id=2):
                N = src_ids.shape[0]
                tgt = torch.Tensor(np.full((N, 1), bos_id, dtype=np.int64), dtype=np.int64)
                for _ in range(max_len):
                    logits = self.forward(src_ids, tgt)
                    next_id = torch.Tensor(np.argmax(logits.data[:, -1:, :], axis=-1).astype(np.int64), dtype=np.int64)
                    tgt = torch.Tensor(np.concatenate([tgt.data, next_id.data.reshape(N, 1)], axis=1).astype(np.int64), dtype=np.int64)
                    if np.all(next_id.data.flatten() == eos_id):
                        break
                return tgt

        return EncDecModel(self.config)

    def _train_transformer(self, on_progress: Optional[Callable]) -> Dict[str, Any]:
        from domains.training.tokenizer import SloBPE

        tok = SloBPE()
        data_path = self.config.data_path
        if not data_path or not Path(data_path).exists():
            raise FileNotFoundError(f"Data not found: {data_path}")
        text = Path(data_path).read_text(encoding="utf-8", errors="replace")
        tok.train([text], vocab_size=self.config.vocab_size)
        ids = tok.encode(text)
        logger.info("Tokenized %d chars → %d ids (vocab=%d)", len(text), len(ids), tok.vocab_size)

        model = self._build_transformer()
        opt = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
        batch_size = self.config.batch_size
        enc_len = self.config.max_src_len
        dec_len = self.config.max_tgt_len
        stride = max(1, (enc_len + dec_len) // 2)
        epochs = self.config.epochs
        total_steps = max(1, (len(ids) // stride) // batch_size) * epochs
        cb = _ProgressCallback(on_progress, total_steps)
        step = 0
        best_loss = float("inf")
        patience = 0

        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_steps = 0
            i = 0
            while i < len(ids) - enc_len - dec_len - 1:
                batch_src, batch_tgt_in, batch_tgt_out = [], [], []
                for _ in range(batch_size):
                    if i >= len(ids) - enc_len - dec_len - 1:
                        break
                    src_chunk = ids[i:i + enc_len]
                    tgt_chunk = ids[i + enc_len:i + enc_len + dec_len + 1]
                    while len(src_chunk) < enc_len:
                        src_chunk.append(tok.pad_id)
                    while len(tgt_chunk) < dec_len + 1:
                        tgt_chunk.append(tok.pad_id)
                    tgt_in = tgt_chunk[:-1]
                    tgt_label = tgt_chunk[1:]
                    batch_src.append(src_chunk)
                    batch_tgt_in.append(tgt_in)
                    batch_tgt_out.append(tgt_label)
                    i += stride

                if not batch_src:
                    break

                opt.zero_grad()
                src_t = torch.Tensor(np.array(batch_src, dtype=np.int64), dtype=np.int64)
                tgt_in_t = torch.Tensor(np.array(batch_tgt_in, dtype=np.int64), dtype=np.int64)
                tgt_label = np.array(batch_tgt_out, dtype=np.int64)

                logits = model(src_t, tgt_in_t)  # (N, Td, vocab)
                V = logits.shape[-1]
                loss = F.cross_entropy(logits.reshape(-1, V), torch.Tensor(tgt_label.reshape(-1), dtype=np.int64),
                                       ignore_index=tok.pad_id)
                loss.backward()
                opt.step()

                epoch_loss += loss.data
                epoch_steps += 1
                step += 1
                cb(epoch + step / max(1, total_steps // epochs), step, float(loss.data))

                if self.config.max_train_steps and step >= self.config.max_train_steps:
                    break

            avg_loss = epoch_loss / max(1, epoch_steps)
            logger.info("Epoch %d/%d — loss: %.4f", epoch + 1, epochs, avg_loss)

            if self.config.early_stop_patience > 0:
                if avg_loss < best_loss - self.config.early_stop_min_delta:
                    best_loss = avg_loss; patience = 0
                else:
                    patience += 1
                    if patience >= self.config.early_stop_patience:
                        break

        out_dir = Path(self.config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        # Save model checkpoint
        state = {"config": self.config, "model": "transformer"}
        for i, p in enumerate(model.parameters()):
            state[f"param_{i}"] = p.data
        np.savez(str(out_dir / "model.npz"), **state)
        tok.save(str(out_dir / "tokenizer.json"))

        logger.info("Training complete — saved to %s", out_dir)
        return {
            "status": "completed",
            "model_path": str(out_dir),
            "method": "transformer",
            "final_loss": float(avg_loss),
            "total_steps": step,
            "epochs": epoch + 1,
        }

    def _train_nanogpt(self, on_progress: Optional[Callable]) -> Dict[str, Any]:
        if not TORCH_AVAILABLE:
            raise ImportError("torch required for NanoGPT training")

        from .models.nanogpt import NanoGPT
        from transformers import AutoTokenizer

        device = torch.device(self.config.device)
        logger.info("Loading NanoGPT on %s", device)

        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = NanoGPT(
            vocab_size=self.config.vocab_size or tokenizer.vocab_size,
            n_embed=self.config.n_embed,
            n_layer=self.config.n_layer,
            n_head=self.config.n_head,
            block_size=self.config.block_size,
            dropout=self.config.dropout,
        ).to(device)

        dataset = TextFileDataset(
            self.config.data_path, tokenizer,
            max_length=self.config.max_seq_length,
        )
        loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True, drop_last=True)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=len(loader) * self.config.epochs,
        )

        scaler = torch.amp.GradScaler(device if self.config.mixed_precision else "cpu") if self.config.mixed_precision else None

        total_steps = len(loader) * self.config.epochs
        cb = _ProgressCallback(on_progress, total_steps)
        step = 0
        best_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.config.epochs):
            model.train()
            epoch_loss = 0.0
            epoch_steps = 0
            optimizer.zero_grad()

            for batch_idx, batch in enumerate(loader):
                x = batch["input_ids"].to(device)
                y = batch["labels"].to(device)

                with torch.amp.autocast(device_type=device.type, enabled=bool(scaler)):
                    _, loss = model(x, y)
                    loss = loss / self.config.gradient_accumulation_steps

                if scaler:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    if scaler:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    if scaler:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                epoch_loss += loss.item() * self.config.gradient_accumulation_steps
                epoch_steps += 1
                step += 1
                cb(epoch + batch_idx / len(loader), step, loss.item() * self.config.gradient_accumulation_steps)

                if self.config.max_train_steps and step >= self.config.max_train_steps:
                    break

            avg_loss = epoch_loss / max(1, epoch_steps)
            logger.info("Epoch %d/%d — loss: %.4f", epoch + 1, self.config.epochs, avg_loss)

            if self.config.early_stop_patience > 0:
                if avg_loss < best_loss - self.config.early_stop_min_delta:
                    best_loss = avg_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.config.early_stop_patience:
                        logger.info("Early stopping at epoch %d", epoch + 1)
                        break

        output_path = Path(self.config.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), str(output_path / "model.pt"))
        tokenizer.save_pretrained(str(output_path))

        config_path = output_path / "config.json"
        with open(config_path, "w") as f:
            json.dump({
                "model": "nanogpt",
                "config": {
                    "vocab_size": model.vocab_size,
                    "n_embed": self.config.n_embed,
                    "n_layer": self.config.n_layer,
                    "n_head": self.config.n_head,
                    "block_size": self.config.block_size,
                },
                "training": {
                    "epochs": epoch + 1,
                    "batch_size": self.config.batch_size,
                    "learning_rate": self.config.learning_rate,
                    "final_loss": avg_loss,
                    "steps": step,
                },
            }, f, indent=2)

        logger.info("NanoGPT training complete — saved to %s", output_path)
        return {
            "status": "completed",
            "model_path": str(output_path),
            "method": "nanogpt",
            "final_loss": avg_loss,
            "total_steps": step,
            "epochs": epoch + 1,
        }

    def _train_hf(self, on_progress: Optional[Callable]) -> Dict[str, Any]:
        from .hf_finetune import HFFineTuner

        model_name = self.config.model_spec
        if model_name.startswith("hf:"):
            model_name = model_name[3:]
        elif model_name.startswith("huggingface:"):
            model_name = model_name[12:]

        tuner = HFFineTuner(
            model_name=model_name or self.config.model_spec,
            data_path=self.config.data_path,
            output_dir=self.config.output_dir,
            use_lora=self.config.use_lora,
            lora_rank=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            max_seq_length=self.config.max_seq_length,
            warmup_steps=self.config.warmup_steps,
            weight_decay=self.config.weight_decay,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            save_steps=self.config.save_steps,
            logging_steps=self.config.logging_steps,
            device=self.config.device if self.config.device != "mps" else "cpu",
        )
        return tuner.train(on_progress=on_progress)

    def _train_slonet(self, on_progress: Optional[Callable]) -> Dict[str, Any]:
        from .slonet import SloAdam, cross_entropy, tensor, import_from_sou
        from .tokenizer import SloBPE

        data_path = Path(self.config.data_path)
        if not data_path.exists():
            raise FileNotFoundError(f"Data not found: {data_path}")

        text = data_path.read_text(encoding="utf-8", errors="replace")

        tokenizer = SloBPE(vocab_size=self.config.vocab_size or 2000)
        tokenizer.train(text)

        ids = tokenizer.encode(text)
        chunk_size = self.config.block_size or 128
        epochs = self.config.epochs
        lr = self.config.learning_rate

        from ..core.soul import SloEngine
        engine = SloEngine(device="cpu")
        engine._tokenizer = tokenizer
        engine._build_network(vocab_size=tokenizer.vocab_size)

        model = engine._model
        optimizer = SloAdam(lr=lr)

        total_steps = max(1, (len(ids) - 1 + chunk_size - 1) // chunk_size) * epochs
        cb = _ProgressCallback(on_progress, total_steps)
        step = 0

        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_steps = 0
            for i in range(0, len(ids) - 1, chunk_size):
                x_chunk = ids[i: i + chunk_size]
                y_chunk = ids[i + 1: i + chunk_size + 1]
                while len(x_chunk) < chunk_size:
                    x_chunk.append(tokenizer.pad_id)
                while len(y_chunk) < chunk_size:
                    y_chunk.append(tokenizer.pad_id)

                x = tensor([[x_chunk]], requires_grad=True)
                y = tensor([[y_chunk]])

                lstm_layer = model.layers[1]
                hidden = lstm_layer.init_hidden()
                logits, _ = lstm_layer.forward(x, hidden)
                loss = cross_entropy(logits, y.reshape(-1))
                loss.backward()
                optimizer.step(model.parameters())

                epoch_loss += loss.data
                epoch_steps += 1
                step += 1
                cb(epoch + i / max(1, len(ids)), step, float(loss.data))

            avg_loss = epoch_loss / max(1, epoch_steps)
            logger.info("SloNet Epoch %d/%d — loss: %.4f", epoch + 1, epochs, avg_loss)

        output_path = Path(self.config.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        from .slonet import export_to_sou
        metadata = {"final_loss": float(avg_loss), "steps": step, "epochs": epochs}
        export_to_sou(model, str(output_path / "model.soul"), metadata=metadata)

        return {
            "status": "completed",
            "model_path": str(output_path),
            "method": "slonet",
            "final_loss": float(avg_loss),
            "total_steps": step,
            "epochs": epochs,
        }
