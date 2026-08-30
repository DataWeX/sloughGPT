"""
Video Caption Trainer — trains a SloNet-based model on video → caption pairs.

Architecture:
  video frames → VisionEncoder (frame embeddings)
    → TemporalEncoder (video-level embedding)
    → SloTransformerDecoder (caption generation)

Training data format (JSONL):
  {"video_path": "/path/to/video.mp4", "caption": "A cat walks across the room"}
  {"video_path": "/path/to/clip.avi", "caption": "Cars driving on a highway"}
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from domains.shared import find_repo_root
from domains.training.slonet import (
    Tensor, SloEmbedding, SloLinear, SloLayerNorm, SloRMSNorm,
    SloTransformerBlock,
    SloAdam, cross_entropy as _cross_entropy, tensor as _tensor,
    zeros, ones,
)

logger = logging.getLogger("slo.video_trainer")

REPO_ROOT = find_repo_root(Path(__file__).resolve())
DEFAULT_OUTPUT_DIR = REPO_ROOT / "models" / "video-training"
CHECKPOINT_DIR = REPO_ROOT / "models" / "video-training" / "checkpoints"


def _causal_mask(seq_len: int) -> np.ndarray:
    """Causal attention mask (lower triangular)."""
    mask = np.tril(np.ones((seq_len, seq_len), dtype=np.float32))
    mask[mask == 0] = -1e9
    return mask


class VideoCaptionTrainer:
    """End-to-end video captioning model.

    Encodes video frames with VisionEncoder, aggregates with TemporalEncoder,
    and generates captions with SloTransformerDecoder.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        hidden_dim: int = 512,
        n_vision_layers: int = 3,
        n_temporal_layers: int = 2,
        n_decoder_layers: int = 3,
        n_heads: int = 4,
        vocab_size: int = 512,
        max_seq_len: int = 128,
        max_frames: int = 8,
        lr: float = 3e-4,
    ):
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.max_frames = max_frames
        self.max_seq_len = max_seq_len
        self.lr = lr

        from domains.multimodal.video import TemporalEncoder
        from domains.multimodal.engine import VisionEncoder, SloTransformerDecoder

        self.vision_encoder = VisionEncoder(embed_dim, n_heads, n_vision_layers)
        self.temporal_encoder = TemporalEncoder(embed_dim, n_heads, n_temporal_layers, max_frames)
        self.decoder = SloTransformerDecoder(
            vocab_size=max(1, vocab_size),
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            n_layers=n_decoder_layers,
            max_seq_len=max_seq_len,
        )

        self.vision_optimizer = SloAdam(lr=lr)
        self.temporal_optimizer = SloAdam(lr=lr)
        self.decoder_optimizer = SloAdam(lr=lr)

        self._trained = False
        self._vocab: Dict[str, int] = {}
        self._rev_vocab: Dict[int, str] = {}

    def build_vocab(self, captions: List[str]):
        """Build character-level vocabulary from training captions."""
        chars = set()
        for c in captions:
            chars.update(c)
        sorted_chars = sorted(chars)
        self._vocab = {ch: i + 2 for i, ch in enumerate(sorted_chars)}
        self._vocab["<PAD>"] = 0
        self._vocab["<BOS>"] = 1
        self._vocab["<EOS>"] = len(self._vocab)
        self._rev_vocab = {v: k for k, v in self._vocab.items()}
        self.decoder.vocab_size = max(1, len(self._vocab))
        logger.info("Built vocab: %d tokens", len(self._vocab),
            extra={"tag": "TRAIN"},)

    def encode_text(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        tokens = [self._vocab.get("<BOS>", 1)]
        for ch in text:
            tokens.append(self._vocab.get(ch, self._vocab.get("<PAD>", 0)))
        tokens.append(self._vocab.get("<EOS>", len(self._vocab) - 1))
        return tokens

    def decode_text(self, token_ids: List[int]) -> str:
        """Decode token IDs back to text."""
        chars = []
        for tid in token_ids:
            if tid == self._vocab.get("<EOS>", len(self._vocab) - 1):
                break
            if tid <= 1:
                continue
            chars.append(self._rev_vocab.get(tid, ""))
        return "".join(chars)

    def load_dataset(self, data_path: str) -> List[Dict[str, str]]:
        """Load JSONL dataset with video_path and caption fields."""
        path = Path(data_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {data_path}")
        entries = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                if "video_path" not in entry or "caption" not in entry:
                    logger.warning("Skipping entry missing video_path or caption: %s", line[:80],
                        extra={"tag": "TRAIN"},)
                    continue
                entries.append(entry)
        logger.info("Loaded %d video-caption pairs from %s", len(entries), data_path,
            extra={"tag": "TRAIN"},)
        return entries

    def _extract_frames(self, video_path: str) -> Optional[np.ndarray]:
        """Extract uniformly spaced frames from a video file.

        Returns (1, N, 224, 224, 3) or None on failure.
        """
        try:
            from domains.multimodal.video import VideoProcessor
            proc = VideoProcessor(max_frames=self.max_frames)
            frames = proc.extract_frames(video_path, self.max_frames)
            if not frames:
                return None
            stacked = np.stack(frames, axis=0)  # (N, 224, 224, 3)
            return stacked.reshape(1, *stacked.shape)  # (1, N, 224, 224, 3)
        except Exception as e:
            logger.warning("Failed to extract frames from %s: %s", video_path, e,
                extra={"tag": "TRAIN"},)
            return None

    def _encode_video(self, frames_np: np.ndarray) -> Tensor:
        """Encode video frames into a video-level embedding.

        Args:
            frames_np: (1, N, 224, 224, 3)
        Returns:
            Tensor: (1, 1, embed_dim) video-level CLS embedding
        """
        B, N, H, W, C = frames_np.shape
        # Flatten batch and frame dims for VisionEncoder
        flat = frames_np.reshape(B * N, H, W, C)
        frame_embeds = self.vision_encoder.get_patch_embeddings(flat)
        # Reshape to (B, N, num_patches+1, embed_dim) and take CLS tokens
        num_tokens = frame_embeds.data.shape[1]
        frame_embeds_3d = frame_embeds.data.reshape(B, N, num_tokens, self.embed_dim)
        # Use CLS token (first token) as frame representation
        frame_cls = frame_embeds_3d[:, :, 0:1, :]  # (B, N, 1, embed_dim)
        frame_cls_flat = frame_cls.reshape(B, N, self.embed_dim)  # (B, N, embed_dim)
        # Temporal encoding
        video_embed = self.temporal_encoder.forward(frame_cls_flat)  # (B, N, embed_dim)
        # Pool to single video embedding (mean over frames)
        pooled = video_embed.data.mean(axis=1, keepdims=True)  # (B, 1, embed_dim)
        return Tensor(pooled, requires_grad=True, _children=(video_embed,))

    def train(
        self,
        data_path: str,
        epochs: int = 5,
        batch_size: int = 2,
        lr: Optional[float] = None,
        output_dir: Optional[str] = None,
        progress_callback=None,
    ) -> Dict:
        """Train the video captioning model on a JSONL dataset.

        Args:
            data_path: Path to JSONL with {video_path, caption} entries
            epochs: Number of training epochs
            batch_size: Videos per batch
            lr: Learning rate (defaults to self.lr)
            output_dir: Directory to save checkpoints
            progress_callback: fn(epoch, step, loss, total_steps) called each step

        Returns:
            dict with training results
        """
        entries = self.load_dataset(data_path)
        if not entries:
            return {"status": "error", "error": "No valid entries in dataset"}

        captions = [e["caption"] for e in entries]
        self.build_vocab(captions)

        lr = lr or self.lr
        for opt in (self.vision_optimizer, self.temporal_optimizer, self.decoder_optimizer):
            opt.lr = lr

        total_steps = epochs * len(entries)
        step = 0
        best_loss = float("inf")

        output_dir = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Starting video training: %d entries, %d epochs", len(entries), epochs,
            extra={"tag": "TRAIN"},)
        t0 = time.time()

        for epoch in range(epochs):
            np.random.shuffle(entries)
            epoch_losses = []

            for i in range(0, len(entries), batch_size):
                batch = entries[i:i + batch_size]
                step += 1

                batch_loss = 0.0
                batch_count = 0

                for entry in batch:
                    frames = self._extract_frames(entry["video_path"])
                    if frames is None:
                        logger.warning("Skipping video: %s", entry["video_path"],
                            extra={"tag": "TRAIN"},)
                        continue

                    video_embed = self._encode_video(frames)
                    text_tokens = self.encode_text(entry["caption"])
                    token_ids = np.array([text_tokens[:self.max_seq_len - 1]], dtype=np.int64)
                    inp = _tensor(token_ids, requires_grad=False)

                    logits, _ = self.decoder.forward(video_embed, inp, None)
                    targets_np = token_ids[:, 1:]
                    if targets_np.shape[1] < 1:
                        continue
                    targets = _tensor(targets_np.reshape(-1), requires_grad=False)
                    loss = _cross_entropy(logits, targets)
                    loss.backward()

                    batch_loss += float(loss.data)
                    batch_count += 1

                if batch_count == 0:
                    continue

                for opt in (self.vision_optimizer, self.temporal_optimizer, self.decoder_optimizer):
                    opt.step(self._all_params())
                    for p in self._all_params():
                        p.grad = None

                avg_loss = batch_loss / batch_count
                epoch_losses.append(avg_loss)

                if progress_callback:
                    progress_callback(epoch + 1, step, avg_loss, total_steps)

                if step % 10 == 0:
                    logger.info(
                        "Epoch %d/%d step %d/%d loss=%.4f (%.1fs)",
                        epoch + 1, epochs, step, total_steps, avg_loss,
                        time.time() - t0,
                        extra={"tag": "TRAIN"},
                    )

            epoch_avg = np.mean(epoch_losses) if epoch_losses else float("inf")
            logger.info("Epoch %d/%d complete — avg loss: %.4f", epoch + 1, epochs, epoch_avg,
                extra={"tag": "TRAIN"},)

            if epoch_avg < best_loss:
                best_loss = epoch_avg
                self._save_checkpoint(output_dir / "checkpoints", f"epoch_{epoch + 1}", epoch + 1, step, epoch_avg)

        elapsed = time.time() - t0
        self._trained = True

        result = {
            "status": "completed",
            "epochs": epochs,
            "total_steps": step,
            "final_loss": float(np.mean([epoch_losses]) if epoch_losses else 0),
            "best_loss": float(best_loss),
            "elapsed_seconds": round(elapsed, 1),
            "examples": len(entries),
            "vocab_size": len(self._vocab),
            "output_dir": str(output_dir),
        }

        self._save_checkpoint(output_dir / "checkpoints", "final", epochs, step, best_loss)

        logger.info("Video training complete in %.1fs: %s", elapsed, result["status"],
            extra={"tag": "TRAIN"},)
        return result

    def _all_params(self):
        """Return all trainable parameters."""
        params = self.vision_encoder.parameters()
        params += self.temporal_encoder.parameters()
        params += self.decoder.parameters()
        return [p for p in params if p.requires_grad]

    def _save_checkpoint(self, checkpoint_dir: Path, name: str, epoch: int, step: int, loss: float):
        """Save a training checkpoint."""
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = checkpoint_dir / f"{name}.npz"

        weights = {}
        params = self._all_params()
        for i, p in enumerate(params):
            weights[f"param_{i}"] = p.data
        weights["vocab_keys"] = np.array(list(self._vocab.keys()), dtype=object)
        weights["vocab_vals"] = np.array(list(self._vocab.values()), dtype=np.int64)

        np.savez_compressed(str(ckpt_path), **weights)

        meta_path = checkpoint_dir / f"{name}_meta.json"
        with open(meta_path, "w") as f:
            json.dump({
                "name": name,
                "epoch": epoch,
                "step": step,
                "loss": loss,
                "vocab_size": len(self._vocab),
                "embed_dim": self.embed_dim,
                "hidden_dim": self.hidden_dim,
                "max_frames": self.max_frames,
            }, f)

        logger.info("Checkpoint saved: %s (loss=%.4f, step=%d)", ckpt_path, loss, step,
            extra={"tag": "TRAIN"},)

    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        ckpt = np.load(str(path), allow_pickle=True)

        vocab_keys = ckpt["vocab_keys"].tolist()
        vocab_vals = ckpt["vocab_vals"].tolist()
        self._vocab = dict(zip(vocab_keys, vocab_vals))
        self._rev_vocab = {v: k for k, v in self._vocab.items()}
        self.decoder.vocab_size = max(1, len(self._vocab))

        params = self._all_params()
        for i, p in enumerate(params):
            key = f"param_{i}"
            if key in ckpt:
                p.data = ckpt[key].copy()

        self._trained = True
        logger.info("Checkpoint loaded: %s (%d params)", path, len(params),
            extra={"tag": "TRAIN"},)

    def generate(self, video_path: str, max_len: int = 50, temperature: float = 0.8) -> str:
        """Generate a caption for a video.

        Args:
            video_path: Path to video file
            max_len: Maximum caption length
            temperature: Sampling temperature (0 = greedy)

        Returns:
            Generated caption string
        """
        if not self._trained:
            return "[model untrained — train on videos first]"

        frames = self._extract_frames(video_path)
        if frames is None:
            return "[failed to process video]"

        video_embed = self._encode_video(frames)
        bos = self._vocab.get("<BOS>", 1)
        eos = self._vocab.get("<EOS>", len(self._vocab) - 1)
        tokens = [bos]

        for _ in range(min(max_len, self.max_seq_len)):
            inp = _tensor(np.array([tokens], dtype=np.int64), requires_grad=False)
            logits, _ = self.decoder.forward(video_embed, inp, None)
            logits_2d = logits.data.reshape(-1, logits.data.shape[-1])
            last_pos = logits_2d[-1]

            if temperature > 0 and self._trained:
                probs_np = np.maximum(last_pos, 1e-8)
                probs_np = probs_np / probs_np.sum()
                next_tok = int(np.random.choice(len(probs_np), p=probs_np))
            else:
                next_tok = int(np.argmax(last_pos))

            if next_tok == eos:
                break
            tokens.append(next_tok)

        return self.decode_text(tokens)


def list_video_checkpoints(base_dir: Optional[str] = None) -> List[Dict]:
    """List all saved video training checkpoints."""
    ckpt_dir = Path(base_dir) / "checkpoints" if base_dir else CHECKPOINT_DIR
    if not ckpt_dir.exists():
        return []

    checkpoints = []
    for meta_file in sorted(ckpt_dir.glob("*_meta.json"), reverse=True):
        try:
            with open(meta_file) as f:
                meta = json.load(f)
            npz_path = ckpt_dir / (meta_file.stem.replace("_meta", "") + ".npz")
            size_mb = round(npz_path.stat().st_size / (1024 * 1024), 3) if npz_path.exists() else 0
            checkpoints.append({
                "name": meta.get("name", meta_file.stem.replace("_meta", "")),
                "path": str(npz_path),
                "size_mb": size_mb,
                "epoch": meta.get("epoch", 0),
                "step": meta.get("step", 0),
                "loss": meta.get("loss"),
                "vocab_size": meta.get("vocab_size", 0),
            })
        except Exception:
            continue
    return checkpoints
