"""
Train the custom SloNet-based multimodal engine (vision + audio + text).

Generates:
- Synthetic shape images with captions (vision → text)
- Synthetic sine-wave audio with captions (audio → text)
- Combined image+audio samples

Usage:
    python scripts/train_multimodal.py [--epochs 50] [--batch-size 1] [--samples 100]
"""

import argparse
import sys
import os
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "packages" / "core-py"))

from domains.multimodal.engine import (
    MultimodalEngine, get_multimodal_engine, ReplayBuffer,
    replay_train_step, contrastive_step,
)

# ── Synthetic vision data ─────────────────────────────────────────────

SHAPES = ["circle", "square", "triangle"]
COLORS = ["red", "green", "blue", "yellow", "orange", "purple"]
BG_COLORS = ["white", "black", "gray", "beige"]

VISION_TEMPLATES = [
    "{color} {shape} on {bg} background",
    "a {color} {shape} centered on {bg}",
    "{color} {shape} over {bg} background",
]

def _draw_shape(img: np.ndarray, shape: str, color: tuple, cx: int, cy: int, size: int):
    h, w, _ = img.shape
    rr, cc = np.ogrid[:h, :w]
    if shape == "circle":
        mask = ((rr - cy) ** 2 + (cc - cx) ** 2) <= size ** 2
    elif shape == "square":
        mask = (np.abs(rr - cy) <= size) & (np.abs(cc - cx) <= size)
    elif shape == "triangle":
        mask = (rr >= cy - size) & (rr <= cy + size) & \
               (np.abs(cc - cx) <= size * (1 - (rr - cy) / (2 * size + 1)))
    else:
        mask = np.zeros((h, w), dtype=bool)
    img[mask] = color

def _color_to_rgb(name: str) -> tuple:
    palette = {
        "red": (1.0, 0.2, 0.2), "green": (0.2, 0.8, 0.2),
        "blue": (0.2, 0.2, 1.0), "yellow": (1.0, 0.9, 0.2),
        "orange": (1.0, 0.6, 0.1), "purple": (0.7, 0.2, 0.8),
        "white": (0.95, 0.95, 0.95), "black": (0.1, 0.1, 0.1),
        "gray": (0.5, 0.5, 0.5), "beige": (0.86, 0.82, 0.74),
    }
    return palette.get(name, (0.5, 0.5, 0.5))

def generate_vision_sample(size: int = 224) -> tuple:
    img = np.zeros((size, size, 3), dtype=np.float32)
    bg_name = BG_COLORS[np.random.randint(len(BG_COLORS))]
    img[:, :] = _color_to_rgb(bg_name)
    shape = SHAPES[np.random.randint(len(SHAPES))]
    color_name = COLORS[np.random.randint(len(COLORS))]
    cx, cy = size // 2 + np.random.randint(-20, 21), size // 2 + np.random.randint(-20, 21)
    shape_size = np.random.randint(20, 45)
    _draw_shape(img, shape, _color_to_rgb(color_name), cx, cy, shape_size)
    template = VISION_TEMPLATES[np.random.randint(len(VISION_TEMPLATES))]
    caption = template.format(color=color_name, shape=shape, bg=bg_name)
    return img.reshape(1, size, size, 3), caption

# ── Synthetic audio data ──────────────────────────────────────────────

FREQUENCIES = [
    (130.81, "low C"), (164.81, "low E"), (220.0, "low A"),
    (261.63, "middle C"), (329.63, "middle E"), (440.0, "middle A"),
    (523.25, "high C"), (659.25, "high E"), (880.0, "high A"),
]

AUDIO_TEMPLATES = [
    "{note} tone playing",
    "a {note} pitch",
    "sound at {note} frequency",
]

def generate_audio_sample(sample_rate: int = 16000, duration: float = 2.0) -> tuple:
    freq_name = FREQUENCIES[np.random.randint(len(FREQUENCIES))]
    freq_hz, note = freq_name
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    waveform = np.sin(2 * np.pi * freq_hz * t).astype(np.float32)
    # Add slight harmonic for realism
    waveform += 0.3 * np.sin(4 * np.pi * freq_hz * t).astype(np.float32)
    waveform /= np.max(np.abs(waveform))
    template = AUDIO_TEMPLATES[np.random.randint(len(AUDIO_TEMPLATES))]
    caption = template.format(note=note)
    return waveform, caption

# ── Dataset generation ────────────────────────────────────────────────

def generate_vision_dataset(n: int) -> tuple:
    images, captions = [], []
    for _ in range(n):
        img, cap = generate_vision_sample()
        images.append(img)
        captions.append(cap)
    return np.concatenate(images, axis=0), captions

def generate_audio_dataset(n: int, sample_rate: int = 16000) -> tuple:
    waveforms, captions = [], []
    for _ in range(n):
        wav, cap = generate_audio_sample(sample_rate)
        waveforms.append(wav)
        captions.append(cap)
    return waveforms, captions

# ── Training ──────────────────────────────────────────────────────────

def train(args):
    np.random.seed(42)
    
    print("Initializing MultimodalEngine...")
    engine = get_multimodal_engine(
        embed_dim=128,
        hidden_dim=256,
        n_vit_layers=3,
        n_heads=4,
        n_decoder_layers=3,
        n_audio_layers=2,
    )

    # Generate all training captions first (for vocab building)
    all_captions = []
    
    # Vision data
    print(f"Generating {args.samples} synthetic image–caption pairs...")
    train_images, train_captions_v = generate_vision_dataset(args.samples)
    all_captions.extend(train_captions_v)
    print(f"  Images shape: {train_images.shape}")
    print(f"  Example: {train_captions_v[0]}")

    # Audio data
    n_audio = args.samples  # same amount
    print(f"Generating {n_audio} synthetic audio–caption pairs...")
    train_waveforms, train_captions_a = generate_audio_dataset(n_audio)
    all_captions.extend(train_captions_a)
    print(f"  First waveform: {train_waveforms[0].shape}")
    print(f"  Example: {train_captions_a[0]}")

    # Build character-level vocabulary from all captions
    print("\nBuilding character vocabulary from all captions...")
    engine.build_vocab(all_captions)
    print(f"  Char vocab size: {engine.text.vocab_size}")

    # Verify token lengths are reasonable (char level: one token per char + BOS/EOS)
    for cap in all_captions[:10]:
        tokens = engine.text.encode(cap)
        assert len(tokens) == len(cap) + 2, f"Char token len mismatch: {cap} → {len(tokens)} tokens (expected {len(cap)+2})"

    # Precompute audio patches to avoid STFT per epoch
    print("Precomputing audio patches (avoids STFT per epoch)...")
    t0 = time.time()
    audio_patches_list = [engine.precompute_audio_patches(w.reshape(1, -1)) for w in train_waveforms]
    print(f"  Precomputed {len(audio_patches_list)} patches in {time.time()-t0:.1f}s")

    # Replay buffer
    buffer = ReplayBuffer(capacity=args.buffer_capacity)

    # Training loop
    print(f"\nTraining for {args.epochs} epochs...")
    losses_v, losses_a = [], []
    n_v, n_a = len(train_images), len(train_waveforms)
    n_epochs = args.epochs

    for epoch in range(n_epochs):
        # Cosine LR schedule: 3e-4 → 3e-5
        frac = epoch / max(n_epochs - 1, 1)
        lr = 3e-5 + 0.5 * (3e-4 - 3e-5) * (1 + np.cos(np.pi * frac))

        # Temperature annealing: 2.0 → 1.0 over training
        # (softer targets early for exploration, sharper later for convergence)
        temp = 2.0 - 1.0 * frac  # 2.0 → 1.0

        # ── Vision batch ──
        idx_v = np.random.permutation(n_v)
        epoch_loss_v = 0.0
        steps_v = 0
        for i in range(0, n_v, args.batch_size):
            batch_idx = idx_v[i:i + args.batch_size]
            for j in batch_idx:
                tokens = engine.text.encode(train_captions_v[j])
                if len(tokens) < 2:
                    continue
                tokens_arr = np.array([tokens], dtype=np.int64)
                loss_val = engine.train_step(train_images[j:j+1], tokens_arr, lr=lr, temperature=temp)
                epoch_loss_v += loss_val
                steps_v += 1
            buffer.add(train_images[batch_idx[0]:batch_idx[0]+1], train_captions_v[batch_idx[0]])

        # ── Audio batch ──
        idx_a = np.random.permutation(n_a)
        epoch_loss_a = 0.0
        steps_a = 0
        for i in range(0, n_a, args.batch_size):
            batch_idx = idx_a[i:i + args.batch_size]
            for j in batch_idx:
                tokens = engine.text.encode(train_captions_a[j])
                if len(tokens) < 2:
                    continue
                tokens_arr = np.array([tokens], dtype=np.int64)
                loss_val = engine.train_step(
                    audio_patches=audio_patches_list[j],
                    text_tokens=tokens_arr,
                    lr=lr,
                    temperature=temp,
                )
                epoch_loss_a += loss_val
                steps_a += 1

        # ── Combined vision+audio batch ──
        combined_loss = 0.0
        combined_steps = 0
        for _ in range(min(5, n_v, n_a)):
            vi = np.random.randint(n_v)
            ai = np.random.randint(n_a)
            cap = f"{train_captions_v[vi]} and {train_captions_a[ai]}"
            tokens = engine.text.encode(cap)
            if len(tokens) < 3:
                continue
            tokens_arr = np.array([tokens], dtype=np.int64)
            loss_val = engine.train_step(
                images_np=train_images[vi:vi+1],
                audio_patches=audio_patches_list[ai],
                text_tokens=tokens_arr,
                lr=lr,
                temperature=temp,
            )
            combined_loss += loss_val
            combined_steps += 1

        # ── Replay & contrastive ──
        replay_loss = replay_train_step(engine, buffer, batch_size=4)
        sample_idx = np.random.randint(n_v)
        contrast_loss = contrastive_step(
            engine, train_images[sample_idx:sample_idx+1], buffer
        )

        avg_loss_v = epoch_loss_v / max(steps_v, 1)
        avg_loss_a = epoch_loss_a / max(steps_a, 1)
        avg_loss_c = combined_loss / max(combined_steps, 1)
        losses_v.append(avg_loss_v)
        losses_a.append(avg_loss_a)

        if (epoch + 1) % max(1, args.epochs // 10) == 0 or epoch == 0:
            print(f"\nEpoch {epoch+1}/{args.epochs} (lr={lr:.1e}, temp={temp:.2f}):")
            print(f"  Vision:  loss={avg_loss_v:.4f} ({steps_v} steps)")
            print(f"  Audio:   loss={avg_loss_a:.4f} ({steps_a} steps)")
            print(f"  V+A:     loss={avg_loss_c:.4f} ({combined_steps} steps)")
            print(f"  Replay:  {replay_loss:.4f}  Contrast: {contrast_loss:.4f}")

            # Generate vision sample
            test_idx = np.random.randint(n_v)
            result = engine.generate(train_images[test_idx:test_idx+1], max_len=12, temperature=0.0)
            print(f"  Vision target:  {train_captions_v[test_idx]}")
            print(f"  Vision gen:     {result.text}")

            # Generate audio sample
            test_ai = np.random.randint(n_a)
            result_a = engine.generate(audio_patches=audio_patches_list[test_ai], max_len=12, temperature=0.0)
            print(f"  Audio target:   {train_captions_a[test_ai]}")
            print(f"  Audio gen:      {result_a.text}")

    # Final evaluation
    print(f"\n{'='*60}")
    print(f"Training complete.")
    print(f"Vision loss:  {losses_v[0]:.4f} → {losses_v[-1]:.4f} ({(1-losses_v[-1]/max(losses_v[0],1e-8))*100:.1f}% reduction)")
    print(f"Audio loss:   {losses_a[0]:.4f} → {losses_a[-1]:.4f} ({(1-losses_a[-1]/max(losses_a[0],1e-8))*100:.1f}% reduction)")
    print(f"{'='*60}")

    # Generate multiple vision samples
    print("\nVision sample generations:")
    for i in range(min(5, n_v)):
        result = engine.generate(train_images[i:i+1], max_len=15, temperature=0.3)
        print(f"  [{i}] Target: {train_captions_v[i]}")
        print(f"      Gen:    {result.text}")

    # Generate multiple audio samples
    print("\nAudio sample generations:")
    for i in range(min(5, n_a)):
        result = engine.generate(audio_patches=audio_patches_list[i], max_len=15, temperature=0.3)
        print(f"  [{i}] Target: {train_captions_a[i]}")
        print(f"      Gen:    {result.text}")

    # Save
    print("\nSaving engine...")
    engine.save("data/multimodal/multimodal_engine.npz")
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SloNet multimodal engine (vision + audio + text)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--buffer-capacity", type=int, default=50)
    args = parser.parse_args()
    train(args)
