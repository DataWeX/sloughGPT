"""Multimodal inference demo — generate captions for image + audio samples."""

import sys, time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "packages" / "core-py"))

from domains.multimodal.engine import MultimodalEngine

def draw_shape(img: np.ndarray, shape: str, color: tuple, cx: int, cy: int, size: int):
    h, w, _ = img.shape
    rr, cc = np.ogrid[:h, :w]
    masks = {
        "circle": ((rr - cy) ** 2 + (cc - cx) ** 2) <= size ** 2,
        "square": (np.abs(rr - cy) <= size) & (np.abs(cc - cx) <= size),
    }
    mask = masks.get(shape, np.zeros((h, w), dtype=bool))
    img[mask] = color

COLORS = {"red": (1.0, 0.2, 0.2), "green": (0.2, 0.8, 0.2), "blue": (0.2, 0.2, 1.0),
          "yellow": (1.0, 0.9, 0.2), "white": (0.95, 0.95, 0.95), "gray": (0.5, 0.5, 0.5)}

# Load trained engine
print("Loading MultimodalEngine...")
t0 = time.time()
engine = MultimodalEngine.load("data/multimodal/multimodal_engine.npz")
print(f"  Loaded in {time.time()-t0:.1f}s — trained={engine._trained}, vocab={engine.text.vocab_size}")
print()

# ── Vision demo: red circle on white background ──
print("=" * 50)
print("VISION: red circle on white background")
print("=" * 50)
img = np.zeros((224, 224, 3), dtype=np.float32)
img[:, :] = COLORS["white"]
draw_shape(img, "circle", COLORS["red"], 112, 112, 40)
result = engine.generate(img.reshape(1, 224, 224, 3), max_len=30, temperature=0.0)
print(f"  Generated: '{result.text}'")
print(f"  Target:     'red circle on white background'")
print()

# ── Vision demo: blue square on gray background ──
print("=" * 50)
print("VISION: blue square on gray background")
print("=" * 50)
img2 = np.zeros((224, 224, 3), dtype=np.float32)
img2[:, :] = COLORS["gray"]
draw_shape(img2, "square", COLORS["blue"], 112, 112, 35)
result2 = engine.generate(img2.reshape(1, 224, 224, 3), max_len=30, temperature=0.0)
print(f"  Generated: '{result2.text}'")
print(f"  Target:     'blue square on gray background'")
print()

# ── Audio demo: sine tone ──
print("=" * 50)
print("AUDIO: middle A pitch (440Hz)")
print("=" * 50)
sr = 16000
t = np.linspace(0, 2.0, int(sr * 2.0), endpoint=False).astype(np.float32)
waveform = np.sin(2 * np.pi * 440 * t).astype(np.float32)
waveform += 0.3 * np.sin(4 * np.pi * 440 * t).astype(np.float32)
waveform /= np.max(np.abs(waveform))
result_a = engine.generate(audio_np=waveform, max_len=30, temperature=0.0)
print(f"  Generated: '{result_a.text}'")
print(f"  Target:     'a middle A pitch'")
print()

# ── Combined vision + audio ──
print("=" * 50)
print("COMBINED: red circle ON white bg + middle A pitch")
print("=" * 50)
result_both = engine.generate(image_np=img.reshape(1, 224, 224, 3), audio_np=waveform, max_len=35, temperature=0.0)
print(f"  Generated: '{result_both.text}'")
print(f"  Target:     'red circle on white background and middle A pitch'")
print()

print("Done — model runs end-to-end.")
print(f"Chars used: {set(result.text + result2.text + result_a.text + result_both.text)}")
