"""Generate caption for a synthetic test image using the trained MultimodalEngine."""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "packages" / "core-py"))

from domains.multimodal.engine import MultimodalEngine

# Load trained engine
engine = MultimodalEngine.load("data/multimodal/multimodal_engine.npz")
print(f"Trained: {engine._trained}, vocab_size: {engine.text.vocab_size}")

# Generate a test image
size = 224
img = np.zeros((size, size, 3), dtype=np.float32)

# Draw a red circle on white background
img[:, :] = (0.95, 0.95, 0.95)  # white bg
cx, cy, r = size // 2, size // 2, 40
rr, cc = np.ogrid[:size, :size]
circle = ((rr - cy) ** 2 + (cc - cx) ** 2) <= r ** 2
img[circle] = (1.0, 0.2, 0.2)  # red

img_batch = img.reshape(1, size, size, 3)

# Generate caption — greedy with temperature=0 for deterministic output
result = engine.generate(img_batch, max_len=30, temperature=0.0)
print(f"\nGenerated caption: '{result.text}'   (confidence: {result.confidence:.3f})")
