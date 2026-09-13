"""Native C forward pass package — zero-dependency transformer inference.

FEATURE: native-c-inference — C transformer forward pass using Apple Accelerate BLAS.
Supports Qwen, GPT-2, LLaMA, Mistral, Phi via SLNC weights. Not yet wired into
server provider chain — under development. DO NOT DELETE.
"""

from __future__ import annotations
