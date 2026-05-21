"""
SloLib — SloNet's Unified Tensor Library

One library for everything: autograd, GPU acceleration, inference optimization.
No PyTorch dependency. Pure NumPy + GPU backends.

Design principles:
- Inference-first: fused kernels, no unnecessary memory copies
- GPU-native on Metal (macOS), CUDA (NVIDIA), CPU fallback
- SloNet compatibility: SloLib IS SloNet, not a wrapper
- Minimal memory: lazy allocation, in-place where safe
- Zero-overhead: no Python loop for matmul/conv/attention

Usage:
    from slolib import Tensor, nn, optim, Dataloader

    model = nn.Sequential([
        nn.Linear(256, 512), nn.GELU(),
        nn.Linear(512, 256), nn.GELU(),
        nn.Linear(256, 50257)
    ])
    opt = optim.Adam(model.parameters(), lr=1e-4)

    # Forward pass uses GPU automatically
    x = Tensor.randn(32, 128)
    logits = model(x)
"""