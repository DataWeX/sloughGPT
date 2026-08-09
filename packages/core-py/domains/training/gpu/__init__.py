"""GPU accelerator — Metal/CUDA/CPU dispatch for training ops."""

from domains.training.gpu.accelerator import get_accelerator, to_gpu, from_gpu, reset_accelerator

__all__ = ["get_accelerator", "to_gpu", "from_gpu", "reset_accelerator"]
