"""
SloughGPT Rust Training Backend
High-performance training compute using native Rust + PyO3.

This module provides a Rust-based compute backend for training operations
that can optionally accelerate the PyTorch trainer.

Status: ⚠️ Requires maturin build to enable Rust backend
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

_RUST_AVAILABLE = False

try:
    from sloughgpt_trainer import Trainer, TrainConfig

    _RUST_AVAILABLE = True
except ImportError:
    logging.debug("Rust trainer not available. Install with: maturin develop")


class RustTrainerBackend:
    """
    Optional Rust compute backend for training.

    Wraps the Rust sloughgpt-trainer crate to provide high-performance
    forward/backward passes for specific operations.

    Falls back to PyTorch when Rust unavailable.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self._trainer: Optional[Any] = None
        self._config = config or {}

        if _RUST_AVAILABLE:
            self._init_rust_trainer()

    def _init_rust_trainer(self) -> None:
        """Initialize the Rust trainer."""
        train_config = TrainConfig(
            vocab_size=self._config.get("vocab_size", 256),
            embedding_dim=self._config.get("embedding_dim", 256),
            hidden_dim=self._config.get("hidden_dim", 1024),
            num_layers=self._config.get("num_layers", 6),
            num_heads=self._config.get("num_heads", 8),
            head_dim=self._config.get("head_dim", 32),
            batch_size=self._config.get("batch_size", 32),
            seq_len=self._config.get("seq_len", 128),
            learning_rate=self._config.get("learning_rate", 1e-3),
            total_steps=self._config.get("total_steps", 10000),
        )
        self._trainer = Trainer(train_config)
        logging.info("Rust trainer backend initialized")

    def step(self, batch: List[int], targets: List[int]) -> float:
        """
        Execute one training step in Rust.

        Args:
            batch: Input token IDs
            targets: Target token IDs

        Returns:
            Loss value
        """
        if self._trainer is not None:
            return self._trainer.step(batch, targets)
        return 0.0

    def forward(self, tokens: List[int]) -> List[float]:
        """
        Forward pass in Rust.

        Args:
            tokens: Input token IDs

        Returns:
            Logits
        """
        if self._trainer is not None:
            return self._trainer.forward(tokens)
        return []

    def get_weights(self) -> Dict[str, List[float]]:
        """Get model weights from Rust."""
        if self._trainer is not None:
            return dict(self._trainer.get_weights())
        return {}

    def load_weights(self, weights: Dict[str, List[float]]) -> None:
        """Load model weights into Rust."""
        if self._trainer is not None:
            self._trainer.load_weights(list(weights.items()))

    def save_checkpoint(self, path: str) -> None:
        """Save checkpoint via Rust."""
        if self._trainer is not None:
            self._trainer.save_checkpoint(path)

    def load_checkpoint(self, path: str) -> None:
        """Load checkpoint via Rust."""
        if self._trainer is not None:
            self._trainer.load_checkpoint(path)

    @property
    def is_available(self) -> bool:
        """Check if Rust backend is available."""
        return _RUST_AVAILABLE and self._trainer is not None

    def __repr__(self) -> str:
        return f"RustTrainerBackend(available={self.is_available})"


def is_rust_available() -> bool:
    """Check if Rust training backend is available."""
    return _RUST_AVAILABLE


def create_rust_backend(config: Optional[Dict[str, Any]] = None) -> RustTrainerBackend:
    """Create a Rust training backend."""
    return RustTrainerBackend(config)


if __name__ == "__main__":
    print(f"Rust training backend available: {is_rust_available()}")
    backend = create_rust_backend({"vocab_size": 1000, "embedding_dim": 128})
    print(f"Backend: {backend}")
