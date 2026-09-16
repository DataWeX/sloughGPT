"""Shared model configuration for ProcessGuard and workers.

Eliminates parameter duplication across ProcessGuard, ModelWorkerProcess,
and all call sites (startup.py, controllers/models.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from apps.api.server.config import ServerConfig


class ExecutionMode(Enum):
    """How ProcessGuard runs model inference."""

    THREAD = "thread"      # PGQ Tree — no subprocess, for autoload / PGQ context
    SUBPROCESS = "subprocess"  # PGQ GuardTree — full OS isolation, for manual API load


@dataclass(frozen=True)
class ModelConfig:
    """Immutable model configuration shared across guard + workers.

    Replaces the 20-parameter mix of guard/slo/hf config that was
    previously duplicated across ProcessGuard.__init__ and
    ModelWorkerProcess.__init__.
    """

    slnc_path: str | None = None
    model_id: str = "default"
    quantize: bool = False
    quant_bits: int = 8
    quant_mode: str = "symmetric"
    quant_clip: float = 0.999
    hf_model_cls_path: str | None = None
    hf_model_kwargs: dict = field(default_factory=dict)

    @property
    def backend(self) -> str:
        """Worker backend: ``'slo'`` or ``'hf'``."""
        return "slo" if self.slnc_path else "hf"

    @property
    def is_slo(self) -> bool:
        return self.slnc_path is not None

    @classmethod
    def from_env(cls, model_id: str, cfg: ServerConfig) -> ModelConfig:
        """Build from ServerConfig + model id.

        Resolves the .slnc path from the model directory.
        """
        import os

        from domain.infrastructure.model_resolver import get_model_dir

        slnc_path = str(get_model_dir(model_id) / "model.slnc")
        if not os.path.exists(slnc_path):
            slnc_path = None

        return cls(
            slnc_path=slnc_path,
            model_id=model_id,
            quantize=cfg.quantize_slonet,
            quant_bits=cfg.quant_bits,
            quant_mode=cfg.quant_mode,
            quant_clip=cfg.quant_clip,
        )
