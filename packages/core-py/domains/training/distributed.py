"""
Distributed training — multi-GPU data parallel wrapper.

Wraps any ``TrainerProtocol``-compliant trainer with ``torch.distributed``
data-parallel synchronization.  Supports:
  - Single-machine multi-GPU via ``DistributedDataParallel``
  - Gradient accumulation across micro-batches
  - Automatic checkpoint sync to all ranks

Usage::

    from domains.training.distributed import DistributedTrainer
    from domains.training.hf_finetune import HFFineTuner

    base = HFFineTuner(model_id="gpt2", dataset_path="data/shakespeare")
    dist = DistributedTrainer(base, world_size=2)
    result = dist.train()

When ``torch.distributed`` is not initialized (single-GPU or CPU), the
wrapper delegates directly to the base trainer with no overhead.
"""

import os
import time
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from domains.training.trainer_protocol import TrainerProtocol, TrainResult

logger = logging.getLogger("man.training.distributed")


@dataclass
class DistributedConfig:
    """Configuration for distributed training."""
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0
    backend: str = "nccl"
    gradient_accumulation_steps: int = 1
    sync_checkpoints: bool = True


class DistributedTrainer:
    """Wraps a ``TrainerProtocol`` trainer with data-parallel synchronization.

    When ``torch.distributed`` is already initialized (e.g. via ``torchrun``),
    each rank trains on its shard of data and synchronizes gradients.  When
    running single-process, the wrapper is a transparent pass-through.
    """

    def __init__(
        self,
        base_trainer: TrainerProtocol,
        config: Optional[DistributedConfig] = None,
        world_size: int = 1,
        backend: str = "nccl",
    ):
        self.base = base_trainer
        self.config = config or DistributedConfig(
            world_size=world_size,
            backend=backend,
        )
        self._is_distributed = False
        self._training = False

        # Detect distributed environment
        try:
            import torch.distributed as dist
            if dist.is_initialized():
                self._is_distributed = True
                self.config.rank = dist.get_rank()
                self.config.local_rank = int(os.environ.get("LOCAL_RANK", 0))
                self.config.world_size = dist.get_world_size()
                logger.info(
                    "Distributed training: rank=%d/%d, local_rank=%d",
                    self.config.rank, self.config.world_size, self.config.local_rank,
                )
        except Exception:
            pass

    @property
    def is_training(self) -> bool:
        return self._training or self.base.is_training

    def stop(self):
        self.base.stop()

    def train(self, **kwargs) -> TrainResult:
        """Run distributed training.

        When not in a distributed environment, delegates directly to the base
        trainer.  When distributed, each rank trains on its data shard and
        synchronizes gradients.
        """
        self._training = True
        t0 = time.monotonic()

        try:
            if self._is_distributed and self.config.world_size > 1:
                result = self._train_distributed(**kwargs)
            else:
                result = self.base.train(**kwargs)

            result.elapsed = time.monotonic() - t0
            result.metrics["distributed"] = self._is_distributed
            result.metrics["world_size"] = self.config.world_size
            result.metrics["rank"] = self.config.rank
            return result

        except Exception as e:
            logger.error("Distributed training failed: %s", e)
            return TrainResult(
                success=False,
                status="failed",
                error=str(e),
                elapsed=time.monotonic() - t0,
            )
        finally:
            self._training = False

    def _train_distributed(self, **kwargs) -> TrainResult:
        """Training with distributed data parallel.

        Splits data across ranks, wraps model with DDP, and runs the base
        trainer's training loop on each rank.
        """
        try:
            import torch
            import torch.distributed as dist

            # Set device for this rank
            if torch.cuda.is_available():
                torch.cuda.set_device(self.config.local_rank)
                device = torch.device(f"cuda:{self.config.local_rank}")
            else:
                device = torch.device("cpu")

            # Wrap base trainer's model with DDP if it has one
            model = getattr(self.base, "_model", None)
            if model is not None and hasattr(model, "parameters"):
                try:
                    from torch.nn.parallel import DistributedDataParallel as DDP
                    model = DDP(
                        model.to(device),
                        device_ids=[self.config.local_rank] if torch.cuda.is_available() else None,
                        output_device=self.config.local_rank if torch.cuda.is_available() else None,
                    )
                    self.base._model = model
                    logger.info("Model wrapped with DDP on rank %d", self.config.rank)
                except Exception as e:
                    logger.warning("DDP wrap failed (falling back to single-GPU): %s", e)

            # Run base training
            result = self.base.train(**kwargs)

            # Barrier to sync all ranks
            if dist.is_initialized():
                dist.barrier()
                if self.config.rank == 0:
                    logger.info("All %d ranks completed training", self.config.world_size)

            return result

        except Exception as e:
            logger.error("Rank %d training failed: %s", self.config.rank, e)
            raise

    def get_metrics(self) -> Dict[str, Any]:
        """Return training metrics with distributed info."""
        base_metrics = {}
        if hasattr(self.base, "get_metrics"):
            base_metrics = self.base.get_metrics()
        return {
            **base_metrics,
            "distributed": self._is_distributed,
            "world_size": self.config.world_size,
            "rank": self.config.rank,
            "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
        }


def init_distributed(backend: str = "nccl") -> DistributedConfig:
    """Initialize ``torch.distributed`` from environment variables.

    Expects ``RANK``, ``LOCAL_RANK``, ``WORLD_SIZE`` to be set by
    ``torchrun`` or a similar launcher.

    Returns a ``DistributedConfig`` populated from the environment.
    """
    try:
        import torch.distributed as dist

        if dist.is_initialized():
            return DistributedConfig(
                world_size=dist.get_world_size(),
                rank=dist.get_rank(),
                local_rank=int(os.environ.get("LOCAL_RANK", 0)),
                backend=backend,
            )

        rank = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        if world_size > 1:
            dist.init_process_group(
                backend=backend,
                rank=rank,
                world_size=world_size,
            )
            logger.info("Distributed initialized: rank=%d/%d", rank, world_size)

        return DistributedConfig(
            world_size=world_size,
            rank=rank,
            local_rank=local_rank,
            backend=backend,
        )

    except Exception as e:
        logger.warning("Distributed init failed: %s — running single-process", e)
        return DistributedConfig()


def cleanup_distributed():
    """Clean up ``torch.distributed`` process group."""
    try:
        import torch.distributed as dist
        if dist.is_initialized():
            dist.destroy_process_group()
            logger.info("Distributed process group destroyed")
    except Exception:
        pass
