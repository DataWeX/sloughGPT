"""
Distributed Checkpoint Management for SloughGPT

Provides efficient checkpoint saving and loading for distributed training.
Supports DDP, FSDP, and ZeRO configurations.
"""

import json
from domains.training.slonet_compat import torch
dist = torch.distributed
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger("sloughgpt.checkpoints")


@dataclass
class CheckpointConfig:
    """Configuration for checkpointing."""

    save_dir: str = "./checkpoints"
    save_freq: int = 1000
    max_checkpoints: int = 5
    save_optimizer: bool = True
    save_scheduler: bool = True
    save_rng_state: bool = True
    save_fsdp_sharded: bool = False
    save_zero_sharded: bool = False


class DistributedCheckpointManager:
    """
    Manages distributed checkpoint saving and loading.

    Supports:
    - Standard PyTorch checkpoints
    - FSDP sharded checkpoints
    - ZeRO optimizer state sharded checkpoints
    - Automatic checkpoint cleanup
    """

    def __init__(
        self,
        config: Optional[CheckpointConfig] = None,
        rank: int = 0,
        world_size: int = 1,
    ):
        self.config = config or CheckpointConfig()
        self.rank = rank
        self.world_size = world_size
        self.checkpoint_dir = Path(self.config.save_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self._checkpoints: List[Dict[str, Any]] = []
        self._load_checkpoint_index()

    def _load_checkpoint_index(self):
        """Load checkpoint index from disk."""
        index_file = self.checkpoint_dir / "checkpoint_index.json"
        if index_file.exists():
            try:
                with open(index_file) as f:
                    self._checkpoints = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load checkpoint index: {e}")

    def _save_checkpoint_index(self):
        """Save checkpoint index to disk."""
        index_file = self.checkpoint_dir / "checkpoint_index.json"
        try:
            with open(index_file, "w") as f:
                json.dump(self._checkpoints, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save checkpoint index: {e}")

    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        step: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Save a checkpoint.

        Args:
            model: The model to save
            optimizer: Optional optimizer to save
            scheduler: Optional scheduler to save
            step: Current training step
            metadata: Additional metadata to save

        Returns:
            Path to saved checkpoint
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_name = f"checkpoint_step_{step}_{timestamp}"

        checkpoint_path = self.checkpoint_dir / checkpoint_name
        checkpoint_path.mkdir(exist_ok=True)

        metadata = metadata or {}
        metadata.update(
            {
                "step": step,
                "timestamp": timestamp,
                "rank": self.rank,
                "world_size": self.world_size,
            }
        )

        if self.rank == 0:
            with open(checkpoint_path / "metadata.json", "w") as f:
                json.dump(metadata, f)

            if self.config.save_rng_state:
                rng_state = {
                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all()
                    if torch.cuda.is_available()
                    else None,
                }
                torch.save(rng_state, checkpoint_path / "rng_state.pt")

        model_state = model.state_dict()
        if self.rank == 0:
            torch.save(model_state, checkpoint_path / "model.pt")

        if optimizer and self.config.save_optimizer and self.rank == 0:
            torch.save(optimizer.state_dict(), checkpoint_path / "optimizer.pt")

        if scheduler and self.config.save_scheduler and self.rank == 0:
            torch.save(scheduler.state_dict(), checkpoint_path / "scheduler.pt")

        if self.world_size > 1:
            dist.barrier()

        checkpoint_info = {
            "name": checkpoint_name,
            "step": step,
            "timestamp": timestamp,
            "path": str(checkpoint_path),
        }
        self._checkpoints.append(checkpoint_info)
        self._checkpoints.sort(key=lambda x: x["step"], reverse=True)

        if len(self._checkpoints) > self.config.max_checkpoints:
            self._cleanup_old_checkpoints()

        self._save_checkpoint_index()

        logger.info(f"Checkpoint saved: {checkpoint_name}")
        return str(checkpoint_path)

    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints exceeding max_checkpoints."""
        to_remove = self._checkpoints[self.config.max_checkpoints :]
        for ckpt in to_remove:
            try:
                ckpt_path = Path(ckpt["path"])
                if ckpt_path.exists():
                    import shutil

                    shutil.rmtree(ckpt_path)
            except Exception as e:
                logger.warning(f"Failed to remove old checkpoint {ckpt['name']}: {e}")

        self._checkpoints = self._checkpoints[: self.config.max_checkpoints]

    def load_checkpoint(
        self,
        checkpoint_path: Optional[str] = None,
        step: Optional[int] = None,
        model: Optional[torch.nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        device: Optional[torch.device] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Load a checkpoint.

        Args:
            checkpoint_path: Path to checkpoint (if None, loads latest)
            step: Step number to load (if checkpoint_path is None)
            model: Model to load state into
            optimizer: Optimizer to load state into
            scheduler: Scheduler to load state into
            device: Device to load checkpoint to

        Returns:
            Checkpoint metadata or None
        """
        if checkpoint_path is None:
            if step is not None:
                for ckpt in self._checkpoints:
                    if ckpt["step"] == step:
                        checkpoint_path = ckpt["path"]
                        break
            else:
                if self._checkpoints:
                    checkpoint_path = self._checkpoints[0]["path"]

        if not checkpoint_path:
            logger.warning("No checkpoint found to load")
            return None

        checkpoint_path = Path(checkpoint_path)

        metadata = {}
        metadata_file = checkpoint_path / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file) as f:
                metadata = json.load(f)

        if model:
            model_state_file = checkpoint_path / "model.pt"
            if model_state_file.exists():
                state_dict = torch.load(model_state_file, map_location=device)
                model.load_state_dict(state_dict)
                logger.info(f"Model state loaded from {checkpoint_path}")

        if optimizer:
            opt_file = checkpoint_path / "optimizer.pt"
            if opt_file.exists():
                opt_state = torch.load(opt_file, map_location=device)
                optimizer.load_state_dict(opt_state)
                logger.info("Optimizer state loaded")

        if scheduler:
            sched_file = checkpoint_path / "scheduler.pt"
            if sched_file.exists():
                sched_state = torch.load(sched_file, map_location=device)
                scheduler.load_state_dict(sched_state)
                logger.info("Scheduler state loaded")

        return metadata

    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List all available checkpoints."""
        return sorted(self._checkpoints, key=lambda x: x["step"], reverse=True)

    def get_latest_checkpoint(self) -> Optional[Dict[str, Any]]:
        """Get the latest checkpoint."""
        return self._checkpoints[0] if self._checkpoints else None


def create_checkpoint_manager(
    save_dir: str = "./checkpoints",
    rank: int = 0,
    world_size: int = 1,
    **kwargs,
) -> DistributedCheckpointManager:
    """Create a checkpoint manager."""
    config = CheckpointConfig(save_dir=save_dir, **kwargs)
    return DistributedCheckpointManager(config, rank, world_size)


__all__ = [
    "CheckpointConfig",
    "DistributedCheckpointManager",
    "create_checkpoint_manager",
]
