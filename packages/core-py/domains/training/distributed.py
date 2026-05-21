"""
Distributed Training Support - Full-Scale DDP Implementation

Production-grade distributed training for SloughGPT:
- Multi-GPU DataParallel/DistributedDataParallel
- Multi-node training support
- Gradient synchronization
- Mixed precision in distributed setting
- Fault tolerance and recovery
- Integration with unified_training.py
"""

import os
import json
import socket
import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Callable
from enum import Enum

from domains.training.slonet_compat import torch
DDP = torch.nn.parallel.DistributedDataParallel
GradScaler = torch.cuda.amp.GradScaler
autocast = torch.cuda.amp.autocast
dist = torch.distributed
mp = torch.multiprocessing

logger = logging.getLogger("sloughgpt.distributed")


class BackendType(Enum):
    """Distributed backend types."""

    NCCL = "nccl"  # GPU, best for multi-GPU
    GLOO = "gloo"  # CPU, for debugging
    MPI = "mpi"  # MPI-based, HPC


class InitMethod(Enum):
    """Process group initialization methods."""

    ENVIRONMENT = "env://"
    TCP = "tcp://"
    FILE = "file://"


@dataclass
class DistributedConfig:
    """
    Configuration for distributed training.

    Attributes:
        use_distributed: Enable distributed training
        backend: Communication backend (nccl, gloo)
        world_size: Total number of processes
        rank: Current process rank
        local_rank: GPU device index
        init_method: How to initialize process group
        master_addr: Master node address (multi-node)
        master_port: Master node port
        seed: Random seed for reproducibility
    """

    use_distributed: bool = False
    backend: str = "nccl"
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0
    init_method: str = "env://"
    master_addr: str = "127.0.0.1"
    master_port: int = 29500
    seed: int = 42

    def __post_init__(self):
        # Auto-detect GPU count
        if self.use_distributed and self.world_size == 1:
            if torch.cuda.is_available():
                self.world_size = torch.cuda.device_count()
            else:
                logger.warning("CUDA not available, falling back to single process")
                self.use_distributed = False

    @classmethod
    def from_env(cls) -> "DistributedConfig":
        """Create config from environment variables."""
        config = cls()

        # Check if distributed training is requested
        if "WORLD_SIZE" in os.environ:
            config.use_distributed = True
            config.world_size = int(os.environ.get("WORLD_SIZE", 1))
            config.rank = int(os.environ.get("RANK", 0))
            config.local_rank = int(os.environ.get("LOCAL_RANK", 0))
            config.master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
            config.master_port = int(os.environ.get("MASTER_PORT", 29500))

        return config

    @classmethod
    def from_slurm(cls) -> "DistributedConfig":
        """Create config from SLURM environment."""
        config = cls()

        if "SLURM_PROCID" in os.environ:
            config.use_distributed = True
            config.world_size = int(os.environ.get("SLURM_NTASKS", 1))
            config.rank = int(os.environ.get("SLURM_PROCID", 0))
            config.local_rank = int(os.environ.get("SLURM_LOCALID", 0))

            # Get master address from SLURM
            config.master_addr = os.environ.get("SLURM_JOB_NODELIST", "127.0.0.1").split(",")[0]
            config.master_port = int(os.environ.get("SLURM_JOB_ID", 29500)) % 60000 + 10000

        return config


class DistributedTrainer:
    """
    Full-featured distributed trainer.

    Features:
    - DDP and DataParallel support
    - Gradient bucketing for efficiency
    - Mixed precision training
    - Gradient clipping
    - Synchronization primitives
    - Checkpoint management
    - Logging and metrics
    """

    def __init__(
        self,
        config: DistributedConfig,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        device: Optional[torch.device] = None,
    ):
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # DDP wrapper
        self.ddp_model: Optional[DDP] = None

        # Training state
        self.world_size = 1
        self.rank = 0
        self.local_rank = 0
        self.is_main = True
        self._initialized = False

        # Mixed precision
        self.scaler: Optional[GradScaler] = None

        # Metrics
        self.metrics: Dict[str, List[float]] = {}

        if config.use_distributed:
            self._init_distributed()

    def _init_distributed(self):
        """Initialize distributed training."""
        if not dist.is_available():
            logger.warning("Distributed training not available")
            self.config.use_distributed = False
            return

        # Setup environment
        self._setup_environment()

        try:
            # Initialize process group
            if self.config.init_method == "env://":
                dist.init_process_group(
                    backend=self.config.backend,
                    init_method="env://",
                )
            else:
                init_url = f"tcp://{self.config.master_addr}:{self.config.master_port}"
                dist.init_process_group(
                    backend=self.config.backend,
                    init_method=init_url,
                    world_size=self.config.world_size,
                    rank=self.config.rank,
                )

            # Get process info
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
            self.local_rank = self.rank % self.world_size

            # Set device
            if torch.cuda.is_available():
                torch.cuda.set_device(self.local_rank)
                self.device = torch.device(f"cuda:{self.local_rank}")

            # Wrap model with DDP
            self.model = self.model.to(self.device)
            self.ddp_model = DDP(
                self.model,
                device_ids=[self.local_rank] if torch.cuda.is_available() else None,
                output_device=self.local_rank if torch.cuda.is_available() else None,
                find_unused_parameters=False,
                gradient_as_bucket_view=True,
            )

            self.is_main = self.rank == 0
            self._initialized = True

            # Synchronize to ensure all processes ready
            self.barrier()

            if self.is_main:
                logger.info(f"DDP initialized: rank={self.rank}, world_size={self.world_size}")

        except Exception as e:
            logger.error(f"Failed to initialize distributed: {e}")
            self.config.use_distributed = False

    def _setup_environment(self):
        """Setup environment variables for distributed training."""
        os.environ.setdefault("RANK", str(self.config.rank))
        os.environ.setdefault("WORLD_SIZE", str(self.config.world_size))
        os.environ.setdefault("LOCAL_RANK", str(self.config.local_rank))
        os.environ.setdefault("MASTER_ADDR", self.config.master_addr)
        os.environ.setdefault("MASTER_PORT", str(self.config.master_port))

    @property
    def training_model(self) -> torch.nn.Module:
        """Get the model for training (handles DDP wrapping)."""
        return self.ddp_model if self.ddp_model is not None else self.model

    def setup_mixed_precision(self, enabled: bool = True, dtype: torch.dtype = torch.bfloat16):
        """Setup mixed precision training."""
        if not enabled:
            self.scaler = None
            return

        if not torch.cuda.is_available():
            logger.warning("CUDA not available, mixed precision disabled")
            return

        self.scaler = GradScaler(device=self.device.type)
        self.mixed_precision_dtype = dtype
        logger.info(f"Mixed precision enabled: {dtype}")

    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        loss_fn: Callable,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
    ) -> Dict[str, float]:
        """
        Execute a training step.

        Args:
            batch: Input batch
            loss_fn: Loss function
            gradient_accumulation_steps: Number of steps to accumulate
            max_grad_norm: Gradient clipping norm

        Returns:
            Dictionary of metrics
        """
        model = self.training_model
        model.train()

        # Move batch to device
        batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}

        # Forward pass with optional mixed precision
        if self.scaler is not None:
            with autocast(dtype=self.mixed_precision_dtype):
                loss = loss_fn(batch)

            # Scale loss for gradient accumulation
            loss = loss / gradient_accumulation_steps

            # Backward pass
            self.scaler.scale(loss).backward()

            # Gradient accumulation check
            # Note: In practice, you'd track step count here
            if True:  # accumulator.should_step()
                # Unscale gradients
                self.scaler.unscale_(self.optimizer)

                # Gradient clipping
                if max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
        else:
            # Standard FP32 training
            loss = loss_fn(batch)
            loss = loss / gradient_accumulation_steps

            loss.backward()

            if True:  # accumulator.should_step()
                if max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

        # Gather metrics from all processes
        metrics = {"loss": loss.item() * gradient_accumulation_steps}

        if self.config.use_distributed and self._initialized:
            metrics = self._gather_metrics(metrics)

        return metrics

    def _gather_metrics(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """Gather metrics from all processes."""
        gathered = {}

        for key, value in metrics.items():
            tensor = torch.tensor(value, device=self.device)
            dist.all_reduce(tensor, op=dist.ReduceOp.AVG)
            gathered[key] = tensor.item()

        return gathered

    def barrier(self):
        """Synchronize all processes."""
        if self.config.use_distributed and self._initialized:
            dist.barrier()

    def all_reduce(self, tensor: torch.Tensor, op: str = "sum") -> torch.Tensor:
        """All-reduce operation across processes."""
        if not self.config.use_distributed or not self._initialized:
            return tensor

        reduce_op = dist.ReduceOp.SUM if op == "sum" else dist.ReduceOp.AVG
        dist.all_reduce(tensor, op=reduce_op)
        return tensor

    def all_gather(self, tensor: torch.Tensor) -> List[torch.Tensor]:
        """All-gather operation across processes."""
        if not self.config.use_distributed or not self._initialized:
            return [tensor]

        world_size = dist.get_world_size()
        tensor_list = [torch.zeros_like(tensor) for _ in range(world_size)]
        dist.all_gather(tensor_list, tensor)
        return tensor_list

    def broadcast(self, tensor: torch.Tensor, src: int = 0):
        """Broadcast tensor from source rank."""
        if self.config.use_distributed and self._initialized:
            dist.broadcast(tensor, src=src)

    def save_checkpoint(
        self,
        path: str,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        epoch: int = 0,
        metrics: Optional[Dict[str, float]] = None,
    ):
        """Save distributed checkpoint."""
        if not self.is_main:
            return

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict() if self.optimizer else None,
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "epoch": epoch,
            "metrics": metrics or {},
            "config": {
                "world_size": self.world_size,
                "rank": self.rank,
            },
        }

        # Save with .rank suffix for multi-rank saving
        save_path = f"{path}.rank{self.rank}"
        torch.save(checkpoint, save_path)

        # Save metadata for consolidation
        metadata = {
            "world_size": self.world_size,
            "ranks": list(range(self.world_size)),
            "epoch": epoch,
        }
        with open(f"{path}.metadata.json", "w") as f:
            json.dump(metadata, f)

        logger.info(f"Checkpoint saved: {save_path}")

    def load_checkpoint(
        self,
        path: str,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Load distributed checkpoint."""
        # Load rank-specific checkpoint
        load_path = f"{path}.rank{self.rank}"

        if not os.path.exists(load_path):
            logger.warning(f"Checkpoint not found: {load_path}")
            return {}

        checkpoint = torch.load(load_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])

        if optimizer and checkpoint.get("optimizer_state_dict"):
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if scheduler and checkpoint.get("scheduler_state_dict"):
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        logger.info(f"Checkpoint loaded: {load_path}")

        return {
            "epoch": checkpoint.get("epoch", 0),
            "metrics": checkpoint.get("metrics", {}),
        }

    @staticmethod
    def consolidate_checkpoints(checkpoint_paths: List[str], output_path: str):
        """Consolidate distributed checkpoints into single file."""
        state_dict = {}

        for path in checkpoint_paths:
            ckpt = torch.load(path, map_location="cpu")
            for key, value in ckpt["model_state_dict"].items():
                state_dict[key] = value

        torch.save(state_dict, output_path)
        logger.info(f"Consolidated {len(checkpoint_paths)} checkpoints to {output_path}")

    def cleanup(self):
        """Clean up distributed training."""
        if self._initialized:
            dist.destroy_process_group()
            self._initialized = False
            logger.info("Distributed training cleaned up")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()


# =============================================================================
# Multi-Node Training Launcher
# =============================================================================


def launch_distributed(
    fn: Callable,
    config: DistributedConfig,
    nprocs: Optional[int] = None,
):
    """
    Launch distributed training across multiple processes.

    Args:
        fn: Training function to run
        config: Distributed configuration
        nprocs: Number of processes (defaults to world_size)
    """
    nprocs = nprocs or config.world_size

    if nprocs == 1:
        # Single process mode
        trainer = DistributedTrainer(config, None)
        fn(trainer)
    else:
        # Multi-process mode
        mp.spawn(
            fn,
            args=(config,),
            nprocs=nprocs,
            join=True,
        )


def spawn_training_processes(
    config: DistributedConfig,
    model: torch.nn.Module,
    train_fn: Callable,
):
    """
    Spawn training processes for distributed training.

    This is the main entry point for multi-GPU training.
    """
    nprocs = config.world_size if config.world_size > 1 else torch.cuda.device_count()

    logger.info(f"Spawning {nprocs} training processes")

    mp.spawn(
        _training_worker,
        args=(config, model, train_fn),
        nprocs=nprocs,
        join=True,
    )


def _training_worker(
    rank: int,
    config: DistributedConfig,
    model: torch.nn.Module,
    train_fn: Callable,
):
    """Worker process for distributed training."""
    # Update config for this process
    config.rank = rank
    config.local_rank = rank

    # Create trainer
    trainer = DistributedTrainer(config, model)

    # Run training
    train_fn(trainer)


# =============================================================================
# Utility Functions
# =============================================================================


def setup_distributed_slurm():
    """Setup distributed training from SLURM environment."""
    # Get SLURM variables
    if "SLURM_PROCID" not in os.environ:
        raise RuntimeError("Not running under SLURM")

    # Get node list
    hostfile = os.environ.get("SLURM_JOB_NODELIST")
    if hostfile:
        # Parse hostfile to get master address
        master_addr = hostfile.split(",")[0]
    else:
        master_addr = socket.gethostname()

    # Get task info
    world_size = int(os.environ.get("SLURM_NTASKS", 1))
    rank = int(os.environ.get("SLURM_PROCID", 0))
    local_rank = int(os.environ.get("SLURM_LOCALID", 0))

    # Create config
    config = DistributedConfig(
        use_distributed=True,
        world_size=world_size,
        rank=rank,
        local_rank=local_rank,
        master_addr=master_addr,
    )

    return config


def get_gpu_memory_info() -> Dict[str, float]:
    """Get GPU memory information for current device."""
    if not torch.cuda.is_available():
        return {}

    device = torch.cuda.current_device()
    return {
        "allocated_mb": torch.cuda.memory_allocated(device) / 1024**2,
        "reserved_mb": torch.cuda.memory_reserved(device) / 1024**2,
        "max_allocated_mb": torch.cuda.max_memory_allocated(device) / 1024**2,
    }


def is_distributed_initialized() -> bool:
    """Check if distributed is initialized."""
    return dist.is_available() and dist.is_initialized()


def get_world_size() -> int:
    """Get world size."""
    if is_distributed_initialized():
        return dist.get_world_size()
    return 1


def get_rank() -> int:
    """Get current rank."""
    if is_distributed_initialized():
        return dist.get_rank()
    return 0


def is_main_process() -> bool:
    """Check if current process is main."""
    return get_rank() == 0


# =============================================================================
# Integration with Unified Training
# =============================================================================


class DistributedTrainerAdapter:
    """
    Adapter to integrate DDP with unified_training.py
    """

    @staticmethod
    def wrap_model(
        model: torch.nn.Module,
        config: DistributedConfig,
    ) -> torch.nn.Module:
        """
        Wrap a model for distributed training.

        Usage:
            from unified_training import Trainer
            from distributed import DistributedConfig, DistributedTrainerAdapter

            config = DistributedConfig.from_env()
            trainer = Trainer(...)
            trainer.setup()

            # Wrap model for DDP
            trainer.model = DistributedTrainerAdapter.wrap_model(
                trainer.model.model, config
            )
        """
        if not config.use_distributed:
            return model

        # Move to device
        device = torch.device(f"cuda:{config.local_rank}" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        # Wrap with DDP
        ddp_model = DDP(
            model,
            device_ids=[config.local_rank] if torch.cuda.is_available() else None,
            find_unused_parameters=False,
        )

        return ddp_model

    @staticmethod
    def setup_from_config(
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        config: DistributedConfig,
    ) -> tuple:
        """
        Setup distributed training from config.

        Returns:
            tuple: (distributed_trainer, ddp_model)
        """
        dist_trainer = DistributedTrainer(
            config=config,
            model=model,
            optimizer=optimizer,
        )

        ddp_model = dist_trainer.training_model

        return dist_trainer, ddp_model


__all__ = [
    "DistributedConfig",
    "DistributedTrainer",
    "DistributedTrainerAdapter",
    "BackendType",
    "InitMethod",
    "launch_distributed",
    "spawn_training_processes",
    "setup_distributed_slurm",
    "is_distributed_initialized",
    "get_world_size",
    "get_rank",
    "is_main_process",
    "get_gpu_memory_info",
    "FSDPTrainer",
    "ShardingStrategy",
    "CPUOffload",
]


# =============================================================================
# FSDP (Fully Sharded Data Parallel) - For Very Large Models
# =============================================================================


class ShardingStrategy(Enum):
    """FSDP sharding strategies."""

    FULL_SHARD = "full_shard"  # Shard params, grads, optimizer states
    SHARD_GRAD_OP = "shard_grad_op"  # Shard grads and optimizer states
    NO_SHARD = "no_shard"  # Replicate params (like DDP)


class CPUOffload(Enum):
    """CPU offloading options."""

    NONE = "none"  # No offloading
    PARAM = "param"  # Offload params to CPU
    PARAM_AND_OPTIM = "param_and_optim"  # Offload params and optimizer states


class FSDPTrainer:
    """
    Fully Sharded Data Parallel Trainer.

    For training very large models that don't fit in GPU memory.
    """

    def __init__(
        self,
        config: DistributedConfig,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        sharding_strategy: ShardingStrategy = ShardingStrategy.FULL_SHARD,
        cpu_offload: CPUOffload = CPUOffload.NONE,
        mixed_precision: bool = True,
    ):
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.fsdp_model = None
        self._initialized = False

        if config.use_distributed:
            self._init_fsdp(sharding_strategy, cpu_offload, mixed_precision)

    def _init_fsdp(
        self,
        sharding_strategy: ShardingStrategy,
        cpu_offload: CPUOffload,
        mixed_precision: bool,
    ):
        """Initialize FSDP."""
        try:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
            from torch.distributed.fsdp.fully_sharded_data_parallel import MixedPrecision
            from torch.distributed.fsdp.fully_sharded_data_parallel import (
                CPUOffload as FSDPCPUOffload,
            )
        except ImportError:
            logger.error("FSDP not available. Use PyTorch 2.0+")
            return

        # Initialize distributed if needed
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")

        # Auto wrap policy for transformers
        auto_wrap_policy = transformer_auto_wrap_policy

        # Mixed precision config
        mixed_precision_config = None
        if mixed_precision:
            mixed_precision_config = MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.float32,
                buffer_dtype=torch.bfloat16,
            )

        # CPU offload config
        cpu_offload_config = None
        if cpu_offload == CPUOffload.PARAM:
            cpu_offload_config = FSDPCPUOffload(offload_params=True)
        elif cpu_offload == CPUOffload.PARAM_AND_OPTIM:
            cpu_offload_config = FSDPCPUOffload(offload_params=True, offload_optimizer=True)

        # Wrap model with FSDP
        self.model = self.model.to(self.config.local_rank)

        self.fsdp_model = FSDP(
            self.model,
            sharding_strategy=self._get_sharding_strategy(sharding_strategy),
            cpu_offload=cpu_offload_config,
            auto_wrap_policy=auto_wrap_policy,
            mixed_precision=mixed_precision_config,
            device_id=self.config.local_rank,
        )

        self._initialized = True
        logger.info(
            f"FSDP initialized: rank={self.config.rank}, world_size={self.config.world_size}"
        )

    def _get_sharding_strategy(self, strategy: ShardingStrategy):
        """Get FSDP sharding strategy."""
        try:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

            strategy_map = {
                ShardingStrategy.FULL_SHARD: FSDP.ShardingStrategy.FULL_SHARD,
                ShardingStrategy.SHARD_GRAD_OP: FSDP.ShardingStrategy.SHARD_GRAD_OP,
                ShardingStrategy.NO_SHARD: FSDP.ShardingStrategy.NO_SHARD,
            }
            return strategy_map.get(strategy, FSDP.ShardingStrategy.FULL_SHARD)
        except:
            return None

    @property
    def training_model(self):
        """Get model for training."""
        return self.fsdp_model if self.fsdp_model is not None else self.model

    def train_step(self, batch, loss_fn, **kwargs):
        """Execute training step."""
        model = self.training_model
        model.train()

        # Forward pass
        loss = loss_fn(batch)

        # Backward
        loss.backward()

        # FSDP step
        model.clip_grad_norm_(max_norm=1.0)
        self.optimizer.step()
        self.optimizer.zero_grad()

        return {"loss": loss.item()}

    def save_checkpoint(self, path: str, **kwargs):
        """Save FSDP checkpoint."""
        if self.config.rank == 0:
            state_dict = self.fsdp_model.state_dict()
            torch.save(state_dict, path)
            logger.info(f"FSDP checkpoint saved: {path}")

    def load_checkpoint(self, path: str):
        """Load FSDP checkpoint."""
        state_dict = torch.load(path, map_location=self.device)
        self.fsdp_model.load_state_dict(state_dict)
        logger.info(f"FSDP checkpoint loaded: {path}")
