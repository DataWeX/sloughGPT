"""
ZeRO Optimizer Implementation for SloughGPT

Implements DeepSpeed-style ZeRO (Zero Redundancy Optimizer) stages:
- Stage 1: Optimizer state sharding
- Stage 2: Gradient sharding
- Stage 3: Parameter sharding
"""

from domains.training.slonet_compat import torch
nn = torch.nn
from typing import Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum


class ZeroStage(Enum):
    """ZeRO optimization stages"""

    DISABLED = 0
    STAGE_1 = 1  # Optimizer state sharding
    STAGE_2 = 2  # Gradient sharding
    STAGE_3 = 3  # Parameter sharding


@dataclass
class ZeroConfig:
    """Configuration for ZeRO optimizer"""

    stage: ZeroStage = ZeroStage.STAGE_1
    reduce_bucket_size: int = 1e6
    allgather_bucket_size: int = 1e6
    overlap_comm: bool = True
    contiguous_gradients: bool = True
    round_robin_binaries: bool = True
    reduce_scatter: bool = True


class ZeroOptimizer:
    """
    ZeRO Optimizer wrapper that shards optimizer states across data parallel ranks.

    This implements Stage 1 ZeRO: optimizer state sharding
    Higher stages would require more complex integration with distributed training.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        config: ZeroConfig,
        device: torch.device,
    ):
        self.optimizer = optimizer
        self.config = config
        self.device = device

        if config.stage == ZeroStage.DISABLED:
            return

        self.param_groups = optimizer.param_groups
        self.state = optimizer.state

        self._init_zero_sharding()

    def _init_zero_sharding(self):
        """Initialize ZeRO sharding structures"""
        if self.config.stage == ZeroStage.STAGE_1:
            self._shard_optimizer_states()
        elif self.config.stage == ZeroStage.STAGE_2:
            self._shard_optimizer_states()
            self._init_gradient_sharding()
        elif self.config.stage == ZeroStage.STAGE_3:
            self._init_parameter_sharding()

    def _shard_optimizer_states(self):
        """Shard optimizer states across ranks"""
        for param_group in self.param_groups:
            for param in param_group["params"]:
                if param in self.state:
                    state = self.state[param]

                    if "exp_avg" in state:
                        state["exp_avg"] = self._shard_tensor(state["exp_avg"])
                    if "exp_avg_sq" in state:
                        state["exp_avg_sq"] = self._shard_tensor(state["exp_avg_sq"])

    def _shard_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Shard a tensor across ranks (reduce to local rank's portion)"""
        if not torch.distributed.is_initialized():
            return tensor

        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()

        if tensor.numel() < world_size:
            return tensor

        chunk_size = (tensor.numel() + world_size - 1) // world_size
        start_idx = rank * chunk_size
        end_idx = min(start_idx + chunk_size, tensor.numel())

        local_tensor = tensor.view(-1)[start_idx:end_idx].clone()
        return local_tensor

    def _init_gradient_sharding(self):
        """Initialize gradient sharding for Stage 2+"""
        self.grad_bucket = {}
        self.bucket_size = self.config.reduce_bucket_size

    def _init_parameter_sharding(self):
        """Initialize parameter sharding for Stage 3"""
        self.param_buckets = {}

    def step(self, closure=None):
        """Perform optimization step"""
        if self.config.stage == ZeroStage.DISABLED:
            return self.optimizer.step(closure)

        if self.config.stage == ZeroStage.STAGE_1:
            self._gather_optimizer_states()

        loss = self.optimizer.step(closure)

        if self.config.stage == ZeroStage.STAGE_1:
            self._scatter_optimizer_states()

        return loss

    def _gather_optimizer_states(self):
        """Gather optimizer states from all ranks for step"""
        if not torch.distributed.is_initialized():
            return

        for param_group in self.param_groups:
            for param in param_group["params"]:
                if param in self.state:
                    state = self.state[param]

                    if "exp_avg" in state:
                        state["exp_avg"] = self._gather_tensor(state["exp_avg"])
                    if "exp_avg_sq" in state:
                        state["exp_avg_sq"] = self._gather_tensor(state["exp_avg_sq"])

    def _scatter_optimizer_states(self):
        """Scatter optimizer states back to sharded form"""
        if not torch.distributed.is_initialized():
            return

        for param_group in self.param_groups:
            for param in param_group["params"]:
                if param in self.state:
                    state = self.state[param]

                    if "exp_avg" in state:
                        state["exp_avg"] = self._shard_tensor(state["exp_avg"])
                    if "exp_avg_sq" in state:
                        state["exp_avg_sq"] = self._shard_tensor(state["exp_avg_sq"])

    def _gather_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Gather sharded tensor from all ranks"""
        if not torch.distributed.is_initialized():
            return tensor

        world_size = torch.distributed.get_world_size()

        if tensor.numel() < world_size:
            return tensor

        gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
        torch.distributed.all_gather(gathered, tensor)

        return torch.cat(gathered)

    def zero_grad(self, set_to_none: bool = True):
        """Zero gradients"""
        self.optimizer.zero_grad(set_to_none)

    def state_dict(self):
        """Return optimizer state"""
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        """Load optimizer state"""
        self.optimizer.load_state_dict(state_dict)

    def param_groups(self):
        """Return parameter groups"""
        return self.optimizer.param_groups

    def get_lr(self):
        """Get current learning rate"""
        for param_group in self.param_groups:
            return param_group["lr"]

    def set_lr(self, lr: float):
        """Set learning rate"""
        for param_group in self.param_groups:
            param_group["lr"] = lr


def create_zero_optimizer(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: Optional[ZeroConfig] = None,
    device: Optional[torch.device] = None,
) -> ZeroOptimizer:
    """
    Create a ZeRO optimizer wrapper

    Args:
        model: The model to optimize
        optimizer: The base optimizer
        config: ZeRO configuration
        device: Device to use

    Returns:
        ZeroOptimizer wrapper
    """
    if config is None:
        config = ZeroConfig(stage=ZeroStage.STAGE_1)

    if device is None:
        device = next(model.parameters()).device

    return ZeroOptimizer(optimizer, config, device)


class MemoryCalculator:
    """Calculate memory savings from ZeRO"""

    @staticmethod
    def calculate_memory_savings(
        num_params: int,
        stage: ZeroStage,
        world_size: int = 1,
        dp_size: int = 1,
    ) -> Dict[str, float]:
        """
        Calculate memory savings from ZeRO optimization

        Args:
            num_params: Number of parameters in model
            stage: ZeRO stage
            world_size: Total world size
            dp_size: Data parallel size

        Returns:
            Dictionary with memory details
        """
        fp32_size = 4
        fp16_size = 2

        base_memory = num_params * fp32_size

        if stage == ZeroStage.DISABLED:
            return {
                "base_memory_gb": base_memory / 1e9,
                "saved_gb": 0,
                "reduction_ratio": 1.0,
            }

        if stage == ZeroStage.STAGE_1:
            reduction = 1 / dp_size
        elif stage == ZeroStage.STAGE_2:
            reduction = 1 / dp_size
        else:
            reduction = 1 / (dp_size * world_size)

        saved = base_memory * (1 - reduction)

        return {
            "base_memory_gb": base_memory / 1e9,
            "saved_gb": saved / 1e9,
            "reduction_ratio": reduction,
            "effective_memory_gb": (base_memory - saved) / 1e9,
        }

    @staticmethod
    def estimate_stage_for_available_memory(
        num_params: int,
        available_memory_gb: float,
        world_size: int = 1,
    ) -> ZeroStage:
        """
        Estimate which ZeRO stage can fit in available memory

        Args:
            num_params: Number of parameters
            available_memory_gb: Available GPU memory in GB
            world_size: World size

        Returns:
            Recommended ZeRO stage
        """
        fp16_size = 2
        model_memory = num_params * fp16_size / 1e9

        dp_size = world_size

        stage3_needed = model_memory * 1.2 / dp_size
        if stage3_needed <= available_memory_gb:
            return ZeroStage.STAGE_3

        stage2_needed = model_memory * 2.5 / dp_size
        if stage2_needed <= available_memory_gb:
            return ZeroStage.STAGE_2

        stage1_needed = model_memory * 4.0 / dp_size
        if stage1_needed <= available_memory_gb:
            return ZeroStage.STAGE_1

        return ZeroStage.DISABLED


def get_zero_config_from_dict(config: Dict[str, Any]) -> ZeroConfig:
    """Create ZeroConfig from dictionary"""
    return ZeroConfig(
        stage=ZeroStage(int(config.get("stage", 1))),
        reduce_bucket_size=config.get("reduce_bucket_size", 1e6),
        allgather_bucket_size=config.get("allgather_bucket_size", 1e6),
        overlap_comm=config.get("overlap_comm", True),
        contiguous_gradients=config.get("contiguous_gradients", True),
        round_robin_binaries=config.get("round_robin_binaries", True),
        reduce_scatter=config.get("reduce_scatter", True),
    )
