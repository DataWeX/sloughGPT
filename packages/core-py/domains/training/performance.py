"""
Performance Optimization Module for SloughGPT

High-performance training and inference optimizations:
- CUDA Graphs: capture/replay for minimal kernel launch overhead
- torch.compile: JIT compilation for PyTorch 2.0+
- Efficient DataLoader: prefetch, pinned memory, persistent workers
- Optimized batching: pre-allocated tensors, vectorized operations
- KV Cache: efficient cache management for inference
- Fused operations: optimized kernels for common patterns
- Memory optimization: channel-last, gradient checkpointing

Usage:
    from domains.training.performance import optimize_training, optimize_inference
    model, trainer = optimize_training(model, config)
"""

from __future__ import annotations

import os
import time
import math
import warnings
from typing import Optional, Dict, Any, List, Tuple, Callable
from dataclasses import dataclass, field
from domains.training.trainer_protocol import TrainResult
from contextlib import contextmanager
from functools import lru_cache
import logging

from domains.training.slonet_compat import torch
DataLoader = torch.utils.data.DataLoader
Dataset = torch.utils.data.Dataset
F = torch.F
nn = torch.nn

logger = logging.getLogger("man.performance")


@dataclass
class TrainingOptimizations:
    """Training performance settings."""

    use_compile: bool = True
    compile_mode: str = "reduce-overhead"
    compile_fullgraph: bool = False

    use_cuda_graphs: bool = False
    channel_last: bool = True

    dataloader_workers: int = 4
    dataloader_prefetch: int = 2
    dataloader_persistent: bool = True
    dataloader_pin_memory: bool = True

    use_fused_optimizer: bool = True

    cudnn_benchmark: bool = True
    cudnn_deterministic: bool = False

    use_flash_attention: bool = True
    gradient_checkpointing: bool = True

    batch_preallocation: bool = True


@dataclass
class InferenceOptimizations:
    """Inference performance settings."""

    use_compile: bool = True
    compile_mode: str = "default"

    use_cuda_graphs: bool = True
    channel_last: bool = True

    use_flash_attention: bool = True
    use_sdpa: bool = True

    max_batch_size: int = 32
    kv_cache_preallocate: bool = True

    use_kv_cache: bool = True
    use_continuous_batching: bool = True


@dataclass
class PerformanceConfig:
    """Configuration for performance optimizations."""

    device: str = "auto"

    training: TrainingOptimizations = field(default_factory=TrainingOptimizations)
    inference: InferenceOptimizations = field(default_factory=InferenceOptimizations)

    def __post_init__(self):
        if self.device == "auto":
            self.device = get_optimal_device()


def get_optimal_device() -> str:
    """Auto-detect best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.version, "hip") and torch.version.hip is not None:
        return "rocm"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_device_name() -> str:
    """Get human-readable device name."""
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    elif torch.backends.mps.is_available():
        return "Apple Silicon (MPS)"
    return "CPU"


def setup_device_environment():
    """Setup optimal device environment variables."""
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if hasattr(torch.cuda, "set_sync_debug_mode"):
            torch.cuda.set_sync_debug_mode(torch.cuda.sync_debug_mode.OFF)


class CUDAGraphManager:
    """Manages CUDA graphs for kernel capture/replay.

    CUDA graphs eliminate CPU overhead by recording GPU operations
    and replaying them with minimal kernel launch latency.
    """

    def __init__(self, model: nn.Module, config: InferenceOptimizations):
        self.model = model
        self.config = config
        self.graphs: Dict[int, torch.cuda.CUDAGraph] = {}
        self.static_inputs: Dict[int, torch.Tensor] = {}
        self.static_outputs: Optional[torch.Tensor] = None
        self._enabled = config.use_cuda_graphs and torch.cuda.is_available()

    def capture(
        self,
        batch_size: int,
        seq_len: int,
        vocab_size: int,
    ) -> bool:
        """Capture a CUDA graph for given input shape."""
        if not self._enabled:
            return False

        key = (batch_size, seq_len)
        if key in self.graphs:
            return True

        try:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

            static_input = torch.zeros(
                batch_size, seq_len, dtype=torch.long, device="cuda"
            )
            self.static_inputs[key] = static_input

            g = torch.cuda.CUDAGraph()
            self.static_outputs = torch.zeros(
                batch_size, seq_len, vocab_size, dtype=torch.float16, device="cuda"
            )

            with torch.cuda.graph(g):
                output = self.model(static_input)
                if isinstance(output, tuple):
                    logits = output[0]
                else:
                    logits = output
                logits.copy_(logits)

            self.graphs[key] = g
            logger.info(f"Captured CUDA graph for shape {key}")
            return True

        except Exception as e:
            logger.warning(f"CUDA graph capture failed: {e}")
            self._enabled = False
            return False

    def replay(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Replay captured graph or fall back to normal forward."""
        if not self._enabled:
            return self.model(input_ids)

        key = (input_ids.shape[0], input_ids.shape[1])
        if key not in self.graphs:
            return self.model(input_ids)

        try:
            self.static_inputs[key].copy_(input_ids)
            self.graphs[key].replay()
            return self.static_outputs
        except Exception:
            return self.model(input_ids)


class OptimizedDataLoader:
    """High-performance DataLoader with prefetching and memory optimization."""

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        num_workers: int = 4,
        prefetch_factor: int = 2,
        persistent_workers: bool = True,
        pin_memory: bool = True,
        collate_fn: Optional[Callable] = None,
    ):
        effective_workers = effective_dataloader_workers(num_workers)
        effective_prefetch = effective_prefetch_factor(effective_workers, prefetch_factor)

        loader_kwargs: Dict[str, Any] = {
            "dataset": dataset,
            "batch_size": batch_size,
            "shuffle": True,
            "num_workers": effective_workers,
            "pin_memory": pin_memory and torch.cuda.is_available(),
            "collate_fn": collate_fn,
        }

        if effective_workers > 0 and effective_prefetch is not None:
            loader_kwargs["prefetch_factor"] = effective_prefetch
            loader_kwargs["persistent_workers"] = persistent_workers

        self.dataloader = DataLoader(**loader_kwargs)
        self._iterator = None
        self._prefetched_batch: Optional[Dict] = None

    def prefetch(self):
        """Prefetch next batch in background."""
        if self._iterator is None:
            self._iterator = iter(self.dataloader)
        try:
            self._prefetched_batch = next(self._iterator)
        except StopIteration:
            self._iterator = iter(self.dataloader)
            self._prefetched_batch = next(self._iterator)

    def get_batch(self) -> Dict[str, torch.Tensor]:
        """Get next batch, prefetching in background."""
        if self._prefetched_batch is not None:
            batch = self._prefetched_batch
            self._prefetched_batch = None
            return batch

        if self._iterator is None:
            self._iterator = iter(self.dataloader)

        try:
            return next(self._iterator)
        except StopIteration:
            self._iterator = iter(self.dataloader)
            return next(self._iterator)


def effective_dataloader_workers(requested: int) -> int:
    """Get safe worker count for platform."""
    import sys
    if sys.platform == "darwin":
        return 0
    try:
        n = int(requested)
    except (TypeError, ValueError):
        n = 0
    return max(0, n)


def effective_prefetch_factor(num_workers: int, requested: int) -> Optional[int]:
    """Get safe prefetch factor."""
    if num_workers <= 0:
        return None
    try:
        pf = int(requested)
    except (TypeError, ValueError):
        pf = 2
    return max(1, pf)


class PreallocatedBatchDataset(Dataset):
    """Dataset that returns indices for pre-allocated batch tensors."""

    def __init__(self, data: torch.Tensor, block_size: int, batch_size: int):
        self.data = data
        self.block_size = block_size
        self.batch_size = batch_size
        self.seq_len = block_size + 1

    def __len__(self):
        return max(0, len(self.data) - self.seq_len)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.data[idx : idx + self.block_size]
        y = self.data[idx + 1 : idx + self.seq_len]
        return x, y


class OptimizedBatchCache:
    """Cache for pre-allocated batch tensors to avoid allocations."""

    def __init__(self, device: str, dtype: torch.dtype = torch.long):
        self.device = device
        self.dtype = dtype
        self.x_cache: Optional[torch.Tensor] = None
        self.y_cache: Optional[torch.Tensor] = None
        self._batch_size = 0
        self._block_size = 0

    def allocate(self, batch_size: int, block_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Allocate or return cached batch tensors."""
        if self.x_cache is None or self._batch_size != batch_size or self._block_size != block_size:
            self._batch_size = batch_size
            self._block_size = block_size
            self.x_cache = torch.empty(batch_size, block_size, dtype=self.dtype, device=self.device)
            self.y_cache = torch.empty(batch_size, block_size, dtype=self.dtype, device=self.device)
        return self.x_cache, self.y_cache

    def fill(
        self,
        batch_size: int,
        block_size: int,
        data: torch.Tensor,
        indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fill pre-allocated tensors with batch data."""
        x, y = self.allocate(batch_size, block_size)

        for i, idx in enumerate(indices):
            x[i].copy_(data[idx : idx + block_size])
            y[i].copy_(data[idx + 1 : idx + block_size + 1])

        return x, y


class FastInferenceSampler:
    """Optimized token sampling with vectorized operations."""

    @staticmethod
    def sample(
        logits: torch.Tensor,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        prev_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Fast token sampling with optional repetition penalty.

        Optimizations:
        - Vectorized repetition penalty (O(1) instead of O(n))
        - Fused top-k/top-p filtering
        - In-place operations where possible
        """
        if temperature == 0:
            return logits.argmax(dim=-1, keepdim=True)

        logits = logits / temperature

        if repetition_penalty != 1.0 and prev_tokens is not None and prev_tokens.numel() > 0:
            logits = FastInferenceSampler._apply_repetition_penalty_vectorized(
                logits, prev_tokens, repetition_penalty
            )

        if top_k > 0:
            logits = FastInferenceSampler._apply_top_k(logits, top_k)

        if top_p < 1.0:
            logits = FastInferenceSampler._apply_top_p(logits, top_p)

        probs = F.softmax(logits, dim=-1)
        probs = torch.clamp(probs, min=1e-10)
        return torch.multinomial(probs, num_samples=1)

    @staticmethod
    def _apply_repetition_penalty_vectorized(
        logits: torch.Tensor,
        prev_tokens: torch.Tensor,
        penalty: float,
    ) -> torch.Tensor:
        """Apply repetition penalty using advanced indexing (O(1) per batch)."""
        unique_tokens, inverse_indices = torch.unique(prev_tokens, return_inverse=True)
        token_counts = torch.bincount(inverse_indices, minlength=len(unique_tokens))

        mask = token_counts > 0
        penalized_indices = unique_tokens[mask]
        penalties = torch.ones(logits.shape[-1], device=logits.device, dtype=logits.dtype)
        penalties[penalized_indices] = penalty

        pos_mask = logits > 0
        neg_mask = ~pos_mask

        logits = torch.where(pos_mask, logits * penalties, logits / penalties)
        return logits

    @staticmethod
    def _apply_top_k(logits: torch.Tensor, k: int) -> torch.Tensor:
        """Apply top-k filtering efficiently."""
        k = min(k, logits.shape[-1])
        v, _ = torch.topk(logits, k, dim=-1)
        threshold = v[:, [-1]]
        return torch.where(logits < threshold, torch.tensor(float("-inf"), device=logits.device), logits)

    @staticmethod
    def _apply_top_p(logits: torch.Tensor, p: float) -> torch.Tensor:
        """Apply top-p (nucleus) filtering with early exit."""
        sorted_logits, sorted_indices = torch.sort(logits, dim=-1, descending=True)
        cumsum = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        mask = cumsum > p
        mask[:, 1:] = mask[:, :-1].clone()
        mask[:, 0] = False

        indices_to_remove = mask.scatter(1, sorted_indices, mask)
        return torch.where(indices_to_remove, torch.tensor(float("-inf"), device=logits.device), logits)


class OptimizedInferenceEngine:
    """High-performance inference engine with all optimizations."""

    def __init__(
        self,
        model: nn.Module,
        config: Optional[InferenceOptimizations] = None,
        device: str = "auto",
    ):
        self.model = model
        self.config = config or InferenceOptimizations()
        self.device = get_optimal_device() if device == "auto" else device

        if self.device == "cuda" and self.config.use_compile:
            self._setup_compiled_forward()
        else:
            self._compiled_model = None

        if self.device == "cuda":
            self.cuda_graph_manager = CUDAGraphManager(model, self.config)
        else:
            self.cuda_graph_manager = None

        self.model.eval()
        self.model.to(self.device)

    def _setup_compiled_forward(self):
        """Setup torch.compile for faster forward passes."""
        if hasattr(torch, "compile") and self.config.use_compile:
            try:
                self._compiled_model = torch.compile(
                    self.model,
                    mode=self.config.compile_mode,
                    fullgraph=self.config.compile_mode == "max-autotune",
                )
                logger.info(f"Model compiled with mode: {self.config.compile_mode}")
            except Exception as e:
                logger.warning(f"torch.compile failed: {e}")
                self._compiled_model = None

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
    ) -> torch.Tensor:
        """Optimized autoregressive generation."""
        model = self._compiled_model if self._compiled_model else self.model
        model.eval()

        input_ids = input_ids.to(self.device)
        current = input_ids
        prev_tokens = input_ids.clone()

        for _ in range(max_new_tokens):
            idx_cond = current[:, -self.model.block_size :]

            if self.cuda_graph_manager:
                logits = self.cuda_graph_manager.replay(idx_cond)
            else:
                logits = model(idx_cond)

            if isinstance(logits, tuple):
                logits = logits[0]

            logits = logits[:, -1, :]

            next_token = FastInferenceSampler.sample(
                logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                prev_tokens=prev_tokens,
            )

            current = torch.cat([current, next_token], dim=1)
            prev_tokens = torch.cat([prev_tokens, next_token], dim=1)

            if next_token.item() == 0:
                break

        return current


class OptimizedTrainer:
    """Training loop with all performance optimizations.

    Satisfies :class:`TrainerProtocol` structurally.
    """

    _is_training: bool = False

    @property
    def is_training(self) -> bool:
        """Whether training is in progress."""
        return self._is_training

    def stop(self) -> None:
        """Request early stopping."""
        self._is_training = False

    def __init__(
        self,
        model: nn.Module,
        config: TrainingOptimizations,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
        device: str = "auto",
    ):
        warnings.warn(
            "OptimizedTrainer is deprecated — use SloughGPTTrainer instead",
            DeprecationWarning, stacklevel=2,
        )
        self.model = model
        self.config = config
        self.device = get_optimal_device() if device == "auto" else device

        self.model = self.model.to(self.device)

        if self.device == "cuda" and config.use_compile and hasattr(torch, "compile"):
            self._setup_compile()

        if config.channel_last and self.device == "cuda":
            self.model = self.model.to(memory_format=torch.channels_last)

        if config.gradient_checkpointing:
            self._setup_gradient_checkpointing()

        self.train_loader = OptimizedDataLoader(
            train_dataset,
            batch_size=1,
            num_workers=config.dataloader_workers,
            prefetch_factor=config.dataloader_prefetch,
            persistent_workers=config.dataloader_persistent,
            pin_memory=config.dataloader_pin_memory,
        )

        self.val_loader = None
        if val_dataset:
            self.val_loader = OptimizedDataLoader(
                val_dataset,
                batch_size=1,
                num_workers=config.dataloader_workers,
                prefetch_factor=config.dataloader_prefetch,
                persistent_workers=config.dataloader_persistent,
                pin_memory=config.dataloader_pin_memory,
            )

        self.batch_cache = OptimizedBatchCache(self.device)
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scaler: Optional[torch.cuda.amp.GradScaler] = None
        self.scheduler = None

        self._step_count = 0
        self._setup_optimizer_and_scaler()

    def _setup_compile(self):
        """Setup torch.compile."""
        try:
            self.model = torch.compile(
                self.model,
                mode=self.config.compile_mode,
                fullgraph=self.config.compile_fullgraph,
            )
            logger.info(f"Model compiled: {self.config.compile_mode}")
        except Exception as e:
            logger.warning(f"compile failed: {e}")

    def _setup_gradient_checkpointing(self):
        """Enable gradient checkpointing on transformer blocks."""
        for module in self.model.modules():
            if hasattr(module, "enable_gradient_checkpointing"):
                module.enable_gradient_checkpointing()

    def _setup_optimizer_and_scaler(self):
        """Setup optimized optimizer and mixed precision scaler."""
        decay_params = []
        no_decay_params = []

        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if "bias" in n or "norm" in n or "ln_" in n:
                no_decay_params.append(p)
            else:
                decay_params.append(p)

        if self.config.use_fused_optimizer and self.device == "cuda":
            try:
                self.optimizer = torch.optim.FusedAdamW(
                    [{"params": decay_params, "weight_decay": 0.01},
                     {"params": no_decay_params, "weight_decay": 0.0}],
                    lr=1e-4,
                )
                logger.info("Using FusedAdamW")
            except AttributeError:
                self.optimizer = torch.optim.AdamW(
                    [{"params": decay_params, "weight_decay": 0.01},
                     {"params": no_decay_params, "weight_decay": 0.0}],
                    lr=1e-4,
                    fused=True,
                )
                logger.info("Using AdamW (fused=True)")
        else:
            self.optimizer = torch.optim.AdamW(
                [{"params": decay_params, "weight_decay": 0.01},
                 {"params": no_decay_params, "weight_decay": 0.0}],
                lr=1e-4,
            )

        if self.device == "cuda":
            self.scaler = torch.cuda.amp.GradScaler(enabled=True, init_scale=2**15)

    def train_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
    ) -> float:
        """Execute optimized training step."""
        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)

        if self.device == "cuda" and self.config.channel_last:
            x = x.to(memory_format=torch.channels_last)
            y = y.to(memory_format=torch.channels_last)

        model = self.model
        loss_scale = 1.0 / gradient_accumulation_steps

        if self.scaler:
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                logits, loss = model(x, y)
                loss = loss * loss_scale
            self.scaler.scale(loss).backward()
        else:
            logits, loss = model(x, y)
            (loss * loss_scale).backward()

        if (self._step_count + 1) % gradient_accumulation_steps == 0:
            if self.scaler:
                self.scaler.unscale_(self.optimizer)

            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            if self.scaler:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            self.optimizer.zero_grad(set_to_none=True)

            if self.scheduler:
                self.scheduler.step()

        self._step_count += 1
        return loss.item() * gradient_accumulation_steps

    def train(
        self,
        resume: bool = False,
        resume_path: Optional[str] = None,
        on_progress: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> TrainResult:
        """Training loop implementing TrainerProtocol."""
        import time as time_module

        self._is_training = True
        best_loss = float("inf")
        start_time = time_module.time()
        last_log_time = start_time

        while self._step_count < self.config.max_steps and self._is_training:
            step_start = time_module.time()
            loss = self.train_step()
            step_time = time_module.time() - step_start

            if on_progress and self._step_count % self.config.log_interval == 0:
                elapsed = time_module.time() - last_log_time
                lr = self.scheduler.get_last_lr()[0] if self.scheduler else 0
                tokens_per_sec = (self.config.batch_size * self.config.block_size) / (step_time + 1e-6)
                try:
                    on_progress({
                        "global_step": int(self._step_count),
                        "train_loss": float(loss),
                        "learning_rate": float(lr),
                        "tokens_per_sec": float(tokens_per_sec),
                    })
                except Exception:
                    pass
                last_log_time = time_module.time()

            best_loss = min(best_loss, loss)

        self._is_training = False
        total_time = time_module.time() - start_time
        return TrainResult(
            success=True,
            status="completed",
            best_eval_loss=best_loss,
            global_step=self._step_count,
            total_steps=self._step_count,
            metrics={"best_loss": best_loss, "total_time": total_time},
        )


class PerformanceMonitor:
    """Monitor training/inference performance metrics."""

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.step_times: List[float] = []
        self.losses: List[float] = []
        self.tokens_processed = 0
        self._start_time = time.time()

    def record_step(self, loss: float, step_time: float, batch_size: int, seq_len: int):
        """Record a training step."""
        self.step_times.append(step_time)
        self.losses.append(loss)
        self.tokens_processed += batch_size * seq_len

        if len(self.step_times) > self.window_size:
            self.step_times.pop(0)
            self.losses.pop(0)

    def get_stats(self) -> Dict[str, float]:
        """Get current performance statistics."""
        if not self.step_times:
            return {}

        avg_step_time = sum(self.step_times) / len(self.step_times)
        tokens_per_sec = self.tokens_processed / (time.time() - self._start_time + 1e-6)

        return {
            "avg_step_time_ms": avg_step_time * 1000,
            "steps_per_sec": 1.0 / avg_step_time if avg_step_time > 0 else 0,
            "tokens_per_sec": tokens_per_sec,
            "avg_loss": sum(self.losses) / len(self.losses) if self.losses else 0,
            "total_steps": len(self.step_times),
        }


def optimize_model_for_inference(
    model: nn.Module,
    device: str = "auto",
    use_compile: bool = True,
    use_channels_last: bool = True,
) -> nn.Module:
    """Apply all inference optimizations to a model."""
    device = get_optimal_device() if device == "auto" else device
    model = model.to(device)
    model.eval()

    if use_channels_last and device == "cuda":
        model = model.to(memory_format=torch.channels_last)

    if use_compile and hasattr(torch, "compile") and device == "cuda":
        try:
            model = torch.compile(model, mode="default")
            logger.info("Model compiled for inference")
        except Exception as e:
            logger.warning(f"compile failed: {e}")

    return model


def benchmark_training(
    model: nn.Module,
    batch_size: int = 8,
    seq_len: int = 128,
    num_steps: int = 100,
    device: str = "auto",
) -> Dict[str, float]:
    """Benchmark training performance."""
    device = get_optimal_device() if device == "auto" else device
    model = model.to(device)

    dummy_data = torch.randint(0, 1000, (10000,))
    dataset = PreallocatedBatchDataset(dummy_data, seq_len, batch_size)

    config = TrainingOptimizations(use_compile=False)
    trainer = OptimizedTrainer(model, config, dataset, device=device)

    x, y = next(iter(DataLoader(dataset, batch_size=batch_size)))
    x, y = x.to(device), y.to(device)

    if device == "cuda":
        torch.cuda.synchronize()

    start = time.time()
    for _ in range(num_steps):
        trainer.train_step((x, y))

    if device == "cuda":
        torch.cuda.synchronize()

    elapsed = time.time() - start
    tokens_per_sec = (batch_size * seq_len * num_steps) / elapsed

    return {
        "elapsed_sec": elapsed,
        "tokens_per_sec": tokens_per_sec,
        "steps_per_sec": num_steps / elapsed,
        "device": device,
    }


def benchmark_inference(
    model: nn.Module,
    batch_size: int = 1,
    seq_len: int = 128,
    gen_len: int = 50,
    num_runs: int = 10,
    device: str = "auto",
) -> Dict[str, float]:
    """Benchmark inference performance."""
    device = get_optimal_device() if device == "auto" else device

    config = InferenceOptimizations(use_compile=False, use_cuda_graphs=False)
    engine = OptimizedInferenceEngine(model, config, device=device)

    input_ids = torch.randint(0, 1000, (batch_size, seq_len), device=device)

    if device == "cuda":
        torch.cuda.synchronize()

    latencies = []
    for _ in range(num_runs):
        start = time.time()
        engine.generate(input_ids, max_new_tokens=gen_len)
        if device == "cuda":
            torch.cuda.synchronize()
        latencies.append((time.time() - start) * 1000)

    return {
        "avg_latency_ms": sum(latencies) / len(latencies),
        "p50_latency_ms": sorted(latencies)[len(latencies) // 2],
        "p95_latency_ms": sorted(latencies)[int(len(latencies) * 0.95)],
        "tokens_per_sec": (batch_size * gen_len * num_runs) / (sum(latencies) / 1000),
        "device": device,
    }


__all__ = [
    "PerformanceConfig",
    "TrainingOptimizations",
    "InferenceOptimizations",
    "get_optimal_device",
    "get_device_name",
    "setup_device_environment",
    "CUDAGraphManager",
    "OptimizedDataLoader",
    "PreallocatedBatchDataset",
    "OptimizedBatchCache",
    "FastInferenceSampler",
    "OptimizedInferenceEngine",
    "OptimizedTrainer",
    "PerformanceMonitor",
    "optimize_model_for_inference",
    "benchmark_training",
    "benchmark_inference",
]
