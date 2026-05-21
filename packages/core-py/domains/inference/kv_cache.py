"""
Unified Key-Value Cache for Transformer Inference.

Pre-allocated, position-based KV cache with per-layer tracking.
Used by InferenceEngine, InferenceOptimizer, and throughput optimization.
"""

from typing import List, Optional, Tuple, Union

try:
    import torch
except ImportError:
    from domains.training.slonet_compat import torch  # type: ignore[no-redef]


class KVCache:
    """
    Pre-allocated key-value cache for transformer autoregressive generation.

    Stores cached keys/values per layer as ``(1, num_heads, max_length, head_dim)``
    tensors.  Update via position-indexed slice assignment; retrieve via
    start/end slices or explicit position lists.

    Attributes
    ----------
    num_layers, num_heads, head_dim, max_length, dtype, device
        Constructor parameters.
    key_cache : List[torch.Tensor]
        ``num_layers`` entries, each ``(1, num_heads, max_length, head_dim)``.
    value_cache : List[torch.Tensor]
        Same shape as ``key_cache``.
    current_lengths : List[int]
        Current valid sequence length per layer (updated on each ``update``).
    """

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        max_length: int = 4096,
        dtype: torch.dtype = torch.float16,
        device: Union[str, torch.device] = "cpu",
    ):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_length = max_length
        self.device = torch.device(device) if isinstance(device, str) else device
        self.dtype = self._resolve_dtype(dtype)

        shape = (1, num_heads, max_length, head_dim)
        self.key_cache = [
            torch.zeros(shape, dtype=self.dtype, device=self.device) for _ in range(num_layers)
        ]
        self.value_cache = [
            torch.zeros(shape, dtype=self.dtype, device=self.device) for _ in range(num_layers)
        ]

        self.current_lengths = [0] * num_layers

    @staticmethod
    def _resolve_dtype(dtype: torch.dtype) -> torch.dtype:
        """Normalise a possibly-compat dtype to a real torch dtype."""
        if isinstance(dtype, torch.dtype):
            return dtype
        name = str(dtype)
        if name.startswith("torch."):
            name = name.split(".")[-1]
        return getattr(torch, name, torch.float16)

    def update(
        self,
        layer_idx: int,
        key: torch.Tensor,
        value: torch.Tensor,
        position: Optional[int] = None,
    ) -> None:
        """
        Store new key/value tensors for a given layer.

        Parameters
        ----------
        layer_idx : int
            Layer index (0 <= layer_idx < num_layers).
        key : torch.Tensor
            Shape ``(1, num_heads, seq_len, head_dim)``.
        value : torch.Tensor
            Same shape as ``key``.
        position : int, optional
            Starting position in the cache.  If ``None``, uses
            ``current_lengths[layer_idx]`` (append mode).
        """
        seq_len = key.shape[2]
        pos = position if position is not None else self.current_lengths[layer_idx]
        self.key_cache[layer_idx][:, :, pos:pos + seq_len, :] = key
        self.value_cache[layer_idx][:, :, pos:pos + seq_len, :] = value
        self.current_lengths[layer_idx] = max(self.current_lengths[layer_idx], pos + seq_len)

    def get(
        self,
        layer_idx: int,
        start: int = 0,
        end: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve cached keys/values for a layer.

        Parameters
        ----------
        layer_idx : int
            Layer index.
        start : int
            Start position (inclusive).
        end : int, optional
            End position (exclusive).  Defaults to ``current_lengths[layer_idx]``.

        Returns
        -------
        (key, value) tensors, each ``(1, num_heads, end-start, head_dim)``.
        """
        if end is None:
            end = self.current_lengths[layer_idx]
        return (
            self.key_cache[layer_idx][:, :, start:end, :],
            self.value_cache[layer_idx][:, :, start:end, :],
        )

    def get_at_positions(
        self,
        layer_idx: int,
        positions: Union[List[int], torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve cached keys/values at specific positions (scatter-read).

        Parameters
        ----------
        layer_idx : int
            Layer index.
        positions : list[int] or torch.Tensor
            Positions to read.

        Returns
        -------
        (key, value) tensors, each ``(1, num_heads, len(positions), head_dim)``.
        """
        if isinstance(positions, torch.Tensor):
            pos_list = positions.tolist() if hasattr(positions, 'tolist') else list(positions)
        else:
            pos_list = positions
        ks = [self.key_cache[layer_idx][:, :, p:p + 1, :] for p in pos_list]
        vs = [self.value_cache[layer_idx][:, :, p:p + 1, :] for p in pos_list]
        return torch.cat(ks, dim=2), torch.cat(vs, dim=2)

    def update_at_positions(
        self,
        layer_idx: int,
        positions: Union[List[int], torch.Tensor],
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        """
        Store key/value tensors at specific positions (scatter-write).

        Parameters
        ----------
        layer_idx : int
            Layer index.
        positions : list[int] or torch.Tensor
            Positions to write.
        key : torch.Tensor
            Shape ``(1, num_heads, seq_len, head_dim)`` where
            ``seq_len == len(positions)``.
        value : torch.Tensor
            Same shape as ``key``.
        """
        if isinstance(positions, torch.Tensor):
            pos_list = positions.tolist() if hasattr(positions, 'tolist') else list(positions)
        else:
            pos_list = positions
        for i, p in enumerate(pos_list):
            self.key_cache[layer_idx][:, :, p, :] = key[:, :, i:i + 1, :]
            self.value_cache[layer_idx][:, :, p, :] = value[:, :, i:i + 1, :]
        max_pos = max(pos_list) + 1 if pos_list else 0
        self.current_lengths[layer_idx] = max(self.current_lengths[layer_idx], max_pos)

    def reset(self) -> None:
        """Zero all cached values and reset length trackers."""
        for i in range(self.num_layers):
            self.key_cache[i].zero_()
            self.value_cache[i].zero_()
        self.current_lengths = [0] * self.num_layers

    def memory_used_mb(self) -> float:
        """Return total memory consumed by the cache in MB."""
        per_layer = self.key_cache[0].element_size() * self.key_cache[0].numel()
        return (per_layer * self.num_layers * 2) / (1024 * 1024)

    @property
    def total_tokens_cached(self) -> int:
        """Total number of token positions currently cached across all layers."""
        return sum(self.current_lengths)


__all__ = ["KVCache"]
