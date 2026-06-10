"""
Unified Key-Value Cache for Transformer Inference.

Pre-allocated, position-based KV cache with per-layer tracking.
Used by InferenceEngine, InferenceOptimizer, and throughput optimization.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

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


def reconstruct_sequence(
    cache: KVCache,
    layer_subset: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    Reconstruct a token-sequence view from the KVCache by correlating
    position indices across layers with stored key/value attention patterns.

    The KV cache stores per-position key and value projections — this function
    rebuilds a structured view of the cached sequence, including:

    - ``positions``: list of cached position indices per layer
    - ``sequence_length``: max contiguous sequence length across layers
    - ``layer_map``: positions cached per layer
    - ``attention_similarity``: pairwise cosine similarity between adjacent
      cached key-vectors (flattened across heads) — useful for detecting
      topic shifts, sentence boundaries, or repeated content.
    - ``value_norms``: L2 norm of value vectors per position per layer
      (high norms may indicate salient tokens).

    Parameters
    ----------
    cache : KVCache
        Populated KV cache instance.
    layer_subset : list[int], optional
        If provided, only examine these layers.  Defaults to all.

    Returns
    -------
    dict with the fields described above.

    Notes
    -----
    True token reconstruction is impossible from KVs alone — keys and values
    are attention projections of hidden states, not token embeddings.  This
    function provides the closest available approximation for debugging and
    cache-inspection purposes.
    """
    layers = layer_subset or list(range(cache.num_layers))
    has_kv = all(
        hasattr(cache, attr) and len(getattr(cache, attr)) == cache.num_layers
        for attr in ("key_cache", "value_cache")
    )
    if not has_kv:
        return {"positions": [], "sequence_length": 0, "layer_map": {}, "attention_similarity": [], "value_norms": {}}

    max_pos = cache.num_layers > 0 and max((cache.current_lengths or [0]))

    layer_map: Dict[int, List[int]] = {}
    for li in layers:
        length = cache.current_lengths[li] if li < len(cache.current_lengths) else 0
        layer_map[li] = list(range(length))

    seq_len = max((len(v) for v in layer_map.values()), default=0)

    value_norms: Dict[int, List[float]] = {}
    for li in layers:
        vals = cache.value_cache[li]
        norms = []
        for pos in range(cache.current_lengths[li] if li < len(cache.current_lengths) else 0):
            v = vals[:, :, pos, :]
            norm = float(v.float().pow(2).sum().sqrt().item())
            norms.append(round(norm, 4))
        value_norms[li] = norms

    sim_matrix: List[Dict[str, Any]] = []
    if seq_len >= 2 and layers:
        ref_layer = layers[0]
        k = cache.key_cache[ref_layer]
        length = cache.current_lengths[ref_layer] if ref_layer < len(cache.current_lengths) else 0
        for i in range(1, min(length, seq_len)):
            a = k[:, :, i - 1, :].float().flatten()
            b = k[:, :, i, :].float().flatten()
            cos = float(torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())
            sim_matrix.append({"from": i - 1, "to": i, "cosine_similarity": round(cos, 4)})

    return {
        "positions": list(range(seq_len)),
        "sequence_length": seq_len,
        "layer_map": {str(k): v for k, v in layer_map.items()},
        "attention_similarity": sim_matrix,
        "value_norms": value_norms,
    }


def reconstruct_tokens(
    kv_cache: KVCache,
    layer_indices: Optional[List[int]] = None,
    similarity_threshold: float = 0.85,
) -> Dict[str, Any]:
    """
    Reconstruct token sequence structure from cached key/value representations.

    KV caches store projected key/value vectors per position per layer — not
    the original tokens themselves. This function infers sequence structure by
    computing cross-position cosine similarity between key vectors: positions
    with highly similar keys likely belong to the same token or repeated content.

    Parameters
    ----------
    kv_cache : KVCache
        Populated key-value cache instance.
    layer_indices : list of int, optional
        Layers to include in the analysis. Defaults to all layers.
    similarity_threshold : float, optional
        Cosine similarity threshold (0-1) for flagging duplicate positions.
        Default 0.85.

    Returns
    -------
    dict with keys:
        - ``total_tokens``: max cached length across layers
        - ``layers``: per-layer metadata (length, shape)
        - ``similarity_matrix``: ``(T, T)`` float matrix of mean cross-position
          cosine similarity (averaged across selected layers), where T is the
          max cached length. Upper-triangular; diagonal is 1.0.
        - ``duplicates``: list of ``(pos_a, pos_b, similarity)`` tuples where
          similarity >= threshold
        - ``attention_context``: per-position dict mapping ``pos -> {layer ->
          (key_norm, value_norm)}`` with L2 norms of cached vectors
    """
    if layer_indices is None:
        layer_indices = list(range(kv_cache.num_layers))

    max_len = max(kv_cache.current_lengths)
    if max_len == 0:
        return {"total_tokens": 0, "layers": {}, "similarity_matrix": [], "duplicates": [], "attention_context": {}}

    n_layers = len(layer_indices)
    all_keys: List[torch.Tensor] = []
    for li in layer_indices:
        k, _ = kv_cache.get(li, 0, max_len)
        all_keys.append(k)

    # Build similarity matrix: (T, T) averaged across layers
    T = all_keys[0].shape[2]
    sim_matrix = torch.zeros((T, T), dtype=torch.float32)
    for k in all_keys:
        flat = k[0].transpose(0, 1).reshape(k.shape[1], -1).unsqueeze(0)
        norms = flat.norm(dim=-1, keepdim=True)
        flat_normed = flat / (norms + 1e-8)
        sim = (flat_normed @ flat_normed.transpose(-2, -1)).squeeze(0)
        sim_matrix += sim
    sim_matrix /= n_layers

    duplicates = []
    for i in range(T):
        for j in range(i + 1, T):
            s = float(sim_matrix[i, j])
            if s >= similarity_threshold:
                duplicates.append((i, j, round(s, 4)))

    attention_context: Dict[int, Dict[str, Any]] = {}
    for pos in range(max_len):
        pos_ctx: Dict[str, Any] = {}
        for li in layer_indices:
            k_slice, v_slice = kv_cache.get(li, pos, pos + 1)
            pos_ctx[str(li)] = {
                "key_norm": round(float(k_slice.norm().item()), 4),
                "value_norm": round(float(v_slice.norm().item()), 4),
            }
        attention_context[pos] = pos_ctx

    layers_meta = {}
    for li in layer_indices:
        layers_meta[str(li)] = {
            "length": kv_cache.current_lengths[li],
            "shape": list(kv_cache.key_cache[li].shape),
        }

    return {
        "total_tokens": max_len,
        "layers": layers_meta,
        "similarity_matrix": sim_matrix.tolist(),
        "duplicates": duplicates,
        "attention_context": attention_context,
    }


__all__ = ["KVCache", "reconstruct_sequence", "reconstruct_tokens"]
