"""
Torch-free .pt checkpoint loader — parses PyTorch serialization format directly.

Reads .pt files (ZIP with pickle + raw tensor bytes) without importing torch.
Parses pickle opcodes to extract tensor metadata, reads raw data files,
and reconstructs numpy arrays.

Usage:
    from domains.infrastructure.pt_loader import load_pt_checkpoint
    state_dict = load_pt_checkpoint("models/my_model.pt")
"""

import io
import logging
import pickle
import pickletools
import zipfile
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger("man.infrastructure.pt_loader")


def _parse_pickle_ops(pkl_data: bytes) -> list:
    """Parse pickle opcodes, returning (opcode_name, arg, pos) tuples."""
    f = io.BytesIO(pkl_data)
    return [(op.name, arg, pos) for op, arg, pos in pickletools.genops(f)]


def _extract_param_names(ops: list) -> list:
    """Extract parameter names (strings with dots) from pickle opcodes."""
    param_names = []
    seen_rebuild = False
    for op_name, arg, pos in ops:
        if op_name in ("SHORT_BINUNICODE", "BINUNICODE"):
            if isinstance(arg, str):
                if "_rebuild_tensor" in arg:
                    seen_rebuild = True
                elif "." in arg and not arg.startswith("torch"):
                    param_names.append(arg)
    return param_names


class _PTUnpickler(pickle.Unpickler):
    """Unpickler that reconstructs torch tensors as numpy arrays.

    Intercepts:
      - persistent_load (storage references) → raw bytes
      - _rebuild_tensor_v2 → numpy array from raw bytes + shape
      - Storage classes → no-ops
    """

    def __init__(self, file, data_arrays: Dict[str, bytes]):
        super().__init__(file)
        self._data = data_arrays
        self._tensors = []  # (name, array) pairs collected in order

    def persistent_load(self, pid):
        if isinstance(pid, tuple) and len(pid) >= 3 and pid[0] == "storage":
            key = str(pid[2])
            raw = self._data.get(key, b"")
            return ("__storage__", key, raw)
        return pid

    def find_class(self, module, name):
        if "_rebuild_tensor_v2" in name:
            return self._rebuild
        if "Storage" in name:
            return lambda *a, **k: None
        if "OrderedDict" in name:
            return OrderedDict
        return super().find_class(module, name)

    def _rebuild(self, *args):
        # args: (storage, offset, size, stride, requires_grad, [metadata_dict])
        storage = args[0]
        offset = args[1] if len(args) > 1 else 0
        size = args[2] if len(args) > 2 else None
        stride = args[3] if len(args) > 3 else None

        if not isinstance(storage, tuple) or storage[0] != "__storage__":
            return storage

        raw = storage[2]
        n_bytes = len(raw)

        # Infer dtype from byte count and expected element count
        n_elements = 1
        if size:
            for s in size:
                if isinstance(s, int):
                    n_elements *= s

        if n_elements > 0 and n_bytes > 0:
            elem_bytes = n_bytes // n_elements
            dtype_map = {1: np.uint8, 2: np.float16, 4: np.float32, 8: np.float64}
            dtype = dtype_map.get(elem_bytes, np.float32)
        else:
            dtype = np.float32

        arr = np.frombuffer(raw, dtype=dtype)

        if offset > 0:
            arr = arr[offset:]

        if size:
            try:
                arr = arr.reshape(size)
            except ValueError:
                pass

        return arr


def load_pt_checkpoint(
    path: Union[str, Path],
    map_location: str = "cpu",
) -> Dict[str, Any]:
    """
    Load a PyTorch .pt checkpoint as a dict of numpy arrays.

    Works WITHOUT torch installed. Parses the ZIP/pickle format directly.

    Args:
        path: Path to .pt file
        map_location: Ignored (always numpy/CPU)

    Returns:
        Dict mapping parameter names to numpy arrays.
        If the checkpoint has a 'model' key, returns the nested dict.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    with zipfile.ZipFile(path, "r") as z:
        # Read all raw tensor data files
        data_arrays = {}
        for name in z.namelist():
            if "/data/" in name and not name.endswith(".pkl"):
                key = name.split("/")[-1]
                data_arrays[key] = z.read(name)

        # Find the pickle file
        pkl_files = [n for n in z.namelist() if n.endswith(".pkl")]
        if not pkl_files:
            raise ValueError(f"No pickle file found in {path}")
        pkl_data = z.read(pkl_files[0])

    # Parse pickle opcodes to get parameter names
    ops = _parse_pickle_ops(pkl_data)
    param_names = _extract_param_names(ops)

    # Unpickle with our custom handler
    unpickler = _PTUnpickler(io.BytesIO(pkl_data), data_arrays)
    raw = unpickler.load()

    # The result is an OrderedDict with tensor values
    if isinstance(raw, OrderedDict):
        # Match param names to tensors
        result = OrderedDict()
        tensors = [v for v in raw.values() if isinstance(v, np.ndarray)]

        for i, name in enumerate(param_names):
            if i < len(tensors):
                result[name] = tensors[i]

        # Check for metadata (chars, stoi, itos, config)
        for key, value in raw.items():
            if isinstance(value, np.ndarray) and key not in result:
                result[key] = value
            elif not isinstance(value, np.ndarray) and key not in ("__storage__",):
                result[key] = value

        return dict(result)

    elif isinstance(raw, dict):
        # Handle nested structures like {'model_state_dict': {...}, 'step': N, ...}
        if "model_state_dict" in raw and isinstance(raw["model_state_dict"], dict):
            return raw
        if "model" in raw and isinstance(raw["model"], dict):
            return raw["model"]
        return raw

    return {"state_dict": raw}


def load_pt_state_dict(path: Union[str, Path]) -> Dict[str, np.ndarray]:
    """
    Load a .pt checkpoint and return only the model weights.

    Handles nested {'model': {...}} and {'model_state_dict': {...}} structures.
    """
    data = load_pt_checkpoint(path)

    # Flatten if nested
    if "model" in data and isinstance(data["model"], dict):
        return data["model"]
    if "model_state_dict" in data and isinstance(data["model_state_dict"], dict):
        return data["model_state_dict"]

    # Filter to only numpy arrays
    return {k: v for k, v in data.items() if isinstance(v, np.ndarray)}


def load_pt_file(path: Union[str, Path]) -> Dict[str, np.ndarray]:
    """Alias for load_pt_state_dict — loads a .pt file and returns weight dict."""
    return load_pt_state_dict(path)


def load_pt_bytes(data: bytes) -> Dict[str, np.ndarray]:
    """Load weights from .pt bytes directly (no temp file needed)."""
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        f.write(data)
        tmp_path = f.name
    try:
        return load_pt_state_dict(tmp_path)
    finally:
        os.unlink(tmp_path)


__all__ = [
    "load_pt_checkpoint",
    "load_pt_state_dict",
    "load_pt_file",
    "load_pt_bytes",
]
