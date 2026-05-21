# utilities for the training pipeline
"""
Utility helpers used across the training codebase.

- ``mkdir`` – ensure a directory exists (creates parent directories as needed).
- ``save_json`` – write a Python object to a JSON file (pretty‑printed, UTF‑8).
- ``load_json`` – read a JSON file and return the deserialized Python object.

All functions raise informative ``OSError`` / ``json.JSONDecodeError``
exceptions which can be caught by callers.
"""

import json
import os
from pathlib import Path
from typing import Any


def mkdir(path: str) -> None:
    """Create a directory ``path`` if it does not already exist.

    The function is **idempotent** – calling it repeatedly will not raise an error.
    It also creates any missing parent directories (``parents=True``) and
    respects the default ``exist_ok=True`` behaviour.
    """
    try:
        Path(path).mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        raise OSError(f"Failed to create directory '{path}': {exc}")


def save_json(data: Any, file_path: str, *, indent: int = 2) -> None:
    """Serialise *data* as JSON and write it to *file_path*.

    The function first ensures that the target directory exists.  It writes the
    file using UTF‑8 encoding and a trailing newline so that tools like ``cat``
    display the file cleanly.
    """
    dir_name = os.path.dirname(file_path)
    if dir_name:
        mkdir(dir_name)
    try:
        with open(file_path, "w", encoding="utf-8") as fp:
            json.dump(data, fp, indent=indent, ensure_ascii=False)
            fp.write("\n")
    except (TypeError, OSError) as exc:
        raise OSError(f"Failed to write JSON to '{file_path}': {exc}")


def load_json(file_path: str) -> Any:
    """Read a JSON file and return the parsed Python object.

    Raises ``FileNotFoundError`` if the file does not exist and ``json.JSONDecodeError``
    if the content cannot be parsed.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as fp:
            return json.load(fp)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"JSON file not found: {file_path}") from exc
    except json.JSONDecodeError as exc:
        raise json.JSONDecodeError(f"Invalid JSON in '{file_path}': {exc.msg}", exc.doc, exc.pos)


import torch


def get_device():
    """Auto-detect best available device (cuda > mps > cpu)."""
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def is_gpu_available():
    """Check if any GPU is available (CUDA or MPS)."""
    return torch.cuda.is_available() or torch.backends.mps.is_available()


def get_device_info():
    """Get detailed GPU information."""
    info = {'device': get_device(), 'gpu_available': is_gpu_available()}
    
    if torch.cuda.is_available():
        info['cuda_device_name'] = torch.cuda.get_device_name(0)
        info['cuda_device_count'] = torch.cuda.device_count()
    elif torch.backends.mps.is_available():
        info['mps_available'] = True
    
    return info
