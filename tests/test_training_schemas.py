"""Pydantic training schemas (``apps/api/server/training/schemas.py``)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SERVER_DIR = _REPO_ROOT / "apps" / "api" / "server"
if str(_SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(_SERVER_DIR))


def test_training_request_normalizes_mixed_precision_dtype_case() -> None:
    from training.schemas import TrainingRequest

    r = TrainingRequest(
        name="n",
        model="m",
        dataset="dummy_ds",
        mixed_precision_dtype=" FP16 ",
    )
    assert r.mixed_precision_dtype == "fp16"


def test_training_request_rejects_invalid_mixed_precision_dtype() -> None:
    from pydantic import ValidationError

    from training.schemas import TrainingRequest

    with pytest.raises(ValidationError):
        TrainingRequest(
            name="n",
            model="m",
            dataset="dummy_ds",
            mixed_precision_dtype="float32",
        )


def test_train_request_device_empty_string_becomes_none() -> None:
    from training.schemas import TrainRequest

    r = TrainRequest(dataset="x", device="  ")
    assert r.device is None


def test_training_request_device_stripped() -> None:
    from training.schemas import TrainingRequest

    r = TrainingRequest(name="n", model="m", dataset="d", device="  cuda  ")
    assert r.device == "cuda"
