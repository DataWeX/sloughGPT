"""Pydantic training schemas (``apps/api/server/training/schemas.py``)."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SERVER_DIR = _REPO_ROOT / "apps" / "api" / "server"
if str(_SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(_SERVER_DIR))


def test_train_request_device_empty_string_becomes_none() -> None:
    from training.schemas import TrainRequest

    r = TrainRequest(dataset="x", device="  ")
    assert r.device is None


def test_training_request_device_stripped() -> None:
    from training.schemas import TrainingRequest

    r = TrainingRequest(name="n", model="m", dataset="d", device="  cuda  ")
    assert r.device == "cuda"


def test_manifests_is_exactly_one_alternative() -> None:
    import pytest
    from pydantic import ValidationError
    from training.schemas import TrainingRequest

    ok = TrainingRequest(name="n", model="m", manifests=["data/crunched/a/v1/manifest.json"])
    assert ok.manifests == ["data/crunched/a/v1/manifest.json"]

    with pytest.raises(ValidationError):
        TrainingRequest(name="n", model="m", dataset="d", manifests=["data/x/v1/manifest.json"])

    with pytest.raises(ValidationError):
        TrainingRequest(name="n", model="m")
