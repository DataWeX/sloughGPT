"""
Activity Recognition Router — train and predict phone sensor activities.

Endpoints:
  POST /activity/data     — submit labeled sensor window (accel + gyro)
  POST /activity/train    — train classifier on accumulated data
  POST /activity/predict  — classify a sensor window
  GET  /activity/status   — model status + dataset stats
  GET  /activity/dataset  — list saved recordings
  DELETE /activity/data   — delete all collected data

Uses SloNet-based ActivityClassifier from domains.activity.
Data stored as .npz in {repo_root}/data/activity_records/
"""

import asyncio
import json
import logging
import threading
import numpy as np
from pathlib import Path
from typing import Optional, List
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/activity", tags=["activity"])

_REPO_ROOT = Path(__file__).resolve().parents[4]
_DATA_DIR = _REPO_ROOT / "data" / "activity_records"

_MODEL_LOCK = threading.Lock()
_MODEL = None

ACTIVITY_NAMES = [
    "stationary",
    "walking",
    "running",
    "shaking",
    "driving",
    "cycling",
]


class SensorWindow(BaseModel):
    """A window of 6-axis sensor data (shape: time_steps x 6).

    Channels: [accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z]
    """
    data: List[List[float]]  # time_steps x 6
    label: Optional[int] = None  # 0..5 or None for unlabeled


class TrainRequest(BaseModel):
    """Training configuration."""
    epochs: int = 30
    lr: float = 0.001
    batch_size: int = 16


class TrainResponse(BaseModel):
    """Training result."""
    status: str
    epochs: int
    final_loss: Optional[float] = None
    val_accuracy: Optional[float] = None
    num_samples: int
    message: str


class PredictRequest(BaseModel):
    """Sensor data to classify."""
    data: List[List[float]]  # time_steps x 6


class PredictResponse(BaseModel):
    """Classification result."""
    activity: str
    class_id: int
    confidence: float
    probabilities: List[float]


class StatusResponse(BaseModel):
    """Activity recognition system status."""
    model_loaded: bool
    num_recordings: int
    num_labels: int
    activities: List[str]
    device: str = "cpu"


def _ensure_data_dir():
    _DATA_DIR.mkdir(parents=True, exist_ok=True)


def _load_all_data() -> tuple[np.ndarray, np.ndarray]:
    """Load all recorded .npz files into arrays."""
    _ensure_data_dir()
    samples, labels = [], []
    for f in sorted(_DATA_DIR.glob("*.npz")):
        try:
            d = np.load(f)
            label = int(d.get("label", -1))
            data = d["data"]
            samples.append(data)
            labels.append(label)
        except Exception as e:
            logger.warning(f"Skipping corrupted {f}: {e}")
    if not samples:
        return np.empty((0, 0, 6), dtype=np.float32), np.empty(0, dtype=np.int64)
    X = np.stack(samples).astype(np.float32)
    y = np.array(labels, dtype=np.int64)
    return X, y


def _get_next_file_id() -> int:
    existing = [int(f.stem) for f in _DATA_DIR.glob("*.npz") if f.stem.isdigit()]
    return max(existing) + 1 if existing else 1


def _maybe_load_model():
    global _MODEL
    with _MODEL_LOCK:
        if _MODEL is not None:
            return True
        # Try loading from disk
        model_path = Path(__file__).resolve().parents[4] / "packages" / "core-py" / "domains" / "activity" / "model.npz"
        if model_path.exists():
            try:
                from domains.activity.classifier import ActivityClassifier
                _MODEL = ActivityClassifier.load(str(model_path))
                logger.info(f"Loaded activity model from {model_path}")
                return True
            except Exception as e:
                logger.warning(f"Could not load activity model: {e}")
    return False


def _build_model(X: np.ndarray, y: np.ndarray, epochs: int, lr: float, batch_size: int):
    """Train a classifier; stores result in global _MODEL."""
    global _MODEL
    from domains.activity import train_classifier, ActivityClassifier
    model = train_classifier(X, y, epochs=epochs, lr=lr, batch_size=batch_size, verbose=False)
    with _MODEL_LOCK:
        _MODEL = model
    return model


@router.post("/data")
async def record_data(body: SensorWindow):
    """Save a labeled or unlabeled sensor window as .npz.

    Args:
        body: SensorWindow with data (time_steps x 6) and optional label.

    Returns:
        Recording ID and file path.
    """
    _ensure_data_dir()
    arr = np.array(body.data, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 6:
        raise HTTPException(status_code=400, detail="data must be (time_steps, 6)")
    file_id = _get_next_file_id()
    path = _DATA_DIR / f"{file_id}.npz"
    kwargs = {"data": arr}
    if body.label is not None:
        kwargs["label"] = np.int64(body.label)
    np.savez_compressed(str(path), **kwargs)
    logger.info(f"Saved recording {file_id}: {arr.shape}, label={body.label}")
    return {"id": file_id, "path": str(path), "samples": len(arr)}


@router.post("/train")
async def train(body: TrainRequest = TrainRequest()):
    """Train activity classifier on all recorded data.

    Args:
        body: epochs, lr, batch_size.

    Returns:
        TrainResponse with loss and accuracy.

    Side effects:
        - Replaces the global _MODEL.
    """
    X, y = _load_all_data()
    if len(X) < 5:
        raise HTTPException(
            status_code=400,
            detail=f"Need at least 5 recordings, have {len(X)}. Record more data first.",
        )
    labeled = y >= 0
    if labeled.sum() < 5:
        raise HTTPException(
            status_code=400,
            detail=f"Need at least 5 labeled recordings, have {int(labeled.sum())}.",
        )
    X_lab = X[labeled]
    y_lab = y[labeled]

    from domains.training.slonet import Tensor, cross_entropy

    model = _build_model(X_lab, y_lab, epochs=body.epochs, lr=body.lr, batch_size=body.batch_size)

    # Final validation accuracy
    from domains.activity.classifier import _accuracy
    xv = Tensor(X_lab, requires_grad=False)
    yv = Tensor(y_lab, requires_grad=False)
    logits = model.forward(xv)
    val_acc = float(_accuracy(logits, y_lab))

    return TrainResponse(
        status="ok",
        epochs=body.epochs,
        final_loss=None,
        val_accuracy=val_acc,
        num_samples=len(X_lab),
        message=f"Trained on {len(X_lab)} labeled samples, {len(X) - int(labeled.sum())} unlabeled",
    )


@router.post("/train/stream")
async def train_stream(body: TrainRequest = TrainRequest()):
    """Train activity classifier with SSE progress stream.

    Yields per-epoch progress events::
        {"epoch": 1, "epochs": 30, "loss": 1.234, "val_loss": 1.345,
         "val_accuracy": 0.75, "lr": 0.001}
        ...
        {"status": "complete", "epochs": 30, "final_loss": 0.876,
         "val_accuracy": 0.875, "num_samples": 42}
        {"status": "error", "message": "..."}
    """
    X, y = _load_all_data()
    if len(X) < 5:
        return StreamingResponse(
            iter([f"data: {json.dumps({'status': 'error', 'message': f'Need at least 5 recordings, have {len(X)}'})}\n\n"]),
            media_type="text/event-stream",
        )
    labeled = y >= 0
    if labeled.sum() < 2:
        return StreamingResponse(
            iter([f"data: {json.dumps({'status': 'error', 'message': f'Need at least 2 labeled recordings, have {int(labeled.sum())}'})}\n\n"]),
            media_type="text/event-stream",
        )
    X_lab = X[labeled]
    y_lab = y[labeled]

    from domains.training.slonet import Tensor
    from domains.activity.classifier import _accuracy

    async def _generate():
        loop = asyncio.get_event_loop()
        from collections import deque
        epoch_q = deque()
        done = False

        def _on_epoch(epoch, epochs, loss, val_loss, val_accuracy, lr):
            epoch_q.append({
                "epoch": epoch, "epochs": epochs, "loss": loss,
                "val_loss": val_loss, "val_accuracy": val_accuracy, "lr": lr,
            })

        def _train():
            global _MODEL
            from domains.activity import train_classifier as tc
            model = tc(
                X_lab, y_lab,
                epochs=body.epochs, lr=body.lr, batch_size=body.batch_size,
                verbose=False, on_epoch=_on_epoch,
            )
            with _MODEL_LOCK:
                _MODEL = model

        train_task = loop.run_in_executor(None, _train)

        while not done or epoch_q:
            while epoch_q:
                yield f"data: {json.dumps(epoch_q.popleft())}\n\n"
            if train_task.done():
                done = True
            if not epoch_q:
                await asyncio.sleep(0.05)

        await train_task

        # Final accuracy
        xv = Tensor(X_lab, requires_grad=False)
        logits = _MODEL.forward(xv)
        final_acc = float(_accuracy(logits, y_lab))

        yield f"data: {json.dumps({'status': 'complete', 'epochs': body.epochs, 'val_accuracy': final_acc, 'num_samples': len(X_lab)})}\n\n"

    return StreamingResponse(_generate(), media_type="text/event-stream")

@router.post("/predict")
async def predict(body: PredictRequest):
    """Classify a sensor window.

    Args:
        body: data (time_steps x 6).

    Returns:
        Predicted activity, class ID, confidence, full probability vector.
    """
    global _MODEL
    if _MODEL is None:
        raise HTTPException(status_code=400, detail="No trained model. POST /activity/train first.")
    arr = np.array(body.data, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 6:
        raise HTTPException(status_code=400, detail="data must be (time_steps, 6)")
    from domains.activity import predict_activity as pa
    cls_id, name, probs = pa(_MODEL, arr)
    return PredictResponse(
        activity=name,
        class_id=int(cls_id),
        confidence=float(probs[cls_id]),
        probabilities=[float(p) for p in probs],
    )


@router.get("/status")
async def status():
    """Get activity recognition system status."""
    _ensure_data_dir()
    recordings = list(_DATA_DIR.glob("*.npz"))
    labeled = 0
    for f in recordings:
        try:
            d = np.load(f)
            if "label" in d:
                labeled += 1
        except Exception:
            pass
    return StatusResponse(
        model_loaded=_maybe_load_model(),
        num_recordings=len(recordings),
        num_labels=labeled,
        activities=ACTIVITY_NAMES,
    )


@router.get("/model")
async def download_model():
    """Download the trained model.npz file.

    Returns:
        The model.npz binary file for local caching.
    """
    model_path = _REPO_ROOT / "packages" / "core-py" / "domains" / "activity" / "model.npz"
    if not model_path.exists():
        raise HTTPException(status_code=404, detail="No trained model available. Train one first.")
    from fastapi.responses import FileResponse
    return FileResponse(str(model_path), media_type="application/octet-stream",
                        filename="model.npz")


@router.get("/dataset")
async def list_dataset():
    """List all recordings with metadata."""
    _ensure_data_dir()
    items = []
    for f in sorted(_DATA_DIR.glob("*.npz")):
        try:
            d = np.load(f)
            data = d["data"]
            label = int(d.get("label", -1))
            items.append({
                "id": int(f.stem),
                "path": str(f),
                "samples": data.shape[0],
                "label": label,
                "activity": ACTIVITY_NAMES[label] if 0 <= label < len(ACTIVITY_NAMES) else "unlabeled",
            })
        except Exception as e:
            logger.warning(f"Corrupted {f}: {e}")
    return {"recordings": items, "total": len(items)}


@router.delete("/data")
async def delete_all_data():
    """Delete all recorded data."""
    _ensure_data_dir()
    count = 0
    for f in _DATA_DIR.glob("*.npz"):
        f.unlink()
        count += 1
    global _MODEL
    with _MODEL_LOCK:
        _MODEL = None
    logger.info(f"Deleted {count} recordings")
    return {"deleted": count}
