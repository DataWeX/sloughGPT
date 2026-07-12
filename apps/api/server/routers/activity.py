"""
Activity router — sensor data ingestion and classification endpoints.

Receives 6-axis motion data (accelerometer + gyroscope) from the mobile app,
stores it for training, and provides real-time predictions via a simple CNN
or nearest-centroid classifier.
"""

import time
import logging
from typing import List, Optional
from fastapi import APIRouter
from pydantic import BaseModel, Field

from schemas.common import success_response

logger = logging.getLogger("man.activity")
router = APIRouter(prefix="/activity", tags=["activity"])

# ── In-memory store (replace with DB in production) ────────────────────

_activity_data: List[dict] = []
_activity_labels: set = set()
_prediction_count = 0


# ── Schemas ────────────────────────────────────────────────────────────


class ActivityRecordRequest(BaseModel):
    """Sensor data from mobile app for activity classification."""
    activity: str = Field(..., description="Activity label (walking, sitting, etc.)")
    readings: List[List[float]] = Field(
        ...,
        description="List of 6-axis readings: [ax, ay, az, gx, gy, gz]",
    )
    timestamps: Optional[List[int]] = Field(None, description="Timestamps in ms")


class ActivityPredictRequest(BaseModel):
    """Request a real-time activity prediction."""
    readings: List[List[float]] = Field(
        ...,
        description="Recent 6-axis readings: [ax, ay, az, gx, gy, gz]",
    )


# ── Endpoints ──────────────────────────────────────────────────────────


@router.post("/record")
async def record_activity(req: ActivityRecordRequest):
    """Record sensor data for activity classification training.

    Stores labeled sensor windows in memory.  Use GET /activity/dataset
    to retrieve collected data for offline training.
    """
    global _prediction_count

    sample = {
        "activity": req.activity,
        "readings": req.readings,
        "timestamps": req.timestamps or [],
        "recorded_at": time.time(),
        "window_size": len(req.readings),
    }
    _activity_data.append(sample)
    _activity_labels.add(req.activity)

    logger.info(
        "Recorded activity: %s (%d readings)",
        req.activity, len(req.readings),
    )

    return success_response(data={
        "activity": req.activity,
        "readings_count": len(req.readings),
        "total_samples": len(_activity_data),
        "known_activities": sorted(_activity_labels),
    })


@router.get("/dataset")
async def get_dataset():
    """Get all collected activity data for offline training."""
    return success_response(data={
        "samples": len(_activity_data),
        "activities": sorted(_activity_labels),
        "data": _activity_data[-100:],  # Last 100 windows
    })


@router.get("/status")
async def activity_status():
    """Get activity collection status."""
    return success_response(data={
        "total_samples": len(_activity_data),
        "known_activities": sorted(_activity_labels),
        "predictions_served": _prediction_count,
    })


@router.post("/predict")
async def predict_activity(req: ActivityPredictRequest):
    """Predict activity from sensor readings.

    Uses a simple nearest-centroid classifier.  For production,
        train a CNN on the collected dataset and swap in here.
    """
    global _prediction_count
    _prediction_count += 1

    if not _activity_data:
        return success_response(data={
            "activity": "unknown",
            "confidence": 0.0,
            "note": "No training data collected yet",
        })

    # Simple nearest-centroid: compute mean reading per activity, pick closest
    import numpy as np

    centroids = {}
    for sample in _activity_data:
        act = sample["activity"]
        readings = np.array(sample["readings"], dtype=np.float32)
        mean = readings.mean(axis=0)
        if act not in centroids:
            centroids[act] = []
        centroids[act].append(mean)

    # Average centroids
    avg_centroids = {act: np.mean(vals, axis=0) for act, vals in centroids.items()}

    query = np.array(req.readings, dtype=np.float32).mean(axis=0)

    best_act = "unknown"
    best_dist = float("inf")
    for act, centroid in avg_centroids.items():
        dist = float(np.linalg.norm(query - centroid))
        if dist < best_dist:
            best_dist = dist
            best_act = act

    # Confidence from distance (closer = higher confidence)
    confidence = max(0.0, 1.0 - best_dist / 10.0)

    return success_response(data={
        "activity": best_act,
        "confidence": round(confidence, 3),
        "distance": round(best_dist, 3),
    })
