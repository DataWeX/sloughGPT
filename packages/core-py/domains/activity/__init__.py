"""
Activity Recognition Domain — train a SloNet-based classifier on phone sensor data.

Components:
- SyntheticDataset: generates realistic accelerometer/gyroscope time-series
- ActivityClassifier: SloNet LSTM + linear head for activity classification
- train_classifier(): training loop with progress
- predict_activity(): single-shot inference

Usage:
    from domains.activity import train_classifier, predict_activity, SyntheticDataset

    ds = SyntheticDataset(samples_per_class=50)
    X, y = ds.generate()
    model = train_classifier(X, y, epochs=20)
    label = predict_activity(model, sensor_data)
"""

from .dataset import SyntheticDataset, ACTIVITIES
from .classifier import ActivityClassifier, train_classifier, predict_activity

__all__ = [
    "SyntheticDataset",
    "ActivityClassifier",
    "train_classifier",
    "predict_activity",
    "ACTIVITIES",
]
