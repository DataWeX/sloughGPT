"""
Synthetic phone sensor dataset for activity recognition.

Generates realistic accelerometer (3-axis) + gyroscope (3-axis) time-series
for common activities using parametric signal models.
"""

import numpy as np
from typing import Tuple, List

ACTIVITIES = [
    "stationary",
    "walking",
    "running",
    "shaking",
    "driving",
    "cycling",
]


class SyntheticDataset:
    """Generate synthetic 6-axis sensor data for activity recognition.

    Each sample is (time_steps, 6) — channels are:
      [accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z]

    Args:
        time_steps: number of timesteps per sample (default 128 ≈ 2s at 64Hz)
        freq_hz: simulated sampling frequency
    """

    def __init__(self, time_steps: int = 128, freq_hz: float = 64.0):
        self.time_steps = time_steps
        self.freq_hz = freq_hz
        self.rng = np.random.RandomState(42)

    def _stationary(self) -> np.ndarray:
        """Phone sitting still: gravity on Z, tiny noise elsewhere."""
        t = np.arange(self.time_steps) / self.freq_hz
        data = np.zeros((self.time_steps, 6))
        data[:, 2] = 9.81  # gravity on Z
        data[:, :3] += self.rng.randn(self.time_steps, 3) * 0.05
        data[:, 3:] += self.rng.randn(self.time_steps, 3) * 0.02
        return data

    def _walking(self) -> np.ndarray:
        """Walking: ~2Hz vertical oscillation + slight rotation."""
        t = np.arange(self.time_steps) / self.freq_hz
        data = np.zeros((self.time_steps, 6))
        data[:, 2] = 9.81 + 2.0 * np.sin(2 * np.pi * 2.0 * t)  # vertical bounce
        data[:, 0] = 0.5 * np.sin(2 * np.pi * 1.0 * t + 0.5)    # lateral sway
        data[:, 3] = 0.3 * np.sin(2 * np.pi * 2.0 * t)           # gyro pitch
        data += self.rng.randn(self.time_steps, 6) * 0.1
        return data

    def _running(self) -> np.ndarray:
        """Running: ~3Hz higher-amplitude oscillation."""
        t = np.arange(self.time_steps) / self.freq_hz
        data = np.zeros((self.time_steps, 6))
        data[:, 2] = 9.81 + 5.0 * np.sin(2 * np.pi * 3.0 * t)
        data[:, 0] = 1.5 * np.sin(2 * np.pi * 1.5 * t + 0.8)
        data[:, 1] = 0.8 * np.sin(2 * np.pi * 3.0 * t + 0.3)
        data[:, 3] = 0.8 * np.sin(2 * np.pi * 3.0 * t)
        data[:, 4] = 0.5 * np.sin(2 * np.pi * 1.5 * t)
        data += self.rng.randn(self.time_steps, 6) * 0.2
        return data

    def _shaking(self) -> np.ndarray:
        """Shaking phone: high-frequency random motion on all axes."""
        t = np.arange(self.time_steps) / self.freq_hz
        data = np.zeros((self.time_steps, 6))
        data[:, :3] = self.rng.randn(self.time_steps, 3) * 4.0
        data[:, 3:] = self.rng.randn(self.time_steps, 3) * 3.0
        return data

    def _driving(self) -> np.ndarray:
        """Driving: low-frequency vibration + constant orientation."""
        t = np.arange(self.time_steps) / self.freq_hz
        data = np.zeros((self.time_steps, 6))
        data[:, 2] = 9.81 + 0.3 * np.sin(2 * np.pi * 0.5 * t)
        data[:, 0] = 0.2 * np.sin(2 * np.pi * 0.8 * t)
        data[:, 1] = 0.1 * np.sin(2 * np.pi * 0.6 * t + 0.4)
        data[:, 3:] += self.rng.randn(self.time_steps, 3) * 0.05
        data[:, :3] += self.rng.randn(self.time_steps, 3) * 0.08
        return data

    def _cycling(self) -> np.ndarray:
        """Cycling: periodic lean + rotation from pedal stroke ~1.5Hz."""
        t = np.arange(self.time_steps) / self.freq_hz
        data = np.zeros((self.time_steps, 6))
        data[:, 2] = 9.81 + 1.0 * np.sin(2 * np.pi * 1.5 * t)
        data[:, 0] = 2.0 * np.sin(2 * np.pi * 0.75 * t + 0.2)
        data[:, 3] = 1.0 * np.sin(2 * np.pi * 1.5 * t)
        data[:, 4] = 0.4 * np.sin(2 * np.pi * 0.75 * t)
        data += self.rng.randn(self.time_steps, 6) * 0.15
        return data

    def generate(self, samples_per_class: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """Generate balanced synthetic dataset.

        Returns:
            X: (num_samples, time_steps, 6) float32
            y: (num_samples,) int64 labels 0..N-1
        """
        generators = [self._stationary, self._walking, self._running,
                      self._shaking, self._driving, self._cycling]
        samples = []
        labels = []
        for cls_idx, gen_fn in enumerate(generators):
            for _ in range(samples_per_class):
                samples.append(gen_fn())
                labels.append(cls_idx)
        X = np.stack(samples).astype(np.float32)
        y = np.array(labels, dtype=np.int64)
        return X, y


def load_recorded_data(data_dir: str = "data/activity_records") -> Tuple[np.ndarray, np.ndarray]:
    """Load user-recorded sensor data from disk.

    Expects .npz files with keys: accel_x/y/z, gyro_x/y/z, label.

    Returns:
        X: (num_samples, time_steps, 6) float32
        y: (num_samples,) int64 labels
    """
    from pathlib import Path
    p = Path(data_dir)
    if not p.exists():
        raise FileNotFoundError(f"No recorded data at {data_dir}")
    samples = []
    labels = []
    for f in sorted(p.glob("*.npz")):
        d = np.load(f)
        accel = np.stack([d["accel_x"], d["accel_y"], d["accel_z"]], axis=1)
        gyro = np.stack([d["gyro_x"], d["gyro_y"], d["gyro_z"]], axis=1)
        samples.append(np.concatenate([accel, gyro], axis=1))
        labels.append(d["label"])
    if not samples:
        return np.empty((0, 0, 6), dtype=np.float32), np.empty(0, dtype=np.int64)
    X = np.stack(samples).astype(np.float32)
    y = np.array(labels, dtype=np.int64)
    return X, y
