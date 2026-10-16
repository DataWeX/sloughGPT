"""ModelBrain — a trainable decision brain for world babies (World–Model Bridge).

Stage A of the world–model bridge: babies can be born with a ModelBrain
instead of a Perceptron. The two are interface-identical (``forward`` /
``update``, ``W``/``b``/``H``/``bh`` tensors, ``to_dict``/``from_dict``) so
``SimBaby``, ``Genome``, evolution, and serialization all work unchanged —
the seam between the model stack and the world is this one duck-typed
contract, exactly as the modularity/interop directive demands.

The capability step (why this is a *model* and not a perceptron):
``Perceptron``'s hidden projection is fixed random features — its own
docstring says "never trained — no backpropagation"; only the readout
learns by the delta rule. ``ModelBrain`` backprops the same reward signal
through the whole stack, so the hidden features themselves are shaped by
lived experience: a genuine two-layer neural net trained in-world by
energy, not by text. (Token-space brains — real transformers deciding
from serialized episodes — are the Stage B follow-up; this is the
weight-space tier.)

Genome compatibility is exact: same tensor names, shapes, and RNG draw
order (H, bh, W) as the hidden-capable ``Perceptron``, so evolution
breeds model brains with zero changes to ``evolution.py`` and a seeded
world consumes the identical noise stream regardless of brain kind.
"""

from __future__ import annotations

import numpy as np


class ModelBrain:
    """Two-layer decision brain with a trainable hidden projection.

    Drop-in replacement for ``Perceptron``: same constructor signature,
    same tensor inventory, same sigmoid readout, same reward-sign update
    convention (positive error reinforces). The difference is entirely in
    ``update`` — the hidden projection learns.
    """

    #: Serialization marker — lets ``brain_from_dict`` rebuild the right class.
    kind = "model"

    def __init__(self, input_dim: int, output_dim: int, hidden_units: int = 0):
        """
        Args:
            input_dim: perception feature count.
            output_dim: decision gate count.
            hidden_units: trainable hidden projection width (0 = plain
                delta-rule readout, mathematically identical to a
                hidden-free Perceptron).
        """
        self.hidden_units = int(hidden_units)
        if self.hidden_units > 0:
            # Same draw order and scales as Perceptron (H, bh, then W) so a
            # seeded world is reproducible regardless of brain kind.
            self.H = np.random.randn(int(input_dim), self.hidden_units).astype(np.float32) * 0.5
            self.bh = np.random.randn(self.hidden_units).astype(np.float32) * 0.1
            readout_in = int(input_dim) + self.hidden_units
        else:
            self.H = None
            self.bh = None
            readout_in = int(input_dim)
        self.W = np.random.randn(readout_in, int(output_dim)).astype(np.float32) * 0.1
        self.b = np.zeros(int(output_dim), dtype=np.float32)

    def _features(self, x: np.ndarray) -> np.ndarray:
        """Representation fed to the readout: raw input (+ hidden features)."""
        arr = np.asarray(x, dtype=np.float32)
        if self.H is None:
            return arr
        h = self._sigmoid(arr @ self.H + self.bh)
        return np.concatenate([arr, h])

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass: sigmoid(features(x) @ W + b)."""
        return self._sigmoid(self._features(x) @ self.W + self.b)

    def update(self, x: np.ndarray, error: np.ndarray, lr: float = 0.01):
        """Delta-rule update with real backprop through the hidden layer.

        Sign convention matches ``Perceptron.update``: positive ``error``
        reinforces the pathway that produced the output (ascent on reward,
        not descent on a text loss). The readout step is the perceptron's
        delta rule, byte for byte; the hidden step is the model upgrade —
        the reward is backpropped through the pre-update readout and the
        sigmoid, so the feature projection is shaped by experience instead
        of staying frozen random.

        Args:
            x: input features.
            error: scalar or vector reward signal (positive = reinforce).
            lr: learning rate (always positive).
        """
        arr = np.asarray(x, dtype=np.float32)
        if self.H is None:
            pred = self._sigmoid(arr @ self.W + self.b)
            d = error * pred * (1.0 - pred)
            self.W += np.outer(arr, d) * lr
            self.b += d * lr
            return
        h = self._sigmoid(arr @ self.H + self.bh)
        features = np.concatenate([arr, h])
        pred = self._sigmoid(features @ self.W + self.b)
        d_out = error * pred * (1.0 - pred)
        n_in = arr.shape[0]
        # Backprop through the PRE-update readout: which hidden features
        # produced this output, and how much did each contribute?
        d_features = self.W @ d_out
        d_h = d_features[n_in:] * h * (1.0 - h)
        # Readout step — identical to the perceptron's delta rule.
        self.W += np.outer(features, d_out) * lr
        self.b += d_out * lr
        # Hidden step — the model upgrade: the projection learns too.
        self.H += np.outer(arr, d_h) * lr
        self.bh += d_h * lr

    def to_dict(self) -> dict:
        """Serialize to a JSON-safe dict (superset of Perceptron's shape)."""
        data: dict = {"kind": self.kind, "W": self.W.tolist(), "b": self.b.tolist()}
        if self.H is not None:
            data["H"] = self.H.tolist()
            data["bh"] = self.bh.tolist()
        return data

    @classmethod
    def from_dict(cls, data: dict) -> ModelBrain:
        """Rebuild from :meth:`to_dict` output without touching RNG."""
        m = object.__new__(cls)
        m.W = np.asarray(data["W"], dtype=np.float32).copy()
        m.b = np.asarray(data["b"], dtype=np.float32).copy()
        if "H" in data:
            m.H = np.asarray(data["H"], dtype=np.float32).copy()
            m.bh = np.asarray(data["bh"], dtype=np.float32).copy()
            m.hidden_units = int(m.H.shape[1])
        else:
            m.H = None
            m.bh = None
            m.hidden_units = 0
        return m

    @staticmethod
    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -10, 10)))
