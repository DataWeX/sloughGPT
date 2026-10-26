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

── The brain registry (one seam, many brains) ────────────────────────────────

``BRAIN_KINDS`` maps a ``WorldParams.brain_kind`` string to a factory, and
``make_brain`` is the single construction seam ``SimBaby`` uses — so the brain
kind is a configuration choice, not a code fork. ``brain_from_dict`` is the
matching deserialization seam; it dispatches on the ``kind`` tag and is
**fail-loud** on an unknown one (a typo in a saved payload must not silently
resurrect a Perceptron).

Registered kinds:

``perceptron``
    The default frozen-feature delta-rule brain (lives in ``simulation.py`` —
    imported lazily here to keep the module cycle-free). Genome-compatible.
``model``
    :class:`ModelBrain` above — hand-rolled numpy 2-layer MLP with a trained
    hidden projection. Genome-compatible (identical tensor inventory).
``slonet``
    :class:`SloNetBrain` — a native ``domain.training._internal.slonet`` stack
    (``SloLinear`` in -> hidden blocks -> readout) trained in-world by real
    ``backward()`` through the whole stack. NOT genome-compatible: the flat
    ``W``/``b``/``H`` genome format cannot represent it, so breeding raises
    ``NotImplementedError`` (``evolution.py``) and offspring inherit whole via
    ``spawn_child``.
``random``
    :class:`RandomBrain` — the coin-flip emergence baseline.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    # Annotation-only: keeps ``-> Perceptron`` resolvable for type checkers
    # without a runtime import (simulation.py imports this module, so a
    # module-scope import here would be circular).
    from domain.shell._internal.simulation import Perceptron

#: Safe fallback gate: sign(0.5 - 0.5) = 0 -> "stay put" on every axis.
#: A dead brain must steer neutral instead of killing the simulation tick.
_FALLBACK_GATE = 0.5


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Elementwise sigmoid clipped to [-10, 10] (Perceptron/ModelBrain parity)."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -10.0, 10.0)))


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


# ── SloNetBrain — a native SloNet stack behind the brain seam ────────────────
#
# Card 7ff49d20: the halves the two Stage A implementations split between them,
# in one brain. The LANDED half is "trained": real backprop flows through the
# whole stack from the in-world energy reward. The STRANDED half is "native":
# every layer is a real ``domain.training._internal.slonet.SloLinear``, so the
# world's decision brain is composed from the same modules the rest of the ML
# infra trains (``backward()`` + ``SloSGD``/``SloAdam``, exercised by
# ``scripts/benchmark_slonet.py``).
#
# There is NO training loop here. ``update`` is the same seam
# ``Perceptron.update``/``ModelBrain.update`` occupy: SimBaby's ``learn()``
# calls ``brain.update(features, error, lr)`` once per tick per brain, and the
# reward is the world's honest energy delta. SloNet supplies autograd; the
# world supplies the teacher.


class SloNetBrain:
    """A native ``SloLinear`` stack trained in-world by real backprop.

    Architecture (fully configurable, fully serializable, composable from the
    shared SloNet module list)::

        x ──SloLinear── tanh ──(SloLinear ─ tanh) × n_layer ──SloLinear── sigmoid
            input_proj          blocks                        head

    The update is gradient **ascent** on the reward-weighted gate value,
    ``L = -Σ (lr · error · pred)`` — the exact generalization of the
    perceptron delta rule (``W += outer(x, error · pred · (1-pred)) · lr``)
    through an arbitrarily deep native stack, so ``error > 0`` still reinforces
    and ``error < 0`` still weakens, with ``lr`` keeping ``learn()``'s
    surprise-scaled meaning.

    Args:
        input_dim: perception feature count.
        output_dim: decision gate count.
        hidden_units: fallback hidden width when ``n_embed`` is 0, so
            ``brain_hidden_units``/``--hidden-units`` size this brain too.
        n_embed: hidden width of the native stack (0 = derive from
            ``hidden_units``, else 16).
        n_layer: number of hidden ``SloLinear`` blocks (0 = proj -> head).
        decide_every: recompute the forward only every N calls; intermediate
            calls reuse the cached action (tick-rate mitigation). ``update``
            deliberately does NOT invalidate the cache — see below.
        learn_every: buffer this many ``update`` calls before one batched
            ``backward`` + optimizer step (mini-batch). Autograd cost is
            per-graph-node, not per-sample, so N buffered rewards cost about
            the same as one — this is the lever that keeps a full backward
            per tick inside the benchmark's throughput gate. ``1`` = step on
            every call.
        optimizer: ``"sgd"`` (default; delta-rule semantics, ``lr`` honoured
            exactly) or ``"adam"`` (``scripts/benchmark_slonet.py``'s
            optimiser — scale-free, uses ``mean(lr)`` over the batch).
    """

    #: Serialization marker — lets ``brain_from_dict`` rebuild the right class.
    kind = "slonet"

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_units: int = 0,
        n_embed: int = 0,
        n_layer: int = 1,
        decide_every: int = 1,
        learn_every: int = 1,
        optimizer: str = "sgd",
    ):
        # Lazy import: the native slonet module is heavy (~0.4s, tens of MB).
        # The perceptron baseline and every non-slonet brain must never pay it,
        # so this is the only place the stack is imported on construction.
        from domain.training._internal.slonet import SloLinear

        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.n_embed = int(n_embed) or (int(hidden_units) if int(hidden_units) > 0 else 16)
        self.n_embed = max(1, self.n_embed)
        self.n_layer = max(0, int(n_layer))
        self.decide_every = max(1, int(decide_every))
        self.learn_every = max(1, int(learn_every))
        self.optimizer = str(optimizer)

        self.proj = SloLinear(self.input_dim, self.n_embed)
        self.blocks = [SloLinear(self.n_embed, self.n_embed) for _ in range(self.n_layer)]
        self.head = SloLinear(self.n_embed, self.output_dim)
        self._layers = [self.proj, *self.blocks, self.head]
        self._params = [p for layer in self._layers for p in layer.parameters()]
        self._opt = self._make_optimizer(lr=0.01)

        self._calls = 0
        self._cached: np.ndarray | None = None
        self._buf_x: list[np.ndarray] = []
        self._buf_e: list[np.ndarray] = []
        self._buf_l: list[float] = []

    # ── construction helpers ──────────────────────────────────────────────

    def _make_optimizer(self, lr: float):
        """Build this brain's optimizer from SloNet (lazy import)."""
        from domain.training._internal.slonet import SloAdam, SloSGD

        if self.optimizer == "adam":
            return SloAdam(lr=float(lr))
        if self.optimizer != "sgd":
            raise ValueError(
                f"unknown brain optimizer {self.optimizer!r} (known: sgd, adam)"
            )
        return SloSGD(lr=float(lr))

    # ── inference ─────────────────────────────────────────────────────────

    def _features_numpy(self, x: np.ndarray) -> np.ndarray:
        """Project a perception vector through the native stack (no autograd)."""
        h = np.tanh(self.proj.forward_numpy(np.asarray(x, dtype=np.float32)))
        for blk in self.blocks:
            h = np.tanh(blk.forward_numpy(h))
        return h

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass: ``sigmoid(head(features(x)))`` with caching + fallback.

        Honours ``decide_every``: recomputes on the first call and every
        ``decide_every``-th call after, reusing the cached action in between —
        "stale by design", because the point of the window is that a
        ``learn()`` every tick does not force a recompute. Any failure inside
        the model yields the safe all-``0.5`` action instead of raising into
        the simulation tick.

        Args:
            x: perception feature vector of shape (input_dim,).

        Returns:
            np.ndarray of shape (output_dim,) with values in [0, 1].
        """
        if (
            self.decide_every > 1
            and self._cached is not None
            and self._calls % self.decide_every != 0
        ):
            self._calls += 1
            return self._cached
        # A raise OR a blown-up stack (NaN/inf gates) means a dead brain: steer
        # neutral instead of killing the tick. The fallback is never cached —
        # the next call gets a fresh chance to compute real gates.
        safe: np.ndarray | None = None
        try:
            cand = _sigmoid(self.head.forward_numpy(self._features_numpy(x)))
            if np.isfinite(cand).all():
                safe = cand
        except Exception:  # noqa: BLE001 — any model failure must stay inside the brain
            safe = None
        if safe is None:
            self._calls += 1
            return np.full(self.output_dim, _FALLBACK_GATE, dtype=np.float32)
        self._cached = safe
        self._calls += 1
        return safe

    # ── learning ──────────────────────────────────────────────────────────

    def update(self, x: np.ndarray, error: np.ndarray, lr: float = 0.01):
        """Queue a reward-weighted update; step every ``learn_every`` calls.

        Same sign convention as ``Perceptron.update``: positive ``error``
        reinforces the pathway that produced the output (ascent on reward, not
        descent on a text loss); ``lr`` is always positive.

        The cached action is deliberately NOT invalidated here. With
        ``decide_every > 1`` the window is "stale by design" — the tick-rate
        mitigation exists precisely so ``learn()`` every tick does not force a
        recompute; the cache refreshes at the N-call boundary instead.

        Args:
            x: perception feature vector of shape (input_dim,).
            error: scalar or (output_dim,) reward signal.
            lr: learning rate (always positive).
        """
        self._buf_x.append(np.asarray(x, dtype=np.float32))
        self._buf_e.append(
            np.broadcast_to(np.asarray(error, dtype=np.float32), (self.output_dim,)).copy()
        )
        self._buf_l.append(float(lr))
        if len(self._buf_x) < self.learn_every:
            return
        self._flush()

    def _flush(self):
        """One batched forward + ``backward()`` + optimizer step, then clear.

        The reward is seeded as the ROOT gradient of ``pred`` instead of being
        built into a loss graph. ``L = -Σ lr·error·pred`` has
        ``∂L/∂pred = -lr·error``, so writing that vector straight onto
        ``pred.grad`` yields the identical gradient while skipping the
        multiply/sum/neg nodes ``backward()`` would otherwise walk — measured
        2x cheaper per step, which is what keeps a real backward inside every
        tick's throughput budget.
        """
        from domain.training._internal.slonet import Tensor, sigmoid, tanh

        xs = np.stack(self._buf_x)  # (N, input_dim)
        errs = np.stack(self._buf_e)  # (N, output_dim)
        lrs = np.asarray(self._buf_l, dtype=np.float32).reshape(-1, 1)  # (N, 1)
        self._buf_x.clear()
        self._buf_e.clear()
        self._buf_l.clear()

        h = tanh(self.proj.forward(Tensor(xs)))
        for blk in self.blocks:
            h = tanh(blk.forward(h))
        pred = sigmoid(self.head.forward(h))

        if self.optimizer == "adam":
            # Adam is scale-free: a per-sample lr riding in the gradient would
            # be normalised straight back out, so it becomes the step size.
            pred.grad = Tensor(-errs)
            self._opt.lr = float(lrs.mean())
        else:
            # SGD: lr rides in the root gradient, so one step is exactly the
            # sum of the individual delta-rule steps the perceptron would have
            # taken across the batch (weights differ only by O(step)).
            pred.grad = Tensor(-(lrs * errs))
            self._opt.lr = 1.0
        pred.backward()
        self._opt.step(self._params)

    # ── serialization ─────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Serialize the whole brain (stack + head + config) to a JSON-safe dict.

        Pending buffered rewards are dropped, not serialized: at most
        ``learn_every - 1`` un-applied rewards exist and the weights are the
        only durable state.

        Returns:
            Dict with the kind tag, the stack config, and every layer's
            weight/bias as nested lists.
        """
        return {
            "kind": self.kind,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "n_embed": self.n_embed,
            "n_layer": self.n_layer,
            "decide_every": self.decide_every,
            "learn_every": self.learn_every,
            "optimizer": self.optimizer,
            "layers": [
                {
                    "weight": layer.weight.data.tolist(),
                    "bias": layer.bias.data.tolist(),
                }
                for layer in self._layers
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SloNetBrain:
        """Rebuild from :meth:`to_dict` output without touching RNG.

        Uses lazily zero-initialized ``SloLinear`` layers and overwrites every
        weight from the payload, so no random draw consumes the global stream —
        the same no-RNG inheritance contract ``Perceptron.from_dict`` guarantees.

        Raises:
            ValueError: if a stored weight shape disagrees with the stored
                config, or the layer count doesn't match ``n_layer``.
        """
        from domain.training._internal.slonet import SloLinear

        brain = object.__new__(cls)
        brain.input_dim = int(data["input_dim"])
        brain.output_dim = int(data["output_dim"])
        brain.n_embed = max(1, int(data["n_embed"]))
        brain.n_layer = max(0, int(data["n_layer"]))
        brain.decide_every = max(1, int(data.get("decide_every", 1)))
        brain.learn_every = max(1, int(data.get("learn_every", 1)))
        brain.optimizer = str(data.get("optimizer", "sgd"))

        stored = data["layers"]
        expected = 2 + brain.n_layer  # proj + blocks + head
        if len(stored) != expected:
            raise ValueError(f"stored {len(stored)} layers, expected {expected}")

        shapes = [(brain.n_embed, brain.input_dim)]
        shapes += [(brain.n_embed, brain.n_embed)] * brain.n_layer
        shapes += [(brain.output_dim, brain.n_embed)]

        brain._layers = []
        for payload, (out_f, in_f) in zip(stored, shapes, strict=True):
            layer = SloLinear(in_f, out_f, _lazy=True)
            w = np.asarray(payload["weight"], dtype=np.float32)
            if w.shape != (out_f, in_f):
                raise ValueError(f"layer weight {w.shape} != ({out_f}, {in_f})")
            layer.weight.data = w.copy()
            layer.bias.data = np.asarray(payload["bias"], dtype=np.float32).copy()
            brain._layers.append(layer)

        brain.proj = brain._layers[0]
        brain.blocks = brain._layers[1:-1]
        brain.head = brain._layers[-1]
        brain._params = [p for layer in brain._layers for p in layer.parameters()]
        brain._opt = brain._make_optimizer(lr=0.01)
        brain._calls = 0
        brain._cached = None
        brain._buf_x, brain._buf_e, brain._buf_l = [], [], []
        return brain


class RandomBrain:
    """The literal emergence baseline: a brain that decides by coin flip.

    A drop-in behind the same ``forward``/``update``/``to_dict``/``from_dict``
    seam whose ``forward`` emits uniform gates in [0, 1] and whose ``update``
    is a no-op (there is nothing to learn). It constructs and serializes
    without consuming any RNG, so enabling it in a world never perturbs the
    perception-noise stream or spawn layout.
    """

    #: Serialization marker.
    kind = "random"

    def __init__(self, input_dim: int, output_dim: int):
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Return fresh uniform gates in [0, 1] — no state, no learning.

        Args:
            x: ignored; the baseline is memoryless by design.

        Returns:
            np.ndarray of shape (output_dim,) with values in [0, 1].
        """
        return np.random.random(self.output_dim).astype(np.float32)

    def update(self, x: np.ndarray, error: np.ndarray, lr: float = 0.01):
        """No-op: the random baseline does not learn (by definition)."""

    def to_dict(self) -> dict[str, Any]:
        """Serialize the brain's identity (dimensions only — no weights).

        Returns:
            Dict with kind tag and dims, JSON-safe.
        """
        return {"kind": self.kind, "input_dim": self.input_dim, "output_dim": self.output_dim}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RandomBrain:
        """Rebuild a random brain from :meth:`to_dict` output.

        Args:
            data: serialized brain dict.

        Returns:
            A new RandomBrain with the stored dims.
        """
        return cls(int(data["input_dim"]), int(data["output_dim"]))


def brain_from_dict(payload: dict[str, Any]) -> Any:
    """Deserialize any registered brain from its ``to_dict`` payload.

    Dispatches on the ``kind`` tag. ``Perceptron`` payloads carry no ``kind``
    key (legacy shape), so an absent tag is the perceptron path — every
    previously saved baby still deserializes.

    Args:
        payload: a dict from some brain's ``to_dict``.

    Returns:
        A brain object honouring the forward/update/to_dict duck-type.

    Raises:
        ValueError: if the payload names a kind the registry does not know
            (fail-loud: a typo must never silently resurrect a Perceptron).
    """
    kind = payload.get("kind")
    if kind is None:
        # Imported lazily: simulation.py imports this module at module scope.
        from domain.shell._internal.simulation import Perceptron

        return Perceptron.from_dict(payload)
    if kind == "model":
        return ModelBrain.from_dict(payload)
    if kind == "slonet":
        return SloNetBrain.from_dict(payload)
    if kind == "random":
        return RandomBrain.from_dict(payload)
    if kind == "perceptron":
        from domain.shell._internal.simulation import Perceptron

        return Perceptron.from_dict(payload)
    known = ", ".join(sorted(BRAIN_KINDS))
    raise ValueError(f"unknown brain kind {kind!r} in payload (known: {known})")


# ── Brain registry ──────────────────────────────────────────────────────────
#
# One seam, many brains. ``make_brain`` is the factory SimBaby uses so the
# brain kind is a WorldParams choice, not a code fork. Extending the world with
# a new brain = register here; the decide_* seam is unchanged.


def _make_perceptron(
    input_dim: int, output_dim: int, hidden_units: int = 0, **_: Any
) -> Perceptron:
    # Imported lazily to avoid a circular import: simulation.py imports this
    # module to route brains through make_brain, so the reverse edge must not
    # exist at module load. ``hidden_units`` must pass through: a deeper
    # Perceptron is a legitimate brain choice and the registry must not
    # silently flatten it. Every other stack knob belongs to SloNetBrain and
    # is ignored here.
    from domain.shell._internal.simulation import Perceptron

    return Perceptron(input_dim, output_dim, hidden_units=int(hidden_units))


def _make_model(input_dim: int, output_dim: int, hidden_units: int = 0, **_: Any) -> ModelBrain:
    return ModelBrain(input_dim, output_dim, hidden_units=int(hidden_units))


def _make_slonet(input_dim: int, output_dim: int, hidden_units: int = 0, **kw: Any) -> SloNetBrain:
    keep = {k: kw[k] for k in ("n_embed", "n_layer", "decide_every", "learn_every", "optimizer") if k in kw}
    return SloNetBrain(input_dim, output_dim, hidden_units=int(hidden_units), **keep)


def _make_random(input_dim: int, output_dim: int, hidden_units: int = 0, **_: Any) -> RandomBrain:
    return RandomBrain(input_dim, output_dim)


#: brain kind -> factory(input_dim, output_dim, hidden_units=0, **params). Every
#: factory returns an object honouring the forward/update/to_dict duck-type.
BRAIN_KINDS: dict[str, Any] = {
    "perceptron": _make_perceptron,
    "model": _make_model,
    "slonet": _make_slonet,
    "random": _make_random,
}


def make_brain(kind: str, input_dim: int, output_dim: int, hidden_units: int = 0, **params: Any):
    """Construct a brain by kind, honouring the shared duck-typed contract.

    Args:
        kind: registered brain name — one of :data:`BRAIN_KINDS`.
        input_dim: perception feature width.
        output_dim: action gate count.
        hidden_units: hidden projection width for the perceptron/model pair,
            and the fallback stack width for ``slonet``.
        **params: extra brain params (``n_embed``/``n_layer``/
            ``decide_every``/``learn_every``/``optimizer`` for the slonet
            brain); each factory takes what it understands and ignores the
            rest, so one call site configures every kind.

    Returns:
        A brain object with ``forward``/``update``/``to_dict``/``from_dict``.

    Raises:
        ValueError: if ``kind`` is not registered.
    """
    try:
        factory = BRAIN_KINDS[kind]
    except KeyError:
        known = ", ".join(sorted(BRAIN_KINDS))
        raise ValueError(f"unknown brain kind {kind!r} (known: {known})") from None
    return factory(input_dim, output_dim, hidden_units=int(hidden_units), **params)
