# Training OOP Refactor Plan (v2 — Simplified Start)

## Problem

Three training files duplicate gradient clipping, zero-grad, vocab construction, and training loops.

## Phase 1: Training Strategies + Simple Loop (NOW)

### New files

1. **`domain/training/_internal/training_strategies.py`** — Protocols + implementations + utilities
2. **`domain/training/_internal/training_loop.py`** — Simple compositional loop

### `training_strategies.py` contents

```python
# --- Protocols ---
class BatchSampler(Protocol):
    def get_batch(self, batch_size: int) -> tuple[np.ndarray, np.ndarray]: ...
    def __len__(self) -> int: ...


class GradientHandler(Protocol):
    def step(self, model, loss, optimizer) -> dict: ...
    def zero_grads(self, model) -> None: ...


class LossTracker(Protocol):
    def update(self, loss_val: float) -> float: ...
    @property
    def current(self) -> float: ...


class CheckpointSaver(Protocol):
    def save(self, model, path, **metadata) -> str: ...
    def load(self, path, model) -> bool: ...
    def latest_path(self) -> str | None: ...


# --- Implementations ---
class RandomBlockSampler: ...  # flat data → random blocks


class PermutationSampler: ...  # fixed sequences → shuffled mini-batches


class ChatPairSampler: ...  # chat pairs → token blocks


class DirectGradientHandler: ...  # single backward → clip → step → zero


class AccumulationGradientHandler: ...  # multi-step accumulation


class RawLossTracker: ...  # simple average


class EMALossTracker: ...  # exponential moving average


class SoulCheckpointSaver: ...  # .soul format, rotation


class NpzCheckpointSaver: ...  # .npz format, filtered tensors


# --- Utilities ---
def clip_gradients(params, max_norm) -> float: ...
def zero_grads(params) -> None: ...
def build_char_vocab(texts, special_tokens=None, offset=0) -> tuple[dict, dict]: ...
```

### `training_loop.py` — Simple version (Phase 1)

```python
@dataclass
class TrainingLoopConfig:
    epochs: int = 5
    batch_size: int = 8
    grad_clip: float = 1.0
    gradient_accumulation_steps: int = 1  # from train_pipeline
    nan_tolerance: int = 10              # from consciousness + train_pipeline
    log_interval: int = 10
    checkpoint_dir: str = "checkpoints"
    checkpoint_every: int = 1

class TrainingLoop:
    """Simple compositional loop — Phase 1."""

    def __init__(
        self, model, optimizer,
        batch_sampler: BatchSampler,
        gradient_handler: GradientHandler,
        loss_tracker: LossTracker,
        checkpoint_saver: CheckpointSaver,
        config: TrainingLoopConfig,
        on_step: Callable | None = None,
        on_epoch: Callable | None = None,
    ): ...

    def train(self) -> TrainResult:
        # Single loop with:
        # - Gradient accumulation
        # - NaN/Inf guard
        # - Checkpoint saving
        # - on_step/on_epoch hooks
```

### Refactor order (Phase 1)

1. Create `training_strategies.py` — all protocols + implementations + utilities
2. Create `training_loop.py` — simple loop
3. Refactor `consciousness/training.py` — simplest, use as proof of concept
4. Refactor `chat_trainer.py` — use as second validation
5. Defer `train_pipeline.py` to Phase 2 (most complex, needs EWC/EMA/eval)

### Phase 2 (later)

Add to `TrainingLoop`:

- EMA loss smoothing
- Cancel/pause events
- Eval integration
- EWC penalty
- Training monitor
- Max steps limit
- Refactor `train_pipeline.py`

### Verification

1. `python -c "from domain.cognition._internal.consciousness.training import ConsciousnessTrainer, TrainingConfig; print('OK')"`
2. `pytest tests/core-py/test_train_pipeline.py tests/core-py/test_slonet_chat_trainer.py`
3. Start server, verify `/consciousness/status` returns 200
