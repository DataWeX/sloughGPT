"""
TrainingUX — programmatic training log formatter.

Receives structured training metrics and formats them for CLI output.
All data is programmatic — no manual string building.

Usage::

    from domains.training.training_ux import TrainingUX

    ux = TrainingUX(log, total_params=124_439_808)
    ux.on_config({"epochs": 10, "lr": 3e-4, "block_size": 128, "dataset": "shakespeare"})
    ux.on_progress({"global_step": 100, "train_loss": 4.23, ...})
    ux.on_eval({"eval_loss": 3.87, "eval_ppl": 48.2})
    ux.on_complete({"best_loss": 3.45, "model_path": "models/shakespeare.soul"})
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, Optional

from domains.logging import CLILogger


class TrainingUX:
    """Programmatic training log formatter.

    Receives structured dicts and formats them for CLI output.
    All formatting is data-driven — no manual string concatenation.
    """

    def __init__(
        self,
        log: CLILogger,
        total_params: Optional[int] = None,
        on_structured: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> None:
        self._log = log
        self._total_params = total_params
        self._on_structured = on_structured
        self._epoch_start: Optional[float] = None
        self._train_start: Optional[float] = None
        self._last_loss: Optional[float] = None
        self._best_loss: Optional[float] = None
        self._step_count = 0

    def _emit(self, data: Dict[str, Any]) -> None:
        """Emit structured data to callback if set."""
        if self._on_structured:
            try:
                self._on_structured(data)
            except Exception:
                pass

    def _fmt_num(self, v: Any, decimals: int = 2) -> str:
        """Format a number with fixed decimals."""
        try:
            f = float(v)
            if f < 0.001 and f > 0:
                return f"{f:.1e}"
            return f"{f:.{decimals}f}"
        except (TypeError, ValueError):
            return str(v)

    def _fmt_eta(self, seconds: Optional[int]) -> str:
        """Format seconds to HH:MM:SS or MM:SS."""
        if seconds is None or seconds < 0:
            return "??:??"
        h, remainder = divmod(int(seconds), 3600)
        m, s = divmod(remainder, 60)
        if h > 0:
            return f"{h}:{m:02d}:{s:02d}"
        return f"{m}:{s:02d}"

    def _fmt_params(self, n: int) -> str:
        """Format parameter count: 124439808 → 124.4M."""
        if n >= 1_000_000_000:
            return f"{n / 1_000_000_000:.1f}B"
        if n >= 1_000_000:
            return f"{n / 1_000_000:.1f}M"
        if n >= 1_000:
            return f"{n / 1_000:.1f}K"
        return str(n)

    # ── Events ───────────────────────────────────────────────────────

    def on_config(self, config: Dict[str, Any]) -> None:
        """Log training configuration (called once before training starts).

        Expected keys: epochs, lr, block_size, batch_size, dataset, model_name, ...
        """
        self._train_start = time.time()

        self._log.header("Training")

        # Build key-value pairs from config (programmatic — no manual strings)
        fields = []
        if "model_name" in config:
            fields.append(("Model", str(config["model_name"])))
        if self._total_params is not None:
            fields.append(("Parameters", self._fmt_params(self._total_params)))
        if "dataset" in config:
            fields.append(("Dataset", str(config["dataset"])))
        if "epochs" in config:
            fields.append(("Epochs", str(config["epochs"])))
        if "lr" in config:
            fields.append(("Learning rate", self._fmt_num(config["lr"], decimals=1e-4)))
        if "block_size" in config:
            fields.append(("Block size", str(config["block_size"])))
        if "batch_size" in config:
            fields.append(("Batch size", str(config["batch_size"])))

        for k, v in fields:
            self._log.key_value(k, v)

        self._emit({"event": "config", **config, "total_params": self._total_params})

    def on_epoch_start(self, epoch: int, total_epochs: int, steps_per_epoch: int) -> None:
        """Log epoch start."""
        self._epoch_start = time.time()
        self._log.section(f"Epoch {epoch}/{total_epochs} ({steps_per_epoch} steps)")
        self._emit({
            "event": "epoch_start",
            "epoch": epoch,
            "total_epochs": total_epochs,
            "steps_per_epoch": steps_per_epoch,
        })

    def on_progress(self, data: Dict[str, Any]) -> None:
        """Log training step progress.

        Expected keys: global_step, total_steps, train_loss, learning_rate,
                       steps_per_sec, eta_s, progress_percent
        """
        step = data.get("global_step", "?")
        total = data.get("total_steps", "?")
        loss = data.get("train_loss")
        lr = data.get("learning_rate")
        sps = data.get("steps_per_sec")
        eta = data.get("eta_s")
        pct = data.get("progress_percent")

        self._last_loss = loss
        self._step_count += 1

        # Build progress line from data (programmatic)
        parts = [f"Step {step}/{total}"]

        if loss is not None:
            parts.append(f"loss {self._fmt_num(loss, 4)}")
        if lr is not None:
            parts.append(f"lr {self._fmt_num(lr, decimals=1e-4)}")
        if pct is not None:
            parts.append(f"{pct}%")
        if sps is not None and sps > 0:
            parts.append(f"{self._fmt_num(sps, 1)} s/s")
        if eta is not None:
            parts.append(f"ETA {self._fmt_eta(eta)}")

        self._log.info(" │ ".join(parts))
        self._emit({"event": "progress", **data})

    def on_eval(self, data: Dict[str, Any]) -> None:
        """Log evaluation results.

        Expected keys: eval_loss, eval_ppl, global_step
        """
        loss = data.get("eval_loss")
        ppl = data.get("eval_ppl")
        step = data.get("global_step", "?")

        parts = [f"Eval @{step}"]
        if loss is not None:
            parts.append(f"loss {self._fmt_num(loss, 4)}")
        if ppl is not None:
            parts.append(f"ppl {self._fmt_num(ppl, 2)}")

        self._log.success(" │ ".join(parts))

        if loss is not None:
            if self._best_loss is None or loss < self._best_loss:
                self._best_loss = loss

        self._emit({"event": "eval", **data})

    def on_checkpoint(self, data: Dict[str, Any]) -> None:
        """Log checkpoint saved.

        Expected keys: path, step, eval_loss
        """
        path = data.get("path", "?")
        self._log.step(f"Checkpoint saved: {path}")
        self._emit({"event": "checkpoint", **data})

    def on_complete(self, data: Dict[str, Any]) -> None:
        """Log training complete.

        Expected keys: best_loss, final_loss, model_path, total_steps, epochs_completed
        """
        elapsed = time.time() - self._train_start if self._train_start else None

        self._log.header("Results")

        fields = []
        if "total_steps" in data:
            fields.append(("Steps", str(data["total_steps"])))
        if "epochs_completed" in data:
            fields.append(("Epochs", str(data["epochs_completed"])))
        if self._best_loss is not None:
            fields.append(("Best loss", self._fmt_num(self._best_loss, 4)))
        if "final_loss" in data:
            fields.append(("Final loss", self._fmt_num(data["final_loss"], 4)))
        if elapsed is not None:
            fields.append(("Duration", self._fmt_eta(int(elapsed))))
        if "model_path" in data:
            fields.append(("Model", str(data["model_path"])))

        for k, v in fields:
            self._log.key_value(k, v)

        self._emit({"event": "complete", "elapsed_s": elapsed, **data})

    def on_error(self, data: Dict[str, Any]) -> None:
        """Log training error.

        Expected keys: error, step, epoch
        """
        error = data.get("error", "Unknown error")
        self._log.error(f"Training failed: {error}")
        self._emit({"event": "error", **data})

    def on_cancel(self, data: Dict[str, Any]) -> None:
        """Log training cancelled."""
        step = data.get("global_step", "?")
        self._log.warning(f"Training cancelled at step {step}")
        self._emit({"event": "cancel", **data})
