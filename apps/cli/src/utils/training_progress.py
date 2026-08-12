"""
Training progress bar for CLI training commands.

A live-updating progress bar that renders per-step training statistics —
step/total, epoch, train/eval loss, learning rate, throughput, ETA, and a
loss sparkline — from the ``on_progress`` dicts emitted by
``SloughGPTTrainer.train()``.

Render behaviour:
- TTY: single line updated in place via carriage return (no flicker).
- Non-TTY (piped / redirected / log file): one complete line per update.

The loss sparkline draws recent train loss as bars; lower loss is better, so
bars shrinking toward the left (``▇▆▄▁``) indicate improvement.
"""
import sys
import time
from typing import Dict, Optional

from utils.progress import ProgressBar, _is_terminal

SPARK_CHARS = "▁▂▃▄▅▆▇█"


class TrainingProgressBar:
    """Renders live training stats from ``SloughGPTTrainer`` progress dicts.

    Attributes:
        desc: Left-aligned description prefix on the rendered line.
        width: Fill width of the progress bar (in characters).
        total_steps: Expected total step count; if ``None`` it is inferred
            from the first progress dict (``epochs * steps_per_epoch``, or
            back-solved from ``progress_percent``).
        sparkline_len: How many recent train losses feed the sparkline.
        last_line: The most recently rendered line (for tests/inspection).
        stats: Latest aggregated stats dict (for tests/inspection).
    """

    def __init__(
        self,
        desc: str = "Training",
        width: int = 36,
        total_steps: Optional[int] = None,
        sparkline_len: int = 12,
    ):
        self.desc = desc
        self.width = max(8, int(width))
        self.total_steps = total_steps
        self.sparkline_len = max(4, int(sparkline_len))

        self._is_tty = _is_terminal()
        self._start = time.time()
        self._last_render_ts = -1e9
        self._throttle = 0.05 if self._is_tty else 0.5
        self._done = False

        self._losses: list = []
        self._eval_loss: Optional[float] = None
        self._best_eval: Optional[float] = None
        self._last_lr: Optional[float] = None

        self.last_line = ""
        self.stats: Dict[str, object] = {}

    # -- public API -----------------------------------------------------

    def update(self, info: Dict[str, object]) -> None:
        """Absorb one ``on_progress`` dict and (re)render the line.

        Args:
            info: Progress dict emitted by the trainer; recognized keys:
                ``global_step``, ``progress_percent``, ``epoch``, ``epochs``,
                ``steps_per_epoch``, ``train_loss``, ``eval_loss``,
                ``learning_rate``, ``done``.

        Side effects:
            - Writes to stdout (unless throttled).
            - Updates ``last_line`` and ``stats``.
        """
        step = int(info.get("global_step") or 0)
        pct = int(info.get("progress_percent") or 0)
        epoch = int(info.get("epoch") or 0)
        epochs = int(info.get("epochs") or 0)

        if self.total_steps is None:
            self._infer_total(info, step, pct)
        else:
            self._refine_total(step, pct)

        loss = info.get("train_loss")
        if loss is not None:
            try:
                self._losses.append(float(loss))
            except (TypeError, ValueError):
                pass

        ev = info.get("eval_loss")
        if ev is not None:
            try:
                ev_f = float(ev)
            except (TypeError, ValueError):
                ev_f = None
            if ev_f is not None:
                self._eval_loss = ev_f
                if self._best_eval is None or ev_f < self._best_eval:
                    self._best_eval = ev_f

        lr = info.get("learning_rate")
        if lr is not None:
            try:
                self._last_lr = float(lr)
            except (TypeError, ValueError):
                pass

        done = bool(info.get("done"))
        if done:
            self._done = True
            pct = 100
        self._render(step=step, pct=pct, epoch=epoch, epochs=epochs, done=done)

    def finish(self, info: Optional[Dict[str, object]] = None) -> None:
        """Finalise the bar: render at 100% and move to a fresh line.

        Args:
            info: Optional final progress dict (``done`` implied).

        Side effects:
            - Writes a completed line to stdout.
        """
        if info:
            self.update(info)
        if not self._done:
            self._done = True
            self._render(
                step=self.stats.get("step") or 0,
                pct=100,
                epoch=self.stats.get("epoch") or 0,
                epochs=self.stats.get("epochs") or 0,
                force=True,
            )
        if self._is_tty and self.last_line:
            sys.stdout.write("\n")
            sys.stdout.flush()

    # -- internals ------------------------------------------------------

    def _infer_total(self, info: Dict[str, object], step: int, pct: int) -> None:
        epochs = info.get("epochs")
        spe = info.get("steps_per_epoch")
        candidate = None
        if epochs and spe:
            candidate = int(epochs) * int(spe)
        if pct > 0 and step > 0:
            pct_total = max(step, round(step * 100.0 / pct))
            candidate = pct_total if candidate is None else min(candidate, pct_total)
        if candidate:
            self.total_steps = max(step, candidate)

    def _refine_total(self, step: int, pct: int) -> None:
        if pct <= 0 or step <= 0 or not self.total_steps:
            return
        candidate = max(step, round(step * 100.0 / pct))
        if candidate < self.total_steps and candidate >= step:
            self.total_steps = candidate

    def _bar(self, pct: int) -> str:
        width = self.width
        fill_units = width * min(100, max(0, pct)) / 100.0
        filled = int(fill_units)
        half = (fill_units - filled) >= 0.5
        bar = "█" * filled
        if half and filled < width:
            bar += "▓" + "░" * (width - filled - 1)
        else:
            bar += "░" * (width - filled)
        return bar

    def _sparkline(self) -> str:
        if len(self._losses) < 2:
            return ""
        vals = self._losses[-self.sparkline_len:]
        lo, hi = min(vals), max(vals)
        rng = hi - lo
        out = []
        for v in vals:
            if rng <= 0:
                out.append("▄")
            else:
                idx = int((v - lo) / rng * (len(SPARK_CHARS) - 1))
                out.append(SPARK_CHARS[max(0, min(len(SPARK_CHARS) - 1, idx))])
        return "".join(out)

    def _build_line(self, step: int, pct: int, epoch: int, epochs: int) -> str:
        elapsed = time.time() - self._start
        speed = step / elapsed if elapsed > 0 else 0.0
        total = self.total_steps or 0
        eta = None
        if speed > 0 and total > 0 and step < total:
            eta = (total - step) / speed

        parts = []
        if self.desc:
            parts.append(self.desc)
        parts.append(f"[{self._bar(pct)}] {min(100, pct):3d}%")

        if total > 0:
            parts.append(f"step {step}/{total}")
        elif step:
            parts.append(f"step {step}")

        if epochs:
            parts.append(f"ep {epoch}/{epochs}")

        if self._losses:
            spark = self._sparkline()
            parts.append(f"loss {self._losses[-1]:.4f}{(' ' + spark) if spark else ''}")

        if self._eval_loss is not None:
            if self._best_eval is not None and self._best_eval != self._eval_loss:
                parts.append(f"eval {self._eval_loss:.4f} (best {self._best_eval:.4f})")
            else:
                parts.append(f"eval {self._eval_loss:.4f}")

        if self._last_lr is not None:
            parts.append(f"lr {self._last_lr:.2e}")

        if speed > 0:
            parts.append(f"{speed:.1f} it/s")

        if eta is not None:
            parts.append(f"eta {ProgressBar._format_time(eta)}")

        parts.append(f"({ProgressBar._format_time(elapsed)})")
        return " ".join(parts)

    def _render(
        self,
        step: int,
        pct: int,
        epoch: int,
        epochs: int,
        done: bool = False,
        force: bool = False,
    ) -> None:
        line = self._build_line(step, pct, epoch, epochs)
        self.last_line = line
        self.stats = {
            "step": step,
            "pct": min(100, pct),
            "epoch": epoch,
            "epochs": epochs,
            "train_loss": self._losses[-1] if self._losses else None,
            "eval_loss": self._eval_loss,
            "best_eval": self._best_eval,
            "learning_rate": self._last_lr,
            "total_steps": self.total_steps,
            "done": done,
        }

        now = time.time()
        if not force and (now - self._last_render_ts) < self._throttle:
            return
        self._last_render_ts = now

        if self._is_tty:
            sys.stdout.write(f"\r\033[2K{line}")
            sys.stdout.flush()
        else:
            print(line)
