"""Targeted race repro: _enforce_disk_limit's unguarded stat vs concurrent prune.

Thread A loops _enforce_disk_limit (glob -> sum(f.stat())). Thread B loops
_prune_stale_checkpoints(keep_final=True) (unlinks everything but newest).
If FileNotFoundError escapes A's loop, the race is proven.
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, "packages/core-py")
sys.path.insert(0, "packages/downcraft")
sys.path.insert(0, "apps/api/server")

CK = Path("models/comprehensive")
ROUNDS = int(sys.argv[1]) if len(sys.argv) > 1 else 200
FILES = 40

errors: list[str] = []
lock = threading.Lock()


class FakeTrainer:
    """Minimal object exposing the two methods under test."""

    config = type("C", (), {"checkpoint_dir": "models/comprehensive", "max_checkpoints": 1})()

    def __init__(self) -> None:
        from domain.training._internal.train_pipeline import SloughGPTTrainer

        self._enforce = SloughGPTTrainer._enforce_disk_limit.__get__(self)
        self._prune = SloughGPTTrainer._prune_stale_checkpoints.__get__(self)
        self._best_model_path = None
        self._last_checkpoint_path = None


def seed(n: int) -> None:
    CK.mkdir(parents=True, exist_ok=True)
    for p in CK.glob("*.soul"):
        p.unlink()
    for i in range(n):
        (CK / f"seed_{i}.soul").write_bytes(b"x" * 4096)


def hammer(t: FakeTrainer) -> None:
    for _ in range(ROUNDS):
        try:
            t._enforce(CK)
        except Exception as exc:  # noqa: BLE001 — we WANT any escape
            with lock:
                errors.append(f"_enforce_disk_limit raised: {type(exc).__name__}: {exc}")
            return


def pruner(t: FakeTrainer) -> None:
    for _ in range(ROUNDS):
        try:
            t._prune(keep_final=True)
        except Exception as exc:  # noqa: BLE001
            with lock:
                errors.append(f"_prune_stale_checkpoints raised: {type(exc).__name__}: {exc}")
            return


def main() -> int:
    seed(FILES)
    ta, tb = FakeTrainer(), FakeTrainer()
    threads = [
        threading.Thread(target=hammer, args=(ta,)),
        threading.Thread(target=pruner, args=(tb,)),
        threading.Thread(target=hammer, args=(ta,)),
    ]
    t0 = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=300)
    print(f"rounds={ROUNDS} elapsed={time.time() - t0:.1f}s errors={len(errors)}")
    for e in errors[:5]:
        print("PROVEN:", e)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
