"""Stress harness: amplify the test_thread_safety contention to capture r.error.

Runs N concurrent run_full_cycle calls (default 6) for R rounds (default 10),
prints any failed result's error + phase errors, and reports checkpoint dir state.
"""

from __future__ import annotations

import sys
import tempfile
import threading
import traceback
from pathlib import Path

sys.path.insert(0, "packages/core-py")
sys.path.insert(0, "packages/downcraft")
sys.path.insert(0, "apps/api/server")

DATA_TEXT = "The quick brown fox jumps over the lazy dog. " * 50
FAST_CONFIG = {
    "method": "sft",
    "data_quality_threshold": 0.0,
    "epochs": 1,
    "batch_size": 8,
    "block_size": 32,
    "max_steps": 2,
    "n_embed": 32,
    "n_layer": 2,
    "n_head": 2,
}

N = int(sys.argv[1]) if len(sys.argv) > 1 else 6
ROUNDS = int(sys.argv[2]) if len(sys.argv) > 2 else 10

failures: list[str] = []
lock = threading.Lock()
counter = {"runs": 0, "fails": 0}


def one(idx: int) -> None:
    from domain.training._internal.comprehensive_trainer import ComprehensiveTrainer

    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(DATA_TEXT)
            f.flush()
            path = f.name
        trainer = ComprehensiveTrainer()
        r = trainer.run_full_cycle(data_path=path, config=FAST_CONFIG)
        with lock:
            counter["runs"] += 1
            if not r.success:
                counter["fails"] += 1
                phases = [(p.phase.value, p.success, p.error) for p in r.phases]
                failures.append(
                    f"[idx={idx}] error={r.error!r} phases={phases}"
                )
    except Exception:
        with lock:
            counter["runs"] += 1
            counter["fails"] += 1
            failures.append(f"[idx={idx}] EXCEPTION:\n{traceback.format_exc()}")


def main() -> int:
    # pre-seed the shared checkpoint dir to exercise prune/disk-limit paths
    ck = Path("models/comprehensive")
    ck.mkdir(parents=True, exist_ok=True)
    for i in range(8):
        (ck / f"seed_{i}.soul").write_bytes(b"seed" * 1000)

    for rnd in range(ROUNDS):
        threads = [threading.Thread(target=one, args=(f"r{rnd}t{i}",)) for i in range(N)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=120)
        alive = sum(1 for t in threads if t.is_alive())
        if alive:
            print(f"round {rnd}: {alive} threads still alive after 120s (hang!)")

    print(f"runs={counter['runs']} fails={counter['fails']}")
    for f in failures[:10]:
        print("---")
        print(f)
    leftover = list(ck.glob("*.soul"))
    print(f"checkpoint dir: {len(leftover)} files")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
