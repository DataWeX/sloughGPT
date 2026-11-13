"""
Soul-index benchmark (card 20261103_085029).

A trained ``.soul`` embeds its Adam optimizer state as JSON —
``metadata.training_state.optimizer`` is ~41 MB of float text — so one
``json.loads`` of a soul config costs ~2.2 s. ``SloManager.__init__`` used to
parse every candidate on construction, which is the 11 s blank window between
"all routers registered" and "Found 5 souls" on the 2026-11-03 cold start.

This measures the three numbers the fix is responsible for:

  init          constructing ``SloManager`` (must not scan — ~free)
  cold index    first scan with no ``data/.soul_index.json`` (the one-time cost)
  warm index    a restart with the index present (stat + JSON only)

and asserts cold and warm discover the *same* souls. Inventory is not
readiness, so only the warm number is gated; the cold number is reported so a
regression in ``_parse_soul_info`` is visible instead of silent.

Boot-level readiness is measured by ``scripts/benchmark_startup.py``
(spawn → ``/health``) — this script stays model-free and finishes in seconds.

Usage:
    ./scripts/python scripts/benchmark_soul_index.py
    ./scripts/python scripts/benchmark_soul_index.py --json out.json
    ./scripts/python scripts/benchmark_soul_index.py --souls-dir models
Exit code is non-zero when init scans, cold and warm disagree, or the indexed
scan exceeds ``--max-index-seconds`` (default 1.0 s).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
for _p in (
    str(_REPO),
    str(_REPO / "packages" / "downcraft"),
    str(_REPO / "packages" / "core-py"),
    str(_REPO / "packages" / "mogdb" / "src"),
    str(_REPO / "apps" / "api" / "server"),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--souls-dir", default=str(_REPO / "models"))
    ap.add_argument("--max-index-seconds", type=float, default=1.0)
    ap.add_argument("--max-init-seconds", type=float, default=0.05)
    ap.add_argument("--json", default="", help="write results to this path")
    args = ap.parse_args()

    from domain.inference._internal.slo_manager import SloManager

    souls_dir = Path(args.souls_dir)
    if not souls_dir.exists():
        print(f"souls dir not found: {souls_dir}", file=sys.stderr)
        return 2

    # A throwaway index keeps the developer's real cache untouched and makes
    # "cold" genuinely cold.
    with tempfile.TemporaryDirectory(prefix="slo-soul-index-") as tmp:
        os.environ["SLO_SOUL_INDEX_PATH"] = str(Path(tmp) / "cold.json")
        try:
            started = time.perf_counter()
            manager = SloManager(souls_dir=str(souls_dir))
            init_seconds = time.perf_counter() - started

            started = time.perf_counter()
            cold_count = manager.warm()
            cold_seconds = time.perf_counter() - started
            cold_souls = sorted(manager._souls_cache)

            restarted = SloManager(souls_dir=str(souls_dir))
            started = time.perf_counter()
            warm_count = restarted.warm()
            warm_seconds = time.perf_counter() - started
            warm_souls = sorted(restarted._souls_cache)
        finally:
            os.environ.pop("SLO_SOUL_INDEX_PATH", None)

    failures: list[str] = []
    if cold_souls != warm_souls:
        failures.append(
            f"indexed scan disagrees with cold scan: cold={cold_souls} warm={warm_souls}"
        )
    if warm_count != cold_count:
        failures.append(f"soul count mismatch: cold={cold_count} warm={warm_count}")
    if init_seconds > args.max_init_seconds:
        failures.append(
            f"SloManager() took {init_seconds:.3f}s — construction must not scan "
            f"(gate {args.max_init_seconds:.3f}s)"
        )
    if warm_seconds > args.max_index_seconds:
        failures.append(
            f"indexed scan {warm_seconds:.3f}s exceeds {args.max_index_seconds:.3f}s gate"
        )

    print(f"souls dir: {souls_dir}")
    print(f"  init (no scan)     {init_seconds * 1000:8.1f} ms")
    print(f"  scan, cold index   {cold_seconds:8.2f} s   ({cold_count} souls)")
    print(f"  scan, warm index   {warm_seconds * 1000:8.1f} ms   ({warm_count} souls)")
    print(f"  souls: {', '.join(cold_souls) or '(none)'}")
    if cold_seconds > 0:
        print(f"  speedup: {cold_seconds / max(warm_seconds, 1e-9):.0f}× on the indexed path")
    print()
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
    else:
        print("PASS: lazy + indexed soul scan")

    payload = {
        "souls_dir": str(souls_dir),
        "init_seconds": round(init_seconds, 4),
        "cold_seconds": round(cold_seconds, 3),
        "warm_seconds": round(warm_seconds, 4),
        "cold_souls": cold_souls,
        "warm_souls": warm_souls,
        "failures": failures,
        "passed": not failures,
    }
    if args.json:
        Path(args.json).write_text(json.dumps(payload, indent=2))
        print(f"wrote {args.json}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
