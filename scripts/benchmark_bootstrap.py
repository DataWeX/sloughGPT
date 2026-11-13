"""Bootstrap benchmark — time ``GET /bootstrap``, the landing page's first paint.

The landing journey makes exactly one domain call before first paint: the
bootstrap envelope (``data.members`` + ``data.errors``). This measures that
call the way the frontend experiences it and gates the two properties that
break the journey:

* **latency** — the frontend's overlay watchdog trips at 8s
  (``overlay_stuck``); the envelope runs members concurrently, so total time
  is the slowest member, not the sum.
* **completeness** — a member error means a degraded landing page (partial
  model card); the envelope reports it rather than 503-ing, so the gate
  catches it here instead of in the journey.

Metrics (seconds; JSON keys match ``benchmark_results.REGRESSION_THRESHOLDS``
for kind ``bootstrap``):

* ``first_s`` — first request in this run. Against a freshly started server
  this is the cold number (one-time imports + cold TTL caches); against a
  warm process it converges toward warm.
* ``warm_mean_s`` / ``warm_p95_s`` / ``warm_max_s`` — over ``--runs``
  subsequent requests.
* ``member_errors`` — keys in the last response's ``data.errors``.
* ``members`` — keys in ``data.members`` (12 with the shallow-wiring build).

Gates (exit 1): ``first_s`` or ``warm_p95_s`` past ``--budget-s`` (default 8s
= the overlay watchdog), ``member_errors > 0``, an empty member map, or a
non-2xx / unparseable response (including a 404 from a server build that
lacks the bootstrap router).

Usage:
    python scripts/benchmark_bootstrap.py                       # 20 warm runs
    python scripts/benchmark_bootstrap.py --runs 5
    python scripts/benchmark_bootstrap.py --json-out out.json
    python scripts/benchmark_bootstrap.py --record              # bench + persist
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
# Running a file under scripts/ puts scripts/ (not the repo root) on
# sys.path, so ``domain`` would not import. Bootstrap it.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def percentile(samples: list[float], p: float) -> float:
    """Nearest-rank percentile of a non-empty list (p in 0..100)."""
    ordered = sorted(samples)
    idx = max(0, math.ceil(p / 100.0 * len(ordered)) - 1)
    return ordered[idx]


def stats(samples: list[float]) -> dict[str, float]:
    """Return mean/p95/max for a non-empty list of seconds."""
    return {
        "mean": sum(samples) / len(samples),
        "p95": percentile(samples, 95),
        "max": max(samples),
    }


def summarize_envelope(payload: dict) -> tuple[dict, dict]:
    """Return ``(members, errors)`` from a bootstrap response envelope.

    Accepts the ``success_response`` shape ``{"status": ..., "data": {...}}``
    as well as a bare ``{"members": ..., "errors": ...}`` body; raises
    ``ValueError`` on anything else so a shape drift fails the benchmark
    instead of silently reporting zeroes.
    """
    if not isinstance(payload, dict):
        raise ValueError(f"non-dict response: {type(payload).__name__}")
    data = payload.get("data") if isinstance(payload.get("data"), dict) else payload
    if "members" not in data:
        raise ValueError(f"response has no members map (keys={sorted(data)!r})")
    members = data.get("members")
    errors = data.get("errors") or {}
    if not isinstance(members, dict) or not isinstance(errors, dict):
        raise ValueError("members/errors must be dicts")
    return members, errors


def measure(url: str, runs: int, timeout: float) -> dict:
    """Time ``first_s`` plus ``runs`` warm requests; return metric payload."""
    samples: list[float] = []
    members: dict = {}
    errors: dict = {}
    for i in range(runs + 1):
        start = time.perf_counter()
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read()
        elapsed = time.perf_counter() - start
        payload = json.loads(body)
        members, errors = summarize_envelope(payload)
        samples.append(elapsed)
    warm = stats(samples[1:]) if len(samples) > 1 else None
    return {
        "first_s": round(samples[0], 4),
        "warm_mean_s": round(warm["mean"], 4) if warm else None,
        "warm_p95_s": round(warm["p95"], 4) if warm else None,
        "warm_max_s": round(warm["max"], 4) if warm else None,
        "runs": runs,
        "members": len(members),
        "member_errors": len(errors),
        "member_error_keys": sorted(errors),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default="http://localhost:8000/bootstrap")
    ap.add_argument("--runs", type=int, default=20, help="warm requests after the first")
    ap.add_argument("--timeout", type=float, default=30.0, help="per-request timeout (s)")
    ap.add_argument(
        "--budget-s",
        type=float,
        default=8.0,
        help="latency budget = frontend overlay watchdog (s)",
    )
    ap.add_argument("--json-out", default=None, help="write raw metrics JSON here")
    ap.add_argument("--record", action="store_true", help="persist via benchmark_results record")
    args = ap.parse_args()

    try:
        metrics = measure(args.url, args.runs, args.timeout)
    except urllib.error.HTTPError as exc:
        hint = ""
        if exc.code == 404:
            hint = " — server build lacks the bootstrap router (error-funnel lineage)"
        print(f"[FAIL] GET {args.url} → HTTP {exc.code}{hint}", file=sys.stderr)
        return 1
    except (urllib.error.URLError, OSError) as exc:
        print(f"[FAIL] GET {args.url} unreachable: {exc}", file=sys.stderr)
        return 1
    except (json.JSONDecodeError, ValueError) as exc:
        print(f"[FAIL] GET {args.url} did not return a bootstrap envelope: {exc}", file=sys.stderr)
        return 1

    payload = {"url": args.url, "budget_s": args.budget_s, **metrics}
    print(json.dumps(payload, indent=2, default=str))

    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(payload, indent=2, default=str))
        print(f"[WROTE] {args.json_out}")

    rc = 0
    if not metrics["members"]:
        print("[FAIL] empty members map — landing page would paint nothing", file=sys.stderr)
        rc = 1
    if metrics["member_errors"]:
        print(
            f"[FAIL] {metrics['member_errors']} member error(s): "
            f"{', '.join(metrics['member_error_keys'])}",
            file=sys.stderr,
        )
        rc = 1
    for name, key in (("first", "first_s"), ("warm_p95", "warm_p95_s")):
        val = metrics[key]
        if val is not None and val > args.budget_s:
            print(
                f"[FAIL] {name} {val:.3f}s > budget {args.budget_s}s — "
                "overlay watchdog would report overlay_stuck",
                file=sys.stderr,
            )
            rc = 1

    if args.record and rc == 0:
        json_file = args.json_out or "/tmp/opencode/bootstrap_bench.json"
        if not args.json_out:
            Path(json_file).write_text(json.dumps(payload, indent=2, default=str))
        sys.path.insert(0, str(REPO_ROOT / "scripts"))
        import benchmark_results  # noqa: PLC0415

        rc = benchmark_results.do_record(
            SimpleNamespace(
                kind="bootstrap",
                json_file=json_file,
                url=args.url,
                runs=args.runs,
                model=None,
                vs="previous",
            )
        )
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
