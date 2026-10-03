"""
Persistent benchmark results + regression tracking.

Records benchmark output (stability, latency) to data/benchmark_results/
as JSON, tagged with model + git commit + timestamp. Compares each new run
against prior runs and reports regressions.

Usage:
    python scripts/benchmark_results.py record --kind stability --json-file out.json
    python scripts/benchmark_results.py record --kind latency --json-file out.json
    python scripts/benchmark_results.py history [--kind latency]
    python scripts/benchmark_results.py compare [--kind latency] [--vs previous|first]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from domain.shared import utc_now_iso

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "data" / "benchmark_results"

REGRESSION_THRESHOLDS = {
    "stability": {
        # higher-is-better metrics (drop = regression) use absolute thresholds
        "overall": 5.0,
        "response_rate": 0.05,
        # lower-is-better metrics (rise = regression): absolute or relative
        "crash_rate": (0.0, "abs"),
        "empty_rate": (0.05, "abs"),
        "latency_degradation": (0.2, "abs"),
        "length_cv": (0.1, "abs"),
    },
    "latency": {
        "mean_ms": (20.0, "rel"),
        "p95_ms": (20.0, "rel"),
    },
    "execution": {
        # fire-and-forget dispatch / execution-consolidation stage 1
        "dispatch_us": (50.0, "rel"),
        "peak_threads": (2, "abs"),
    },
    "startup": {
        # spawn → GET /health; the CLI kills the server past API_STARTUP_TIMEOUT
        "time_to_health_s": (20.0, "rel"),
        "time_to_ready_s": (20.0, "rel"),
        # cold imports left for the background model-load thread = race risk
        "preload_warnings": (0, "abs"),
    },
}


def git_commit() -> str | None:
    """Return short git commit of repo root, or None."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=5,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def timestamp() -> str:
    """Return ISO timestamp with timezone."""
    return utc_now_iso()


def results_path(kind: str, model: str, stamp: str) -> Path:
    """Return file path for a run result."""
    safe_model = model.replace("/", "--")
    return RESULTS_DIR / kind / f"{safe_model}_{stamp}.json"


def collect_records(kind: str) -> list[Path]:
    """Return all stored result files for a kind, newest first."""
    d = RESULTS_DIR / kind
    if not d.exists():
        return []
    return sorted(d.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)


def _extract_from_stability(raw: str) -> dict:
    """Pull the JSON report from benchmark_stability --json stdout."""
    # Report is a trailing JSON object. Try every '{' position from the
    # last one backward; the correct one parses the full trailing object.
    decoder = json.JSONDecoder()
    candidates = [i for i in range(len(raw)) if raw[i] == "{"]
    for start in reversed(candidates):
        try:
            obj, end = decoder.raw_decode(raw, start)
        except json.JSONDecodeError:
            continue
        if raw[end:].strip() == "" or raw[end:].strip().startswith("\n"):
            return obj
    raise ValueError("no complete JSON object found in stability output")


def _run_stability(url: str, runs: int) -> dict:
    """Execute benchmark_stability.py and capture its JSON report."""
    out = subprocess.run(
        [
            sys.executable,
            "scripts/benchmark_stability.py",
            "--runs",
            str(runs),
            "--json",
            "--url",
            url,
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=3600,
    )
    if out.returncode != 0 and "Error" in out.stdout:
        raise RuntimeError(f"stability benchmark failed: {out.stdout[-500:]}")
    return _extract_from_stability(out.stdout + out.stderr)


def _run_latency(url: str, runs: int, update_baseline: bool) -> dict:
    """Execute benchmark_latency.py, returning a flat metric dict."""
    args = [sys.executable, "scripts/benchmark_latency.py", "--url", url]
    if update_baseline:
        args.append("--update")
    out = subprocess.run(args, cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=1800)
    text = out.stdout + out.stderr
    metrics = {}
    for line in text.splitlines():
        m = line.strip()
        if ":" in m and "BENCH" not in m and "BASELINE" not in m:
            key, _, val = m.partition(":")
            key = key.strip().lower()
            if key.endswith("ms"):
                key = key.replace(" ", "_").replace("ms", "ms")
            if key in ("mean", "min", "max", "p50", "p95", "sample_count"):
                try:
                    metrics[
                        "mean_ms"
                        if key == "mean"
                        else "min_ms"
                        if key == "min"
                        else "max_ms"
                        if key == "max"
                        else "p50_ms"
                        if key == "p50"
                        else "p95_ms"
                        if key == "p95"
                        else key
                    ] = float(val.strip())
                except ValueError:
                    pass
    if not metrics:
        raise RuntimeError(f"no latency metrics parsed from output:\n{text}")
    return metrics


def load_result(path: Path) -> dict:
    """Load one stored result file."""
    with open(path) as f:
        return json.load(f)


# higher-is-better metric names (a drop means regression)
HIGHER_IS_BETTER = {"overall", "response_rate"}


def _threshold(kind: str, metric: str) -> tuple:
    """Return (limit, mode) where mode is 'abs' or 'rel'.

    'rel' limits are stored as percentages (e.g. 20.0 = 20%) and are
    normalized to fractions for comparison.
    """
    spec = REGRESSION_THRESHOLDS[kind][metric]
    if isinstance(spec, tuple):
        limit, mode = spec
        return (limit / 100.0, mode) if mode == "rel" else (limit, mode)
    return (spec, "abs")


def _regression_deltas(kind: str, new: dict, old: dict) -> dict:
    """Compute delta for each tracked metric. Positive = regression."""
    deltas = {}
    for metric in REGRESSION_THRESHOLDS[kind]:
        nv = _dig(new, metric)
        ov = _dig(old, metric)
        if nv is None or ov is None:
            continue
        deltas[metric] = nv - ov
    return deltas


def _dig(d: dict, dotted: str):
    """Fetch nested value by dotted path (e.g. score.overall)."""
    if dotted in d:
        return d[dotted]
    for container in ("score", "metrics"):
        if isinstance(d.get(container), dict) and dotted in d[container]:
            return d[container][dotted]
    cur = d
    for part in dotted.split("."):
        if not isinstance(cur, dict):
            return None
        if part in cur:
            cur = cur[part]
        else:
            return None
    return cur


def is_regression(kind: str, new: dict, old: dict) -> bool:
    """Return True if any tracked metric regressed beyond threshold.

    Direction-aware: higher-is-better metrics regress when they drop;
    lower-is-better metrics regress when they rise.
    """
    for metric in REGRESSION_THRESHOLDS[kind]:
        limit, mode = _threshold(kind, metric)
        nv = _dig(new, metric)
        ov = _dig(old, metric)
        if nv is None or ov is None:
            continue
        delta = nv - ov
        if mode == "rel":
            if ov == 0:
                continue
            delta = delta / abs(ov)
        if metric in HIGHER_IS_BETTER:
            if delta < -limit:
                return True
        else:
            if delta > limit:
                return True
    return False


def do_record(args) -> int:
    """Persist a run from an existing JSON file, or run live if none given."""
    kind = args.kind
    if args.json_file:
        with open(args.json_file) as f:
            raw = f.read()
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            # stability script writes progress then a trailing JSON object
            data = _extract_from_stability(raw)
    else:
        if kind == "stability":
            data = _run_stability(args.url, args.runs)
        elif kind == "latency":
            data = _run_latency(args.url, args.runs, update_baseline=False)
        elif kind == "startup":
            print(
                "[ERR] startup kind requires --json-file (from benchmark_startup.py)",
                file=sys.stderr,
            )
            return 1
        elif kind == "training":
            print(
                "[ERR] training kind requires --json-file (from benchmark_slonet_training.py)",
                file=sys.stderr,
            )
            return 1
        else:
            print(f"[ERR] unknown kind {kind}", file=sys.stderr)
            return 1

    if kind == "stability":
        model = data.get("model", "unknown")
        stamp = timestamp()
        record = {
            "kind": kind,
            "model": model,
            "commit": git_commit(),
            "timestamp": stamp,
            "runs": data.get("runs", args.runs),
            "passed": data.get("passed"),
            "score": data.get("score"),
            "elapsed_s": data.get("elapsed_s"),
        }
    else:
        model = args.model or "served"
        stamp = timestamp()
        record = {
            "kind": kind,
            "model": model,
            "commit": git_commit(),
            "timestamp": stamp,
            "metrics": data,
        }

    path = results_path(kind, model, stamp.replace(":", "").replace(".", ""))
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(record, f, indent=2, default=str)

    print(f"[RECORD] {kind} → {path}")

    prior = collect_records(kind)
    if len(prior) > 1:
        return do_compare(args)
    return 0


# Which metrics each kind stores, in display order. `metrics` is not one
# shape across kinds: latency/Qwen is a dict of *_ms, startup is *_s,
# execution is *_us, and latency/served is a list of compression samples
# rather than a dict at all. Rendering mean_ms/p95_ms for every kind aborted
# on that list and printed a row of '?' for the other two.
HISTORY_FIELDS = {
    "latency": (("mean", "mean_ms"), ("p50", "p50_ms"), ("p95", "p95_ms")),
    "startup": (("health", "time_to_health_s"), ("ready", "time_to_ready_s")),
    "execution": (("dispatch", "dispatch_us"), ("threads", "peak_threads")),
}


def _scalars(value) -> dict:
    """Flatten a record's metrics into {name: number}.

    Accepts the dict most kinds store and the list some latency files hold.
    Booleans and nested structures are dropped — the renderer wants numbers
    it can print, not structure it would have to interpret.
    """
    if isinstance(value, dict):
        items = list(value.items())
    elif isinstance(value, list):
        items = [(k, v) for item in value if isinstance(item, dict) for k, v in item.items()]
    else:
        return {}
    return {k: v for k, v in items if isinstance(v, (int, float)) and not isinstance(v, bool)}


def _fmt(value) -> str:
    """Render one metric: '?' when absent, trimmed digits when floating."""
    if value is None:
        return "?"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _describe(metrics) -> str:
    """Say what the payload actually is when none of the expected keys land.

    A row of '?' reads like missing data; naming the shape does not.
    """
    if isinstance(metrics, list):
        return f"list({len(metrics)}) non-comparable samples"
    if isinstance(metrics, dict):
        keys = ", ".join(str(k) for k in list(metrics)[:4])
        return f"metrics[{keys}]"
    return "no metrics"


def _kinds_with_runs() -> list[str]:
    """Every kind directory holding at least one run, sorted.

    The default was hard-coded to ['stability', 'latency'], which hid the
    startup and execution runs on disk — 9 of the 12 stored records.
    """
    if not RESULTS_DIR.exists():
        return []
    return sorted(p.name for p in RESULTS_DIR.iterdir() if p.is_dir() and any(p.glob("*.json")))


def _history_metrics(r: dict, kind: str) -> str:
    """Render one run's metrics for `history`, using the keys that kind has."""
    metrics = r.get("metrics")
    vals = _scalars(metrics)
    fields = HISTORY_FIELDS.get(kind, ())
    if not fields or all(vals.get(key) is None for _, key in fields):
        return _describe(metrics)
    line = " ".join(f"{label}={_fmt(vals.get(key))}" for label, key in fields)
    if kind == "startup":
        timeout, ready = vals.get("api_starting_timeout_s"), vals.get("time_to_ready_s")
        if isinstance(timeout, (int, float)) and isinstance(ready, (int, float)):
            if ready > timeout:
                line += f" [past {timeout}s timeout]"
    return line


def do_history(args) -> int:
    """List stored runs.

    With no --kind, every kind that holds runs is listed.
    """
    for kind in [args.kind] if args.kind else _kinds_with_runs():
        runs = collect_records(kind)
        print(f"── {kind}: {len(runs)} runs ──")
        for p in runs:
            r = load_result(p)
            stamp = r.get("timestamp", p.name)[:19]
            model = r.get("model", "?")
            if kind == "stability":
                sc = r.get("score", {})
                print(
                    f"  {stamp}  {model:<24} overall={sc.get('overall', '?'):<4} "
                    f"passed={'✓' if r.get('passed') else '✗'}  {p.name}"
                )
            elif kind == "training":
                ms = r.get("metrics", {})
                if isinstance(ms, dict):
                    ms = [ms]
                for c in ms:
                    conf = c.get("config") if isinstance(c, dict) else "?"
                    gate = c.get("gate_converged") if isinstance(c, dict) else None
                    final = c.get("final_loss") if isinstance(c, dict) else None
                    print(
                        f"  {stamp}  {conf:<10} gate={'✓' if gate else '✗'} final={final}  {p.name}"
                    )
            else:
                print(f"  {stamp}  {model:<24} {_history_metrics(r, kind)}  {p.name}")
    return 0


def do_compare(args) -> int:
    """Compare newest run against a prior run and report regressions."""
    runs = collect_records(args.kind)
    if len(runs) < 2:
        print(f"[INFO] need ≥2 runs of kind '{args.kind}' to compare (have {len(runs)})")
        return 0

    new = load_result(runs[0])
    if args.vs == "first":
        old = load_result(runs[-1])
    else:
        old = load_result(runs[1])

    print(
        f"── compare {args.kind}: {old.get('timestamp', '?')[:19]} → {new.get('timestamp', '?')[:19]} ──"
    )

    if args.kind == "training":

        def _best_final(r):
            ms = r.get("metrics", {})
            if isinstance(ms, dict):
                ms = [ms]
            vals = [
                float(c.get("final_loss"))
                for c in ms
                if isinstance(c, dict) and c.get("final_loss") is not None
            ]
            return min(vals) if vals else None

        nf, of = _best_final(new), _best_final(old)
        print(f"  best_final_loss        {of:<10} → {nf:<10}")
        if nf is not None and of is not None and nf > of * 1.5:
            print("  → [FAIL] training final_loss regressed vs prior run")
            return 1
        print("  → [OK] no regression vs prior run")
        return 0

    deltas = _regression_deltas(args.kind, new, old)
    regressed = is_regression(args.kind, new, old)

    for metric, delta in deltas.items():
        nv = _dig(new, metric)
        ov = _dig(old, metric)
        limit, mode = _threshold(args.kind, metric)
        flag = ""
        comp_delta = delta / abs(ov) if mode == "rel" and ov else delta
        if metric in HIGHER_IS_BETTER:
            if comp_delta < -limit:
                flag = "  ⚠ REGRESSION"
        else:
            if comp_delta > limit:
                flag = "  ⚠ REGRESSION"
        print(f"  {metric:<24} {ov:<10} → {nv:<10} Δ{delta:+.3f}{flag}")

    if regressed:
        print(f"  → [FAIL] regression detected in {args.kind}")
        return 1
    print("  → [OK] no regression vs prior run")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_rec = sub.add_parser("record", help="record a benchmark run")
    p_rec.add_argument(
        "--kind",
        required=True,
        choices=["stability", "latency", "execution", "training", "startup"],
    )
    p_rec.add_argument("--json-file", default=None, help="existing JSON output file to ingest")
    p_rec.add_argument("--url", default="http://localhost:8000")
    p_rec.add_argument("--runs", type=int, default=20)
    p_rec.add_argument("--model", default=None, help="model name (latency only)")
    p_rec.add_argument("--vs", default="previous", choices=["previous", "first"])
    p_rec.set_defaults(fn=do_record)

    p_h = sub.add_parser("history", help="list stored runs")
    p_h.add_argument(
        "--kind", default=None, choices=["stability", "latency", "execution", "training", "startup"]
    )
    p_h.set_defaults(fn=do_history)

    p_c = sub.add_parser("compare", help="compare newest vs prior run")
    p_c.add_argument(
        "--kind",
        default="stability",
        choices=["stability", "latency", "execution", "training", "startup"],
    )
    p_c.add_argument("--vs", default="previous", choices=["previous", "first"])
    p_c.set_defaults(fn=do_compare)

    args = parser.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
