"""
Persistent benchmark results + regression tracking.

Records benchmark output (stability, latency, inference, ...) as JSON, tagged
with model + git commit + timestamp + source. Compares each new run against
prior runs and reports regressions.

The store is CANONICAL and checkout-independent (card 20261031_133226): the
old script-relative ``REPO_ROOT/data/benchmark_results`` forked one history per
checkout — worktrees and the api-std lane each stranded their own records, so
no single place ever held the series that regression gating needs. Resolution:
``SLO_BENCH_DATA_DIR`` env var, else ``~/.local/share/sloughgpt/benchmark_results``.
``migrate --from <dir>`` copies an old per-checkout store in (union: never
overwrites an existing file).

Usage:
    python scripts/benchmark_results.py record --kind stability --json-file out.json
    python scripts/benchmark_results.py record --kind latency --json-file out.json
    python scripts/benchmark_results.py record --kind inference --json-file out.json
    python scripts/benchmark_results.py history [--kind latency]
    python scripts/benchmark_results.py compare [--kind latency] [--vs previous|first]
    python scripts/benchmark_results.py migrate --from <old-checkout>/data/benchmark_results
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from domain.shared import utc_now_iso

REPO_ROOT = Path(__file__).resolve().parents[1]
# Canonical store — env override for CI/containers, else the user-level
# default. NEVER repo-relative: a worktree must record into the same history
# as main or the series silently forks (the 2026-10-31 stranding: 5 decode
# runs visible only from the api-std checkout).
RESULTS_DIR = Path(
    os.environ.get("SLO_BENCH_DATA_DIR")
    or Path.home() / ".local" / "share" / "sloughgpt" / "benchmark_results"
)

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
    "vm_life": {
        # scripts/benchmark_vm_life.py — step counts are deterministic, wall
        # time carries machine-load noise (hence the wider rel limit)
        "batch_wall_ms": (50.0, "rel"),
        "step_frame_steps": (10.0, "rel"),
        "first_seed_steps.600": (25.0, "rel"),
        "reseed_steps.600": (25.0, "rel"),
    },
    "effects": {
        # show_effects.py engine-level report (card 20261024_095912).
        # Load-sensitive metrics (attention, prefill, decode) get generous
        # thresholds — runs happen on a shared box at unknown loadavg,
        # which is recorded alongside for interpretation.
        # ms-scale metrics use abs deltas: rel% on a 15ms baseline is pure
        # scheduler jitter (measured 15→38ms at loadavg 23, no real change).
        "rag_context_chars": (30.0, "rel"),
        "rag_query_ms": (100.0, "abs"),
        "attn_single_ms": (1.0, "abs"),
        "attn_multi_ms": (50.0, "rel"),
        "first_token_short_s": (50.0, "rel"),
        "first_token_rag_s": (50.0, "rel"),
        "decode_tok_s": (30.0, "rel"),
    },
    "bootstrap": {
        # GET /bootstrap → landing first paint; frontend overlay watchdog = 8s
        # rel limits are PERCENT (100.0 = 100%): load-tolerant but still
        # catching fan-out storms (pre-fix cold envelope ~19.5s vs ~1.0s)
        "first_s": (100.0, "rel"),
        "warm_p95_s": (50.0, "rel"),
        # a degraded member = partial landing page; zero tolerance
        "member_errors": (0, "abs"),
    },
    "inference": {
        # benchmark_decode_slope.py --json (greedy decode, int8 model).
        # Shared box: decode swings ±50% with loadavg alone (measured
        # 16.13 @ load 4.3 vs 11.6 @ load 8.8 on the same commit), so the
        # regression bar is generous; load_s is recorded for interpretation.
        # Headline = marginal slope; naive_tok_s.<n> gates the two ends of
        # the context-growth curve (short-context throughput and the
        # long-context fade the slope alone can hide).
        "decode_tok_s": (30.0, "rel"),
        "naive_tok_s.32": (30.0, "rel"),
        "naive_tok_s.256": (30.0, "rel"),
    },
}


# Single declaration of the recorded kinds — argparse choices and the
# history default must never drift apart (parity is mechanical, not vibes).
KINDS = (
    "stability",
    "latency",
    "execution",
    "training",
    "startup",
    "effects",
    "bootstrap",
    "inference",
)


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
HIGHER_IS_BETTER = {"overall", "response_rate", "decode_tok_s"}


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
        elif kind == "effects":
            print(
                "[ERR] effects kind requires --json-file (from show_effects.py --json)",
                file=sys.stderr,
            )
        elif kind == "bootstrap":
            print(
                "[ERR] bootstrap kind requires --json-file (from benchmark_bootstrap.py)",
                file=sys.stderr,
            )
            return 1
        elif kind == "inference":
            print(
                "[ERR] inference kind requires --json-file (from benchmark_decode_slope.py --json)",
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
            "source": "ci" if os.environ.get("CI") else "local",
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
            "source": "ci" if os.environ.get("CI") else "local",
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


def do_migrate(args) -> int:
    """Union-copy an old per-checkout store into the canonical dir.

    Copy semantics, never move: the source checkout may still be in use, and a
    failed migrate must never destroy history (same lesson as the board
    export wipe — preserve before touching). Existing canonical files are
    skipped, so re-running is idempotent.
    """
    src = Path(args.source).expanduser()
    if not src.is_dir():
        print(f"[ERR] not a directory: {src}", file=sys.stderr)
        return 1
    copied = skipped = 0
    for f in sorted(src.rglob("*.json")):
        rel = f.relative_to(src)  # <kind>/<file>.json
        dest = RESULTS_DIR / rel
        if dest.exists():
            skipped += 1
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, dest)
        copied += 1
        print(f"  copied {rel}")
    print(f"[MIGRATE] {copied} copied, {skipped} already present → {RESULTS_DIR}")
    return 0


def _metric_entries(record: dict) -> list[dict]:
    """Normalize a record's metrics into a list of dicts (never crashes).

    ``benchmark_slonet_training --results`` merges per-config results into a
    LIST; metric kinds store a single dict. The renderer must handle both —
    an un-normalized list used to kill history with
    ``'list' object has no attribute 'get'``.
    """
    ms = record.get("metrics", {})
    if isinstance(ms, dict):
        return [ms]
    if isinstance(ms, list):
        return [m for m in ms if isinstance(m, dict)]
    return []


def do_history(args) -> int:
    """List stored runs (all kinds unless --kind narrows it)."""
    for kind in [args.kind] if args.kind else list(KINDS):
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
                continue
            for c in _metric_entries(r) or [{}]:
                if kind == "training":
                    conf = str(c.get("config", "?"))[:10]
                    gate = c.get("gate_converged")
                    final = c.get("final_loss")
                    print(
                        f"  {stamp}  {conf:<10} gate={'✓' if gate else '✗'} final={final}  {p.name}"
                    )
                elif kind == "startup":
                    print(
                        f"  {stamp}  {model:<24} health={str(c.get('time_to_health_s', '?')):<7} "
                        f"ready={str(c.get('time_to_ready_s', '?')):<7}  {p.name}"
                    )
                elif kind == "execution":
                    print(
                        f"  {stamp}  {model:<24} dispatch_us={str(c.get('dispatch_us', '?')):<9} "
                        f"threads={str(c.get('peak_threads', '?')):<7}  {p.name}"
                    )
                elif kind == "effects":
                    print(
                        f"  {stamp}  {model:<24} decode={str(c.get('decode_tok_s', '?')):<7} "
                        f"attn1={str(c.get('attn_single_ms', '?')):<7} "
                        f"rag={str(c.get('rag_context_chars', '?')):<6} "
                        f"load={str(c.get('loadavg', '?')):<5}  {p.name}"
                    )
                elif kind == "bootstrap":
                    print(
                        f"  {stamp}  {model:<24} first={str(c.get('first_s', '?')):<7} "
                        f"p95={str(c.get('warm_p95_s', '?')):<7} "
                        f"errs={c.get('member_errors', '?')}  {p.name}"
                    )
                elif kind == "inference":
                    naive = c.get("naive_tok_s") or {}
                    print(
                        f"  {stamp}  {model:<24} decode={str(c.get('decode_tok_s', '?')):<7} "
                        f"r2={str(c.get('r2', '?')):<7} "
                        f"spread={str(c.get('spread_pct', '?')):<5} "
                        f"naive32={str(naive.get('32', '?')):<7} "
                        f"naive256={str(naive.get('256', '?')):<7} "
                        f"load={c.get('loadavg_end', c.get('loadavg', '?'))}  {p.name}"
                    )
                else:  # latency and unknown kinds
                    print(
                        f"  {stamp}  {model:<24} mean={str(c.get('mean_ms', '?')):<7} "
                        f"p95={str(c.get('p95_ms', '?')):<7}  {p.name}"
                    )
    return 0


# inference compare trust band — loadavg is a confounder of decode tok/s
# (measured 18.5 @ load 6.86 vs 12.86 @ load 7.96 on the same commit), so a
# FAIL across materially different load bands is box noise until proven
# otherwise. r² floor from healthy runs is 0.9977 — 0.99 keeps headroom.
INFERENCE_LOAD_BAND = 1.0
INFERENCE_R2_FLOOR = 0.99


def _inference_trust_reason(new: dict, old: dict) -> str | None:
    """Return why an inference compare verdict is untrustworthy, else None.

    Missing fields (pre-trust records) never suppress anything — the guard
    only acts on evidence, it cannot invent innocence from absent data.
    """

    def _metric(record: dict, key: str) -> float | None:
        v = record.get("metrics", {}).get(key)
        try:
            return float(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    for label, rec in (("new", new), ("old", old)):
        r2 = _metric(rec, "r2")
        if r2 is not None and r2 < INFERENCE_R2_FLOOR:
            return f"fit untrustworthy ({label} r2={r2:.4f} < {INFERENCE_R2_FLOOR})"
    nl = _metric(new, "loadavg_end") or _metric(new, "loadavg")
    ol = _metric(old, "loadavg_end") or _metric(old, "loadavg")
    if nl is not None and ol is not None and abs(nl - ol) > INFERENCE_LOAD_BAND:
        return (
            f"load band Δ{abs(nl - ol):.2f} > {INFERENCE_LOAD_BAND} (old {ol:.2f} → new {nl:.2f})"
        )
    return None


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

    if regressed and args.kind == "inference":
        reason = _inference_trust_reason(new, old)
        if reason:
            print(
                f"  → [INCONCLUSIVE] {reason} — FAIL suppressed; re-run both sides under similar load"
            )
            return 0

    if regressed:
        print(f"  → [FAIL] regression detected in {args.kind}")
        return 1
    print("  → [OK] no regression vs prior run")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_rec = sub.add_parser("record", help="record a benchmark run")
    p_rec.add_argument("--kind", required=True, choices=list(KINDS))
    p_rec.add_argument("--json-file", default=None, help="existing JSON output file to ingest")
    p_rec.add_argument("--url", default="http://localhost:8000")
    p_rec.add_argument("--runs", type=int, default=20)
    p_rec.add_argument("--model", default=None, help="model name (latency only)")
    p_rec.add_argument("--vs", default="previous", choices=["previous", "first"])
    p_rec.set_defaults(fn=do_record)

    p_h = sub.add_parser("history", help="list stored runs")
    p_h.add_argument("--kind", default=None, choices=list(KINDS))
    p_h.set_defaults(fn=do_history)

    p_c = sub.add_parser("compare", help="compare newest vs prior run")
    p_c.add_argument("--kind", default="stability", choices=list(KINDS))
    p_c.add_argument("--vs", default="previous", choices=["previous", "first"])
    p_c.set_defaults(fn=do_compare)

    p_m = sub.add_parser(
        "migrate", help="union-copy an old per-checkout store into the canonical dir"
    )
    p_m.add_argument(
        "--from", dest="source", required=True, help="old results dir (…/data/benchmark_results)"
    )
    p_m.set_defaults(fn=do_migrate)

    args = parser.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
