"""
Latency benchmark: measures inference speed and compares against a stored baseline.

Usage:
    python scripts/benchmark_latency.py          # run and compare to baseline
    python scripts/benchmark_latency.py --update  # update baseline
    python scripts/benchmark_latency.py --ci      # exit non-zero if regression >20%
    python scripts/benchmark_latency.py --url http://localhost:8010  # target another server
    python scripts/benchmark_latency.py --json-out run.json  # dump metrics for
        `benchmark_results.py record --kind latency --json-file run.json`
"""

import json
import os
import sys
import time
import uuid
from pathlib import Path

BASELINE_FILE = Path("data/benchmark_latency_baseline.json")
SAMPLE_PROMPTS = [
    "hello world",
    "what is the meaning of life",
    "tell me a story about a cat",
    "how does machine learning work",
    "write a poem about winter",
]

# Mirror of /chat's rate limit (apps/api/server/infrastructure/
# rate_limit_middleware.py: RATE_LIMIT_REQUESTS per RATE_LIMIT seconds —
# 10 req / 60 s at time of writing). Spacing must stay comfortably UNDER
# that budget: the pre-spacing version fired 15 requests back-to-back,
# requests 11+ got 429'd and silently dropped, and the survivors reported
# a fake ~8 ms min (the "rate-limit trap" recorded on card
# 20261102_001616_latency-regression-56x). Keep the margin >= 1 s.
RATE_LIMIT_REQUESTS = 10
RATE_LIMIT_WINDOW_S = 60.0
SPACING_S = RATE_LIMIT_WINDOW_S / RATE_LIMIT_REQUESTS + 1.0  # 7.0 s


def _host_snapshot() -> dict:
    """Host context for comparing runs — the Sep 30 baseline recorded it.

    Load + memory + swap directly serve the regression card's leads: the
    Nov 1 session ran with swap fully exhausted (3.8/3.8 GiB) and nobody
    knows whether Sep 30 did too. A latency number without this is
    half a data point.
    """
    snap: dict = {}
    try:
        snap["load1"] = os.getloadavg()[0]
    except OSError:
        pass
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, _, rest = line.partition(":")
            if key in ("MemAvailable", "SwapTotal", "SwapFree"):
                snap[key + "_kB"] = int(rest.split()[0])
    except (OSError, ValueError, IndexError):
        pass
    return snap


def measure_latency(url: str = "http://localhost:8000", runs: int = 5) -> dict:
    """Measure chat latency for sample prompts (rate-limit-safe spacing)."""
    import json as _json
    import urllib.error
    import urllib.request

    latencies = []
    ttbs = []
    rate_limited = 0
    for prompt in SAMPLE_PROMPTS:
        for _ in range(runs):
            payload = _json.dumps(
                {
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 20,
                    "temperature": 0.01,
                    "session_id": f"bench-{uuid.uuid4().hex[:8]}",
                }
            ).encode()
            req = urllib.request.Request(
                f"{url}/chat",
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            start = time.perf_counter()
            try:
                with urllib.request.urlopen(req, timeout=120) as resp:
                    # /chat is an SSE stream: urlopen returns when the
                    # HEADERS land (8-28 ms), so clocking here measures
                    # time-to-first-byte, NOT generation — drain the body
                    # to EOF first. (Found by a "98.8% improvement" run
                    # that was measuring TTFB all along.)
                    first_byte = time.perf_counter() - start
                    while resp.read(65536):
                        pass
                latencies.append(time.perf_counter() - start)
                ttbs.append(first_byte)
            except urllib.error.HTTPError as e:
                if e.code == 429:
                    rate_limited += 1
                print(f"[WARN] Request failed: {e}", file=sys.stderr)
            except Exception as e:
                print(f"[WARN] Request failed: {e}", file=sys.stderr)
            # Pace to the rate-limit budget AFTER the request (its own
            # latency time is part of the window too).
            time.sleep(SPACING_S)

    if not latencies:
        return {"error": "no successful requests"}

    result = {
        "mean_ms": (sum(latencies) / len(latencies)) * 1000,
        "min_ms": min(latencies) * 1000,
        "max_ms": max(latencies) * 1000,
        "p50_ms": sorted(latencies)[len(latencies) // 2] * 1000,
        "p95_ms": sorted(latencies)[int(len(latencies) * 0.95)] * 1000,
        "sample_count": len(latencies),
        # Time-to-first-byte ≈ queue + prefill + first token; the tail
        # (total - ttfb) ≈ decode — the split the regression bisect needs.
        "ttfb_mean_ms": (sum(ttbs) / len(ttbs)) * 1000,
        "ttfb_max_ms": max(ttbs) * 1000,
        "timestamp": time.time(),
        "host": _host_snapshot(),
        "spacing_s": SPACING_S,
        "rate_limited": rate_limited,
    }
    if rate_limited:
        # Loud, never silent: dropped samples poison the stats (429s also
        # come back FAST, which fakes a great min).
        result["error"] = f"{rate_limited} requests rate-limited (429) — stats unreliable"
    return result


def load_baseline() -> dict:
    if BASELINE_FILE.exists():
        with open(BASELINE_FILE) as f:
            return json.load(f)
    return {}


def save_baseline(data: dict):
    BASELINE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(BASELINE_FILE, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"[BASELINE] Saved to {BASELINE_FILE}")


def main():
    argv = sys.argv[1:]
    args = set(argv)
    update = "--update" in args
    ci_mode = "--ci" in args
    url = "http://localhost:8000"
    if "--url" in args:
        idx = argv.index("--url")
        if idx + 1 < len(argv):
            url = argv[idx + 1]
        else:
            print("[FAIL] --url requires a value", file=sys.stderr)
            sys.exit(2)

    json_out = None
    if "--json-out" in args:
        idx = argv.index("--json-out")
        if idx + 1 < len(argv):
            json_out = argv[idx + 1]
        else:
            print("[FAIL] --json-out requires a value", file=sys.stderr)
            sys.exit(2)

    print(f"[BENCH] Measuring latency against {url}...")
    result = measure_latency(url=url, runs=3)

    if json_out:
        Path(json_out).write_text(json.dumps(result, indent=2, default=str))
        print(f"[BENCH] JSON written to {json_out}")

    if "error" in result:
        print(f"[FAIL] {result['error']}")
        sys.exit(1)

    print(f"  mean: {result['mean_ms']:.1f} ms  ({result['sample_count']} samples)")
    print(f"  min:  {result['min_ms']:.1f} ms")
    print(f"  max:  {result['max_ms']:.1f} ms")
    print(f"  p50:  {result['p50_ms']:.1f} ms")
    print(f"  p95:  {result['p95_ms']:.1f} ms")
    if "ttfb_mean_ms" in result:
        print(f"  ttfb: {result['ttfb_mean_ms']:.1f} ms mean / {result['ttfb_max_ms']:.1f} ms max")

    if update:
        save_baseline(result)
        print("[BENCH] Baseline updated")
        return

    baseline = load_baseline()
    if not baseline:
        print("[BENCH] No baseline found. Run with --update to create one.")
        return

    change = ((result["mean_ms"] - baseline["mean_ms"]) / baseline["mean_ms"]) * 100
    print(f"  vs baseline: {baseline['mean_ms']:.1f} ms → Δ{change:+.1f}%")

    if change > 20:
        print(f"[REGRESSION] {change:.1f}% slower than baseline (>20% threshold)")
        if ci_mode:
            sys.exit(1)
    elif change < -20:
        print(f"[IMPROVEMENT] {change:.1f}% faster than baseline")
    else:
        print("[OK] Within acceptable range (±20%)")


if __name__ == "__main__":
    main()
