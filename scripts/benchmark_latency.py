"""
Latency benchmark: measures inference speed and compares against a stored baseline.

Usage:
    python scripts/benchmark_latency.py          # run and compare to baseline
    python scripts/benchmark_latency.py --update  # update baseline
    python scripts/benchmark_latency.py --ci      # exit non-zero if regression >20%
"""

from __future__ import annotations

import json
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
MAX_TOKENS = 20
TEMPERATURE = 0.01


def _host_context() -> dict:
    """Thermal/load context for a recorded baseline.

    This laptop clamps all-core clocks ~40% once the package hits ~97C
    (measured 2026-09-30: 2.3-2.5GHz vs 4.1GHz max turbo), so baselines
    recorded hot vs cool are not comparable without this context.
    """
    ctx = {}
    try:
        for zone in sorted(Path("/sys/class/thermal").glob("thermal_zone*")):
            if (zone / "type").read_text().strip() == "x86_pkg_temp":
                ctx["pkg_temp_c"] = round(int((zone / "temp").read_text()) / 1000, 1)
                break
    except Exception:
        pass
    try:
        ctx["load1"] = float(Path("/proc/loadavg").read_text().split()[0])
    except Exception:
        pass
    return ctx


def _wait_for_thermal(max_c: float = 88.0, timeout_s: float = 120.0) -> dict:
    """Pause until the package has thermal headroom before measuring.

    This laptop clamps all-core clocks ~40% at ~97C: consecutive hot runs
    drifted 1915 -> 2253 -> 3018 ms (2026-09-30), which would bake the
    throttle into the baseline and false-positive the 20% CI gate. Waits up
    to ``timeout_s``; proceeds anyway (recording ok=False) rather than hang.
    """
    deadline = time.monotonic() + timeout_s
    waited = 0.0
    while True:
        temp = _host_context().get("pkg_temp_c")
        if temp is None or temp <= max_c:
            return {"waited_s": round(waited, 1), "start_temp_c": temp, "ok": True}
        if time.monotonic() >= deadline:
            return {"waited_s": round(waited, 1), "start_temp_c": temp, "ok": False}
        time.sleep(5)
        waited += 5


def measure_latency(url: str = "http://localhost:8000", runs: int = 5) -> dict:
    """Measure chat latency for sample prompts."""
    import json as _json
    import urllib.request

    latencies = []
    for prompt in SAMPLE_PROMPTS:
        for _ in range(runs):
            payload = _json.dumps(
                {
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": MAX_TOKENS,
                    "temperature": TEMPERATURE,
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
                urllib.request.urlopen(req, timeout=120)
                elapsed = time.perf_counter() - start
                latencies.append(elapsed)
            except Exception as e:
                print(f"[WARN] Request failed: {e}", file=sys.stderr)

    if not latencies:
        return {"error": "no successful requests"}

    model = None
    try:
        with urllib.request.urlopen(f"{url}/health", timeout=5) as resp:
            model = _json.loads(resp.read()).get("data", {}).get("model_type")
    except Exception:
        pass

    return {
        "mean_ms": (sum(latencies) / len(latencies)) * 1000,
        "min_ms": min(latencies) * 1000,
        "max_ms": max(latencies) * 1000,
        "p50_ms": sorted(latencies)[len(latencies) // 2] * 1000,
        "p95_ms": sorted(latencies)[int(len(latencies) * 0.95)] * 1000,
        "sample_count": len(latencies),
        "timestamp": time.time(),
        "model": model,
        "endpoint": "POST /chat",
        "params": {"max_tokens": MAX_TOKENS, "temperature": TEMPERATURE},
    }


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
    args = set(sys.argv[1:])
    update = "--update" in args
    ci_mode = "--ci" in args
    url = "http://localhost:8000"

    thermal = _wait_for_thermal()
    print(
        f"[BENCH] thermal: start_temp={thermal.get('start_temp_c')}C "
        f"waited={thermal['waited_s']}s ok={thermal['ok']}"
    )

    print(f"[BENCH] Measuring latency against {url}...")
    result = measure_latency(url=url, runs=3)

    if "error" in result:
        print(f"[FAIL] {result['error']}")
        sys.exit(1)

    result["host"] = _host_context()
    result["thermal_wait"] = thermal

    print(f"  mean: {result['mean_ms']:.1f} ms  ({result['sample_count']} samples)")
    print(f"  min:  {result['min_ms']:.1f} ms")
    print(f"  max:  {result['max_ms']:.1f} ms")
    print(f"  p50:  {result['p50_ms']:.1f} ms")
    print(f"  p95:  {result['p95_ms']:.1f} ms")

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

