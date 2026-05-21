"""
Latency benchmark: measures inference speed and compares against a stored baseline.

Usage:
    python scripts/benchmark_latency.py          # run and compare to baseline
    python scripts/benchmark_latency.py --update  # update baseline
    python scripts/benchmark_latency.py --ci      # exit non-zero if regression >20%
"""
import json
import time
import sys
from pathlib import Path

BASELINE_FILE = Path("data/benchmark_latency_baseline.json")
SAMPLE_PROMPTS = [
    "hello world",
    "what is the meaning of life",
    "tell me a story about a cat",
    "how does machine learning work",
    "write a poem about winter",
]


def measure_latency(url: str = "http://localhost:8000", runs: int = 5) -> dict:
    """Measure chat latency for sample prompts."""
    import urllib.request
    import json as _json

    latencies = []
    for prompt in SAMPLE_PROMPTS:
        for _ in range(runs):
            payload = _json.dumps({"prompt": prompt, "max_tokens": 20, "temperature": 0.8}).encode()
            req = urllib.request.Request(
                f"{url}/labs/chat",
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            start = time.perf_counter()
            try:
                urllib.request.urlopen(req, timeout=30)
                elapsed = time.perf_counter() - start
                latencies.append(elapsed)
            except Exception as e:
                print(f"[WARN] Request failed: {e}", file=sys.stderr)

    if not latencies:
        return {"error": "no successful requests"}

    return {
        "mean_ms": (sum(latencies) / len(latencies)) * 1000,
        "min_ms": min(latencies) * 1000,
        "max_ms": max(latencies) * 1000,
        "p50_ms": sorted(latencies)[len(latencies) // 2] * 1000,
        "p95_ms": sorted(latencies)[int(len(latencies) * 0.95)] * 1000,
        "sample_count": len(latencies),
        "timestamp": time.time(),
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

    print(f"[BENCH] Measuring latency against {url}...")
    result = measure_latency(url=url, runs=3)

    if "error" in result:
        print(f"[FAIL] {result['error']}")
        sys.exit(1)

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
