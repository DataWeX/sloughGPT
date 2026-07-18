"""
Stability Benchmark — Sequential Chat Request Test

Measures model stability across N sequential inference requests:
  - Crash rate (0 crashes = gold)
  - Latency degradation (p95 first 5 vs p95 last 5)
  - Response consistency (length variance, empty rate)
  - Memory growth trend

Usage:
    python scripts/benchmark_stability.py                  # runs against localhost:8000
    python scripts/benchmark_stability.py --url http://...    # custom server
    python scripts/benchmark_stability.py --runs 50          # 50 requests
    python scripts/benchmark_stability.py --verbose          # per-request logging

Gold Standard (pass/fail thresholds):
  - Crash rate: 0%
  - Latency degradation: p95(last 5) <= p95(first 5) * 1.20
  - Empty response rate: 0%
  - Response length variance (CV): <= 0.30
"""

import sys
import json
import time
import math
import argparse
from dataclasses import dataclass, asdict, field
from typing import List, Optional
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError


# ── Gold Standard Thresholds ────────────────────────────────────────────────

GOLD = {
    "max_crash_rate": 0.0,            # 0% — no crashes allowed
    "max_latency_degradation": 1.20,  # p95 last 5 / p95 first 5 <= 1.20
    "max_empty_rate": 0.0,            # 0% — no empty responses
    "max_length_cv": 0.30,            # coefficient of variation of response length
    "min_response_rate": 1.0,         # 100% — all requests must return 200
}


# ── Data Structures ─────────────────────────────────────────────────────────

@dataclass
class RequestRecord:
    index: int
    status: int
    latency_s: float
    response_length: int
    error: Optional[str] = None
    token_count: Optional[int] = None


@dataclass
class StabilityScore:
    crash_rate: float
    latency_degradation: float
    empty_rate: float
    length_cv: float
    response_rate: float
    overall: float  # 0-100

    def passed(self) -> bool:
        return (
            self.crash_rate <= GOLD["max_crash_rate"]
            and self.latency_degradation <= GOLD["max_latency_degradation"]
            and self.empty_rate <= GOLD["max_empty_rate"]
            and self.length_cv <= GOLD["max_length_cv"]
            and self.response_rate >= GOLD["min_response_rate"]
        )


@dataclass
class StabilityResult:
    model: str
    runs: int
    records: List[RequestRecord]
    score: StabilityScore
    elapsed_s: float
    passed: bool


# ── HTTP Helpers ────────────────────────────────────────────────────────────

def _chat_request(url: str, prompt: str, timeout: int = 120) -> tuple:
    """Send one chat request, return (status, latency_s, response_text)."""
    body = json.dumps({
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 50,
        "temperature": 0.01,  # near-deterministic for stable length CV
    }).encode()
    req = Request(f"{url}/chat", data=body, headers={"Content-Type": "application/json"})
    start = time.time()
    try:
        with urlopen(req, timeout=timeout) as resp:
            latency = time.time() - start
            data = json.loads(resp.read())
            text = data.get("message", "") or data.get("text", "")
            return resp.status, latency, text
    except HTTPError as e:
        latency = time.time() - start
        return e.code, latency, ""
    except (URLError, OSError) as e:
        latency = time.time() - start
        return 0, latency, str(e)


def _health_check(url: str) -> Optional[dict]:
    """Check if server is alive, return health JSON or None."""
    try:
        with urlopen(f"{url}/health", timeout=5) as resp:
            return json.loads(resp.read())
    except Exception:
        return None


def _resolve_model(url: str) -> str:
    """Get the loaded model name from health endpoint."""
    health = _health_check(url)
    if health:
        return health.get("model", health.get("model_type", "unknown"))
    return "unknown"


# ── Prompts ─────────────────────────────────────────────────────────────────

# Use a single prompt for all requests — response lengths are consistent,
# and we measure real latency degradation (not prompt-induced variance).
BENCHMARK_PROMPT = "Say hi in 3 words."


# ── Scoring ─────────────────────────────────────────────────────────────────

def compute_score(records: List[RequestRecord]) -> StabilityScore:
    """Compute all stability metrics from request records."""
    total = len(records)
    if total == 0:
        return StabilityScore(1.0, 99.0, 1.0, 99.0, 0.0, 0.0)

    crashes = sum(1 for r in records if r.status == 0 or r.status >= 500)
    empties = sum(1 for r in records if r.status == 200 and r.response_length == 0)
    ok = [r for r in records if r.status == 200]

    crash_rate = crashes / total
    empty_rate = empties / total if total > 0 else 1.0
    response_rate = len(ok) / total

    # Latency degradation: compare p95 of first 5 vs last 5 ok requests
    n_first = min(5, max(1, len(ok) // 2))
    n_last = min(5, max(1, len(ok) // 2))
    first_latencies = sorted([r.latency_s for r in ok[:n_first]])
    last_latencies = sorted([r.latency_s for r in ok[-n_last:]])
    p95_first = first_latencies[int(len(first_latencies) * 0.95)] if first_latencies else 0.001
    p95_last = last_latencies[int(len(last_latencies) * 0.95)] if last_latencies else 0.001
    latency_degradation = (p95_last / p95_first) if p95_first > 0 else 99.0

    # Response length variance (CV)
    lengths = [r.response_length for r in ok]
    if lengths:
        mean_l = sum(lengths) / len(lengths)
        var_l = sum((l - mean_l) ** 2 for l in lengths) / len(lengths)
        length_cv = math.sqrt(var_l) / mean_l if mean_l > 0 else 99.0
    else:
        length_cv = 99.0

    # Overall score: 0-100, weighted composite
    crash_ok = 1.0 if crash_rate == 0 else max(0, 1.0 - crash_rate * 5)
    latency_ok = 1.0 if latency_degradation <= 1.20 else max(0, 1.0 - (latency_degradation - 1.20) * 2)
    empty_ok = 1.0 if empty_rate == 0 else max(0, 1.0 - empty_rate * 5)
    cv_ok = 1.0 if length_cv <= 0.30 else max(0, 1.0 - (length_cv - 0.30) * 2)
    response_ok = 1.0 if response_rate == 1.0 else max(0, response_rate)
    overall = round(100 * (crash_ok * 0.35 + latency_ok * 0.25 + empty_ok * 0.15 + cv_ok * 0.10 + response_ok * 0.15))

    return StabilityScore(
        crash_rate=crash_rate,
        latency_degradation=latency_degradation,
        empty_rate=empty_rate,
        length_cv=length_cv,
        response_rate=response_rate,
        overall=overall,
    )


# ── Main ────────────────────────────────────────────────────────────────────

def run_benchmark(url: str, runs: int = 20, verbose: bool = False) -> StabilityResult:
    """Run stability benchmark against a live server."""
    model = _resolve_model(url)

    if verbose:
        print(f"  Server: {url}")
        print(f"  Model:  {model}")
        print(f"  Runs:   {runs}")
        print()

    records: List[RequestRecord] = []
    start_time = time.time()

    for i in range(runs):
        prompt = BENCHMARK_PROMPT
        status, latency, text = _chat_request(url, prompt)

        if verbose:
            icon = "✓" if status == 200 else "✗"
            print(f"  [{i+1:2d}/{runs}] {icon} {status} {latency*1000:6.1f}ms  len={len(text)}")

        records.append(RequestRecord(
            index=i,
            status=status,
            latency_s=latency,
            response_length=len(text),
            error=None if status == 200 else text,
        ))

    elapsed = time.time() - start_time
    score = compute_score(records)
    passed = score.passed()

    return StabilityResult(
        model=model,
        runs=runs,
        records=records,
        score=score,
        elapsed_s=elapsed,
        passed=passed,
    )


def print_report(result: StabilityResult, verbose: bool = False):
    """Print human-readable report."""
    s = result.score
    print()
    print("=" * 56)
    print("  Stability Benchmark Report")
    print("=" * 56)
    print(f"  Model:      {result.model}")
    print(f"  Runs:       {result.runs}")
    print(f"  Duration:   {result.elapsed_s:.1f}s")
    print(f"  Avg/req:    {result.elapsed_s/result.runs*1000:.0f}ms")
    print()

    failures = [r for r in result.records if r.status != 200]
    print(f"  Crashes:            {len(failures)}/{result.runs}  "
          f"({s.crash_rate*100:.0f}%)  {'✅' if s.crash_rate == 0 else '❌'}  ≤{GOLD['max_crash_rate']*100:.0f}%")
    print(f"  Latency degr.:      {s.latency_degradation:.2f}x  "
          f"{'✅' if s.latency_degradation <= GOLD['max_latency_degradation'] else '❌'}  ≤{GOLD['max_latency_degradation']}x")
    print(f"  Empty responses:    {sum(1 for r in result.records if r.status == 200 and r.response_length == 0)}/{result.runs}  "
          f"({s.empty_rate*100:.0f}%)  {'✅' if s.empty_rate == 0 else '❌'}  ≤{GOLD['max_empty_rate']*100:.0f}%")
    print(f"  Length CV:          {s.length_cv:.2f}  "
          f"{'✅' if s.length_cv <= GOLD['max_length_cv'] else '❌'}  ≤{GOLD['max_length_cv']}")
    print(f"  Response rate:      {s.response_rate*100:.0f}%  "
          f"{'✅' if s.response_rate >= GOLD['min_response_rate'] else '❌'}  ≥{GOLD['min_response_rate']*100:.0f}%")
    print()

    # Per-request detail
    if verbose:
        print("  ── Per-Request Latencies ──")
        for r in result.records:
            icon = "✓" if r.status == 200 else "✗"
            marker = " ← CRASH" if r.status == 0 else ""
            print(f"  [{r.index+1:2d}] {icon} {r.status} {r.latency_s*1000:7.1f}ms  len={r.response_length}{marker}")

    # Trend analysis
    ok = [r for r in result.records if r.status == 200]
    if len(ok) >= 4:
        half = len(ok) // 2
        first_half_avg = sum(r.latency_s for r in ok[:half]) / half
        last_half_avg = sum(r.latency_s for r in ok[-half:]) / half
        trend = (last_half_avg / first_half_avg - 1) * 100
        print(f"  ── Trend ──")
        print(f"  First {half} avg: {first_half_avg*1000:.0f}ms   Last {half} avg: {last_half_avg*1000:.0f}ms   Δ {trend:+.0f}%")
        if trend > 20:
            print(f"  ⚠ Latency increasing — possible memory leak")
        elif trend < -10:
            print(f"  ✴ Latency improving (warmup effect)")

    # Summary verdict
    print()
    print(f"  Stability Score:  {s.overall}/100")
    print(f"  Verdict:          {'✅ GOLD STANDARD' if result.passed else '❌ FAILED'}")
    print()

    if failures:
        print(f"  Failed requests:")
        for r in failures:
            print(f"    [{r.index+1}] status={r.status} error={r.error[:80] if r.error else 'unknown'}")

    print("=" * 56)
    print()


def main():
    parser = argparse.ArgumentParser(description="Stability Benchmark — Sequential Chat Request Test")
    parser.add_argument("--url", default="http://localhost:8000", help="Server URL (default: http://localhost:8000)")
    parser.add_argument("--runs", type=int, default=20, help="Number of sequential requests (default: 20)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print per-request latencies")
    parser.add_argument("--json", action="store_true", help="Output JSON report")
    args = parser.parse_args()

    # Health check
    health = _health_check(args.url)
    if health is None:
        print(f"❌ Server at {args.url} is not reachable")
        return 1

    # Health may be wrapped in {status, data} envelope
    payload = health.get("data", health)
    model_loaded = payload.get("model_loaded", health.get("status") == "healthy")
    if not model_loaded:
        print(f"⚠  Server reachable but no model loaded")
        print(f"   Health: {json.dumps(health, indent=2)[:200]}")
        proceed = input("   Continue anyway? [y/N] ")
        if proceed.lower() != "y":
            return 1

    print(f"🔍 Server at {args.url} is alive — running {args.runs} sequential requests...")
    result = run_benchmark(args.url, runs=args.runs, verbose=args.verbose)

    if args.json:
        print(json.dumps(asdict(result), indent=2, default=str))
    else:
        print_report(result, verbose=args.verbose)

    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())
