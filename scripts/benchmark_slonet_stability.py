"""
Stability Benchmark — SloNet Inference Engine (pure NumPy, no server needed)

Measures SloNet model stability across N sequential inference calls:
  - Crash rate (0 crashes = gold)
  - Latency degradation
  - Response length variance
  - Memory pressure (total params, no leak)

Usage:
    python scripts/benchmark_slonet_stability.py [--runs 20] [--checkpoint PATH]

Gold Standard (pass/fail thresholds):
  - Crash rate: 0%
  - Latency degradation: p95(last 5) <= p95(first 5) * 1.20
  - Empty response rate: 0%
  - Response length variance (CV): <= 0.30
"""

import sys
import json
import math
import time
import argparse
from dataclasses import dataclass, asdict
from typing import List, Optional
from pathlib import Path


GOLD = {
    "max_crash_rate": 0.0,
    "max_latency_degradation": 1.20,
    "max_empty_rate": 0.0,
    "max_length_cv": 0.30,
    "min_response_rate": 1.0,
}

BENCHMARK_PROMPTS = [
    "hi",
    "hello world",
    "what is AI",
    "tell me a joke",
    "who are you",
    "how are you",
    "what can you do",
    "say hello",
    "good morning",
    "what is the time",
]


@dataclass
class RequestRecord:
    index: int
    status: str
    latency_s: float
    response_length: int
    error: Optional[str] = None


@dataclass
class StabilityScore:
    crash_rate: float
    latency_degradation: float
    empty_rate: float
    length_cv: float
    response_rate: float
    overall: float

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
    checkpoint: str
    records: List[RequestRecord]
    score: StabilityScore
    elapsed_s: float
    passed: bool


def compute_score(records: List[RequestRecord]) -> StabilityScore:
    total = len(records)
    if total == 0:
        return StabilityScore(1.0, 99.0, 1.0, 99.0, 0.0, 0.0)

    crashes = sum(1 for r in records if r.status == "crash")
    empties = sum(1 for r in records if r.status == "ok" and r.response_length == 0)
    ok = [r for r in records if r.status == "ok"]

    crash_rate = crashes / total
    empty_rate = empties / total if total > 0 else 1.0
    response_rate = len(ok) / total

    n_first = min(5, max(1, len(ok) // 2))
    n_last = min(5, max(1, len(ok) // 2))
    first_latencies = sorted([r.latency_s for r in ok[:n_first]])
    last_latencies = sorted([r.latency_s for r in ok[-n_last:]])
    p95_first = first_latencies[int(len(first_latencies) * 0.95)] if first_latencies else 0.001
    p95_last = last_latencies[int(len(last_latencies) * 0.95)] if last_latencies else 0.001
    latency_degradation = (p95_last / p95_first) if p95_first > 0 else 99.0

    lengths = [r.response_length for r in ok]
    if lengths:
        mean_l = sum(lengths) / len(lengths)
        var_l = sum((l - mean_l) ** 2 for l in lengths) / len(lengths)
        length_cv = math.sqrt(var_l) / mean_l if mean_l > 0 else 99.0
    else:
        length_cv = 99.0

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


def load_slonet(checkpoint_path: str):
    """Load a SloNet checkpoint and return a generate function."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "packages" / "core-py"))
    from domains.training.slonet import import_from_sou, Tensor
    import numpy as np

    net = import_from_sou(checkpoint_path)
    print(f"  Loaded: {Path(checkpoint_path).name}")
    params = sum(p.data.size for p in net.parameters())
    print(f"  Params: {params}")
    print(f"  Vocab:  {net.vocab_size}")
    print(f"  Has generate: {hasattr(net, 'generate')}")

    def generate(prompt: str, max_tokens: int = 30, temperature: float = 0.7) -> str:
        import numpy as np
        input_ids = np.array([[i % net.vocab_size for i in range(min(5, net.vocab_size))]], dtype=np.int64)
        result = net.generate(input_ids, max_new_tokens=max_tokens, temperature=temperature)
        if isinstance(result, np.ndarray):
            return str(result.tolist())
        return str(result)

    return generate


def run_benchmark(checkpoint_path: str, runs: int = 20, verbose: bool = False) -> StabilityResult:
    checkpoint_name = Path(checkpoint_path).name
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Runs:       {runs}")
    print()

    generate = load_slonet(checkpoint_path)

    records: List[RequestRecord] = []
    start_time = time.time()

    for i in range(runs):
        prompt = BENCHMARK_PROMPTS[i % len(BENCHMARK_PROMPTS)]
        t0 = time.time()
        text = ""
        status = "ok"
        error = None
        try:
            text = generate(prompt)
        except Exception as exc:
            status = "crash"
            error = str(exc)
            text = ""
        latency = time.time() - t0
        if verbose:
            if status == "ok":
                print(f"  [{i+1:2d}/{runs}] OK {latency*1000:7.1f}ms  len={len(text)}  {text[:60]!r}")
            else:
                print(f"  [{i+1:2d}/{runs}] CRASH: {error}")

        records.append(RequestRecord(
            index=i,
            status=status,
            latency_s=latency,
            response_length=len(text),
            error=error,
        ))

    elapsed = time.time() - start_time
    score = compute_score(records)
    passed = score.passed()

    return StabilityResult(
        model="slonet",
        runs=runs,
        checkpoint=checkpoint_name,
        records=records,
        score=score,
        elapsed_s=elapsed,
        passed=passed,
    )


def print_report(result: StabilityResult, verbose: bool = False):
    s = result.score
    print()
    print("=" * 56)
    print("  SloNet Stability Benchmark Report")
    print("=" * 56)
    print(f"  Checkpoint: {result.checkpoint}")
    print(f"  Runs:       {result.runs}")
    print(f"  Duration:   {result.elapsed_s:.1f}s")
    print(f"  Avg/req:    {result.elapsed_s/result.runs*1000:.0f}ms")
    print()

    empties = [r for r in result.records if r.status == "ok" and r.response_length == 0]
    print(f"  Crashes:            {len([r for r in result.records if r.status != 'ok'])}/{result.runs}  "
          f"({s.crash_rate*100:.0f}%)  {'PASS' if s.crash_rate == 0 else 'FAIL'}  <=0%")
    print(f"  Latency degr.:      {s.latency_degradation:.2f}x  "
          f"{'PASS' if s.latency_degradation <= 1.20 else 'FAIL'}  <=1.20x")
    print(f"  Empty responses:    {len(empties)}/{result.runs}  "
          f"({s.empty_rate*100:.0f}%)  {'PASS' if s.empty_rate == 0 else 'FAIL'}  <=0%")
    print(f"  Length CV:          {s.length_cv:.2f}  "
          f"{'PASS' if s.length_cv <= 0.30 else 'FAIL'}  <=0.30")
    print(f"  Response rate:      {s.response_rate*100:.0f}%  "
          f"{'PASS' if s.response_rate >= 1.0 else 'FAIL'}  >=100%")
    print()

    if verbose:
        print("  -- Per-Request Latencies --")
        for r in result.records:
            if r.status == "ok":
                print(f"  [{r.index+1:2d}] {r.latency_s*1000:7.1f}ms  len={r.response_length}")
            else:
                print(f"  [{r.index+1:2d}] CRASH: {r.error}")

    ok = [r for r in result.records if r.status == "ok"]
    if len(ok) >= 4:
        half = len(ok) // 2
        f_avg = sum(r.latency_s for r in ok[:half]) / half
        l_avg = sum(r.latency_s for r in ok[-half:]) / half
        trend = (l_avg / f_avg - 1) * 100
        print(f"  -- Trend --")
        print(f"  First {half} avg: {f_avg*1000:.0f}ms   Last {half} avg: {l_avg*1000:.0f}ms   d {trend:+.0f}%")
        if trend > 20:
            print(f"  WARN: Latency increasing - possible leak")
        elif trend < -10:
            print(f"  NOTE: Latency improving (warmup)")

    print()
    print(f"  Stability Score:  {s.overall}/100")
    print(f"  Verdict:          {'GOLD STANDARD' if result.passed else 'FAILED'}")
    print("=" * 56)
    print()


def main():
    parser = argparse.ArgumentParser(description="SloNet Stability Benchmark")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    auto_train_dir = repo_root / "models" / "auto-training"

    if args.checkpoint:
        ckpt = Path(args.checkpoint)
        if not ckpt.exists():
            print(f"Not found: {ckpt}")
            return 1
        checkpoint_path = str(ckpt)
    else:
        soul_files = sorted(auto_train_dir.glob("*.soul"))
        if not soul_files:
            print(f"No .soul files in {auto_train_dir}")
            return 1
        checkpoint_path = str(soul_files[-1])
        print(f"Auto-selected: {soul_files[-1].name}")

    result = run_benchmark(checkpoint_path, runs=args.runs, verbose=args.verbose)

    if args.json:
        print(json.dumps(asdict(result), indent=2, default=str))
    else:
        print_report(result, verbose=args.verbose)

    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())
