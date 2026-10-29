"""Dataset import benchmark — time the GitHub-import hot path offline.

Measures the CPU-side of ``datasets.import_from_github`` with no network:

* **export** — ``RepoImporter.export_to_corpus`` over a synthetic repo tree
  (mixed .py/.js/.ts/.md/.json files): the read + truncate + language-detect
  + JSONL-write loop that dominates import time once the clone is done.
* **import_cache_hit** — end-to-end ``RepoImporter.import_from_github`` with
  a pre-seeded clone cache (marker validated): export + corpus assembly +
  ImportResult, the work a repeat import of the same repo pays.

Both are wall-clock over ``--runs`` iterations (first run reported separately
as cold-ish; mean/p95 over all runs). Metrics: ``export_files_per_s``,
``export_mb_per_s``, ``import_cache_hit_s``.

Gates (exit 1): produced file count != requested (a filter/ignore regression
silently empties corpora), an empty corpus, or ``--budget-export-files-per-s``
(default off) exceeded. Clone network time is intentionally out of scope —
``clone_repo`` shells out to git and is covered by unit tests for timeout,
filter fallback, and cache keying.

Usage:
    ./scripts/python scripts/benchmark_dataset_import.py
    ./scripts/python scripts/benchmark_dataset_import.py --files 2000 --runs 10
    ./scripts/python scripts/benchmark_dataset_import.py --json-out out.json
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
# Running a file under scripts/ puts scripts/ (not the repo root) on
# sys.path, so ``domain`` would not import. Bootstrap it.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from domain.training._internal.data_import import RepoImporter  # noqa: E402

SYNTHETIC_EXTENSIONS = (".py", ".js", ".ts", ".md", ".json")


def build_synthetic_repo(root: Path, files: int, avg_bytes: int) -> int:
    """Create a deterministic synthetic repo tree; returns bytes written."""
    written = 0
    for i in range(files):
        ext = SYNTHETIC_EXTENSIONS[i % len(SYNTHETIC_EXTENSIONS)]
        sub = root / f"pkg{i % 16}" / f"mod{i % 64}"
        sub.mkdir(parents=True, exist_ok=True)
        # Vary size around avg_bytes so truncation/skip paths stay realistic.
        payload = (f"# file {i}\n" + "x" * avg_bytes)[: avg_bytes + (i % 512)]
        (sub / f"file_{i}{ext}").write_text(payload)
        written += len(payload)
    return written


def bench_export(repo: Path, out: Path, runs: int) -> tuple[list[float], int]:
    """Time export_to_corpus; returns (seconds per run, files written)."""
    importer = RepoImporter(cache_dir=str(out / "cache"))
    samples: list[float] = []
    files = 0
    for _ in range(runs):
        t0 = time.perf_counter()
        files = importer.export_to_corpus(repo, str(out / "corpus.jsonl"))
        samples.append(time.perf_counter() - t0)
    return samples, files


def bench_import_cache_hit(repo: Path, out: Path, runs: int) -> list[float]:
    """Time import_from_github with a pre-seeded (marker-valid) clone cache."""
    url = "https://github.com/bench/synthetic"
    cache_dir = out / "cache"
    target = cache_dir / RepoImporter._repo_cache_key(url)
    target.mkdir(parents=True, exist_ok=True)
    (target / ".slough-source-url").write_text(url)

    importer = RepoImporter(cache_dir=str(cache_dir))
    samples: list[float] = []
    for i in range(runs):
        t0 = time.perf_counter()
        result = importer.import_from_github(url, f"bench_ds_{i}", output_dir=str(out / "datasets"))
        samples.append(time.perf_counter() - t0)
        if not result.success:
            raise RuntimeError(f"import failed: {result.error}")
    return samples


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--files", type=int, default=500, help="synthetic repo file count")
    parser.add_argument("--size", type=int, default=2048, help="avg bytes per file")
    parser.add_argument("--runs", type=int, default=5, help="timed iterations")
    parser.add_argument("--json-out", type=str, default="", help="write metrics JSON here")
    parser.add_argument(
        "--budget-export-files-per-s",
        type=float,
        default=0.0,
        help="gate: fail if export throughput drops below this (0 = informational)",
    )
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="slo-bench-import-") as tmp:
        tmp_path = Path(tmp)
        repo = tmp_path / "repo"
        total_bytes = build_synthetic_repo(repo, args.files, args.size)

        export_samples, files = bench_export(repo, tmp_path / "export", args.runs)
        import_samples = bench_import_cache_hit(repo, tmp_path / "import", args.runs)

        export_mean = sum(export_samples) / len(export_samples)
        metrics = {
            "kind": "dataset_import",
            "files": args.files,
            "avg_bytes": args.size,
            "runs": args.runs,
            "export_mean_s": export_mean,
            "export_files_per_s": args.files / export_mean if export_mean > 0 else 0.0,
            "export_mb_per_s": (total_bytes / (1024 * 1024)) / export_mean
            if export_mean > 0
            else 0.0,
            "export_files_written": files,
            "import_cache_hit_mean_s": sum(import_samples) / len(import_samples),
            "import_cache_hit_max_s": max(import_samples),
        }

    failures: list[str] = []
    if files != args.files:
        failures.append(
            f"export wrote {files} files, expected {args.files} (filter/ignore regression)"
        )
    if (
        args.budget_export_files_per_s > 0
        and metrics["export_files_per_s"] < args.budget_export_files_per_s
    ):
        failures.append(
            f"export throughput {metrics['export_files_per_s']:.0f} files/s "
            f"< budget {args.budget_export_files_per_s:.0f} files/s"
        )

    print("dataset import benchmark")
    print(
        f"  export        : {export_mean * 1000:8.1f} ms  ({metrics['export_files_per_s']:.0f} files/s, {metrics['export_mb_per_s']:.1f} MB/s)"
    )
    print(
        f"  import (cache) : {metrics['import_cache_hit_mean_s'] * 1000:8.1f} ms mean, {metrics['import_cache_hit_max_s'] * 1000:.1f} ms max"
    )

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(metrics, indent=2))
        print(f"  wrote {args.json_out}")

    for f in failures:
        print(f"[GATE] {f}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
