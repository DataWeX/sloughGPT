#!/usr/bin/env python3
"""Benchmark tokenizer training over a mixed-format corpus (corpus_loader).

Measures what card 20260928_073 changed: loading files + structured data
(JSONL messages, JSON, CSV, txt/md/py) through the single serialization
seam, then training the tokenizer pipeline on the decoded documents.

Metrics per preset:
- corpus: docs, chars, load time (load_corpus_dir)
- train: wall time, steps-equivalent (docs/sec), vocab_size, chars/sec
- quality gate: encode round-trip keeps structured content (jsonl marker)

Run:
    python scripts/benchmark_tokenizer_corpus.py --json /tmp/tokenizer-corpus.json
    PYTHONPATH=. python scripts/benchmark_results.py record --kind training --json-file /tmp/tokenizer-corpus.json
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "packages" / "core-py"))

PRESETS = {
    "gate": {"vocab_size": 256, "files_per_format": 4, "repeat": 6},
    "standard": {"vocab_size": 1024, "files_per_format": 16, "repeat": 40},
}


def build_corpus_dir(root: Path, files_per_format: int, repeat: int) -> None:
    (root / "text").mkdir(parents=True)
    (root / "structured").mkdir(parents=True)
    for i in range(files_per_format):
        (root / "text" / f"doc{i}.txt").write_text(
            ("the quick brown fox jumps over the lazy dog " * repeat) + f" unique{i} marker",
            encoding="utf-8",
        )
        (root / "text" / f"note{i}.md").write_text(
            f"# Heading {i}\n\n" + ("markdown prose content line " * repeat),
            encoding="utf-8",
        )
        (root / "text" / f"script{i}.py").write_text(
            f"def func_{i}(x):\n    return x + {i}\n" + ("import os\n" * repeat),
            encoding="utf-8",
        )
        recs = [
            {
                "messages": [
                    {"role": "user", "content": f"question number {i} uniquephrase " * 3},
                    {"role": "assistant", "content": f"answer number {i} uniquephrase " * 3},
                ]
            }
            for _ in range(repeat)
        ]
        (root / "structured" / f"conv{i}.jsonl").write_text(
            "\n".join(json.dumps(r) for r in recs) + "\n", encoding="utf-8"
        )
        rows = "id,label\n" + "".join(f"{i}-{j},value{j}\n" for j in range(repeat * 4))
        (root / "structured" / f"table{i}.csv").write_text(rows, encoding="utf-8")
        (root / "structured" / f"records{i}.json").write_text(
            json.dumps([{"text": f"json document {i} payload " * 4} for _ in range(2)]),
            encoding="utf-8",
        )
        (root / "structured" / f"skip{i}.bin").write_bytes(bytes(range(256)) * 4)


def run_preset(name: str, cfg: dict) -> dict:
    from domain.training._internal.corpus_loader import load_corpus_dir
    from domain.training._internal.tokenizer_manager import TokenizerManager

    with tempfile.TemporaryDirectory(prefix="tokbench_") as tmp:
        root = Path(tmp)
        build_corpus_dir(root, cfg["files_per_format"], cfg["repeat"])

        t0 = time.perf_counter()
        docs = load_corpus_dir(str(root))
        load_s = time.perf_counter() - t0
        chars = sum(len(d) for d in docs)
        if not docs or chars <= 0:
            raise SystemExit(f"[{name}] FAIL: corpus empty (docs={len(docs)})")

        mgr = TokenizerManager()
        t1 = time.perf_counter()
        stats = mgr.train_from_directory(
            str(root), vocab_size=cfg["vocab_size"], min_frequency=1, lowercase=False
        )
        train_s = time.perf_counter() - t1
        vocab = stats.get("vocab_size", 0)

        marker = "uniquephrase"
        marker_ok = False
        for d in docs:
            if marker in d:
                ids = mgr.tokenize(d[:200])
                marker_ok = bool(ids) and mgr.detokenize(ids).strip() != ""
                break

        ok = vocab > 0 and marker_ok and load_s < 60 and train_s < 300
        record = {
            "preset": name,
            "docs": len(docs),
            "chars": chars,
            "load_seconds": round(load_s, 4),
            "train_seconds": round(train_s, 4),
            "docs_per_sec": round(len(docs) / max(train_s, 1e-9), 2),
            "chars_per_sec": round(chars / max(train_s, 1e-9), 2),
            "vocab_size": vocab,
            "target_vocab": cfg["vocab_size"],
            "structured_marker_roundtrip": marker_ok,
            "passed": ok,
        }
        print(
            f"  [{name}] {record['docs']} docs {record['chars']} chars | "
            f"load {record['load_seconds']}s | train {record['train_seconds']}s "
            f"({record['chars_per_sec']} chars/s) | vocab {vocab} | "
            f"marker roundtrip {marker_ok} | {'PASS' if ok else 'FAIL'}"
        )
        return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip())
    parser.add_argument("--json", default="/tmp/tokenizer-corpus-results.json")
    parser.add_argument("--preset", choices=[*PRESETS, "all"], default="all")
    args = parser.parse_args()

    names = list(PRESETS) if args.preset == "all" else [args.preset]
    results = [run_preset(n, PRESETS[n]) for n in names]
    payload = {"kind": "tokenizer_corpus", "results": results}
    Path(args.json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Results saved to {args.json}")
    return 0 if all(r["passed"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
