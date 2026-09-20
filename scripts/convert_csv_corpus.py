"""Dataset adapter: convert a directory of CSV files into a training corpus.

Reads every ``*.csv`` in the dataset dir and writes one JSON object per row
to ``corpus.jsonl`` (``{"text": ...}``), the format the training resolvers
(``find_corpus_file``) and ``SloughGPTTrainer`` already understand.

This does NOT duplicate any training loop — it only adapts data into the
existing ``SloughGPTTrainer``/turbo input contract.

Usage:
    python3 scripts/convert_csv_corpus.py data/world-happiness
    python3 scripts/convert_csv_corpus.py data/my-dataset --out train.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


def row_to_text(headers: list[str], row: list[str]) -> str:
    """Render one CSV row as a plain training sentence, skipping empties."""
    parts = []
    for h, v in zip(headers, row):
        v = (v or "").strip()
        if v:
            parts.append(f"{h.strip()}: {v}")
    return ". ".join(parts)


def convert(dataset_dir: Path, out_name: str = "corpus.jsonl") -> Path:
    csv_files = sorted(dataset_dir.glob("*.csv"))
    if not csv_files:
        raise SystemExit(f"No CSV files in {dataset_dir}")
    out_path = dataset_dir / out_name
    n_rows = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for csv_file in csv_files:
            with open(csv_file, newline="", encoding="utf-8-sig") as f:
                reader = csv.reader(f)
                try:
                    headers = next(reader)
                except StopIteration:
                    continue
                for row in reader:
                    text = row_to_text(headers, row)
                    if len(text) >= 20:
                        out.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                        n_rows += 1
    print(f"Wrote {n_rows} rows -> {out_path} ({out_path.stat().st_size} bytes)")
    return out_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir", help="Directory containing *.csv files")
    parser.add_argument("--out", default="corpus.jsonl", help="Output file name")
    args = parser.parse_args(argv)
    convert(Path(args.dataset_dir), args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
