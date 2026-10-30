#!/usr/bin/env python3
"""Build a sealed training corpus: acquire → Cruncher (clean/dedup) → snapshot.

The missing wire between the three halves that already existed (AGENTS.md —
check before building): ``bawl`` crawls, ``data_import`` fetches, the cruncher
cleans and snapshots — nothing connected them, and ``data/crunched/`` had never
been written. This script is that connection, one source per run:

  python scripts/build_corpus.py --dataset-id site_docs \\
      --source bawl --src crawl_data/site.jsonl

  python scripts/build_corpus.py --dataset-id py_repos \\
      --source github --src https://github.com/org/repo --license MIT

Output is an immutable Standard v1 snapshot under ``data/crunched/<id>/v<N>/``
(manifest + train.txt + dedup_state); the last line prints the
``trainer_data_path`` to hand ``SloughGPTTrainer``. No network in tests — the
``local``/``bawl`` sources read files, and acquisition helpers are injectable.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterator
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from domain.training._internal.cruncher import (  # noqa: E402
    Cruncher,
    SnapshotStore,
    trainer_data_path,
)

_SOURCES = ("bawl", "local", "github", "url", "hf")


def _records_from_jsonl(path: Path, keys: tuple[str, ...] = ("text", "content")) -> Iterator[str]:
    """Yield the first present non-empty string under *keys* per JSONL line."""
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{lineno}: invalid JSON: {exc}") from exc
            if not isinstance(obj, dict):
                continue
            for key in keys:
                value = obj.get(key)
                if isinstance(value, str) and value.strip():
                    yield value
                    break


def _records_from_text_file(path: Path) -> Iterator[str]:
    """Yield a plain text file's content as a single record."""
    text = path.read_text(encoding="utf-8")
    if text.strip():
        yield text


def acquire_records(source: str, src: str, *, work_dir: Path) -> Iterator[str]:
    """Yield raw text records for the cruncher from one source."""
    if source == "local":
        path = Path(src)
        if not path.is_file():
            raise SystemExit(f"local source not found: {path}")
        if path.suffix == ".jsonl":
            yield from _records_from_jsonl(path)
        else:
            yield from _records_from_text_file(path)
        return

    if source == "bawl":
        path = Path(src)
        if not path.is_file():
            raise SystemExit(f"bawl crawl file not found: {path}")
        yield from _records_from_jsonl(path, keys=("text",))
        return

    if source == "github":
        from domain.training import RepoImporter

        importer = RepoImporter()
        repo_path = importer.clone_repo(src)
        corpus_path = work_dir / "github_corpus.jsonl"
        importer.export_to_corpus(repo_path, str(corpus_path))
        yield from _records_from_jsonl(corpus_path, keys=("content",))
        return

    if source == "url":
        from domain.training import URLImporter

        result = URLImporter().import_from_url(
            src, dataset_name="url_import", output_dir=str(work_dir)
        )
        if not result.success:
            raise SystemExit(f"url import failed: {result.error}")
        yield from _records_from_jsonl(Path(result.output_path), keys=("content",))
        return

    if source == "hf":
        from domain.training import HuggingFaceImporter

        importer = HuggingFaceImporter()
        result = importer.download_dataset(src, output_dir=str(work_dir))
        if not result.success:
            raise SystemExit(f"hf download failed: {result.error}")
        out = Path(result.output_path)
        if out.is_dir():
            jsonls = sorted(out.rglob("*.jsonl"))
            if not jsonls:
                raise SystemExit(f"hf download produced no jsonl under {out}")
            out = jsonls[0]
        yield from _records_from_jsonl(out)
        return

    raise SystemExit(f"unknown source {source!r} (choose from {_SOURCES})")


def source_entry(source: str, src: str, license_spdx: str | None = None) -> dict:
    """Standard v1 source object for the manifest (provenance + license)."""
    type_ = {
        "bawl": "https",
        "local": "local_path",
        "github": "github",
        "url": "https",
        "hf": "huggingface",
    }[source]
    entry: dict = {"type": type_, "uri": src}
    if license_spdx:
        entry["license_spdx"] = license_spdx
    return entry


def build_corpus(
    dataset_id: str,
    source: str,
    src: str,
    *,
    domain: str = "general",
    license_spdx: str | None = None,
    store_root: Path | None = None,
    work_dir: Path | None = None,
) -> str:
    """Acquire, crunch, and seal one corpus; return the trainer data_path."""
    work = work_dir or Path("runs") / "build_corpus"
    work.mkdir(parents=True, exist_ok=True)
    store = SnapshotStore(store_root) if store_root else SnapshotStore()
    cruncher = Cruncher(dataset_id, store=store)
    # Provenance is carried solely by the explicit sources= on seal() — do not
    # also pass source= to crunch_stream() or it lands as a duplicate entry.
    result = cruncher.crunch_stream(acquire_records(source, src, work_dir=work))
    ref = cruncher.seal(
        sources=[source_entry(source, src, license_spdx)],
        domain=domain,
    )
    print(
        f"sealed {dataset_id} v{ref.version}: kept={result.records_kept} "
        f"dropped_quality={result.dropped_quality} "
        f"dropped_duplicate={result.dropped_duplicate} "
        f"pii_replacements={result.pii_replacements}",
        file=sys.stderr,
    )
    return trainer_data_path(ref)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--dataset-id", required=True, help="Snapshot identity ([a-zA-Z0-9_-]+)")
    parser.add_argument("--source", required=True, choices=_SOURCES)
    parser.add_argument("--src", required=True, help="Source argument (path, URL, or HF id)")
    parser.add_argument("--domain", default="general", help="Manifest domain (general|code|...)")
    parser.add_argument(
        "--license", dest="license_spdx", default=None, help="SPDX id for provenance"
    )
    parser.add_argument(
        "--store-root", type=Path, default=None, help="Snapshot store root (default data/crunched)"
    )
    args = parser.parse_args(argv)
    path = build_corpus(
        args.dataset_id,
        args.source,
        args.src,
        domain=args.domain,
        license_spdx=args.license_spdx,
        store_root=args.store_root,
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
