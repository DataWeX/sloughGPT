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


# SPDX aliases for common non-SPDX license names returned by GitHub/HF APIs.
_LICENSE_ALIASES = {
    "mit": "MIT",
    "apache-2.0": "Apache-2.0",
    "apache 2.0": "Apache-2.0",
    "bsd-2-clause": "BSD-2-Clause",
    "bsd-3-clause": "BSD-3-Clause",
    "gpl-3.0": "GPL-3.0-only",
    "gpl-3.0-only": "GPL-3.0-only",
    "gpl-3.0-or-later": "GPL-3.0-or-later",
    "lgpl-2.1": "LGPL-2.1-only",
    "mpl-2.0": "MPL-2.0",
    "isc": "ISC",
    "unlicense": "Unlicense",
    "cc0-1.0": "CC0-1.0",
    "cc-by-4.0": "CC-BY-4.0",
    "cc-by-sa-4.0": "CC-BY-SA-4.0",
}


def normalize_license(raw: str | None) -> str | None:
    """Best-effort SPDX id from a raw license name, or None when unknown.

    Never invents a license — an unrecognized name returns None and the
    manifest omits ``license_spdx`` (no provenance, no seal).
    """
    if not raw:
        return None
    key = raw.strip().lower()
    if key in _LICENSE_ALIASES:
        return _LICENSE_ALIASES[key]
    # Already an SPDX-looking id (e.g. "MIT", "Apache-2.0", "GPL-3.0-or-later")
    if raw.strip() and " " not in raw.strip() and len(raw.strip()) <= 32:
        return raw.strip()
    return None


def fetch_github_license(repo_url: str) -> str | None:
    """Fetch SPDX license for a GitHub repo via the public API (no auth).

    Returns None on any network/parse failure — caller omits the field.
    """
    import urllib.request

    # https://github.com/owner/repo[.git] -> owner/repo
    parts = repo_url.rstrip("/").removesuffix(".git").split("/")
    if len(parts) < 2:
        return None
    owner, repo = parts[-2], parts[-1]
    api = f"https://api.github.com/repos/{owner}/{repo}/license"
    try:
        with urllib.request.urlopen(api, timeout=10) as resp:  # noqa: S310 — fixed https host
            data = json.loads(resp.read().decode("utf-8"))
        spdx = (data.get("license") or {}).get("spdx_id")
        if spdx and spdx != "NOASSERTION":
            return normalize_license(spdx)
    except Exception:  # noqa: BLE001 — any API failure just omits the field
        pass
    return None


def fetch_hf_license(dataset_id: str) -> str | None:
    """Fetch SPDX license for an HF dataset via the Hub metadata API.

    Returns None on any network/parse failure — caller omits the field.
    """
    import urllib.request

    api = f"https://huggingface.co/api/datasets/{dataset_id}"
    try:
        with urllib.request.urlopen(api, timeout=10) as resp:  # noqa: S310 — fixed https host
            data = json.loads(resp.read().decode("utf-8"))
        card = data.get("cardData") or {}
        raw = card.get("license") or data.get("license")
        if isinstance(raw, list) and raw:
            raw = raw[0]
        return normalize_license(raw if isinstance(raw, str) else None)
    except Exception:  # noqa: BLE001 — any API failure just omits the field
        pass
    return None


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
    probe: int = 0,
) -> str:
    """Acquire, crunch, and seal one corpus; return the trainer data_path.

    For ``hf`` and ``github`` sources, when *license_spdx* is not given the
    license is auto-fetched from the HF Hub card / GitHub API. An unknown
    license is omitted from the manifest (no provenance, no seal).

    ``probe > 0`` holds that many records OUT of ``train.txt`` after the
    quality/dedup/PII gate and writes them to ``probe.jsonl`` beside the
    manifest — a leakage-free eval set for sequential-training standings
    (dedup state keeps them, so later versions drop them instead of
    leaking them back into train).
    """
    if license_spdx is None and source == "github":
        license_spdx = fetch_github_license(src)
    if license_spdx is None and source == "hf":
        license_spdx = fetch_hf_license(src)

    work = work_dir or Path("runs") / "build_corpus"
    work.mkdir(parents=True, exist_ok=True)
    store = SnapshotStore(store_root) if store_root else SnapshotStore()
    cruncher = Cruncher(dataset_id, store=store)
    # Provenance is carried solely by the explicit sources= on seal() — do not
    # also pass source= to crunch_stream() or it lands as a duplicate entry.
    result = cruncher.crunch_stream(acquire_records(source, src, work_dir=work))
    held: list[str] = []
    if probe > 0:
        # Take the holdout from the END of the post-gate buffer BEFORE seal()
        # (seal() consumes it), always leaving >= 1 record for train.txt.
        take = min(probe, max(0, len(cruncher._buffer) - 1))
        if take:
            held = list(cruncher._buffer[-take:])
            del cruncher._buffer[-take:]
            result.records_kept -= len(held)
        else:
            print("probe: skipped (corpus too small)", file=sys.stderr)
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
    if held:
        probe_path = Path(trainer_data_path(ref)).parent / "probe.jsonl"
        with probe_path.open("w", encoding="utf-8") as fh:
            for record in held:
                fh.write(json.dumps({"text": record}) + "\n")
        print(f"probe: {len(held)} held-out records -> {probe_path}", file=sys.stderr)
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
    parser.add_argument(
        "--probe",
        type=int,
        default=0,
        help="Hold N records out of train.txt into probe.jsonl (sequential-training standings)",
    )
    args = parser.parse_args(argv)
    path = build_corpus(
        args.dataset_id,
        args.source,
        args.src,
        domain=args.domain,
        license_spdx=args.license_spdx,
        store_root=args.store_root,
        probe=args.probe,
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
