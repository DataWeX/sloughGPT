#!/usr/bin/env python
"""Migrate fat ``*.soul.meta.json`` sidecars: Adam moments → binary state file + state_ref.

Measured 2026-11-07: 17 sidecars under ``models/`` totalled 1.3 GB, of which
~92 MB per periodic checkpoint was ``metadata.training_state.optimizer.state``
as pretty JSON float text (the weights beside them are ~98 MB). This rewrites
each fat sidecar to keep a hash-pinned ``state_ref`` and moves the moments to
a ``<x>.optimizer.dat`` sibling — the ``bin-v1`` format the writer
(``domain/inference/_internal/soul_meta_slim.py``) now emits for every save
(legacy ``.optimizer.json.gz`` siblings are still read).

Safety contract:
- **Dry-run by default** — nothing is written without ``--apply``.
- **Original preserved byte-for-byte** under ``<root>/.meta-backups/<stamp>/``
  before the sidecar is replaced (user-trained artifacts are user data).
- **Atomic replace** — slim JSON is written to a ``.tmp``, re-parsed and
  checked, then ``os.replace``d; the state file lands (fsync) before the ref does.
- **Idempotent** — already-slim metas are skipped, and ``.meta-backups`` is
  never re-migrated (those are the preserved originals).
- **stdlib-only** — the helper is loaded standalone via importlib, so this
  runs at 2am without the domain stack.

Exit code is non-zero only when a file errored (unreadable JSON, verify
failure); skipped files are not errors.

Usage:
    ./scripts/python scripts/migrate_soul_meta_optimizer.py --root models
    ./scripts/python scripts/migrate_soul_meta_optimizer.py --root models --apply
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
HELPER_PATH = REPO_ROOT / "domain" / "inference" / "_internal" / "soul_meta_slim.py"
BACKUP_DIRNAME = ".meta-backups"


def _load_helper():
    spec = importlib.util.spec_from_file_location("soul_meta_slim", HELPER_PATH)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load helper: {HELPER_PATH}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _movable_state(doc: dict) -> bool:
    if not isinstance(doc, dict):
        return False
    md = doc.get("metadata")
    if not isinstance(md, dict):
        return False
    ts = md.get("training_state")
    if not isinstance(ts, dict):
        return False
    opt = ts.get("optimizer")
    if not isinstance(opt, dict):
        return False
    state = opt.get("state")
    return isinstance(state, dict) and bool(state)


def _iter_sidecars(root: Path):
    for path in sorted(root.rglob("*.meta.json")):
        # preserved originals are never re-migrated (idempotency)
        if BACKUP_DIRNAME in path.relative_to(root).parts:
            continue
        yield path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", required=True, help="directory to scan (e.g. models)")
    ap.add_argument("--apply", action="store_true", help="write changes (default: dry-run)")
    ap.add_argument("--json", dest="json_out", help="write a machine-readable report here")
    args = ap.parse_args(argv)

    root = Path(args.root)
    if not root.is_dir():
        print(f"error: root is not a directory: {root}", file=sys.stderr)
        return 2

    helper = _load_helper()
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    migrated = skipped = errors = 0
    saved_bytes = 0
    rows: list[dict] = []

    for meta_path in _iter_sidecars(root):
        rel = str(meta_path.relative_to(root))
        try:
            raw = meta_path.read_text(encoding="utf-8")
            doc = json.loads(raw)
        except Exception as exc:  # noqa: BLE001 — per-file error, keep going
            errors += 1
            rows.append({"file": rel, "error": str(exc)})
            print(f"error: {rel}: {exc}", file=sys.stderr)
            continue

        if not _movable_state(doc):
            skipped += 1
            continue

        before = meta_path.stat().st_size
        try:
            if not args.apply:
                _slim, moved = helper.externalize_optimizer_state(doc, meta_path, write=False)
                rows.append(
                    {
                        "file": rel,
                        "before": before,
                        "after": None,
                        "would_move": moved,
                    }
                )
                if moved:
                    print(f"would migrate: {rel} ({before} bytes)")
                continue

            slim, moved = helper.externalize_optimizer_state(doc, meta_path, write=True)
            if not moved:
                skipped += 1
                continue

            # verify what we just wrote before touching the original
            ref = slim["metadata"]["training_state"]["optimizer"]["state_ref"]
            gz_path = meta_path.parent / ref["file"]
            if hashlib.sha256(gz_path.read_bytes()).hexdigest() != ref["sha256"]:
                raise RuntimeError("state file sha256 mismatch after write")

            # backup first (copy, never move — the original stays until swap)
            backup = root / BACKUP_DIRNAME / stamp / rel
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(meta_path, backup)

            # slim sidecar: write .tmp, re-parse, then atomic replace
            tmp = meta_path.with_name(meta_path.name + ".tmp")
            tmp.write_text(json.dumps(slim, indent=2, default=str), encoding="utf-8")
            reparsed = json.loads(tmp.read_text(encoding="utf-8"))
            if not isinstance(reparsed, dict) or _movable_state(reparsed):
                tmp.unlink(missing_ok=True)
                raise RuntimeError("slim verify failed: state still inline or unparsable")
            tmp.replace(meta_path)

            after = meta_path.stat().st_size
            saved_bytes += before - after
            migrated += 1
            rows.append({"file": rel, "before": before, "after": after})
            print(f"migrated: {rel}  {before} -> {after} bytes")
        except Exception as exc:  # noqa: BLE001 — per-file error, keep going
            errors += 1
            rows.append({"file": rel, "error": str(exc)})
            print(f"error: {rel}: {exc}", file=sys.stderr)

    mode = "apply" if args.apply else "dry-run"
    print(f"root: {root}")
    print(f"mode: {mode}")
    print(f"migrated: {migrated}")
    print(f"skipped: {skipped}")
    print(f"errors: {errors}")
    if args.apply and migrated:
        print(f"saved: {saved_bytes} bytes")

    if args.json_out:
        report = {
            "root": str(root),
            "mode": mode,
            "migrated": migrated,
            "skipped": skipped,
            "errors": errors,
            "saved_bytes": saved_bytes,
            "files": rows,
        }
        Path(args.json_out).write_text(json.dumps(report, indent=2), encoding="utf-8")

    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
