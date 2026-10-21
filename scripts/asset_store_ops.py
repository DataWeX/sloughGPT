"""Asset store operator CLI — external utility tier (never a chat/agent tool).

Maintenance surface for the unified asset store (``domain/infrastructure``
AssetStore: content-addressed blobs + per-asset metadata + event history).
Same tier as the site doctor: tooling of the app itself, callable by an
operator from the shell — it never enters the chat/agent tool registry, the
``domain/tools`` engine, or anything the model reasons over.

Subcommands:

- ``verify``   read-only integrity report (blobs vs metadata: missing blobs,
  sha mismatches, orphan CAS objects). Mutates nothing; run it before acting.
- ``prune``    remove CAS blobs no metadata row references. Only unreferenced
  blob files are touched — metadata, events, and legacy directories (user
  data) are never in scope (file-safety).
- ``reindex``  reconcile metadata drift from blob content (the blob wins):
  repairs sha256/size_bytes; rows whose blob is gone are reported, never
  deleted — losing the last reference to user data is a human decision.
- ``history``  print recent asset events (newest first), optionally filtered
  by action (put/download/delete/migrate).

The store root follows ``$SLO_DATA_DIR/assets`` (or ``<repo>/data/assets``),
so pointing ``SLO_DATA_DIR`` at a scratch dir scopes every command there.
Exit code is non-zero only on operational failure (e.g. reindex hitting an
unwritable record); ``verify`` on a damaged store still exits 0 — its job is
to report, and the JSON carries ``healthy: false`` for gates to judge.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from domain.infrastructure.asset_store import get_asset_store  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("verify", help="read-only integrity report (blobs vs metadata)")
    sub.add_parser("prune", help="remove CAS blobs no metadata references")
    sub.add_parser("reindex", help="reconcile metadata drift from blob content")
    p_h = sub.add_parser("history", help="print recent asset events (newest first)")
    p_h.add_argument("--limit", type=int, default=50, help="max events to print")
    p_h.add_argument("--action", default="", help="filter by action (put/download/delete/migrate)")
    args = parser.parse_args()

    store = get_asset_store()
    try:
        if args.cmd == "verify":
            print(json.dumps(store.verify_integrity(), indent=2))
        elif args.cmd == "prune":
            print(json.dumps(store.prune_orphan_blobs(), indent=2))
        elif args.cmd == "reindex":
            print(json.dumps(store.reindex(), indent=2))
        elif args.cmd == "history":
            for event in store.history(limit=args.limit, action=args.action):
                print(json.dumps(event, default=str))
    except OSError as exc:
        print(f"asset-store-ops: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
