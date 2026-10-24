"""FileHub standalone server — python -m apps.filehub [--host 0.0.0.0 --port 8400]

Binds its own interface so the LAN pulls files (and HF pull-through
caches) from the hub instead of the upstream. Token auth comes from
``SLO_FILEHUB_TOKEN`` (or --token); with no token the hub is open —
loopback/dev only. Served by our own wire layer (wire-py), not uvicorn.
"""

from __future__ import annotations

import argparse
import logging


def main() -> int:
    ap = argparse.ArgumentParser(prog="filehub", description="Standalone file download hub")
    ap.add_argument("--host", default="0.0.0.0", help="bind address (own IP)")
    ap.add_argument("--port", type=int, default=8400)
    ap.add_argument("--token", default=None, help="auth token (default: $SLO_FILEHUB_TOKEN)")
    ap.add_argument("--root", default=None, help="blob root directory (default: data/uploads)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    from pathlib import Path

    from domain.files import FileHub, FileStore

    store = FileStore(root=Path(args.root) if args.root else None)
    hub = FileHub(store=store, token=args.token)
    if not hub.token and args.host not in ("127.0.0.1", "::1", "localhost"):
        logging.getLogger("slo.filehub").warning(
            "NO TOKEN on a non-loopback bind (%s) — the hub is open to the network",
            args.host,
        )

    from wire import WireServer

    from apps.filehub.app import make_app

    server = WireServer(asgi_app=make_app(hub), host=args.host, port=args.port, lifespan="off")
    server.start()  # binds + spawns the accept thread; port 0 resolves here
    # _bound is updated by start() from getsockname — resolves port 0.
    bound = getattr(server, "_bound", (args.host, args.port))
    logging.getLogger("slo.filehub").info(
        "filehub serving %s on http://%s:%s (token %s)",
        store.root,
        bound[0],
        bound[1],
        "on" if hub.token else "OFF",
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
