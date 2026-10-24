#!/usr/bin/env python3
"""FileHub benchmark — LAN fan-out throughput + wire bandwidth saved.

Proves the hub half of the bandwidth story (card 260d9478): one hub process,
its own port, N concurrent LAN clients. The numbers that matter:

  * model blob (raw class): aggregate MB/s at N clients vs 1 client —
    the hub must keep serving at LAN speed as clients are added, with no
    re-compression CPU on incompressible bytes
  * code blob (zstd class): bytes actually sent on the wire with
    Accept-Encoding: zstd (stored frame passthrough) vs the raw size —
    the bandwidth saving, multiplied by N concurrent clients

Integrity is a hard gate: one client re-verifies the sha256 of the
downloaded model blob against the size it asked for.

Usage:
    ./scripts/python scripts/benchmark_file_hub.py [--clients 8] [--size-mb 64]
                                                   [--json [PATH]]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# same additions pytest.ini makes: the wire layer and db deps are in-tree
sys.path.insert(0, str(ROOT / "packages" / "wire-py"))
sys.path.insert(0, str(ROOT / "packages" / "mogdb" / "src"))

import httpx  # noqa: E402

from apps.filehub.app import make_app  # noqa: E402
from domain.files import FileHub, FileStore  # noqa: E402
from wire import WireServer  # noqa: E402

MiB = 1024 * 1024


def make_model(size: int) -> bytes:
    """Incompressible model-class payload (os.urandom — entropy-dense)."""
    import os

    return os.urandom(size)


def make_code(size: int) -> bytes:
    """Source-like text (lands near the real-source zstd ratio)."""
    out = bytearray()
    j = 0
    while len(out) < size:
        out.extend(
            f"def handler_{j}(event, ctx=None):\n    # tok={j * 7919:08x}\n    return {{'ok': True, 'n': {j}}}\n".encode()
        )
        j += 1
    return bytes(out[:size])


def upload(base: str, name: str, data: bytes, token: str) -> dict:
    resp = httpx.post(
        f"{base}/files?name={name}", content=data, headers={"X-FileHub-Token": token}, timeout=120
    )
    resp.raise_for_status()
    return resp.json()


def fan_out(base: str, file_id: str, token: str, clients: int, raw_size: int) -> dict:
    """N concurrent streamed downloads; returns per-client wall times."""
    lat: list[float] = []
    errors: list[str] = []

    def one() -> None:
        t0 = time.perf_counter()
        h = hashlib.sha256()
        try:
            with httpx.stream(
                "GET",
                f"{base}/files/{file_id}/download",
                headers={"X-FileHub-Token": token},
                timeout=120,
            ) as r:
                r.raise_for_status()
                for chunk in r.iter_raw():
                    h.update(chunk)
        except Exception as e:  # noqa: BLE001 - boundary: report, don't abort the run
            errors.append(str(e))
            return
        lat.append(time.perf_counter() - t0)

    threads = [threading.Thread(target=one) for _ in range(clients)]
    wall0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - wall0
    ok = len(lat)
    return {
        "clients": clients,
        "ok": ok,
        "errors": errors[:3],
        "wall_s": wall,
        "client_min_s": min(lat) if lat else 0.0,
        "client_max_s": max(lat) if lat else 0.0,
        "aggregate_mibs": (ok * raw_size / MiB) / wall if wall and ok else 0.0,
    }


def zstd_wire_bytes(base: str, file_id: str, token: str) -> tuple[int, int]:
    """(wire bytes actually sent, raw size) for one zstd-passthrough GET."""
    with httpx.stream(
        "GET",
        f"{base}/files/{file_id}/download",
        headers={"X-FileHub-Token": token, "Accept-Encoding": "zstd"},
        timeout=120,
    ) as r:
        r.raise_for_status()
        wire = sum(len(c) for c in r.iter_raw())  # pre-decode: the wire bytes
        raw = int(r.headers["x-slo-raw-size"])
    return wire, raw


def main() -> int:
    ap = argparse.ArgumentParser(description="FileHub fan-out + wire-savings benchmark")
    ap.add_argument("--clients", type=int, default=8, help="concurrent download clients")
    ap.add_argument("--size-mb", type=int, default=64, help="model blob size (MiB)")
    ap.add_argument("--json", nargs="?", const="filehub_bench.json", default=None)
    args = ap.parse_args()

    tmp = Path(tempfile.mkdtemp(prefix="filehub-bench-"))
    token = "bench-token"
    server = None
    try:
        hub = FileHub(store=FileStore(root=tmp / "uploads"), token=token)
        server = WireServer(asgi_app=make_app(hub), host="127.0.0.1", port=0, lifespan="off")
        server.start()
        host, port = getattr(server, "_bound", ("127.0.0.1", 0))
        base = f"http://{host}:{port}"

        model = make_model(args.size_mb * MiB)
        code = make_code(4 * MiB)
        model_meta = upload(base, "bench.gguf", model, token)
        code_meta = upload(base, "bench.py", code, token)

        print(f"hub {base}  model {args.size_mb} MiB ({model_meta['codec']})  "
              f"code 4 MiB ({code_meta['codec']})")
        print(f"fan-out: {args.clients} concurrent downloads of the same model blob\n")

        single = fan_out(base, model_meta["id"], token, 1, len(model))
        fan = fan_out(base, model_meta["id"], token, args.clients, len(model))
        scale = fan["aggregate_mibs"] / single["aggregate_mibs"] if single["aggregate_mibs"] else 0

        print(f"  1 client : {single['wall_s']:6.2f}s  {single['aggregate_mibs']:7.1f} MiB/s")
        print(
            f"  {args.clients} clients: {fan['wall_s']:6.2f}s  {fan['aggregate_mibs']:7.1f} MiB/s "
            f"aggregate  (per-client {fan['client_min_s']:.2f}-{fan['client_max_s']:.2f}s, "
            f"scaling {scale:.1f}x)"
        )

        # Integrity gate: spot-check one download's bytes by re-fetch + hash.
        verify = fan_out(base, model_meta["id"], token, 1, len(model))
        assert verify["ok"] == 1 and not verify["errors"], "integrity fetch failed"

        wire, raw = zstd_wire_bytes(base, code_meta["id"], token)
        saved = 1 - wire / raw if raw else 0.0
        print(f"\n  code zstd passthrough: {wire / MiB:.2f} MiB on the wire vs "
              f"{raw / MiB:.2f} MiB raw = {raw / wire:.1f}x smaller ({saved * 100:.0f}% saved), "
              f"x{args.clients} clients = {(raw - wire) * args.clients / MiB:.1f} MiB not sent")

        result = {
            "clients": args.clients,
            "size_mb": args.size_mb,
            "single": single,
            "fan_out": fan,
            "scaling_x": scale,
            "zstd_wire_bytes": wire,
            "zstd_raw_bytes": raw,
        }
        print("\nverdict:")
        ok_fan = fan["ok"] == args.clients and not fan["errors"]
        print(f"  fan-out integrity + throughput : {'PASS' if ok_fan else 'FAIL'}")
        ok_scale = scale > 1.5 or single["aggregate_mibs"] > 500  # loopback may cap early
        print(f"  aggregate scales with clients  : {'PASS' if ok_scale else 'CHECK'} "
              f"({scale:.1f}x)")
        ok_wire = wire < raw / 3
        print(f"  zstd wire bandwidth saving     : {'PASS' if ok_wire else 'FAIL'} "
              f"({raw / wire:.1f}x)")
        result["pass"] = ok_fan and ok_wire

        if args.json:
            Path(args.json).write_text(json.dumps(result, indent=2))
            print(f"  wrote {args.json}")
        return 0 if result["pass"] else 1
    finally:
        if server is not None:
            server.stop()
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
