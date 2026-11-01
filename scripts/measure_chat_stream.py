#!/usr/bin/env python3
"""Measure /chat/stream: TTFB, inter-token gaps, tok/s, done/error frames.

Usage: scripts/python scripts/measure_chat_stream.py [--max-tokens N] [--prompt TEXT]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--path", default="/chat")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument(
        "--prompt",
        default="Write a short paragraph about the pros and cons of enhancing online presence.",
    )
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--use-rag", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()

    body = json.dumps(
        {
            "messages": [{"role": "user", "content": args.prompt}],
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "use_rag": args.use_rag,
            "session_id": "measure-1",
        }
    ).encode()
    req = urllib.request.Request(
        f"{args.base}{args.path}",
        data=body,
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )

    t0 = time.perf_counter()
    ttfb = None
    first_tok = None
    tok_times: list[float] = []
    got: list[str] = []
    frames: list[str] = []
    done = False
    errors: list[str] = []
    heartbeats = 0

    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            status = resp.status
            ctype = resp.headers.get("content-type", "")
            print(f"HTTP {status} content-type={ctype}")
            buf = b""
            while True:
                chunk = resp.read(1)
                if not chunk:
                    break
                if ttfb is None:
                    ttfb = (time.perf_counter() - t0) * 1000
                buf += chunk
                if not buf.endswith(b"\n\n"):
                    continue
                frame = buf.decode("utf-8", "replace").strip()
                buf = b""
                if not frame:
                    continue
                frames.append(frame)
                now = time.perf_counter()
                if frame.startswith(":"):
                    heartbeats += 1
                    print(f"  t={now - t0:7.2f}s HEARTBEAT: {frame[:60]!r}")
                    continue
                data_lines = [
                    ln[len("data:") :].strip()
                    for ln in frame.splitlines()
                    if ln.startswith("data:")
                ]
                payload = "\n".join(data_lines)
                if not payload:
                    continue
                try:
                    obj = json.loads(payload)
                except json.JSONDecodeError:
                    got.append(payload)
                    tok_times.append(now)
                    print(f"  t={now - t0:7.2f}s RAW: {payload[:80]!r}")
                    continue
                ev = obj.get("event") or obj.get("type")
                data = obj.get("data") or {}
                if isinstance(data, str):
                    try:
                        data = json.loads(data)
                    except json.JSONDecodeError:
                        data = {"text": data}
                status = obj.get("status")
                if status == "error" or ev in ("error", "ERROR") or data.get("error"):
                    errors.append(json.dumps(obj)[:300])
                    print(f"  t={now - t0:7.2f}s ERROR: {json.dumps(obj)[:300]}")
                    continue
                if status in ("complete", "success") or data.get("done"):
                    done = True
                    print(f"  t={now - t0:7.2f}s DONE status={status} meta={obj.get('meta')}")
                    continue
                text = data.get("text") or data.get("token") or ""
                if text:
                    if first_tok is None:
                        first_tok = now
                    got.append(text)
                    tok_times.append(now)
    except urllib.error.HTTPError as e:
        print(f"HTTP {e.code}: {e.read()[:400]}")
        return 1
    except Exception as e:  # noqa: BLE001
        print(f"STREAM ABORTED after {(time.perf_counter() - t0):.2f}s: {type(e).__name__}: {e}")
        errors.append(f"{type(e).__name__}: {e}")

    total = time.perf_counter() - t0
    n = len(tok_times)
    text = "".join(got)
    print()
    print("=" * 60)
    print(
        f"frames={len(frames)} heartbeats={heartbeats} tokens={n} done={done} errors={len(errors)}"
    )
    print(
        f"ttfb_ms={ttfb and round(ttfb, 1)} first_token_ms={first_tok and round((first_tok - t0) * 1000, 1)}"
    )
    print(f"total_s={total:.2f}")
    if n >= 2:
        decode_span = tok_times[-1] - tok_times[0]
        print(
            f"decode_tok_s={n - 1:.0f} tok / {decode_span:.2f}s = {(n - 1) / decode_span:.2f} tok/s"
        )
        gaps = [round((b - a) * 1000) for a, b in zip(tok_times, tok_times[1:])]
        gaps_sorted = sorted(gaps)
        med = gaps_sorted[len(gaps_sorted) // 2]
        print(
            f"gap_ms: min={min(gaps)} med={med} p90={gaps_sorted[int(0.9 * len(gaps_sorted))]} max={max(gaps)}"
        )
        print(f"gaps={gaps}")
    elif n == 1:
        print("only 1 token — cannot compute rate")
    print(f"chars={len(text)}")
    print(f"text={text[:200]!r}")
    if errors:
        print("errors:")
        for e in errors:
            print(f"  {e}")
    return 0 if done and not errors else 2


if __name__ == "__main__":
    sys.exit(main())
