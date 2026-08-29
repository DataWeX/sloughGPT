"""Standalone inference engine process.

Runs a TCP server that hosts the model and serves generate/stream/health
requests.  Separated from the API server so inference work (large memory,
CPU-bound) runs in its own process.

Usage::

    # As a subprocess launched by the API server:
    python -m domains.infrastructure.inference_engine \\
        --model-id Qwen/Qwen2.5-0.5B-Instruct \\
        --host 127.0.0.1 --port 9100

    # Or programmatically:
    engine = InferenceEngine(model_id="Qwen/Qwen2.5-0.5B-Instruct")
    engine.start()  # non-blocking, runs in background thread
    engine.wait_ready()
    engine.stop()
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import json
import logging
import os
import selectors
import socket
from .constants import DEFAULT_GENERATE_TIMEOUT
import struct
import sys
import threading
import time
from typing import Any, Dict, Optional

from domains.infrastructure.inference_protocol import (
    HEADER_FMT,
    HEADER_SIZE,
    decode_header,
    encode_message,
)

logger = logging.getLogger(__name__)


class InferenceEngine:
    """Standalone inference engine that listens on a TCP socket.

    Wraps ``SloNetChatProvider`` in its own process, handling model load,
    generate, stream, and health requests over a simple length-prefixed
    JSON protocol.

    Args:
        model_id: HuggingFace model ID (e.g. ``"Qwen/Qwen2.5-0.5B-Instruct"``)
        slnc_path: Optional direct path to a .slnc file
        host: Bind address
        port: Bind port (0 = auto-assign)
        quantize: Whether to quantize the model
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-0.5B-Instruct",
        slnc_path: Optional[str] = None,
        host: str = "127.0.0.1",
        port: int = 9100,
        quantize: bool = True,
        quant_bits: int = 8,
        quant_mode: str = "symmetric",
        quant_clip: float = 0.999,
    ):
        self.model_id = model_id
        self.slnc_path = slnc_path
        self.host = host
        self.port = port
        self.quantize = quantize
        self.quant_bits = quant_bits
        self.quant_mode = quant_mode
        self.quant_clip = quant_clip

        self._provider = None
        self._server_socket: Optional[socket.socket] = None
        self._thread: Optional[threading.Thread] = None
        self._ready = threading.Event()
        self._stop = threading.Event()
        self._active_streams: Dict[str, bool] = {}
        self._active_streams_lock = threading.Lock()
        self._pid_file: Optional[str] = None

        # Metrics
        self._metrics_lock = threading.Lock()
        self._request_count: int = 0
        self._error_count: int = 0
        self._total_latency: float = 0.0
        self._max_latency: float = 0.0
        self._reload_count: int = 0

        # Config
        self._request_timeout: float = DEFAULT_GENERATE_TIMEOUT

    @property
    def addr(self) -> tuple:
        if self._server_socket is not None:
            return self._server_socket.getsockname()
        return (self.host, self.port)

    def start(self) -> None:
        """Start the engine in a background thread."""
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="inference-engine")
        self._thread.start()

    def wait_ready(self, timeout: float = 300) -> bool:
        """Block until the engine is ready to serve requests."""
        return self._ready.wait(timeout)

    def stop(self) -> None:
        """Shut down the engine gracefully.

        Signals all active streams to stop, waits up to 5s for them to drain,
        then closes the server socket and joins the accept thread.
        """
        self._stop.set()

        # Signal all active streams to stop
        with self._active_streams_lock:
            for req_id in list(self._active_streams):
                self._active_streams[req_id] = False

        # Wait for active streams to drain (max 5s)
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            with self._active_streams_lock:
                if not self._active_streams:
                    break
            time.sleep(0.1)

        if self._server_socket is not None:
            try:
                self._server_socket.close()
            except Exception as e:
                logger.warning("Failed to close server socket: %s", e)
        if self._thread is not None:
            self._thread.join(timeout=10)

        # Remove PID file
        if self._pid_file is not None:
            try:
                os.remove(self._pid_file)
            except Exception as e:
                logger.warning("Failed to remove PID file %s: %s", self._pid_file, e)

    def _record_request(self, latency: float, error: bool = False) -> None:
        """Record request metrics (thread-safe)."""
        with self._metrics_lock:
            self._request_count += 1
            self._total_latency += latency
            self._max_latency = max(self._max_latency, latency)
            if error:
                self._error_count += 1

    def _record_reload(self) -> None:
        """Record a reload event (thread-safe)."""
        with self._metrics_lock:
            self._reload_count += 1

    def get_metrics(self) -> dict:
        """Return current engine metrics."""
        with self._metrics_lock:
            count = self._request_count
            return {
                "request_count": count,
                "error_count": self._error_count,
                "avg_latency_ms": round((self._total_latency / count * 1000), 1) if count else 0,
                "max_latency_ms": round(self._max_latency * 1000, 1),
                "reload_count": self._reload_count,
                "active_streams": len(self._active_streams),
            }

    @property
    def active_stream_count(self) -> int:
        with self._active_streams_lock:
            return len(self._active_streams)

    # ── Internal ──────────────────────────────────────────────────────

    def _run(self) -> None:
        """Main loop: load model, bind socket, accept connections."""
        import os
        try:
            self._load_model()
        except Exception as e:
            logger.error("Inference engine failed to load model: %s", e)
            return

        # Write PID file for external monitoring
        pid_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".inference_engine.pid")
        try:
            with open(pid_path, "w") as f:
                f.write(str(os.getpid()))
            self._pid_file = pid_path
        except Exception as exc:
            logger.debug("Failed to write PID file: %s", exc)

        try:
            self._server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._server_socket.settimeout(1.0)
            self._server_socket.bind((self.host, self.port))
            self._server_socket.listen(4)
            logger.info(
                "Inference engine listening on %s:%d (model=%s)",
                *self._server_socket.getsockname(), self.model_id,
            )
            self._ready.set()

            while not self._stop.is_set():
                try:
                    client, addr = self._server_socket.accept()
                except socket.timeout:
                    continue
                except OSError:
                    break
                t = threading.Thread(
                    target=self._handle_client, args=(client, addr), daemon=True
                )
                t.start()
        except Exception as e:
            logger.error("Inference engine socket error: %s", e)
        finally:
            self._ready.clear()

    def _load_model(self) -> None:
        """Load the model via SloNetChatProvider with progress logging."""
        import time as _time

        logger.info("Inference engine: loading model %s ...", self.model_id)
        _st = _time.monotonic()

        from domains.inference.slonet_provider import SloNetChatProvider

        load_kwargs: Dict[str, Any] = {
            "quantize": self.quantize,
            "quant_bits": self.quant_bits,
            "quant_mode": self.quant_mode,
            "quant_clip": self.quant_clip,
        }
        if self.slnc_path:
            logger.info("Inference engine: loading from .slnc %s", self.slnc_path)
            self._provider = SloNetChatProvider.from_slnc(
                self.slnc_path, model_id=self.model_id, **load_kwargs,
            )
        else:
            from domains.infrastructure.safetensors_loader import _get_model_dir
            _cache_dir = _get_model_dir(self.model_id)
            _slnc = _cache_dir / "model.slnc"
            if not _slnc.exists():
                raise FileNotFoundError(f"No .slnc file for {self.model_id} at {_slnc}")
            logger.info("Inference engine: loading from .slnc %s", _slnc)
            self._provider = SloNetChatProvider.from_slnc(
                str(_slnc), model_id=self.model_id, **load_kwargs,
            )

        elapsed = _time.monotonic() - _st
        logger.info(
            "Inference engine: model loaded in %.1fs (quantize=%s, bits=%d)",
            elapsed, self.quantize, self.quant_bits,
        )

    def _handle_client(self, client: socket.socket, addr: tuple) -> None:
        """Handle a single client connection (blocking)."""
        logger.debug("Inference engine: client connected from %s", addr)
        try:
            while not self._stop.is_set():
                msg = self._recv_message(client)
                if msg is None:
                    break
                self._dispatch(client, msg)
        except (ConnectionResetError, BrokenPipeError, OSError):
            pass
        finally:
            try:
                client.close()
            except Exception as exc:
                logger.debug("client close failed: %s", exc)
            logger.debug("Inference engine: client disconnected %s", addr)

    def _recv_message(self, client: socket.socket) -> Optional[dict]:
        """Read a length-prefixed JSON message from the socket."""
        header = self._recv_exact(client, HEADER_SIZE)
        if header is None:
            return None
        length = decode_header(header)
        if length > 100 * 1024 * 1024:
            logger.error("Inference engine: message too large (%d bytes)", length)
            return None
        payload = self._recv_exact(client, length)
        if payload is None:
            return None
        try:
            return json.loads(payload)
        except json.JSONDecodeError as e:
            logger.error("Inference engine: bad JSON: %s", e)
            return None

    def _recv_exact(self, client: socket.socket, n: int) -> Optional[bytes]:
        """Read exactly n bytes from the socket."""
        buf = bytearray()
        while len(buf) < n:
            if self._stop.is_set():
                return None
            try:
                chunk = client.recv(n - len(buf))
            except (OSError, ValueError):
                return None
            if not chunk:
                return None
            buf.extend(chunk)
        return bytes(buf)

    def _send_message(self, client: socket.socket, msg: dict) -> None:
        """Send a length-prefixed JSON message."""
        data = encode_message(msg)
        client.sendall(data)

    def _dispatch(self, client: socket.socket, msg: dict) -> None:
        """Route a message to the appropriate handler."""
        msg_type = msg.get("type")
        if msg_type == "health":
            self._handle_health(client, msg)
        elif msg_type == "generate":
            self._handle_generate(client, msg)
        elif msg_type == "stream_start":
            self._handle_stream_start(client, msg)
        elif msg_type == "stream_stop":
            self._handle_stream_stop(msg)
        elif msg_type == "reload":
            self._handle_reload(client, msg)
        else:
            self._send_message(client, {
                "type": "error",
                "id": msg.get("id", ""),
                "message": f"Unknown message type: {msg_type}",
            })

    def _handle_health(self, client: socket.socket, msg: dict) -> None:
        meta = {}
        if self._provider is not None:
            meta = getattr(self._provider, "_meta", {}) or {}
        metrics = self.get_metrics()
        self._send_message(client, {
            "type": "health_ok",
            "model_id": self.model_id,
            "loaded": self._provider is not None and getattr(self._provider, "_loaded", False),
            "model_type": self.model_id,
            "quantized": meta.get("quantized", False),
            "metrics": metrics,
        })

    def _handle_generate(self, client: socket.socket, msg: dict) -> None:
        import time as _time
        req_id = msg.get("id", "")
        messages = msg.get("messages", [{"role": "user", "content": msg.get("prompt", "")}])
        params = msg.get("params", {})
        _st = _time.monotonic()
        try:
            result = self._provider._generate_sync(
                messages,
                max_tokens=params.get("max_new_tokens", 256),
                temperature=params.get("temperature", 0.7),
                top_k=params.get("top_k"),
                top_p=params.get("top_p"),
                repetition_penalty=params.get("repetition_penalty", 1.0),
                session_id=params.get("session_id"),
            )
            elapsed = _time.monotonic() - _st
            self._record_request(elapsed)
            self._send_message(client, {
                "type": "result",
                "id": req_id,
                "text": result,
                "meta": {"model": self.model_id, "elapsed_ms": round(elapsed * 1000, 1)},
            })
        except Exception as e:
            elapsed = _time.monotonic() - _st
            self._record_request(elapsed, error=True)
            self._send_message(client, {
                "type": "error",
                "id": req_id,
                "message": str(e),
            })

    def _handle_stream_start(self, client: socket.socket, msg: dict) -> None:
        import time as _time
        req_id = msg.get("id", "")
        messages = msg.get("messages", [{"role": "user", "content": msg.get("prompt", "")}])
        params = msg.get("params", {})
        _st = _time.monotonic()
        with self._active_streams_lock:
            self._active_streams[req_id] = True

        def _stream_worker():
            import asyncio

            async def _run_stream():
                async for token in self._provider.chat_stream(
                    messages,
                    max_tokens=params.get("max_new_tokens", 256),
                    temperature=params.get("temperature", 0.7),
                    top_k=params.get("top_k"),
                    top_p=params.get("top_p"),
                    repetition_penalty=params.get("repetition_penalty", 1.0),
                    session_id=params.get("session_id"),
                ):
                    with self._active_streams_lock:
                        if not self._active_streams.get(req_id, False):
                            break
                    elapsed = _time.monotonic() - _st
                    if elapsed > self._request_timeout:
                        logger.warning("Inference engine: stream %s timed out after %.1fs", req_id, elapsed)
                        break
                    try:
                        self._send_message(client, {
                            "type": "token",
                            "id": req_id,
                            "token": token,
                        })
                    except (OSError, BrokenPipeError):
                        break

            try:
                asyncio.run(_run_stream())
                elapsed = _time.monotonic() - _st
                self._record_request(elapsed)
                self._send_message(client, {
                    "type": "stream_done",
                    "id": req_id,
                    "meta": {"model": self.model_id, "elapsed_ms": round(elapsed * 1000, 1)},
                })
            except Exception as e:
                elapsed = _time.monotonic() - _st
                self._record_request(elapsed, error=True)
                try:
                    self._send_message(client, {
                        "type": "error",
                        "id": req_id,
                        "message": str(e),
                    })
                except (OSError, BrokenPipeError):
                    pass
            finally:
                with self._active_streams_lock:
                    self._active_streams.pop(req_id, None)

        t = threading.Thread(target=_stream_worker, daemon=True, name=f"stream-{req_id[:8]}")
        t.start()

    def _handle_stream_stop(self, msg: dict) -> None:
        req_id = msg.get("id", "")
        with self._active_streams_lock:
            self._active_streams[req_id] = False

    def _handle_reload(self, client: socket.socket, msg: dict) -> None:
        """Hot-reload: swap the model at runtime without restarting the process.

        Rejects if active streams are running (data race risk).
        """
        import time as _time

        # Safety: reject if streams are active
        active = self.active_stream_count
        if active > 0:
            self._send_message(client, {
                "type": "error",
                "id": msg.get("id", ""),
                "message": f"Reload rejected: {active} active stream(s) in progress",
            })
            return

        new_model_id = msg.get("model_id", self.model_id)
        new_slnc_path = msg.get("slnc_path", self.slnc_path)

        logger.info("Inference engine: reloading model %s -> %s", self.model_id, new_model_id)
        _st = _time.monotonic()

        try:
            from domains.inference.slonet_provider import SloNetChatProvider

            load_kwargs: Dict[str, Any] = {
                "quantize": self.quantize,
                "quant_bits": self.quant_bits,
                "quant_mode": self.quant_mode,
                "quant_clip": self.quant_clip,
            }
            if new_slnc_path:
                new_provider = SloNetChatProvider.from_slnc(
                    new_slnc_path, model_id=new_model_id, **load_kwargs,
                )
            else:
                from domains.infrastructure.safetensors_loader import _get_model_dir
                _slnc = _get_model_dir(new_model_id) / "model.slnc"
                if not _slnc.exists():
                    raise FileNotFoundError(f"No .slnc file for {new_model_id} at {_slnc}")
                new_provider = SloNetChatProvider.from_slnc(
                    str(_slnc), model_id=new_model_id, **load_kwargs,
                )

            old_provider = self._provider
            self._provider = new_provider
            self.model_id = new_model_id
            self.slnc_path = new_slnc_path
            self._record_reload()

            elapsed = _time.monotonic() - _st
            logger.info("Inference engine: model reloaded in %.1fs (%s)", elapsed, new_model_id)
            self._send_message(client, {
                "type": "reload_ok",
                "model_id": new_model_id,
                "elapsed": round(elapsed, 2),
            })
        except Exception as e:
            elapsed = _time.monotonic() - _st
            logger.error("Inference engine: reload failed in %.1fs: %s", elapsed, e)
            self._send_message(client, {
                "type": "error",
                "id": msg.get("id", ""),
                "message": str(e),
            })


# ── CLI entry point ───────────────────────────────────────────────────

def main():
    import signal

    parser = argparse.ArgumentParser(description="Standalone inference engine")
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--slnc-path", default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9100)
    parser.add_argument("--quantize", action="store_true", default=True)
    parser.add_argument("--quant-bits", type=int, default=8)
    parser.add_argument("--quant-mode", default="symmetric")
    parser.add_argument("--quant-clip", type=float, default=0.999)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")

    engine = InferenceEngine(
        model_id=args.model_id,
        slnc_path=args.slnc_path,
        host=args.host,
        port=args.port,
        quantize=args.quantize,
        quant_bits=args.quant_bits,
        quant_mode=args.quant_mode,
        quant_clip=args.quant_clip,
    )

    def _handle_signal(signum, frame):
        logger.info("Inference engine: received signal %d, shutting down", signum)
        engine.stop()

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    engine.start()
    engine.wait_ready()
    print(f"ENGINE_READY port={engine.addr[1]}", flush=True)
    try:
        while engine._thread.is_alive():
            engine._thread.join(timeout=1.0)
    except KeyboardInterrupt:
        engine.stop()


if __name__ == "__main__":
    main()
