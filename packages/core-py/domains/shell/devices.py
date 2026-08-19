"""
Shell Device Nodes — /dev/llm, /dev/embedding, /dev/vision, /dev/knowledge, /dev/null, /dev/random.

Each device behaves like a Unix device file:
  - cat /dev/llm       → read generates text
  - echo text > /dev/llm → write sends prompt
  - cat /dev/random    → reads random content
"""

from __future__ import annotations

import os
import io
import re
import json
import time
import random
import string
import logging
from typing import Any, Callable

logger = logging.getLogger("slo.shell.devices")


# ── Device Base ────────────────────────────────────────────────────────────


class AIDevice:
    """Base class for AI device nodes. Subclass and register via DeviceManager."""

    name: str = ""
    description: str = ""

    def read(self, args: str = "") -> str:
        """Read from device. Args passed from command line (e.g. cat /dev/llm prompt)."""
        raise NotImplementedError

    def write(self, data: str) -> str:
        """Write to device. Returns response text (printed to stdout)."""
        raise NotImplementedError


# ── Device Implementations ─────────────────────────────────────────────────


class NullDevice(AIDevice):
    name = "null"
    description = "Discards all written data, returns empty on read"

    def read(self, args: str = "") -> str:
        return ""

    def write(self, data: str) -> str:
        return ""


class RandomDevice(AIDevice):
    name = "random"
    description = "Returns random tokens on read"

    def read(self, args: str = "") -> str:
        try:
            n = int(args.strip()) if args.strip() else 64
        except ValueError:
            n = 64
        n = max(1, min(n, 4096))
        return "".join(random.choices(string.ascii_letters + string.digits + " \n", k=n))

    def write(self, data: str) -> str:
        return f"  wrote {len(data)} bytes to /dev/random (discarded)"


class LLMDevice(AIDevice):
    name = "llm"
    description = "AI text generation — read with prompt in args, write prompt and get response"

    def __init__(self, generate_fn: Callable[[str], str] | None = None):
        self._generate_fn = generate_fn

    def read(self, args: str = "") -> str:
        prompt = args.strip() or "continue this thought"
        return self._call_llm(prompt)

    def write(self, data: str) -> str:
        prompt = data.strip()
        if not prompt:
            return "  Usage: echo <prompt> > /dev/llm"
        return self._call_llm(prompt)

    def _call_llm(self, prompt: str) -> str:
        if self._generate_fn:
            return self._generate_fn(prompt)
        try:
            import requests
            from .config import get_api_base
            r = requests.post(
                f"{get_api_base()}/inference/generate",
                json={"prompt": prompt, "max_new_tokens": 128},
                timeout=30,
            )
            if r.status_code == 200:
                return r.json().get("text", "")
            return f"  llm: API error {r.status_code}"
        except ImportError:
            return "  llm: requests not available"
        except Exception as e:
            return f"  llm: {e}"


class EmbeddingDevice(AIDevice):
    name = "embedding"
    description = "Text embedding — write text, read returns embedding vector"

    def __init__(self, embed_fn: Callable[[str], list[float]] | None = None):
        self._embed_fn = embed_fn
        self._last_embedding: list[float] = []

    def read(self, args: str = "") -> str:
        if not self._last_embedding:
            return "  No embedding computed yet. Write text first: echo <text> > /dev/embedding"
        vec = self._last_embedding
        # Show first few non-zero values for readability
        nonzero = [v for v in vec if abs(v) > 1e-6]
        if nonzero:
            preview = nonzero[:6]
            return f"[{', '.join(f'{v:.4f}' for v in preview)}...] ({len(vec)} dims, {len(nonzero)} non-zero)"
        return f"[all zeros] ({len(vec)} dims)"

    def write(self, data: str) -> str:
        text = data.strip()
        if not text:
            return "  Usage: echo <text> > /dev/embedding"
        self._last_embedding = self._compute_embedding(text)
        return f"  embedding: {len(self._last_embedding)} dims"

    def _compute_embedding(self, text: str) -> list[float]:
        if self._embed_fn:
            return self._embed_fn(text)
        # Use the project's real embedder (sentence-transformers or n-gram fallback)
        try:
            from domains.inference.vector_store import simple_embed
            return simple_embed(text)
        except Exception:
            # Absolute fallback — deterministic based on text content
            import hashlib
            h = hashlib.sha256(text.encode()).digest()
            return [b / 255.0 for b in h[:64]]


class KnowledgeDevice(AIDevice):
    name = "knowledge"
    description = "Knowledge base — read returns random fact, write stores a fact"

    def __init__(self, api_base: str | None = None):
        from .config import get_api_base
        self._api_base = api_base or get_api_base()

    def read(self, args: str = "") -> str:
        try:
            import requests
            r = requests.get(f"{self._api_base}/knowledge", timeout=10)
            if r.status_code == 200:
                facts = r.json()
                if facts:
                    f = random.choice(facts)
                    content = f.get("content", f.get("text", str(f)))
                    return f"[{f.get('topic', 'general')}] {content}"
                return "  Knowledge base is empty."
            return f"  knowledge: API error {r.status_code}"
        except ImportError:
            return "  knowledge: requests not available"
        except Exception as e:
            return f"  knowledge: {e}"

    def write(self, data: str) -> str:
        text = data.strip()
        if not text:
            return "  Usage: echo <fact> > /dev/knowledge"
        try:
            import requests
            r = requests.post(
                f"{self._api_base}/knowledge",
                json={"content": text},
                timeout=10,
            )
            if r.status_code in (200, 201):
                return f"  Stored: {text[:60]}..."
            return f"  knowledge: API error {r.status_code}"
        except ImportError:
            return "  knowledge: requests not available"
        except Exception as e:
            return f"  knowledge: {e}"


class VisionDevice(AIDevice):
    name = "vision"
    description = "Image analysis — write image path, read returns classification"

    def read(self, args: str = "") -> str:
        return "  Write an image path: echo <path> > /dev/vision"

    def write(self, data: str) -> str:
        path = data.strip()
        if not path or not os.path.isfile(path):
            return f"  File not found: {path}"
        # Delegate to VisionCNN if available
        try:
            from domains.multimodal.vision import VisionCNN
            cnn = VisionCNN()
            from PIL import Image
            img = Image.open(path).convert("RGB")
            result = cnn.caption(img)
            return f"  Vision: {result.text}"
        except ImportError:
            return f"  VisionCNN not available — file exists: {path} ({os.path.getsize(path)} bytes)"


class ProcDevice(AIDevice):
    """Virtual /proc filesystem — process tree, resource stats, uptime."""

    name = "proc"
    description = "Virtual process filesystem — /proc/uptime, /proc/loadavg, /proc/pid/status"

    def __init__(self, get_kernel):
        self._get_kernel = get_kernel

    def read(self, args: str = "") -> str:
        path = args.strip().lstrip("/")
        kernel = self._get_kernel() if callable(self._get_kernel) else self._get_kernel

        if path == "uptime" or path == "":
            return f"{kernel.uptime:.2f}" if kernel else "0.00"

        if path == "loadavg":
            procs = len(kernel.list_processes()) if kernel else 0
            return f"0.00 0.00 0.00 1/{procs}"

        if path == "stat":
            if not kernel:
                return "kernel not available"
            procs = kernel.list_processes()
            lines = [f"processes {len(procs)}"]
            for p in procs:
                lines.append(f"pid {p.pid}  {p.name}  {p.state}  {p.uptime:.1f}s")
            return "\n".join(lines)

        # /proc/<pid>/status
        pid_match = re.match(r"(\d+)/status", path)
        if pid_match and kernel:
            pid = int(pid_match.group(1))
            p = kernel.get_process(pid)
            if p:
                return (
                    f"Name:\t{p.name}\n"
                    f"Pid:\t{p.pid}\n"
                    f"State:\t{p.state}\n"
                    f"Uptime:\t{p.uptime:.1f}s\n"
                )
            return f"  No such process: {pid}"

        return f"  /proc/{path}: No such file or directory"

    def write(self, data: str) -> str:
        return "  /proc is read-only"


# ── Device Manager ─────────────────────────────────────────────────────────


class DeviceManager:
    """Manages all registered AI device nodes. Provides read/write dispatch."""

    def __init__(self):
        self._devices: dict[str, AIDevice] = {}

    def register(self, device: AIDevice) -> AIDevice:
        self._devices[device.name] = device
        logger.debug("registered device /dev/%s", device.name)
        return device

    def get(self, name: str) -> AIDevice | None:
        return self._devices.get(name)

    @property
    def names(self) -> list[str]:
        return sorted(self._devices.keys())

    def list_devices(self) -> str:
        lines = [f"  {'/dev/' + n:<20} {self._devices[n].description}" for n in self.names]
        return "\n".join(lines)

    def read(self, path: str, args: str = "") -> str:
        """Read from a device by path (e.g. /dev/llm or /dev/proc/uptime)."""
        cleaned = path.lstrip("/")
        if cleaned.startswith("dev/"):
            cleaned = cleaned[4:]
        parts = cleaned.split("/", 1)
        dev_name = parts[0]
        subpath = parts[1] if len(parts) > 1 else ""
        device = self._devices.get(dev_name)
        if device is None:
            return f"  /dev/{dev_name}: No such device"
        combined = (subpath + " " + args).strip() if subpath else args
        return device.read(args=combined)

    def write(self, path: str, data: str) -> str:
        """Write to a device by path (e.g. /dev/llm)."""
        cleaned = path.lstrip("/").replace("dev/", "", 1)
        parts = cleaned.split("/", 1)
        dev_name = parts[0]
        device = self._devices.get(dev_name)
        if device is None:
            return f"  /dev/{dev_name}: No such device"
        return device.write(data)

    @staticmethod
    def is_device_path(path: str) -> bool:
        return path.startswith("/dev/") or path.startswith("dev/")


# ── Default devices ────────────────────────────────────────────────────────


def create_default_devices(get_kernel: Callable | None = None) -> DeviceManager:
    """Create and register all built-in device nodes."""
    mgr = DeviceManager()
    mgr.register(NullDevice())
    mgr.register(RandomDevice())
    mgr.register(LLMDevice())
    mgr.register(EmbeddingDevice())
    mgr.register(KnowledgeDevice())
    mgr.register(VisionDevice())
    mgr.register(ProcDevice(get_kernel))
    return mgr
