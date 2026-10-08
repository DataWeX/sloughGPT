"""
Shell Device Nodes — /dev/ai (unified) and the per-capability nodes.

``/dev/ai`` is the single AI node: ``generate``, ``embed``, ``health`` and
``info`` all go through one driver's ``ioctl``. ``/dev/llm`` and
``/dev/embedding`` are aliases of it, so old habits keep working.
The remaining nodes stay separate: /dev/null, /dev/random, /dev/knowledge,
/dev/vision and /dev/proc. Hardware drivers (tensor, npu, storage, network,
display, input) live in ``kernel_devices`` and share the same registry.

Each device behaves like a Unix device file:
  - cat /dev/ai                 → capability card (no API call)
  - echo prompt > /dev/ai       → generate
  - echo "embed: text" > /dev/ai → pick an op with an <op>: prefix
  - cat /dev/random             → reads random content
"""

from __future__ import annotations

import logging
import os
import random
import re
import string
from collections.abc import Callable
from typing import Any

from .kernel_devices import DeviceDriver, DeviceType
from .kernel_syscall import SyscallResult

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

    def info(self) -> dict[str, Any]:
        """Device info dict — same shape DeviceDriver.info() returns."""
        return {
            "name": self.name,
            "type": "ai",
            "state": "READY",
            "description": self.description,
        }


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
            from domain.inference._internal.vector_store import simple_embed

            return simple_embed(text)
        except Exception as e:
            import logging

            logging.getLogger("slo.devices").warning(
                "simple_embed failed, using hash fallback: %s", e
            )
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
            from domain.multimodal._internal.vision import VisionCNN

            cnn = VisionCNN()
            from PIL import Image

            img = Image.open(path).convert("RGB")
            result = cnn.caption(img)
            return f"  Vision: {result.text}"
        except ImportError:
            return (
                f"  VisionCNN not available — file exists: {path} ({os.path.getsize(path)} bytes)"
            )


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
                    f"Name:\t{p.name}\nPid:\t{p.pid}\nState:\t{p.state}\nUptime:\t{p.uptime:.1f}s\n"
                )
            return f"  No such process: {pid}"

        return f"  /proc/{path}: No such file or directory"

    def write(self, data: str) -> str:
        return "  /proc is read-only"


# ── Unified AI node ────────────────────────────────────────────────────────


class AIDeviceDriver(DeviceDriver, AIDevice):
    """The unified AI device node — ``/dev/ai``.

    One node, several operations, all dispatched through the kernel's
    existing ``ioctl`` seam rather than a per-capability device file::

        ioctl("generate", prompt)   text generation   (LLMDevice)
        ioctl("embed", text)        embedding         (EmbeddingDevice)
        ioctl("health")             API probe
        ioctl("info")               capability card
        ioctl("auto", payload)      ``<op>:`` prefix selects, else generate

    The VFS projects this onto ``/dev/ai`` plus the alias nodes
    ``/dev/llm`` and ``/dev/embedding`` (see :meth:`vfs_ops`), so the shell's
    plain ``cat`` / ``echo >`` reach the driver without any device-specific
    plumbing. Behaviour is *composed* from the existing ``LLMDevice`` and
    ``EmbeddingDevice`` — never duplicated.

    Training is deliberately not an operation here: it is long-running and
    checkpointed, with its own ``TRAIN_START``/``TRAIN_STOP`` syscalls.
    """

    #: Operations this node advertises (``DeviceTable.capabilities`` reads it).
    OPS: tuple[str, ...] = ("generate", "embed", "health", "info")

    description = "Unified AI device — generate, embed, health, info"

    #: VFS projection per node name: ``(read_op, write_op)``.
    #: ``"auto"`` lets the written payload pick the operation with an
    #: ``<op>:`` prefix; the suffix names the default when there is no
    #: prefix (``auto`` → generate, ``auto:embed`` → embed) so
    #: ``echo text > /dev/embedding`` keeps embedding while
    #: ``echo prompt > /dev/ai`` keeps generating.
    VFS_OPS: dict[str, tuple[str, str]] = {
        "ai": ("info", "auto"),
        "llm": ("info", "auto"),
        "embedding": ("info", "auto:embed"),
    }

    #: Write default per node, derived from :attr:`VFS_OPS` once at class
    #: definition — the card renders often and shouldn't parse op strings.
    _WRITE_DEFAULT: dict[str, str] = {
        node: (ops[1].partition(":")[2] or "generate") for node, ops in VFS_OPS.items()
    }

    def __init__(
        self,
        name: str = "ai",
        *,
        generate_fn: Callable | None = None,
        embed_fn: Callable | None = None,
    ):
        super().__init__(name, DeviceType.INFERENCE)
        self._llm = LLMDevice(generate_fn=generate_fn)
        self._embedding = EmbeddingDevice(embed_fn=embed_fn)
        self._aliases: set[str] = {name}

    # ── Kernel seam ────────────────────────────────────────────────────

    def ioctl(self, command: str, *args: Any) -> SyscallResult:
        """Run an operation on this node. Payload, if any, is ``args[0]``."""
        op = (command or "").strip().lower()
        payload = str(args[0]) if args else ""
        if op.startswith("auto"):
            # ``auto`` → the payload's ``<op>:`` prefix, else generate;
            # ``auto:<op>`` → same, but fall back to that node's default.
            default = op.partition(":")[2].strip()
            if default and default not in self.OPS:
                return SyscallResult.fail(
                    f"/dev/{self.name}: bad default op '{default}' (ops: {', '.join(self.OPS)})"
                )
            op, payload = self._split_op(payload, default or "generate")
        elif not op:
            op = "info"
        if op not in self.OPS:
            return SyscallResult.fail(
                f"/dev/{self.name}: unknown operation '{command}' (ops: {', '.join(self.OPS)})"
            )
        return SyscallResult.ok(self._run(op, payload))

    def list_commands(self) -> list[str]:
        """Advertised capabilities — ``DeviceTable.capabilities()`` reads this."""
        return list(self.OPS)

    def info(self) -> dict[str, Any]:
        info = super().info()
        info.update(
            {
                "description": self.description,
                "ops": list(self.OPS),
                "aliases": sorted(self._aliases - {self.name}),
            }
        )
        return info

    # ── AIDevice projection (cat / echo >) ─────────────────────────────

    def read(self, args: str = "") -> str:
        """``cat /dev/ai`` → capability card; ``cat /dev/ai <prompt>`` → generate."""
        payload = (args or "").strip()
        if not payload:
            return self._run("info", "")
        return self._run("generate", payload)

    def write(self, data: str) -> str:
        """Write payload — an ``<op>:`` prefix selects the operation."""
        op, payload = self._split_op(data)
        return self._run(op, payload)

    # ── VFS projection ─────────────────────────────────────────────────

    def vfs_ops(self, name: str) -> tuple[str, str] | None:
        """``(read_op, write_op)`` for node ``name``, or None if we don't own it."""
        if name in self.VFS_OPS:
            return self.VFS_OPS[name]
        if name in self._aliases:
            return ("info", "auto")
        return None

    def add_alias(self, name: str) -> bool:
        """Track an extra node name for this driver."""
        if not name or name == self.name:
            return False
        self._aliases.add(name)
        return True

    # ── Operation implementations ──────────────────────────────────────

    @staticmethod
    def _split_op(data: str, default_op: str = "generate") -> tuple[str, str]:
        """``"embed: hello"`` → ``("embed", "hello")``; anything else → ``default_op``.

        ``echo`` prints its arguments verbatim, so a quoted payload reaches the
        device with the quotes intact (``'"embed: hello"'``). Strip one matched
        outer quote pair before matching the ``<op>:`` prefix — but only when the
        inner text holds no same-kind quote, so ``'it's fine'`` and half-quoted
        expressions are passed through untouched.
        """
        text = (data or "").strip()
        if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
            inner = text[1:-1]
            if text[0] not in inner:
                text = inner.strip()
        head, sep, rest = text.partition(":")
        key = head.strip().lower()
        if sep and key in AIDeviceDriver.OPS:
            return key, rest.strip()
        return default_op, text

    def _run(self, op: str, payload: str) -> str:
        if op == "generate":
            return self._generate(payload)
        if op == "embed":
            return self._embed(payload)
        if op == "health":
            return self._health()
        # ``info``: the VFS passes the node name it was opened through.
        node = payload if payload in self._aliases else ""
        return self._info_text(node)

    def _generate(self, prompt: str) -> str:
        prompt = (prompt or "").strip()
        if not prompt:
            return f'  no prompt — usage: echo "<prompt>" > /dev/{self.name}'
        return self._llm.read(prompt)

    def _embed(self, text: str) -> str:
        text = (text or "").strip()
        if not text:
            # Empty read previews the last embedding (or explains how to make one).
            return self._embedding.read().strip()
        dims = self._embedding.write(text).strip()
        return f"{dims}\n{self._embedding.read().strip()}"

    def _health(self) -> str:
        try:
            from .config import get_api_base
            from .runtime import _probe_api
        except Exception as exc:  # pragma: no cover - import guarded
            return f"  /dev/{self.name}: health unavailable ({exc})"
        status = _probe_api(get_api_base())
        if not status.get("available"):
            return f"  /dev/{self.name}: API unreachable — {status.get('error', 'unknown')}"
        model = status.get("model_id") or "none"
        state = "loaded" if status.get("model_loaded") else "not loaded"
        return f"  /dev/{self.name}: API {status.get('status', 'ok')} — model {model} ({state})"

    def _info_text(self, node: str = "") -> str:
        node = node or self.name
        # The card must state THIS node's write default: /dev/embedding
        # embeds, /dev/ai and /dev/llm generate.
        default = self._default_write_op(node)
        example = "text" if default == "embed" else "prompt"
        lines = [
            f"  /dev/{node} — {self.description}",
            f"  ops: {', '.join(self.OPS)}",
            f'  echo "<{example}>" > /dev/{node}      # {default}',
            f'  echo "embed: <text>" > /dev/{node}  # any op as <op>: <payload>',
            f"  cat /dev/{node}                    # this card",
        ]
        if node != self.name:
            lines.append(f"  canonical node: /dev/{self.name}")
        others = sorted(n for n in self._aliases if n != node)
        if others:
            lines.append("  also reachable as: " + ", ".join(f"/dev/{n}" for n in others))
        return "\n".join(lines)

    @classmethod
    def _default_write_op(cls, node: str) -> str:
        """Fallback operation for a node's write when the payload has no ``<op>:``."""
        return cls._WRITE_DEFAULT.get(node, "generate")


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
        """Human-readable device listing (``lsdev`` prints this)."""
        lines = []
        for n in self.names:
            dev = self._devices[n]
            desc = getattr(dev, "description", "") or str(dev.info().get("type", "unknown"))
            lines.append(f"  {'/dev/' + n:<20} {desc}")
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
    """Create and register all built-in device nodes.

    Two families share one registry:

    - **AI** — ``/dev/ai`` plus its aliases ``/dev/llm`` and ``/dev/embedding``
      (one driver instance, three node names), then ``null``, ``random``,
      ``knowledge``, ``vision`` and ``proc``.
    - **hardware** — ``tensor``, ``npu``, ``storage``, ``network``,
      ``display``, ``input``.

    The VFS builds ``/dev`` from ``mgr.names``, so both families appear there;
    only nodes whose driver exposes ``vfs_ops()`` get a live (ioctl-backed)
    entry — everything else keeps the inert placeholder it has always had.
    """
    from .display_device import DisplayDevice
    from .input_device import InputDevice
    from .kernel_devices import DeviceManager as KernelDeviceManager
    from .network_device import NetworkDevice
    from .npu_device import NPUDevice
    from .storage_device import StorageDevice
    from .tensor_device import TensorDevice

    mgr = KernelDeviceManager()

    # AI family — one unified node, aliased twice.
    ai = AIDeviceDriver("ai")
    mgr.register(ai)
    mgr.alias("llm", "ai")
    mgr.alias("embedding", "ai")
    mgr.register(NullDevice())
    mgr.register(RandomDevice())
    mgr.register(KnowledgeDevice())
    mgr.register(VisionDevice())
    mgr.register(ProcDevice(get_kernel))

    # Hardware family.
    mgr.register(TensorDevice("tensor"))
    mgr.register(NPUDevice("npu"))
    mgr.register(StorageDevice("storage"))
    mgr.register(NetworkDevice("network"))
    mgr.register(DisplayDevice("display"))
    mgr.register(InputDevice("input"))
    return mgr
