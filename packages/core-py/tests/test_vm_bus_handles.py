"""
Tests for the VM bus-value discipline — TensorArena slots + resolve().

The register file holds machine-representable values only (the fd
precedent from FileDevice): ``cycles render`` returns a slot id, never
a live ndarray. Covers the arena itself, the resolve() boundary
function, the CyclesDevice slot/encode_png contract, MultimodalDevice
slot consumption, and the flagship pure-assembly pipeline:

    render -> encode_png -> FileDevice.write   (writes a valid PNG)
    cycles render -> multimodal embed          (slot crosses devices)
"""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from domain.shell._internal.cycles_device import CyclesDevice
from domain.shell._internal.tensor_arena import TensorArena, resolve
from domain.shell._internal.vm import (
    _STANDALONE_DEVICES,
    DeviceFault,
    FileDevice,
    VMRunner,
    register_standalone_device,
)
from domain.shell._internal.vm_devices import MultimodalDevice

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


# ── TensorArena ─────────────────────────────────────────────────────────


class TestTensorArena:
    def test_put_get_roundtrip(self):
        arena = TensorArena()
        arr = np.ones((4, 4, 3), dtype=np.float32)
        h = arena.put(arr)
        assert arena.get(h) is arr

    def test_slots_monotonic(self):
        arena = TensorArena()
        hs = [arena.put(np.zeros(3)) for _ in range(5)]
        assert hs == [0, 1, 2, 3, 4]

    def test_len(self):
        arena = TensorArena()
        arena.put(np.zeros(3))
        arena.put(np.zeros(3))
        assert len(arena) == 2

    def test_free(self):
        arena = TensorArena()
        h = arena.put(np.zeros(3))
        arena.free(h)
        assert len(arena) == 0
        with pytest.raises(DeviceFault, match="bad slot"):
            arena.get(h)

    def test_free_unknown_raises(self):
        arena = TensorArena()
        with pytest.raises(DeviceFault, match="bad slot"):
            arena.free(99)

    def test_get_unknown_raises(self):
        arena = TensorArena()
        with pytest.raises(DeviceFault, match="bad slot"):
            arena.get(99)

    def test_clear(self):
        arena = TensorArena()
        arena.put(np.zeros(3))
        arena.put(np.zeros(3))
        arena.clear()
        assert len(arena) == 0

    def test_freed_slot_ids_not_reused(self):
        arena = TensorArena()
        h0 = arena.put(np.zeros(3))
        arena.free(h0)
        h1 = arena.put(np.zeros(3))
        assert h1 != h0


# ── resolve() ───────────────────────────────────────────────────────────


class TestResolve:
    def test_ndarray_passthrough(self):
        arena = TensorArena()
        arr = np.zeros((2, 2, 3))
        assert resolve(arr, arena) is arr

    def test_none_passthrough(self):
        assert resolve(None, TensorArena()) is None

    def test_int_slot_lookup(self):
        arena = TensorArena()
        arr = np.zeros(5)
        h = arena.put(arr)
        assert resolve(h, arena) is arr

    def test_np_integer_slot_lookup(self):
        arena = TensorArena()
        arr = np.zeros(5)
        h = arena.put(arr)
        assert resolve(np.int64(h), arena) is arr

    def test_bad_type_raises(self):
        with pytest.raises(DeviceFault, match="expected ndarray or arena slot"):
            resolve("not a bus value", TensorArena())

    def test_bad_slot_raises(self):
        with pytest.raises(DeviceFault, match="bad slot"):
            resolve(42, TensorArena())


# ── CyclesDevice slot contract ──────────────────────────────────────────


class TestCyclesRenderSlot:
    def setup_method(self):
        self.dev = CyclesDevice(width=32, height=24, samples=2)

    def test_render_returns_int_slot(self):
        self.dev.call("add_sphere")
        self.dev.call("add_light")
        slot = self.dev.call("render")
        assert isinstance(slot, int)

    def test_slot_resolves_to_image(self):
        self.dev.call("add_sphere")
        self.dev.call("add_light")
        slot = self.dev.call("render")
        assert self.dev.arena.get(slot).shape == (24, 32, 3)

    def test_renders_get_distinct_slots(self):
        self.dev.call("add_sphere")
        self.dev.call("add_light")
        h1 = self.dev.call("render")
        h2 = self.dev.call("render")
        assert h1 != h2
        assert len(self.dev.arena) == 2

    def test_encode_png_magic(self):
        self.dev.call("add_sphere")
        self.dev.call("add_light")
        self.dev.call("render")
        png = self.dev.call("encode_png")
        assert png[:8] == PNG_MAGIC

    def test_encode_png_before_render_faults(self):
        with pytest.raises(DeviceFault, match="render first"):
            self.dev.call("encode_png")

    def test_encode_png_for_slot(self):
        slot = self.dev.arena.put(np.full((5, 7, 3), 0.5))
        png = self.dev.call("encode_png", slot)
        img = Image.open(io.BytesIO(png))
        assert img.size == (7, 5)  # (width, height)

    def test_encode_png_legacy_live_array_resolves(self):
        # resolve() passthrough: a live array put by Python still encodes
        slot = self.dev.arena.put(np.zeros((4, 4, 3)))
        assert self.dev.call("encode_png", slot)[:8] == PNG_MAGIC

    def test_free_slot(self):
        self.dev.call("add_sphere")
        self.dev.call("add_light")
        slot = self.dev.call("render")
        self.dev.call("free_slot", slot)
        with pytest.raises(DeviceFault, match="bad slot"):
            self.dev.arena.get(slot)

    def test_free_slot_unknown_raises(self):
        with pytest.raises(DeviceFault, match="bad slot"):
            self.dev.call("free_slot", 12345)

    def test_shared_arena_between_devices(self):
        arena = TensorArena()
        cycles = CyclesDevice(width=16, height=12, samples=1, arena=arena)
        assert cycles.arena is arena


# ── MultimodalDevice slot consumption ───────────────────────────────────


class _FakeVision:
    embed_dim = 64


class _FakeEngine:
    def __init__(self):
        self.received = []

    def generate(self, image_np, max_len, temperature):
        self.received.append(image_np)
        return _FakeOut()

    def _concat_modalities(self, img, text, extra):
        self.received.append(img)
        return _FakeEmbed(), np.zeros((1, 7, 16)), None


class _FakeOut:
    text = "a caption"


class _FakeEmbed:
    data = np.zeros((1, 64), dtype=np.float32)


class TestMultimodalResolve:
    def test_embed_accepts_slot(self):
        arena = TensorArena()
        engine = _FakeEngine()
        dev = MultimodalDevice(engine, arena=arena)
        slot = arena.put(np.zeros((8, 8, 3), dtype=np.float32))
        assert dev.call("embed", slot).shape == (1, 64)
        assert isinstance(engine.received[-1], np.ndarray)

    def test_generate_accepts_slot(self):
        arena = TensorArena()
        engine = _FakeEngine()
        dev = MultimodalDevice(engine, arena=arena)
        slot = arena.put(np.zeros((8, 8, 3), dtype=np.float32))
        assert dev.call("generate", slot) == "a caption"
        assert isinstance(engine.received[-1], np.ndarray)

    def test_legacy_live_array_still_works(self):
        dev = MultimodalDevice(_FakeEngine())
        assert dev.call("embed", np.zeros((8, 8, 3))).shape == (1, 64)

    def test_bad_value_raises(self):
        dev = MultimodalDevice(_FakeEngine())
        with pytest.raises(DeviceFault, match="expected ndarray or arena slot"):
            dev.call("embed", "not an image")


# ── Pure-assembly pipelines ─────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _clean_standalone_registry():
    yield
    _STANDALONE_DEVICES.clear()


class TestAsmPipeline:
    def test_asm_render_encode_write_png(self, tmp_path):
        """render -> encode_png -> FileDevice.write writes a valid PNG."""
        cycles = CyclesDevice(width=32, height=24, samples=2)
        register_standalone_device("cycles", cycles)
        register_standalone_device("file", FileDevice())
        out = str(tmp_path / "frame.png")

        source = f"""
        DEV_OPEN  R0, "cycles"
        DEV_CALL  R1, R0, "add_sphere"
        DEV_CALL  R2, R0, "add_light"
        DEV_CALL  R3, R0, "render"
        DEV_CALL  R4, R0, "encode_png", R3
        DEV_OPEN  R5, "file"
        DEV_CALL  R6, R5, "open", "{out}", "wb"
        DEV_CALL  R7, R5, "write", R6, R4
        DEV_CALL  R8, R5, "close", R6
        HALT
        """
        runner = VMRunner()
        runner.assemble_and_run(source)

        assert runner.cpu.regs[3] == 0  # first slot id
        data = (tmp_path / "frame.png").read_bytes()
        assert data[:8] == PNG_MAGIC
        assert runner.cpu.regs[7] > 0  # bytes written

    def test_asm_slot_crosses_devices(self, tmp_path):
        """A cycles render slot embeds in multimodal — pure assembly."""
        arena = TensorArena()
        engine = _FakeEngine()
        cycles = CyclesDevice(width=32, height=24, samples=2, arena=arena)
        multimodal = MultimodalDevice(engine, arena=arena)
        register_standalone_device("cycles", cycles)
        register_standalone_device("multimodal", multimodal)

        source = """
        DEV_OPEN  R0, "cycles"
        DEV_CALL  R1, R0, "add_sphere"
        DEV_CALL  R2, R0, "add_light"
        DEV_CALL  R3, R0, "render"
        DEV_OPEN  R4, "multimodal"
        DEV_CALL  R5, R4, "embed", R3
        HALT
        """
        runner = VMRunner()
        runner.assemble_and_run(source)

        assert isinstance(runner.cpu.regs[5], np.ndarray)
        assert isinstance(engine.received[-1], np.ndarray)
