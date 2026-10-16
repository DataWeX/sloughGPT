"""
Tests for the world frame stream — StreamDevice + world_stream.asm.

Covers the three Card C pieces:
  * StreamDevice        — ConsoleDevice-callback frame sink (ring, tags,
                          host pull, faults)
  * WorldDevice         — step/render/stats driver over the live sim,
                          render returns an arena slot (bus values only)
  * the asm app itself  — world_stream.asm runs step -> render ->
                          encode_png -> stream write in machine code, and
                          the run_stream() kernel driver wraps it
"""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from domain.shell._internal.cycles_device import CyclesDevice
from domain.shell._internal.realm_view import make_live_scene
from domain.shell._internal.tensor_arena import TensorArena
from domain.shell._internal.vm import (
    _STANDALONE_DEVICES,
    DeviceFault,
    StreamDevice,
    VMRunner,
    register_standalone_device,
)
from domain.shell._internal.vm_devices import WorldDevice
from domain.shell._internal.world_render import RenderConfig
from domain.shell._internal.world_stream import _ASM_PATH, run_stream

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _tiny_scene():
    return make_live_scene(
        grid=(6, 4, 6), population=2, organic_pools=1, day_ticks=12, seed=3
    )


# ── StreamDevice ─────────────────────────────────────────────────────────


class TestStreamDevice:
    def test_write_invokes_callback_with_tag(self):
        seen = []
        dev = StreamDevice(frame_fn=lambda data, tag: seen.append((data, tag)))
        n = dev.write(b"png-bytes", tag=42)
        assert n == len(b"png-bytes")
        assert seen == [(b"png-bytes", 42)]

    def test_write_without_tag(self):
        seen = []
        dev = StreamDevice(frame_fn=lambda data, tag: seen.append((data, tag)))
        dev.write(b"x")
        assert seen == [(b"x", None)]

    def test_str_write_encodes_utf8(self):
        dev = StreamDevice()
        assert dev.write("héllo") == len("héllo".encode())

    def test_numeric_write_faults(self):
        dev = StreamDevice()
        with pytest.raises(DeviceFault, match="expects bytes"):
            dev.write(4096)

    def test_ring_keeps_newest_and_counts_drops(self):
        dev = StreamDevice(max_pending=2)
        for i in range(4):
            dev.write(b"f%d" % i, tag=i)
        info = dev.info()
        assert info["written"] == 4
        assert info["dropped"] == 2
        assert info["pending"] == 2
        assert dev.read() == (2, b"f2")
        assert dev.read() == (3, b"f3")
        assert dev.read() is None

    def test_clear(self):
        dev = StreamDevice()
        dev.write(b"a")
        dev.write(b"b")
        assert dev.clear() == 2
        assert dev.read() is None
        assert dev.info()["pending"] == 0

    def test_unknown_method_faults(self):
        with pytest.raises(DeviceFault, match="does not support"):
            StreamDevice().call("teleport")


# ── WorldDevice ──────────────────────────────────────────────────────────


class TestWorldDevice:
    def test_step_returns_increasing_tick(self):
        scene = _tiny_scene()
        dev = WorldDevice(scene, config=RenderConfig(width=24, height=18, samples=1))
        assert dev.call("step") == 1
        assert dev.call("step") == 2
        assert dev.call("step") == 3

    def test_render_returns_slot_with_image(self):
        scene = _tiny_scene()
        dev = WorldDevice(scene, config=RenderConfig(width=24, height=18, samples=1))
        dev.call("step")
        slot = dev.call("render")
        assert isinstance(slot, (int, np.integer))
        img = dev.arena.get(int(slot))
        assert img.shape == (18, 24, 3)
        assert np.isfinite(img).all()

    def test_free_slot_then_render_again(self):
        scene = _tiny_scene()
        dev = WorldDevice(scene, config=RenderConfig(width=24, height=18, samples=1))
        slot = dev.call("render")
        assert dev.call("free_slot", slot) == 0
        slot2 = dev.call("render")  # arena must be reusable
        assert isinstance(slot2, (int, np.integer))

    def test_stats_shape(self):
        scene = _tiny_scene()
        dev = WorldDevice(scene, config=RenderConfig(width=24, height=18, samples=1))
        stats = dev.call("stats")
        assert stats["tick"] == 0
        assert 0 <= stats["alive"] <= stats["population"]
        assert stats["energy_total"] >= 0.0

    def test_info(self):
        scene = _tiny_scene()
        dev = WorldDevice(scene, config=RenderConfig(width=24, height=18, samples=1))
        info = dev.info()
        assert info["type"] == "world"
        assert info["resolution"] == [24, 18]
        assert info["samples"] == 1

    def test_unknown_method_faults(self):
        dev = WorldDevice(_tiny_scene(), config=RenderConfig(width=24, height=18, samples=1))
        with pytest.raises(DeviceFault, match="does not support"):
            dev.call("time_travel")


# ── The asm app ──────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _clean_standalone_registry():
    yield
    for name in ("world", "cycles", "stream"):
        _STANDALONE_DEVICES.pop(name, None)


class TestWorldStreamAsm:
    def test_asm_app_streams_png_frames(self):
        """world_stream.asm: step -> render -> encode_png -> stream write."""
        scene = _tiny_scene()
        arena = TensorArena()
        received = []
        world = WorldDevice(
            scene, config=RenderConfig(width=24, height=18, samples=1), arena=arena
        )
        cycles = CyclesDevice(width=24, height=18, samples=1, arena=arena)
        stream = StreamDevice(frame_fn=lambda data, tag: received.append((tag, data)))
        register_standalone_device("world", world)
        register_standalone_device("cycles", cycles)
        register_standalone_device("stream", stream)

        source = _ASM_PATH.read_text().replace("__FRAMES__", "2")
        runner = VMRunner()
        runner.assemble_and_run(source, max_steps=1000)

        assert len(received) == 2
        ticks = [tag for tag, _ in received]
        assert ticks == [1, 2]  # tag = simulation tick, streamed in order
        for _, png in received:
            assert png[:8] == PNG_MAGIC
            img = Image.open(io.BytesIO(png))
            assert img.size == (24, 18)
        assert stream.info()["written"] == 2
        assert scene.tick == 2

    def test_asm_render_slot_stays_machine_value(self):
        """The register file never holds the frame ndarray — slot + bytes only."""
        scene = _tiny_scene()
        arena = TensorArena()
        register_standalone_device(
            "world",
            WorldDevice(scene, config=RenderConfig(width=24, height=18, samples=1), arena=arena),
        )
        register_standalone_device("cycles", CyclesDevice(arena=arena))

        source = """
        DEV_OPEN  R6, "world"
        DEV_OPEN  R7, "cycles"
        DEV_CALL  R1, R6, "render"
        DEV_CALL  R2, R7, "encode_png", R1
        DEV_CALL  R3, R7, "free_slot", R1
        HALT
        """
        runner = VMRunner()
        runner.assemble_and_run(source, max_steps=100)

        assert isinstance(runner.cpu.regs[1], (int, np.integer))  # slot, not ndarray
        assert runner.cpu.regs[2][:8] == PNG_MAGIC  # bytes, not ndarray


# ── Kernel driver (run_stream) ───────────────────────────────────────────


class TestRunStream:
    def test_run_stream_end_to_end(self, tmp_path):
        result = run_stream(
            2,
            grid=(6, 4, 6),
            population=2,
            pools=1,
            day=12,
            seed=3,
            width=24,
            height=18,
            samples=1,
            out_dir=tmp_path,
        )
        assert result["frames"] == 2
        assert result["ticks"] == [1, 2]
        assert result["bytes"] > 0
        assert result["tick"] == 2
        assert result["fps"] > 0
        files = sorted(tmp_path.glob("frame_*.png"))
        assert len(files) == 2
        for f in files:
            assert f.read_bytes()[:8] == PNG_MAGIC

    def test_run_stream_restores_device_registry(self):
        sentinel = object()
        register_standalone_device("stream", sentinel)
        try:
            run_stream(1, grid=(6, 4, 6), population=1, pools=1, seed=3,
                       width=24, height=18, samples=1)
            assert _STANDALONE_DEVICES["stream"] is sentinel  # restored, not clobbered
            assert "world" not in _STANDALONE_DEVICES
            assert "cycles" not in _STANDALONE_DEVICES
        finally:
            _STANDALONE_DEVICES.pop("stream", None)

    def test_run_stream_zero_frames(self):
        result = run_stream(
            0, grid=(6, 4, 6), population=1, pools=1, seed=3, width=24, height=18, samples=1
        )
        assert result["frames"] == 0
        assert result["stream"]["written"] == 0
