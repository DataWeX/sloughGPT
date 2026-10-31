"""
World frame stream — ``python -m domain.shell._internal.world_stream``.

Runs the VM assembly app ``world_stream.asm``: the machine loops
step-sim -> render -> encode_png -> stream write; Python only builds the
drivers (kernel + drivers + socket), registers them on the device bus,
and attaches the frame consumer to ``StreamDevice``.  Each frame arrives
as PNG bytes tagged with its simulation tick.

    python -m domain.shell._internal.world_stream                     # 30 frames, 96x72
    python -m domain.shell._internal.world_stream --frames 60 --out /tmp/frames
    python -m domain.shell._internal.world_stream --width 160 --height 120 --samples 16

``--out DIR`` writes every streamed frame as ``frame_<tick>.png``; the
summary line prints totals (summarize, don't dump).  The stream socket
(web/SSE) attaches later at ``StreamDevice(frame_fn=...)`` — same seam.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from .cycles_device import CyclesDevice
from .realm_view import make_live_scene
from .tensor_arena import TensorArena
from .vm import (
    _STANDALONE_DEVICES,
    StreamDevice,
    VMRunner,
    register_standalone_device,
)
from .vm_devices import WorldDevice
from .world_render import RenderConfig

_ASM_PATH = Path(__file__).with_name("world_stream.asm")
_DEVICE_NAMES = ("world", "cycles", "stream")


def run_stream(
    frames: int = 30,
    *,
    grid: tuple[int, int, int] = (24, 12, 24),
    population: int = 8,
    pools: int = 3,
    day: int = 24,
    seed: int = 7,
    width: int = 96,
    height: int = 72,
    samples: int = 4,
    out_dir: str | Path | None = None,
    max_pending: int = 8,
    on_frame=None,
) -> dict:
    """Run ``world_stream.asm`` for ``frames`` frames; returns a summary dict.

    ``on_frame(data, tag)`` — if given — is called per streamed frame
    (PNG bytes, tick).  ``out_dir`` additionally writes each frame to
    ``frame_<tick>.png``.  Both are host-side consumers of the one
    StreamDevice callback seam.
    """
    scene = make_live_scene(
        grid=grid,
        population=population,
        organic_pools=pools,
        day_ticks=day,
        seed=seed,
    )
    arena = TensorArena()
    out_path = Path(out_dir) if out_dir else None
    if out_path:
        out_path.mkdir(parents=True, exist_ok=True)

    received: list[tuple[int, int]] = []

    def _consume(data: bytes, tag) -> None:
        received.append((int(tag) if tag is not None else -1, len(data)))
        if out_path is not None:
            (out_path / f"frame_{int(tag):05d}.png").write_bytes(data)
        if on_frame is not None:
            on_frame(data, tag)

    world = WorldDevice(
        scene,
        config=RenderConfig(width=width, height=height, samples=samples),
        arena=arena,
    )
    cycles = CyclesDevice(width=width, height=height, samples=samples, arena=arena)
    stream = StreamDevice(frame_fn=_consume, max_pending=max_pending)

    saved = {name: _STANDALONE_DEVICES.get(name) for name in _DEVICE_NAMES}
    elapsed = 0.0
    try:
        # LOOP decrements before it tests, so a zero counter would run
        # away — the guard lives in the kernel, not the machine.
        if int(frames) > 0:
            for name, dev in zip(_DEVICE_NAMES, (world, cycles, stream)):
                register_standalone_device(name, dev)

            source = _ASM_PATH.read_text().replace("__FRAMES__", str(int(frames)))
            runner = VMRunner()
            t0 = time.perf_counter()
            runner.assemble_and_run(source, max_steps=int(frames) * 16 + 1000)
            elapsed = time.perf_counter() - t0
    finally:
        for name, prev in saved.items():
            if prev is None:
                _STANDALONE_DEVICES.pop(name, None)
            else:
                _STANDALONE_DEVICES[name] = prev

    stats = world.call("stats")
    return {
        "frames": len(received),
        "bytes": sum(n for _, n in received),
        "ticks": [t for t, _ in received],
        "elapsed_s": elapsed,
        "fps": len(received) / elapsed if elapsed > 0 else 0.0,
        "tick": stats["tick"],
        "alive": stats["alive"],
        "population": stats["population"],
        "stream": stream.info(),
        "out_dir": str(out_path) if out_path else None,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="world_stream",
        description="Stream live world frames: the VM runs step -> render -> encode -> write.",
    )
    parser.add_argument("--frames", type=int, default=30, help="frames to stream (default 30)")
    parser.add_argument(
        "--grid",
        type=lambda v: tuple(int(p) for p in v.split(",")),  # type: ignore[misc]
        default=(24, 12, 24),
        help="world dimensions x,y,z (default 24,12,24)",
    )
    parser.add_argument("--population", type=int, default=8, help="babies on the surface")
    parser.add_argument("--pools", type=int, default=3, help="organic food pools")
    parser.add_argument("--day", type=int, default=24, help="ticks per day/night cycle")
    parser.add_argument("--seed", type=int, default=7, help="world + RNG seed")
    parser.add_argument("--width", type=int, default=96, help="frame width (default 96)")
    parser.add_argument("--height", type=int, default=72, help="frame height (default 72)")
    parser.add_argument("--samples", type=int, default=4, help="samples per pixel (default 4)")
    parser.add_argument("--out", default=None, help="directory to write frame_<tick>.png files")
    parser.add_argument(
        "--pending", type=int, default=8, help="stream ring size (frames kept for a slow consumer)"
    )
    args = parser.parse_args(argv)

    result = run_stream(
        args.frames,
        grid=args.grid,  # type: ignore[arg-type]
        population=args.population,
        pools=args.pools,
        day=args.day,
        seed=args.seed,
        width=args.width,
        height=args.height,
        samples=args.samples,
        out_dir=args.out,
        max_pending=args.pending,
    )

    dest = f"  out={result['out_dir']}" if result["out_dir"] else ""
    size = result["bytes"]
    size_str = f"{size / 1048576:.2f} MB" if size >= 1048576 else f"{size / 1024:.1f} KB"
    print(
        f"world stream: {result['frames']} frames in {result['elapsed_s']:.1f}s = "
        f"{result['fps']:.2f} fps  {size_str}  "
        f"tick={result['tick']} alive={result['alive']}/{result['population']}"
        f"{dest}"
    )
    return 0 if result["frames"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
