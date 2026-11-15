"""Compile-once loader: one C source, two engines.

This is the seam the machine-first stack routes hot kernels through. A
kernel is written ONCE as plain C (``SLO_Q``-tagged functions, no
``#ifdef`` logic in the algorithm itself); this loader compiles that
single source to whichever engine the caller needs:

* CPU — ``gcc -O3 -fopenmp -shared -fPIC`` → ``ctypes`` function. The
  entry is an OpenMP ``parallel for`` over the work units. ctypes
  releases the GIL for the call, so OpenMP fans out across cores.
* GPU — NVRTC (via ``cupy``) → ``RawKernel``. The entry is a
  ``__global__`` kernel with one thread per work unit.

``SLO_Q`` is the ONLY target-specific token: empty on CPU, ``__device__``
on GPU. That is what makes parity structural rather than tested — both
engines execute the same instruction stream, so "nearly identical
performance" and "identical results" stop being hopes and become the
compilation model.

Fallback chain (each step logs once at INFO, never raises):
  no C compiler → CPU returns None (caller uses its numpy oracle)
  no CUDA      → GPU returns None (caller uses the CPU engine)
Both None → caller stays on numpy. Determinism: every kernel gets a
``seed`` mixed into its counter RNG, so a kernel is reproducible in
isolation and across engines regardless of thread/scheduling order.

Ported from the ``quant_core`` gcc-on-demand + numpy-fallback pattern;
generalized to also cover NVRTC so ONE source serves both engines.
"""

from __future__ import annotations

import ctypes
import hashlib
import logging
import os
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger("slo.native")

# Artifact cache: one dir per source-hash, holds the .so. Lives beside
# this module by default; SLO_NATIVE_CACHE redirects it (tests/CI).
_CACHE_ROOT = Path(os.environ.get("SLO_NATIVE_CACHE") or (Path(__file__).parent / ".native_cache"))

# gcc target flags. Baseline only — the C kernels are portable scalar
# code (no intrinsics), so we never pass -mavx* here the way quant_core
# does; gcc autovectorizes and the ISA always matches the runner.
# -fopenmp is tried first (real parallelism), downgraded to serial if
# the toolchain lacks it (still correct; just single-core).
_GCC_FLAGS = ["-O3", "-shared", "-fPIC"]


def _find_compiler() -> str | None:
    """gcc, then clang. None if neither compiles C on this box."""
    for cc in ("gcc", "clang"):
        path = shutil.which(cc)
        if path:
            return path
    return None


def _source_hash(source: str, flags: list[str]) -> str:
    return hashlib.sha256((source + "\x00" + "\x00".join(flags)).encode()).hexdigest()[:16]


def _has_cuda() -> bool:
    """A working CUDA device via cupy — import AND a real kernel launch."""
    try:
        import cupy as cp

        probe = cp.arange(8, dtype=cp.float32) * 2.0
        float(probe.sum())  # sync: a kernel must actually run
        return True
    except Exception:
        return False


def compile_cpu(source: str, entry: str) -> ctypes.CDLL | None:
    """Compile ``source`` with gcc/clang → shared lib, return ctypes handle.

    The compiled entry is a C function ``void entry(...)`` taking raw
    pointers/scalars (the caller sets argtypes). Returns None if no
    compiler or the compile fails — caller falls back to numpy.
    """
    cc = _find_compiler()
    if cc is None:
        logger.info("native: no C compiler (gcc/clang); CPU engine unavailable — numpy fallback")
        return None

    # Try with OpenMP, downgrade to serial on failure.
    for flags in (["-fopenmp"], []):
        h = _source_hash(source, _GCC_FLAGS + flags)
        _CACHE_ROOT.mkdir(parents=True, exist_ok=True)
        so_path = _CACHE_ROOT / f"core_{h}.so"
        if not so_path.exists():
            src_path = _CACHE_ROOT / f"core_{h}.c"
            src_path.write_text(source)
            cmd = [cc, *_GCC_FLAGS, *flags, "-o", str(so_path), str(src_path)]
            try:
                res = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            except (subprocess.TimeoutExpired, OSError) as exc:
                logger.warning("native: compile invoke failed (%s); numpy fallback", exc)
                return None
            if res.returncode != 0:
                logger.warning(
                    "native: compile failed (%s %s): %s",
                    cc,
                    flags,
                    res.stderr[-400:] if res.stderr else "(no stderr)",
                )
                if flags:  # -fopenmp failed → try serial
                    continue
                return None
        try:
            return ctypes.CDLL(str(so_path))
        except OSError as exc:
            logger.warning("native: dlopen failed (%s); numpy fallback", exc)
            return None
    return None


def compile_gpu(source: str, entry: str):
    """NVRTC-compile ``source`` → cupy RawKernel for ``entry``.

    Returns the kernel (callable with grid/block/args) or None if CUDA
    is unavailable or compilation fails — caller uses the CPU engine.
    """
    if not _has_cuda():
        return None
    try:
        import cupy as cp

        # NVRTC via RawModule. __CUDACC__ is predefined by NVRTC, which
        # (a) flips SLO_Q to __device__ inside the source and (b) selects
        # the __global__ entry out of the #ifdef. One source, verbatim.
        mod = cp.RawModule(code=source)
        return mod.get_function(entry)
    except Exception as exc:
        logger.warning("native: NVRTC compile failed (%s); CPU engine fallback", exc)
        return None


class NativeEngines:
    """Both compiled views of ONE source, lazily built, cached forever.

    ``cpu``: ctypes.CDLL (argtypes set by caller) or None.
    ``gpu``: cupy RawKernel or None.
    ``any``: True when at least one engine compiled.
    """

    def __init__(self, source: str, cpu_entry: str, gpu_entry: str):
        self._source = source
        self._cpu_entry = cpu_entry
        self._gpu_entry = gpu_entry
        self._cpu = "unset"
        self._gpu = "unset"

    @property
    def cpu(self) -> ctypes.CDLL | None:
        if self._cpu == "unset":
            self._cpu = compile_cpu(self._source, self._cpu_entry) or None
        return self._cpu

    @property
    def gpu(self):
        if self._gpu == "unset":
            self._gpu = compile_gpu(self._source, self._gpu_entry) or None
        return self._gpu

    @property
    def any(self) -> bool:
        return self.cpu is not None or self.gpu is not None
