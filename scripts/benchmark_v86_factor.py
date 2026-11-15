#!/usr/bin/env python3
"""V86 factor benchmark — card 20261103_121721 (C1): decode tok/s on the target.

Card A measured decode on bare metal (368 tok/s @ 141K params) and judged
it against the 3.0 tok/s v86 floor — a verdict for a machine it never
ran. C1 calibrates the emulation factor instead of guessing it:

1. ONE freestanding i386 float workload (GEMM-shaped — the compute
   decode spends its time in) is compiled once; the v86 guest is a
   32-bit buildroot with no glibc, so the workload brings no libc
   either side;
2. the SAME binary runs (a) native on the host and (b) inside the v86
   guest, injected over serial (base64, wrapped lines — the canonical
   tty input queue is 4095 bytes) and md5-verified;
3. host wall-clock times both sides (the guest clock runs on emulated
   TSC time and is NOT comparable to the host's);
4. factor = guest_s / native_s scales Card A's bare-metal decode rate
   into an in-guest estimate, carrying the assumption explicitly.

Logic lives here (tests: tests/core-py/test_benchmark_v86_factor.py);
scripts/v86_factor_guest.mjs is a dumb serial driver.
"""

from __future__ import annotations

import argparse
import json
import statistics
import struct
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

#: Card A's bare-metal decode rate (own-docs corpus, 141K params, CPU).
DEFAULT_BASELINE_TPS = 368.0
#: Where that number came from — recorded, never re-derived here.
DEFAULT_BASELINE_SOURCE = "Card A gap benchmark (own-docs corpus, recorded)"
#: The floor Card A's verdict was stated against.
DEFAULT_SLO_TPS = 3.0
#: Kernel cmdline — the copy.sh buildroot6 preset the app boots with
#: (apps/web/hooks/useV86.ts KERNEL_CMDLINE).
DEFAULT_CMDLINE = "earlyprintk=serial,ttyS0,115200 tsc=reliable mitigations=off random.trust_cpu=on"

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ASSETS = Path("/tmp/opencode/v86-assets")
DEFAULT_WASM = REPO_ROOT / "node_modules" / "v86" / "build" / "v86.wasm"
DEFAULT_DRIVER = Path(__file__).resolve().parent / "v86_factor_guest.mjs"

ASSET_URLS = {
    "bzimage68.bin": "https://i.copy.sh/buildroot-bzimage68.bin",
    "seabios.bin": "https://copy.sh/v86/bios/seabios.bin",
    "vgabios.bin": "https://copy.sh/v86/bios/vgabios.bin",
}

#: The guest-side workload: a GEMM float loop, the compute shape decode
#: spends its time in. Freestanding i386 (the v86 guest is a 32-bit
#: buildroot with no glibc; syscalls via int 0x80), the SAME binary runs
#: native and emulated so the factor is apples-to-apples.
WORKLOAD_C = r"""
/* V86 factor workload — freestanding i386, no libc. */
typedef unsigned int u32;

#define N 64
#ifndef ITERS
#define ITERS 150
#endif
static int sys_write(int fd, const char *buf, u32 len) {
    int ret;
    __asm__ volatile ("int $0x80"
        : "=a" (ret)
        : "a" (4), "b" (fd), "c" (buf), "d" (len)
        : "memory");
    return ret;
}

static void sys_exit(int code) {
    __asm__ volatile ("int $0x80" : : "a" (1), "b" (code));
    __builtin_unreachable();
}

/* i386 __NR_clock_gettime = 265, CLOCK_MONOTONIC = 1. Guest-clock time
   is diagnostic only — the factor uses host wall-clock. */
static void mono(u32 *sec, u32 *nsec) {
    struct { u32 s; u32 ns; } ts;
    __asm__ volatile ("int $0x80"
        :
        : "a" (265), "b" (1), "c" (&ts)
        : "memory");
    *sec = ts.s;
    *nsec = ts.ns;
}

/* gcc may turn loops into these calls in freestanding mode. */
void *memset(void *dst, int c, unsigned long n) {
    char *d = dst;
    while (n--) *d++ = (char)c;
    return dst;
}
void *memcpy(void *dst, const void *src, unsigned long n) {
    char *d = dst;
    const char *s = src;
    while (n--) *d++ = *s++;
    return dst;
}

static int fmt_u(char *o, u32 v) {
    char tmp[12];
    int n = 0;
    do { tmp[n++] = (char)('0' + (v % 10)); v /= 10; } while (v);
    for (int i = 0; i < n; i++) o[i] = tmp[n - 1 - i];
    return n;
}

static double a[N * N], b[N * N], c[N * N];

void _start(void) {
    for (int i = 0; i < N * N; i++) {
        a[i] = (double)((i * 1103515245u + 12345u) % 1000) / 1000.0;
        b[i] = (double)((i * 22695477u + 1u) % 1000) / 1000.0;
        c[i] = 0.0;
    }
    u32 s0, ns0, s1, ns1;
    mono(&s0, &ns0);
    for (int it = 0; it < ITERS; it++) {
        for (int i = 0; i < N; i++)
            for (int k = 0; k < N; k++) {
                double aik = a[i * N + k];
                for (int j = 0; j < N; j++)
                    c[i * N + j] += aik * b[k * N + j];
            }
    }
    mono(&s1, &ns1);

    long total = (long)(s1 - s0) * 1000000L + (long)(ns1 - ns0) / 1000L;
    double sum = 0.0;
    for (int i = 0; i < N * N; i++) sum += c[i];

    char msg[80];
    int p = 0;
    const char *m = "V86FACTOR_OK ";
    while (*m) msg[p++] = *m++;
    p += fmt_u(msg + p, (u32)total);
    msg[p++] = ' ';
    p += fmt_u(msg + p, (u32)sum);
    msg[p++] = '\n';
    sys_write(1, msg, (u32)p);
    sys_exit(0);
}
"""


def compute_factor(native_s: float, guest_s: float) -> float:
    """guest/native — how many times slower the emulated machine is."""
    if native_s <= 0 or guest_s <= 0:
        raise ValueError(f"times must be positive, got native={native_s} guest={guest_s}")
    return guest_s / native_s


def apply_factor(baseline_tps: float, factor: float) -> float:
    """Scale a bare-metal rate down by the slowdown factor.

    A factor below 1 would mean the emulated machine beat native at the
    same binary — impossible; that is broken data, not a fast guest.
    """
    if factor < 1.0:
        raise ValueError(f"factor must be >= 1.0 (guest cannot beat native), got {factor}")
    return baseline_tps / factor


def verdict(estimate_tps: float, slo_tps: float) -> dict:
    """Meets the floor? With margin — never a bare bool."""
    if estimate_tps <= 0:
        raise ValueError(f"estimate must be positive, got {estimate_tps}")
    if slo_tps <= 0:
        raise ValueError(f"slo must be positive, got {slo_tps}")
    return {
        "estimate_tps": estimate_tps,
        "slo_tps": slo_tps,
        "margin_x": estimate_tps / slo_tps,
        "meets_slo": estimate_tps >= slo_tps,
    }


def parse_guest_output(raw: str) -> dict:
    """Parse the driver's single JSON line — the contract between halves."""
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError) as e:
        raise ValueError(f"driver output is not JSON: {e}") from e
    if not isinstance(data, dict) or "guest_s" not in data or "method" not in data:
        raise ValueError("driver output must contain guest_s and method")
    if not isinstance(data["guest_s"], (int, float)) or data["guest_s"] <= 0:
        raise ValueError(f"guest_s must be positive, got {data.get('guest_s')!r}")
    return data


def build_report(
    baseline_decode_tps: float,
    baseline_source: str,
    native_s: list[float],
    guest_s: float,
    method: str,
    slo_tps: float,
) -> dict:
    """The C1 report: factor, scaled estimate, verdict — assumption attached."""
    if not native_s:
        raise ValueError("native_s needs at least one timing")
    native_med = statistics.median(native_s)
    factor = compute_factor(native_med, guest_s)
    estimate = apply_factor(baseline_decode_tps, factor)
    return {
        "target": "v86-browser-vm",
        "baseline_decode_tps": baseline_decode_tps,
        "baseline_source": baseline_source,
        "native_s_median": native_med,
        "native_repeats": len(native_s),
        "native_s_all": list(native_s),
        "guest_s": guest_s,
        "factor": factor,
        "v86_decode_tps": estimate,
        "method": method,
        "workload": "freestanding i386 GEMM float loop (64x64), same binary native and in-guest",
        "assumption": (
            "calibrated estimate: the guest cannot run numpy, so the factor comes "
            "from a same-binary C float workload and is applied to Card A's "
            "bare-metal decode rate"
        ),
        "slo_tps": slo_tps,
        "verdict": verdict(estimate, slo_tps),
    }


def _is_dynamically_linked(binary: Path) -> bool:
    """True if the ELF carries a PT_INTERP (needs a loader the guest lacks)."""
    data = Path(binary).read_bytes()
    if data[:4] != b"\x7fELF":
        raise ValueError(f"not an ELF binary: {binary}")
    if data[4] == 2:  # ELF64
        (e_phoff,) = struct.unpack_from("<Q", data, 0x20)
        (e_phentsize,) = struct.unpack_from("<H", data, 0x36)
        (e_phnum,) = struct.unpack_from("<H", data, 0x38)
    elif data[4] == 1:  # ELF32 (the i386 workload)
        (e_phoff,) = struct.unpack_from("<I", data, 0x1C)
        (e_phentsize,) = struct.unpack_from("<H", data, 0x2A)
        (e_phnum,) = struct.unpack_from("<H", data, 0x2C)
    else:
        raise ValueError(f"unknown ELF class {data[4]}: {binary}")
    for i in range(e_phnum):
        off = e_phoff + i * e_phentsize
        (p_type,) = struct.unpack_from("<I", data, off)
        if p_type == 3:  # PT_INTERP
            return True
    return False


def compile_workload(binary: Path, cc: str = "gcc", iters: int = 150, sse2: bool = False) -> Path:
    """Compile the freestanding i386 GEMM workload — one binary, both sides.

    -m32 -nostdlib: the guest is a 32-bit buildroot without glibc, and
    the host has no 32-bit libc either — neither side needs one.
    sse2=True targets the guest's SSE2 (the set an in-guest decode would
    be compiled for); default is the x87 baseline.
    """
    binary = Path(binary)
    src = binary.with_suffix(".c")
    src.write_text(WORKLOAD_C)
    cmd = [
        cc,
        "-m32",
        "-O2",
        "-nostdlib",
        "-static",
        "-ffreestanding",
        "-fno-pie",
        "-no-pie",
        "-fno-stack-protector",
        *(["-msse2", "-mfpmath=sse"] if sse2 else []),
        f"-DITERS={iters}",
        "-o",
        str(binary),
        str(src),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError:
        raise
    if proc.returncode != 0:
        raise RuntimeError(f"compile failed ({cc}): {proc.stderr.strip()[:400]}")
    if _is_dynamically_linked(binary):
        raise RuntimeError(f"workload is dynamically linked — guest cannot run it: {binary}")
    return binary


def run_native(binary: Path) -> float:
    """Host wall-clock seconds for one native run (fork/exec is ~1 ms)."""
    t0 = time.perf_counter()
    subprocess.run([str(binary)], check=True, stdout=subprocess.DEVNULL)
    return time.perf_counter() - t0


def guest_driver_argv(
    node: str,
    driver: str,
    wasm: str,
    kernel: str,
    bios: str,
    vga_bios: str,
    binary: str,
    timeout_s: int,
) -> list[str]:
    """argv for the serial driver — a flag rename on either side fails a test."""
    return [
        node,
        str(driver),
        "--wasm",
        str(wasm),
        "--kernel",
        str(kernel),
        "--bios",
        str(bios),
        "--vga-bios",
        str(vga_bios),
        "--binary",
        str(binary),
        "--timeout",
        str(int(timeout_s)),
    ]


def ensure_assets(assets: Path) -> dict[str, Path]:
    """Fetch the boot trio (kernel + BIOS pair) if absent — idempotent."""
    assets = Path(assets)
    assets.mkdir(parents=True, exist_ok=True)
    for name, url in ASSET_URLS.items():
        dest = assets / name
        if not dest.exists() or dest.stat().st_size == 0:
            with urllib.request.urlopen(url, timeout=120) as resp, open(dest, "wb") as f:
                f.write(resp.read())
    return {name: assets / name for name in ASSET_URLS}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--report", default="/tmp/slobench/v86_factor_report.json")
    ap.add_argument("--baseline", type=float, default=DEFAULT_BASELINE_TPS)
    ap.add_argument("--baseline-source", default=DEFAULT_BASELINE_SOURCE)
    ap.add_argument("--slo", type=float, default=DEFAULT_SLO_TPS)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--assets", default=str(DEFAULT_ASSETS))
    ap.add_argument("--wasm", default=str(DEFAULT_WASM))
    ap.add_argument("--node", default="node")
    ap.add_argument("--timeout", type=int, default=480)
    ap.add_argument("--cc", default="gcc")
    ap.add_argument(
        "--sse2",
        action="store_true",
        help="compile the workload with SSE2 (guest-supported) instead of x87",
    )
    ap.add_argument("--native-only", action="store_true", help="stop after native timings")
    args = ap.parse_args(argv)

    with tempfile.TemporaryDirectory(prefix="v86factor-") as td:
        binary = compile_workload(Path(td) / "bm", cc=args.cc, iters=args.iters, sse2=args.sse2)
        native = [run_native(binary) for _ in range(max(1, args.repeats))]
        print(f"native: median {statistics.median(native):.3f}s over {len(native)} runs")
        if args.native_only:
            return 0

        assets = ensure_assets(Path(args.assets))
        argv_drv = guest_driver_argv(
            node=args.node,
            driver=str(DEFAULT_DRIVER),
            wasm=args.wasm,
            kernel=str(assets["bzimage68.bin"]),
            bios=str(assets["seabios.bin"]),
            vga_bios=str(assets["vgabios.bin"]),
            binary=str(binary),
            timeout_s=args.timeout,
        )
        proc = subprocess.run(argv_drv, capture_output=True, text=True, timeout=args.timeout + 60)
        lines = [ln for ln in proc.stdout.strip().splitlines() if ln.strip()]
        if not lines:
            raise SystemExit(
                f"driver produced no stdout (exit {proc.returncode}): {proc.stderr.strip()[-600:]}"
            )
        guest = parse_guest_output(lines[-1])

        report = build_report(
            baseline_decode_tps=args.baseline,
            baseline_source=args.baseline_source,
            native_s=native,
            guest_s=guest["guest_s"],
            method=guest["method"],
            slo_tps=args.slo,
        )
        report["boot_s"] = guest.get("boot_s")
        report["guest_md5_ok"] = guest.get("md5_ok")
        report["fp_mode"] = "sse2" if args.sse2 else "x87"
        report["driver_stderr_tail"] = proc.stderr.strip()[-400:]

        out = Path(args.report)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2))

    v = report["verdict"]
    print(
        f"factor x{report['factor']:.1f} (guest {report['guest_s']:.1f}s vs "
        f"native {report['native_s_median']:.3f}s) -> "
        f"v86 decode ~{report['v86_decode_tps']:.1f} tok/s vs {args.slo} floor: "
        f"{'MEETS' if v['meets_slo'] else 'MISS'} (margin x{v['margin_x']:.1f})"
    )
    print(f"report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
