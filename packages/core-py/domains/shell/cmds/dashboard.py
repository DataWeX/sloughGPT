"""dashboard — show live server dashboard inline."""

from __future__ import annotations

import sys
import time
from datetime import datetime, timezone

from ..console import Console
from ..commands import ShellCommands, _api_get

help = "Show live server dashboard (health, processes, events)"

_CLEAR = "\033[2J\033[H"


def _format_uptime(seconds: int) -> str:
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        m, s = divmod(seconds, 60)
        return f"{m}m {s:02d}s"
    h, m = seconds // 3600, (seconds % 3600) // 60
    return f"{h}h {m:02d}m"


def _format_ts(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%H:%M:%S")


def _sparkline(values: list[float], width: int = 12) -> str:
    if not values:
        return ""
    blocks = " \u2581\u2582\u2583\u2584\u2585\u2586\u2587\u2588"
    mn, mx = min(values), max(values)
    rng = mx - mn if mx != mn else 1
    if len(values) > width:
        step = len(values) / width
        sampled = [values[int(i * step)] for i in range(width)]
    else:
        sampled = values
    return "".join(blocks[min(int((v - mn) / rng * 8) + 1, 8)] for v in sampled)


def _status_icon(status: str) -> str:
    icons = {
        "running": "\033[32m\u25b6\033[0m",
        "queued": "\033[33m\u25c6\033[0m",
        "starting": "\033[36m\u21bb\033[0m",
        "complete": "\033[32m\u2713\033[0m",
        "completed": "\033[32m\u2713\033[0m",
        "error": "\033[31m\u2717\033[0m",
        "exited": "\033[90m\u25a0\033[0m",
        "idle": "\033[90m\u00b7\033[0m",
        "stopped": "\033[90m\u25a0\033[0m",
    }
    return icons.get(status, "?")


def _progress_bar(progress: float, width: int = 16) -> str:
    filled = int(width * progress / 100)
    return f"\033[32m{'█' * filled}\033[90m{'░' * (width - filled)}\033[0m"


def _fetch_snapshot() -> dict | None:
    """Fetch dashboard data from the API. Returns None on error."""
    try:
        health = _api_get("/health/detailed")
        events_resp = _api_get("/dashboard/events")
    except Exception:
        return None

    health_data = health.get("data", health) if isinstance(health, dict) else {}
    events_data = events_resp.get("data", {}) if isinstance(events_resp, dict) else {}
    return {
        "health": health_data,
        "events": events_data.get("events", []),
        "errors": health_data.get("recent_errors", []),
    }


def _render(data: dict, compact: bool = False, io=None) -> None:
    """Render the dashboard to the given IO object."""
    health = data["health"]
    events = data["events"]
    errors = data["errors"]

    loaded = health.get("model_loaded", False)
    model = health.get("model_type", "")
    uptime = health.get("uptime_seconds", 0)
    status_str = "\033[32monline\033[0m" if loaded else "\033[33mno model\033[0m"
    model_str = model if model else "\033[2mno model\033[0m"

    def _w(line: str):
        if io:
            io.write(line)
        else:
            sys.stdout.write(line + "\n")

    _w(f"\n  \033[1mSloughGPT\033[0m  {_format_ts(time.time())}")
    _w(f"  {'─' * 56}")
    _w(f"  \033[1mSERVER\033[0m {status_str}  \033[36m{model_str}\033[0m  up {_format_uptime(uptime)}")

    # MODEL line — device, params, quantization
    if loaded:
        model_parts = []
        device = health.get("device", "")
        if device:
            model_parts.append(device)
        gpu = health.get("gpu", {})
        if gpu and gpu.get("backend") not in ("unknown", None):
            vram = gpu.get("vram_gb", 0)
            if vram:
                model_parts.append(f"{vram}GB")
        quant = health.get("quantization", {})
        if quant:
            bits = quant.get("bits", 0)
            if bits:
                model_parts.append(f"{bits}bit")
            mode = quant.get("mode", "")
            if mode:
                model_parts.append(mode)
        if model_parts:
            _w(f"  \033[1mMODEL\033[0m   {'  '.join(model_parts)}")

    cpu = health.get("cpu_percent", 0)
    mem = health.get("memory_percent", 0)
    mem_mb = health.get("memory_used_mb", 0)
    reqs = health.get("request_count", 0)
    errs = health.get("error_count", 0)
    rpm_history = health.get("rpm_history", [])
    mem_history = health.get("memory_history", [])

    sys_line = f"  \033[1mSYS\033[0m   cpu {cpu:.0f}%  mem {mem:.0f}% ({mem_mb}MB)  reqs {reqs}  err {errs}"
    if rpm_history:
        sys_line += f"  \033[90m{_sparkline(rpm_history)}\033[0m"
    if mem_history:
        sys_line += f"  \033[90m{_sparkline(mem_history)}\033[0m"
    _w(sys_line)

    tps = health.get("tokens_per_sec", 0)
    lat = health.get("avg_latency_ms", 0)
    gen_parts = []
    if tps > 0:
        gen_parts.append(f"{tps:.1f} tok/s")
    if lat > 0:
        gen_parts.append(f"{lat:.0f}ms avg")
    if gen_parts:
        _w(f"  \033[1mGEN\033[0m   {'  '.join(gen_parts)}")

    # Health score
    health_score = health.get("health_score", {})
    if health_score:
        score = health_score.get("score", 0)
        status = health_score.get("status", "")
        if score > 0:
            score_color = "\033[32m" if score >= 80 else "\033[33m" if score >= 50 else "\033[31m"
            _w(f"  \033[1mHEALTH\033[0m  {score_color}{score}/100\033[0m  \033[2m{status}\033[0m")

    if not compact:
        procs = health.get("active_processes", {})
        _w("")
        _w("  \033[1mPROCESSES\033[0m")
        _w(f"  {'─' * 56}")
        if not procs:
            _w("  \033[2m  (none active)\033[0m")
        else:
            for proc_id, proc in procs.items():
                status = proc.get("status", "unknown")
                label = proc.get("label", proc_id)
                detail = proc.get("detail", "")
                progress = proc.get("progress", 0)
                icon = _status_icon(status)
                name = f"\033[1m{label:<14}\033[0m"
                if progress > 0 and status == "running":
                    bar = _progress_bar(progress)
                    pct = f"{progress:.0f}%".rjust(4)
                    detail_str = f"{detail}  {bar} {pct}" if detail else f"{bar} {pct}"
                else:
                    detail_str = detail
                _w(f"  {icon} {name} \033[2m{detail_str}\033[0m")

        _w("")
        _w("  \033[1mEVENTS\033[0m")
        _w(f"  {'─' * 56}")
        if not events:
            _w("  \033[2m  (no events yet)\033[0m")
        else:
            cat_colors = {
                "TRAIN": "\033[32m", "MODEL": "\033[36m", "INFERENCE": "\033[35m",
                "SYSTEM": "\033[90m", "ERROR": "\033[31m", "INFRA": "\033[90m",
                "CHAT": "\033[36m", "SOUL": "\033[33m", "START": "\033[90m",
                "IDLE": "\033[90m", "DOWNLOAD": "\033[36m", "SLOW": "\033[33m",
                "WORKFLOW": "\033[90m",
            }
            for ev in events[:8]:
                ts_str = f"\033[90m{_format_ts(ev.get('ts', 0))}\033[0m"
                cat = ev.get("category", "")
                msg = ev.get("message", "")
                color = cat_colors.get(cat, "")
                cat_str = f"{color}{cat.ljust(10)}\033[0m"
                max_msg = 40
                if len(msg) > max_msg:
                    msg = msg[:max_msg - 1] + "\u2026"
                _w(f"  {ts_str} {cat_str} {msg}")

        if errors:
            _w("")
            _w("  \033[1;31mRECENT ERRORS\033[0m")
            _w(f"  {'─' * 56}")
            for err in errors[:3]:
                path = err.get("path", "")
                msg = err.get("message", "")[:38]
                _w(f"  \033[2m{path:<18}\033[0m \033[31m{msg}\033[0m")

    _w("")
    _w("  \033[2mCtrl+C to exit\033[0m")
    if io:
        io.flush()
    else:
        sys.stdout.flush()


def run(argv: list[str], out: Console, api: ShellCommands,
        env: dict[str, str]) -> int:
    compact = "--compact" in argv or "-c" in argv
    watch = "--watch" in argv or "-w" in argv
    interval = 2.0

    # Parse --interval
    for arg in argv:
        if arg.startswith("--interval="):
            try:
                interval = float(arg.split("=", 1)[1])
            except ValueError:
                pass
        elif arg == "--interval":
            idx = argv.index(arg)
            if idx + 1 < len(argv):
                try:
                    interval = float(argv[idx + 1])
                except ValueError:
                    pass

    if watch:
        # Auto-refresh mode
        sys.stdout.write("\033[?25l")  # hide cursor
        try:
            while True:
                data = _fetch_snapshot()
                if data is None:
                    sys.stdout.write(_CLEAR)
                    print("\n  \033[31mCannot connect to server\033[0m")
                    print("  \033[2mRetrying in 3s...\033[0m")
                    sys.stdout.flush()
                    time.sleep(3)
                    continue
                sys.stdout.write(_CLEAR)
                _render(data, compact=compact)
                time.sleep(interval)
        except KeyboardInterrupt:
            sys.stdout.write("\033[?25h\n")
            sys.stdout.flush()
            return 0
    else:
        # Single shot
        data = _fetch_snapshot()
        if data is None:
            out.status("error", "Dashboard fetch failed: Cannot connect to server")
            out.note("Use 'api start' to launch the API server.")
            return 1
        _render(data, compact=compact, io=out._io)
        return 0
