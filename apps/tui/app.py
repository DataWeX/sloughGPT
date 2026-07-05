"""SloughGPT TUI - OpenCode-inspired clean terminal UI."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

from rich.console import Console
from rich.panel import Panel

from apps.tui.components import (
    CONSOLE,
    Color,
    StatusTable,
    ChoiceMenu,
    HealthIndicator,
    LiveProgress,
    header,
    section,
    status_table,
    choice_menu,
    health_indicator,
    info,
    divider,
    live_progress,
)

if TYPE_CHECKING:
    from apps.tui.adapters.http_api import ApiJsonResult
    from apps.tui.adapters.local_status import LocalStatusSnapshot


def _print_local_status(snap: "LocalStatusSnapshot") -> None:
    header("Local Repository", f"Path: {snap.repo_root}")
    CONSOLE.print()

    tbl = StatusTable("Models")
    tbl.add("Total files", str(snap.model_file_count))
    tbl.add("First 5", "")
    for p in snap.model_sample_paths:
        tbl.add("", p)
    tbl.render()
    CONSOLE.print()

    tbl2 = StatusTable("Datasets")
    tbl2.add("Total entries", str(snap.dataset_entry_count))
    tbl2.add("First 5", "")
    for name in snap.dataset_sample_names:
        tbl2.add("", name)
    tbl2.render()


def _print_api_json(label: str, r: "ApiJsonResult") -> None:
    header("API Response", label)

    status_map = {
        200: "healthy",
        400: "degraded",
        500: "error",
    }
    status_text = status_map.get(r.status_code, "unknown")
    CONSOLE.print(f"  [{getattr(Color, status_text.upper(), Color.MUTED)}]●[/{getattr(Color, status_text.upper(), Color.MUTED)}] status: {status_text}")
    CONSOLE.print()

    if r.error:
        CONSOLE.print(f"  error: {r.error}", style=Color.ERROR)
    elif r.payload:
        text = json.dumps(r.payload, indent=2)
        CONSOLE.print(Panel(text[:2000], title="Response", border_style=Color.BORDER))
    else:
        CONSOLE.print("  (empty response)")


def main(argv: Optional[List[str]] = None) -> None:
    args = argv if argv is not None else sys.argv[1:]
    parser = argparse.ArgumentParser(
        prog="sloughgpt-tui",
        description="SloughGPT terminal UI",
    )
    parser.add_argument("--host", default="127.0.0.1", help="API host")
    parser.add_argument("--port", type=int, default=8000, help="API port")
    parser.add_argument("--repo-root", type=Path, default=None, metavar="PATH")
    parser.add_argument("--local-status", action="store_true", help="Scan models/datasets")
    parser.add_argument("--api-health", action="store_true", help="GET /health")
    parser.add_argument("--api-metrics", action="store_true", help="GET /metrics")
    parser.add_argument("--api-health-detailed", action="store_true", help="GET /health/detailed")
    parser.add_argument("--train", action="store_true", help="Local training")
    parser.add_argument("--train-api", action="store_true", help="API training")
    parser.add_argument("--dataset", default="shakespeare")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--docker-start", action="store_true", help="Start Docker")
    parser.add_argument("--docker-stop", action="store_true", help="Stop Docker")
    parser.add_argument("--docker-status", action="store_true", help="Docker status")
    parser.add_argument("--docker-logs", action="store_true", help="Docker logs")
    parser.add_argument("--docker-dev", action="store_true")
    parser.add_argument("--docker-gpu", action="store_true")
    parser.add_argument("--compose-file", type=Path, default=None, metavar="PATH")
    parser.add_argument("--interactive", action="store_true", help="Interactive menu")
    parser.add_argument("--version", action="version", version="%(prog)s 0.1.0")
    ns = parser.parse_args(args)

    base = f"http://{ns.host}:{ns.port}"

    if ns.local_status:
        from apps.tui.adapters.local_status import scan_local_repo
        from apps.tui.session import discover_repo_root

        root = ns.repo_root.resolve() if ns.repo_root else (discover_repo_root() or Path.cwd())
        snap = scan_local_repo(root)
        _print_local_status(snap)

    elif ns.api_health:
        from apps.tui.adapters.http_api import fetch_health
        _print_api_json("/health", fetch_health(base))

    elif ns.api_metrics:
        from apps.tui.adapters.http_api import fetch_metrics
        _print_api_json("/metrics", fetch_metrics(base))

    elif ns.api_health_detailed:
        from apps.tui.adapters.http_api import fetch_health_detailed
        _print_api_json("/health/detailed", fetch_health_detailed(base))

    elif ns.train:
        from apps.tui.adapters.training import LocalTrainAdapter, TrainConfig
        from apps.tui.session import discover_repo_root

        header("Training", f"dataset={ns.dataset}")
        info(f"epochs={ns.epochs} batch={ns.batch_size} lr={ns.lr}")
        CONSOLE.print()

        root = ns.repo_root.resolve() if ns.repo_root else (discover_repo_root() or Path.cwd())
        config = TrainConfig(
            dataset=ns.dataset,
            epochs=ns.epochs,
            batch_size=ns.batch_size,
            max_steps=ns.max_steps,
            learning_rate=ns.lr,
        )
        adapter = LocalTrainAdapter(config, root)

        pbar = LiveProgress()
        with pbar:
            task = pbar.add_task("Training", total=ns.max_steps or (ns.epochs * 100))
            try:
                for prog in adapter.train():
                    pbar.progress.update(task, advance=1, description=f"step {prog.step} loss={prog.loss:.4f}")
            except KeyboardInterrupt:
                CONSOLE.print("\n  interrupted", style=Color.WARNING)

        if adapter._result and adapter._result.success:
            info(f"saved: {adapter._result.save_path}")
        else:
            info(f"failed: {adapter._result.error if adapter._result else 'unknown'}")

    elif ns.train_api:
        from apps.tui.adapters.training import HttpTrainAdapter, TrainConfig

        header("API Training", base)
        config = TrainConfig(
            dataset=ns.dataset,
            epochs=ns.epochs,
            batch_size=ns.batch_size,
            max_steps=ns.max_steps,
            learning_rate=ns.lr,
        )
        adapter = HttpTrainAdapter(base)
        result = adapter.start_training(config)

        if result.status == "started":
            info(f"job: {result.job_id}")
            CONSOLE.print("  polling...", style=Color.MUTED)
            for _ in range(30):
                status = adapter.get_job_status(result.job_id)
                if status:
                    s = status.get("status", "?")
                    CONSOLE.print(f"\r    {s} {status.get('progress', 0)}%", end="")
                    if s in ("completed", "failed"):
                        break
                time.sleep(2)
            CONSOLE.print()
        else:
            info(f"failed: {result.message}")

    elif ns.docker_start:
        from apps.tui.adapters.docker import DockerAdapter
        adapter = DockerAdapter(compose_file=ns.compose_file)
        if not adapter.is_available():
            info("docker not available")
        elif adapter.start(dev=ns.docker_dev, gpu=ns.docker_gpu):
            info("services started")

    elif ns.docker_stop:
        from apps.tui.adapters.docker import DockerAdapter
        adapter = DockerAdapter(compose_file=ns.compose_file)
        if not adapter.is_available():
            info("docker not available")
        elif adapter.stop():
            info("services stopped")

    elif ns.docker_status:
        from apps.tui.adapters.docker import DockerAdapter
        adapter = DockerAdapter(compose_file=ns.compose_file)
        if not adapter.is_available():
            info("docker not available")
        else:
            status = adapter.status()
            if status:
                CONSOLE.print(status)

    elif ns.docker_logs:
        from apps.tui.adapters.docker import DockerAdapter
        adapter = DockerAdapter(compose_file=ns.compose_file)
        if adapter.is_available():
            adapter.logs()

    elif ns.interactive:
        from apps.tui.interactive import main as interactive_main
        interactive_main(["--interactive", "--host", ns.host, "--port", str(ns.port)])
        return

    else:
        header("SloughGPT TUI", "Terminal UI")
        divider()

        menu = ChoiceMenu("Quick Start")
        menu.add("--local-status", "Scan repository models/datasets")
        menu.add("--api-health", "Check API health")
        menu.add("--train --epochs 1", "Quick training test")
        menu.add("--interactive", "Interactive menu")
        menu.render()

        CONSOLE.print()
        CONSOLE.print("  run with --help for all options", style=Color.MUTED)


if __name__ == "__main__":
    main()
