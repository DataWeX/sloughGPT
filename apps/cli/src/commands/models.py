"""
Model commands - Model listing, export, and soul management.
"""
import sys
import os
import json
from pathlib import Path
from typing import Optional

from core.printer import printer
from utils.formatting import format_size, format_number, truncate


def cmd_models(args):
    """List available models. Use subcommands for info, download, compare, personalities."""
    from utils.helpers import local_soul_candidate_paths

    models_dir = Path("models")

    printer.header("Available Models")

    # Slo files
    printer.section("Soul Files (.soul)")
    soul_files = local_soul_candidate_paths(models_dir)
    if soul_files:
        rows = []
        for f in soul_files:
            size = f.stat().st_size
            rows.append([f.name, format_size(size)])
        printer.table(["Name", "Size"], rows)
    else:
        printer.info("No soul files found")

    # Compiled models (.slnc mmap format)
    printer.section("Compiled Models (.slnc)")
    slnc_files = sorted(models_dir.rglob("*.slnc")) if models_dir.is_dir() else []
    if slnc_files:
        rows = []
        for f in slnc_files:
            size = f.stat().st_size
            rows.append([f.name, format_size(size)])
        printer.table(["Name", "Size"], rows)
    else:
        printer.info("No .slnc files found")

    # SafeTensors
    printer.section("SafeTensors (.safetensors)")
    st_files = list(models_dir.glob("*.safetensors"))
    if st_files:
        rows = []
        for f in sorted(st_files):
            size = f.stat().st_size
            rows.append([f.name, format_size(size)])
        printer.table(["Name", "Size"], rows)
    else:
        printer.info("No .safetensors files found")

    printer.blank()
    printer.section("Available Architectures")
    architectures = [
        ("gpt2", "GPT-2", "124M params"),
        ("gpt2-medium", "GPT-2 Medium", "355M params"),
        ("gpt2-large", "GPT-2 Large", "774M params"),
        ("llama", "LLaMA", "Meta model"),
        ("phi", "Phi", "Microsoft model"),
    ]
    printer.table(["ID", "Name", "Info"], architectures)


def _cmd_models_info(args):
    """Show .soul checkpoint info."""
    import numpy as np
    from domains.training.slonet import import_from_sou

    model_path = Path(args.model)
    if not model_path.exists():
        printer.error(f"Model not found: {model_path}")
        return

    printer.header(f"Model: {model_path}")

    try:
        net = import_from_sou(str(model_path))
    except Exception as e:
        printer.error(f"Failed to load: {e}")
        return

    printer.key_value("Soul Name", getattr(net, "soul_name", "?"))
    if getattr(net, "soul_traits", None):
        printer.key_value("Traits", str(net.soul_traits))
    params = list(net.parameters())
    total_params = sum(int(np.prod(p.shape)) for p in params)
    printer.key_value("Parameters", f"{total_params:,}")

    meta = getattr(net, "metadata", None) or {}
    for k in ("vocab_size", "n_embed", "n_layer", "n_head", "block_size"):
        if meta.get(k) is not None:
            printer.key_value(k.replace("n_", "Num "), str(meta[k]))
    if meta.get("tokenizer"):
        printer.key_value("Tokenizer", str(meta["tokenizer"].get("type", "?")))
    training = meta.get("training") or {}
    for k, v in training.items():
        printer.key_value(str(k), str(v))


def _interactive_download_select():
    """Show an interactive fuzzy-searchable list of popular HuggingFace models.

    Queries the HuggingFace Hub API for trending text-generation models,
    displays them in a curses-based fuzzy selector, and returns the selected
    model ID. Returns None if the user cancels.

    Returns:
        Selected model ID string, or None if cancelled.
    """
    import curses
    import requests

    printer.step("Fetching popular models from HuggingFace Hub...")

    try:
        resp = requests.get(
            "https://huggingface.co/api/models",
            params={
                "pipeline_tag": "text-generation",
                "sort": "downloads",
                "direction": "-1",
                "limit": 50,
            },
            timeout=15,
        )
        if resp.status_code != 200:
            printer.error(f"Failed to fetch models: HTTP {resp.status_code}")
            return None
        raw_models = resp.json()
    except Exception as e:
        printer.error(f"Failed to fetch models: {e}")
        return None

    # Build list: (display_name, model_id, downloads)
    model_list = []
    for m in raw_models:
        mid = m.get("modelId") or m.get("id", "")
        if not mid:
            continue
        downloads = m.get("downloads", 0)
        tags = m.get("tags", [])
        # Short display: model name + download count
        dl_str = f"{downloads:,}" if downloads else "?"
        model_list.append((f"{mid}  ({dl_str} downloads)", mid, downloads))

    if not model_list:
        printer.info("No models found")
        return None

    # Sort by downloads descending
    model_list.sort(key=lambda x: x[2], reverse=True)

    # ── curses fuzzy selector ──
    def _run_selector(stdscr):
        curses.curs_set(0)
        curses.use_default_colors()
        stdscr.nodelay(False)

        height, width = stdscr.getmaxyx()
        query = ""
        selected = 0
        scroll_offset = 0

        while True:
            stdscr.clear()
            stdscr.addstr(0, 0, " Download Model from HuggingFace ", curses.A_REVERSE)
            stdscr.addstr(2, 0, f" Search: {query}_")
            stdscr.addstr(3, 0, "─" * min(width - 1, 60))

            # Fuzzy filter
            filtered = [
                (display, mid, dl)
                for display, mid, dl in model_list
                if query.lower() in mid.lower() or query.lower() in display.lower()
            ]

            if not filtered:
                stdscr.addstr(5, 0, " No matching models — type to search")
                key = stdscr.getch()
                if key == 27:
                    return None
                elif key == 263 or key == 127:
                    query = query[:-1]
                elif 32 <= key <= 126:
                    query += chr(key)
                continue

            selected = min(selected, len(filtered) - 1)
            if selected < scroll_offset:
                scroll_offset = selected
            if selected >= scroll_offset + height - 6:
                scroll_offset = selected - height + 7

            max_display = height - 6
            for i, (display, mid, dl) in enumerate(filtered[scroll_offset:scroll_offset + max_display]):
                line_y = 5 + i
                prefix = "▸ " if i + scroll_offset == selected else "  "
                label = f"{prefix}{display}"
                if len(label) > width - 1:
                    label = label[:width - 4] + "..."
                if i + scroll_offset == selected:
                    stdscr.addstr(line_y, 0, label, curses.A_REVERSE)
                else:
                    stdscr.addstr(line_y, 0, label)

            footer = f" {len(filtered)} models  ↑↓ navigate  Enter select  ESC cancel  / search"
            stdscr.addstr(height - 1, 0, footer[:width - 1])

            key = stdscr.getch()
            if key == 27:
                return None
            elif key in (10, 13):
                return filtered[selected]
            elif key == 259:
                selected = max(0, selected - 1)
            elif key == 258:
                selected = min(len(filtered) - 1, selected + 1)
            elif key == 338 or key == 261:
                selected = min(len(filtered) - 1, selected + 10)
            elif key == 339 or key == 260:
                selected = max(0, selected - 10)
            elif key == 263 or key == 127:
                query = query[:-1]
            elif 32 <= key <= 126:
                query += chr(key)

    try:
        result = curses.wrapper(_run_selector)
    except Exception as e:
        printer.error(f"Selector error: {e}")
        return None

    if result is None:
        printer.info("Selection cancelled")
        return None

    name, model_id, _ = result
    printer.success(f"Selected: {model_id}")
    return model_id


def _cmd_models_download(args):
    """Download a HuggingFace model with size confirmation and live progress.

    Enforces the bandwidth policy: queries HuggingFace Hub for model size,
    shows a Rich panel with the estimate, and requires user confirmation
    for downloads over 50 MB. Use ``--yes`` or ``SLO_AUTO_DOWNLOAD=1``
    to skip the prompt.

    Shows a live ``rich.progress`` bar with percentage, speed, ETA, and
    current filename. Ctrl+C cancels gracefully.

    If no model_id is provided, shows an interactive fuzzy-searchable list
    of popular text-generation models from HuggingFace Hub.

    Side effects:
        - May download model files to HF cache directory
        - Prints to stdout via Rich panels and progress bars
    """
    from core.permissions import PermissionsManager
    from rich.console import Console
    from rich.progress import (
        Progress,
        SpinnerColumn,
        BarColumn,
        TextColumn,
        TimeRemainingColumn,
        TransferSpeedColumn,
        MofNCompleteColumn,
    )

    # ── Interactive model selection if no model_id ──────────
    if not getattr(args, "model_id", None):
        model_id = _interactive_download_select()
        if not model_id:
            return
        args.model_id = model_id

    printer.header("Download Model")
    printer.key_value("Model ID", args.model_id)
    printer.blank()

    console = Console(highlight=False)

    try:
        from domains.infrastructure.download_manager import get_download_manager
        import asyncio

        mgr = get_download_manager()

        if mgr.is_cached(args.model_id):
            printer.success(f"Model already cached: {args.model_id}")
            return

        # ── Confirmation gate ──────────────────────────────
        pm = PermissionsManager(auto_yes=getattr(args, "yes", False))
        if not pm.confirm_download(args.model_id):
            printer.info("Download cancelled by user")
            return

        # ── Live progress bar ──────────────────────────────
        cancel_requested = False
        progress_task_id = None
        current_file = ""

        progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=30),
            MofNCompleteColumn(),
            TransferSpeedColumn(),
            TimeRemainingColumn(),
            console=console,
        )

        def _render_progress(progress_dict):
            nonlocal current_file
            pct = progress_dict.get("percentage", 0)
            downloaded = progress_dict.get("bytes_downloaded", 0)
            total = progress_dict.get("total_bytes", 0)
            fname = progress_dict.get("current_file", "")
            if fname:
                current_file = fname.split("/")[-1][:40]

            if progress_task_id is not None and total > 0:
                progress.update(
                    progress_task_id,
                    completed=downloaded,
                    total=total,
                    description=current_file or "Downloading",
                )

        async def _do_download():
            nonlocal progress_task_id
            mgr.on_progress(args.model_id, _render_progress)

            with progress:
                progress_task_id = progress.add_task(
                    "Downloading", total=None, completed=0
                )
                result = await mgr.download(args.model_id)
                return result

        result = asyncio.run(_do_download())

        if result.get("status") == "complete":
            printer.success(
                f"Downloaded in {result.get('elapsed_seconds', '?')}s "
                f"→ {result.get('cache_dir', '')}"
            )
        elif result.get("status") == "failed":
            printer.error(f"Download failed: {result.get('error', 'unknown error')}")
        elif result.get("status") == "cancelled":
            printer.warning("Download cancelled")
    except KeyboardInterrupt:
        printer.warning("Download interrupted by user")
        try:
            mgr = get_download_manager()
            mgr.cancel(args.model_id)
        except Exception:
            pass
    except Exception as e:
        printer.error(f"Download failed: {e}")


def _cmd_models_status(args):
    """Show status of all cached/downloaded HuggingFace models.

    Scans the HuggingFace cache directory for downloaded models, shows
    their sizes, and indicates whether each has been converted to .slnc
    format. Useful for managing disk space and verifying downloads.
    """
    from rich.console import Console
    from rich.table import Table

    console = Console(highlight=False)

    hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
    if not hf_cache.exists():
        printer.info("No HuggingFace cache found")
        printer.key_value("Cache path", str(hf_cache))
        return

    # Scan for model directories
    models = []
    for entry in sorted(hf_cache.iterdir()):
        if not entry.name.startswith("models--") or not entry.is_dir():
            continue
        model_id = entry.name[len("models--"):].replace("--", "/")

        # Calculate total size of weight files
        total_bytes = 0
        file_count = 0
        has_slnc = False
        has_safetensors = False

        for f in entry.rglob("*"):
            if not f.is_file():
                continue
            try:
                size = f.stat().st_size
            except OSError:
                continue
            if size < 1024:
                continue
            if f.suffix in (".safetensors", ".bin", ".slnc", ".onnx"):
                total_bytes += size
                file_count += 1
                if f.suffix == ".slnc":
                    has_slnc = True
                if f.suffix == ".safetensors":
                    has_safetensors = True

        if total_bytes == 0:
            continue

        # Determine format status
        if has_slnc:
            status = "[green].slnc[/]"
        elif has_safetensors:
            status = "[yellow]safetensors[/]"
        else:
            status = "[dim]other[/]"

        models.append({
            "id": model_id,
            "size": total_bytes,
            "files": file_count,
            "status": status,
        })

    if not models:
        printer.info("No cached models found")
        printer.key_value("Cache path", str(hf_cache))
        return

    # Sort by size descending
    models.sort(key=lambda m: m["size"], reverse=True)

    total_cache = sum(m["size"] for m in models)

    printer.header(f"Cached Models ({len(models)} models, {format_size(total_cache)} total)")

    table = Table(show_header=True, header_style="bold")
    table.add_column("Model", style="cyan")
    table.add_column("Size", justify="right")
    table.add_column("Files", justify="right")
    table.add_column("Format")

    for m in models:
        table.add_row(
            m["id"],
            format_size(m["size"]),
            str(m["files"]),
            m["status"],
        )

    console.print(table)
    console.print()
    console.print(f"  [dim]Cache: {hf_cache}[/]")


def _cmd_models_compare(args):
    """Compare benchmark results or models."""
    printer.header("Model Comparison")

    # Compare benchmark results
    benchmarks_dir = Path("data/experiments/benchmarks")
    if benchmarks_dir.exists():
        benchmarks = list(benchmarks_dir.glob("*.json"))
        if benchmarks:
            printer.section("Benchmark Results")
            rows = []
            for bf in sorted(benchmarks)[:5]:
                with open(bf) as f:
                    data = json.load(f)
                rows.append([
                    data.get("model", "unknown")[:18],
                    f'{data.get("tokens_per_second", 0):.2f}',
                    f'{data.get("latency_ms", 0):.1f}',
                    f'{data.get("memory_mb", 0):.1f}',
                ])
            printer.table(["Model", "Tokens/s", "Latency (ms)", "Memory (MB)"], rows, align=["l", "r", "r", "r"])

    # Compare models
    printer.section("Model Specifications")
    model_specs = [
        ("gpt2", "124M", "~250MB", "Fast"),
        ("gpt2-medium", "355M", "~700MB", "Medium"),
        ("gpt2-large", "774M", "~1.5GB", "Slow"),
        ("phi-2", "2.7B", "~5.4GB", "Medium"),
        ("mistral-7b", "7.3B", "~14GB", "Slow"),
        ("llama-2-7b", "7B", "~13GB", "Slow"),
    ]
    printer.table(["Model", "Params", "Size", "Speed"], model_specs)

    printer.blank()
    printer.info("Run benchmarks: cli.py eval --checkpoint <path> --benchmark")


def _cmd_models_personalities(args):
    """List available personalities."""
    try:
        from domains.ai_personality import PERSONALITIES
    except ImportError:
        printer.error("Personalities module not found")
        return

    printer.header("Available Personalities")
    rows = []
    for ptype, personality in PERSONALITIES.items():
        rows.append([ptype.value.upper(), personality.name, personality.description[:50], ", ".join(personality.traits)])
    printer.table(["Type", "Name", "Description", "Traits"], rows)


def cmd_export_cli(args):
    """Export a .soul model to different formats."""
    import numpy as np
    from domains.training.export import export_model, list_export_formats, ExportConfig

    printer.header("Model Export")

    # List formats
    printer.section("Supported Formats")
    formats = list_export_formats()
    for fmt, desc in formats.items():
        printer.key_value(fmt, desc)

    model_path = Path(args.model)
    if not model_path.exists():
        printer.error(f"Model not found: {args.model}")
        return

    printer.blank()
    printer.step(f"Loading: {args.model}")
    from domains.training.slonet import import_from_sou
    net = import_from_sou(str(model_path))
    metadata = dict(getattr(net, "metadata", None) or {})
    metadata.setdefault("name", getattr(net, "soul_name", "SloughGPT"))

    total_params = sum(int(np.prod(p.shape)) for p in net.parameters())
    printer.success(f"Loaded: {format_number(total_params)} parameters")

    output_path = args.output or str(model_path.with_suffix(""))

    cli_metadata = {}
    if args.metadata:
        for item in args.metadata:
            if "=" in item:
                key, value = item.split("=", 1)
                try:
                    if value.replace(".", "", 1).isdigit():
                        value = float(value) if "." in value else int(value)
                    elif value.lower() in ("true", "false"):
                        value = value.lower() == "true"
                except ValueError:
                    pass
                cli_metadata[key] = value

    meta_with_name = {**metadata, **cli_metadata}
    if args.soul_name:
        meta_with_name["name"] = args.soul_name

    from domains.training.export import ExportConfig

    config = ExportConfig(
        input_path=args.model,
        output_path=output_path,
        format=args.format,
        quantization=args.quantization,
        metadata=meta_with_name,
        seq_len=args.seq_len,
        opset_version=args.opset,
        n_ctx=args.n_ctx if hasattr(args, "n_ctx") else 2048,
    )

    printer.blank()
    printer.section("Export Configuration")
    printer.key_value("Format", args.format)
    printer.key_value("Quantization", args.quantization or "N/A")
    printer.key_value("Sequence Length", str(args.seq_len))
    printer.key_value("Output", output_path)

    printer.blank()
    printer.step("Exporting...")
    results = export_model(config, model=net)

    if results:
        printer.blank()
        printer.success("Export successful!")
        for fmt, path in results.items():
            file_size = Path(path).stat().st_size if Path(path).exists() else 0
            printer.key_value(fmt, f"{path} ({format_size(file_size)})")
    else:
        printer.error("Export failed")


def cmd_soul(args):
    """Load, inspect, or create .soul files."""
    if args.load:
        import requests

        base_url = f"http://{args.host}:{args.port}"
        try:
            resp = requests.post(
                f"{base_url}/load-soul",
                json={"soul_path": args.load},
                timeout=30,
            )
            if resp.status_code == 200:
                data = resp.json()
                printer.header("Slo Loaded")
                printer.key_value("Name", data.get("soul_name", "unknown"))
                printer.key_value("Lineage", data.get("lineage", "unknown"))
                printer.key_value("Born", data.get("born_at", ""))
                printer.blank()
                printer.section("Generation Params")
                for k, v in data.get("generation_params", {}).items():
                    printer.key_value(k, str(v))
                printer.section("Personality")
                for k, v in data.get("personality", {}).items():
                    printer.key_value(k, str(v))
            else:
                printer.error(f"Failed: {resp.json()}")
        except Exception as e:
            printer.error(str(e))
        return

    if args.info:
        from domains.inference.slo_format import SouParser

        try:
            soul = SouParser.load(args.info)
            printer.header(f"Slo: {soul.name}")
            printer.key_value("Version", soul.version)
            printer.key_value("Lineage", soul.lineage)
            printer.key_value("Born", soul.born_at)
            printer.key_value("Tags", ", ".join(soul.tags))
            printer.blank()
            printer.section("Personality")
            if soul.personality:
                for k, v in soul.personality.to_dict().items():
                    printer.key_value(k, str(v))
            printer.section("Behavior")
            if soul.behavior:
                for k, v in soul.behavior.to_dict().items():
                    printer.key_value(k, str(v))
        except Exception as e:
            printer.error(str(e))
        return

    if args.create:
        from domains.inference.slo_format import create_soul_profile, SouParser
        from domains.training.slonet import export_to_sou, import_from_sou

        soul = create_soul_profile(
            name=args.name or "SloughGPT-Slo",
            base_model="slonet",
            training_dataset=args.dataset or "",
            epochs_trained=args.epochs or 0,
            lineage=args.lineage or "slonet",
            tags=args.tags.split(",") if args.tags else ["sloughgpt", "soul"],
        )

        if args.model:
            net = import_from_sou(args.model)
            export_to_sou(
                net,
                args.create,
                metadata={
                    "soul_profile": soul.to_dict() if hasattr(soul, "to_dict") else str(soul),
                    "name": soul.name,
                },
            )
            printer.success(f"Created: {args.create}")
        else:
            SouParser.save(soul, args.create)
            printer.success(f"Created: {args.create}")


def cmd_benchmark(args):
    """Benchmark a .soul checkpoint using pure-numpy SloNet inference."""
    import time
    import statistics
    import numpy as np
    from domains.training.slonet import _get_accelerator
    from domains.inference.slonet_provider import SloNetChatProvider

    acc = _get_accelerator()
    backend = acc.name if acc is not None else "cpu"

    printer.header(f"Benchmark - {args.model}")
    printer.key_value("Backend", backend)
    printer.key_value("Device", getattr(acc, "device_name", "CPU") if acc is not None else "CPU")

    if not Path(args.model).exists():
        printer.error(f"Checkpoint not found: {args.model}")
        return

    printer.step("Loading checkpoint...")
    start_time = time.time()
    provider = SloNetChatProvider.from_soul(args.model, model_id="bench")
    load_time = time.time() - start_time
    net = provider._get_model()
    params = sum(int(np.prod(p.shape)) for p in net.parameters())
    printer.key_value("Load Time", f"{load_time:.1f}s")
    printer.key_value("Parameters", f"{params:,}")

    printer.step("Warming up...")
    provider.generate(args.prompt, max_new_tokens=10)

    if args.test in ("all", "latency"):
        printer.section("Latency Test")
        latencies = []
        for _ in range(args.runs):
            start = time.perf_counter()
            provider.generate(args.prompt, max_new_tokens=args.tokens)
            elapsed = (time.perf_counter() - start) * 1000
            latencies.append(elapsed)
        latencies.sort()
        p50 = latencies[int(len(latencies) * 0.50)]
        p95 = latencies[int(len(latencies) * 0.95)]
        avg = statistics.mean(latencies)
        printer.key_value("P50", f"{p50:.1f}ms")
        printer.key_value("P95", f"{p95:.1f}ms")
        printer.key_value("Mean", f"{avg:.1f}ms")

    if args.test in ("all", "throughput"):
        printer.section("Throughput Test")
        throughputs = []
        for _ in range(min(args.runs, 5)):
            start = time.perf_counter()
            text = provider.generate(args.prompt, max_new_tokens=args.tokens)
            elapsed = time.perf_counter() - start
            tokenizer = getattr(provider, "_tokenizer", None)
            n_tokens = len(tokenizer.encode(text)) if tokenizer is not None else len(text)
            tps = n_tokens / elapsed if elapsed > 0 else 0.0
            throughputs.append(tps)
        if throughputs:
            printer.key_value("Average", f"{statistics.mean(throughputs):.1f} tok/s")

    printer.blank()
    printer.success("Benchmark complete!")


def _cmd_models_select(args):
    """Interactive model selector with fuzzy search."""
    import curses
    import requests

    base_url = f"http://{args.host}:{args.port}"

    # Fetch available models
    printer.step("Fetching available models...")
    try:
        resp = requests.get(f"{base_url}/models/hf", timeout=10)
        hf_models = resp.json() if resp.status_code == 200 else []
    except Exception as e:
        printer.warn(f"HuggingFace models: {e}")
        hf_models = []

    try:
        resp = requests.get(f"{base_url}/models", timeout=10)
        local_models = resp.json() if resp.status_code == 200 else []
    except Exception as e:
        printer.warn(f"Local models: {e}")
        local_models = []

    # Build model list: (display_name, model_id, source)
    model_list = []
    for m in hf_models:
        mid = m.get("id", m.get("model_id", ""))
        name = m.get("name", mid)
        model_list.append((name, mid, "hf"))
    seen = set()
    for m in local_models:
        mid = m.get("id", m.get("model_id", ""))
        if mid not in seen:
            seen.add(mid)
            name = m.get("name", mid)
            model_list.append((name, mid, "local"))

    if not model_list:
        printer.info("No models available. Use 'model download <id>' to add one.")
        return

    model_list.sort(key=lambda x: x[0].lower())
    printer.success(f"Found {len(model_list)} models")

    # ── curses interactive selector ──
    def _run_selector(stdscr):
        curses.curs_set(0)
        curses.use_default_colors()
        stdscr.nodelay(False)

        height, width = stdscr.getmaxyx()
        query = ""
        selected = 0
        scroll_offset = 0

        while True:
            stdscr.clear()
            # Header
            stdscr.addstr(0, 0, " SloughGPT Model Selector ", curses.A_REVERSE)
            stdscr.addstr(2, 0, f" Search: {query}")
            stdscr.addstr(3, 0, "─" * min(width - 1, 60))

            # Filter models
            filtered = [(n, i, s) for n, i, s in model_list
                        if query.lower() in n.lower() or query.lower() in i.lower()]

            if not filtered:
                stdscr.addstr(5, 0, " No matching models")
                key = stdscr.getch()
                if key == 27:  # ESC
                    return None
                elif key in (10, 13):  # Enter
                    continue
                elif key == 263 or key == 127:  # Backspace
                    query = query[:-1]
                elif 32 <= key <= 126:
                    query += chr(key)
                continue

            # Ensure selected index is valid
            selected = min(selected, len(filtered) - 1)
            if selected < scroll_offset:
                scroll_offset = selected
            if selected >= scroll_offset + height - 6:
                scroll_offset = selected - height + 7

            # Draw list
            max_display = height - 6
            for i, (name, mid, src) in enumerate(filtered[scroll_offset:scroll_offset + max_display]):
                line_y = 5 + i
                prefix = "▸ " if i + scroll_offset == selected else "  "
                src_tag = " [HF]" if src == "hf" else " [LOCAL]"
                label = f"{prefix}{name}{src_tag}"
                if len(label) > width - 1:
                    label = label[:width - 4] + "..."
                if i + scroll_offset == selected:
                    stdscr.addstr(line_y, 0, label, curses.A_REVERSE)
                else:
                    stdscr.addstr(line_y, 0, label)

            # Footer
            stdscr.addstr(height - 1, 0, f" {len(filtered)} matches  ↑↓ navigate  Enter select  ESC cancel  / search")

            key = stdscr.getch()
            if key == 27:  # ESC
                return None
            elif key in (10, 13):  # Enter
                return filtered[selected]
            elif key == 259:  # Up
                selected = max(0, selected - 1)
            elif key == 258:  # Down
                selected = min(len(filtered) - 1, selected + 1)
            elif key == 338 or key == 261:  # PageDown / Right
                selected = min(len(filtered) - 1, selected + 10)
            elif key == 339 or key == 260:  # PageUp / Left
                selected = max(0, selected - 10)
            elif key == 263 or key == 127:  # Backspace
                query = query[:-1]
            elif 32 <= key <= 126:
                query += chr(key)

    try:
        result = curses.wrapper(_run_selector)
    except Exception as e:
        printer.error(f"Selector error: {e}")
        return

    if result is None:
        printer.info("Selection cancelled")
        return

    name, model_id, source = result
    printer.success(f"Selected: {name} ({model_id})")

    # Load the model
    printer.step(f"Loading {model_id}...")
    try:
        resp = requests.post(f"{base_url}/models/load", json={"model_id": model_id}, timeout=120)
        if resp.status_code == 200:
            printer.success(f"Loaded {model_id}")
        else:
            printer.error(f"Load failed: {resp.json().get('detail', resp.text)}")
    except Exception as e:
        printer.error(f"Load error: {e}")


def register(subparsers):
    """Register model commands with argparse."""
    # Models (with subcommands)
    models_parser = subparsers.add_parser(
        "models",
        help="List models. Subcommands: info, download, compare, personalities",
    )
    models_sub = models_parser.add_subparsers(dest="models_cmd", metavar="SUBCOMMAND")

    # Select (interactive)
    models_select = models_sub.add_parser("select", help="Interactive model selector with fuzzy search")
    models_select.add_argument("--host", default="localhost", help="API host")
    models_select.add_argument("--port", type=int, default=8000, help="API port")
    models_select.set_defaults(func=_cmd_models_select)

    # List (default)
    models_list = models_sub.add_parser("list", help="List available models")
    models_list.set_defaults(func=cmd_models)

    # Info
    info_parser = models_sub.add_parser("info", help="Show checkpoint info")
    info_parser.add_argument("model", help="Path to model checkpoint")
    info_parser.set_defaults(func=_cmd_models_info)

    # Download
    download_parser = models_sub.add_parser("download", help="Download model from HuggingFace (interactive if no model given)")
    download_parser.add_argument("model_id", nargs="?", default=None, help="HuggingFace model ID (e.g., gpt2) — omit for interactive selection")
    download_parser.add_argument("--yes", "-y", action="store_true", help="Override: skip confirmation for this download (default from config: confirm on/off)")
    download_parser.set_defaults(func=_cmd_models_download)

    # Status
    status_parser = models_sub.add_parser("status", help="Show cached/downloaded models with sizes")
    status_parser.set_defaults(func=_cmd_models_status)

    # Compare
    compare_parser = models_sub.add_parser("compare", help="Compare models or benchmarks")
    compare_parser.set_defaults(func=_cmd_models_compare)

    # Personalities
    pers_parser = models_sub.add_parser("personalities", help="List built-in personalities")
    pers_parser.set_defaults(func=_cmd_models_personalities)

    # Keep top-level models as default list
    models_parser.set_defaults(func=lambda a: cmd_models(a) if not a.models_cmd else None)

    # Export
    export_parser = subparsers.add_parser(
        "export",
        help="Export model to different formats",
    )
    export_parser.add_argument("model", nargs="?", default="models/sloughgpt.soul", help="Input model (.soul)")
    export_parser.add_argument("--output", "-o", help="Output path")
    export_parser.add_argument(
        "--format", "-f",
        default="safetensors",
        choices=["safetensors", "safetensors_bf16", "onnx", "gguf_q4_k_m", "gguf_fp16", "gguf_q5_k_m", "gguf_q8_0", "sou", "all"],
        help="Export format",
    )
    export_parser.add_argument("--quantize", dest="quantization", choices=["Q4_K_M", "Q5_K_M", "Q8_0", "F16", "F32"])
    export_parser.add_argument("--seq-len", type=int, default=128, help="Sequence length for ONNX")
    export_parser.add_argument("--opset", type=int, default=17, help="ONNX opset")
    export_parser.add_argument("--ctx", type=int, dest="n_ctx", default=2048, help="Context length for GGUF")
    export_parser.add_argument("--soul-name", type=str, default=None, help="Slo name")
    export_parser.add_argument("--metadata", type=str, nargs="+", default=None, help="Metadata KEY=VALUE")
    export_parser.set_defaults(func=cmd_export_cli)

    # Slo
    soul_parser = subparsers.add_parser(
        "soul",
        help="Manage .soul files",
    )
    soul_parser.add_argument("--load", "-l", metavar="PATH", help="Load soul via API")
    soul_parser.add_argument("--info", "-i", metavar="PATH", help="Inspect soul file")
    soul_parser.add_argument("--create", "-c", metavar="PATH", help="Create new soul")
    soul_parser.add_argument("--model", "-m", metavar="PATH", help="Weights for --create")
    soul_parser.add_argument("--name", "-n", metavar="NAME", help="Slo name")
    soul_parser.add_argument("--dataset", "-d", metavar="PATH", help="Dataset citation")
    soul_parser.add_argument("--epochs", "-e", type=int, default=0, help="Epoch count")
    soul_parser.add_argument("--lineage", default="nanogpt", help="Architecture label")
    soul_parser.add_argument("--tags", default="", help="Comma-separated tags")
    soul_parser.set_defaults(func=cmd_soul)

    # Benchmark
    bench_parser = subparsers.add_parser("benchmark", help="Run performance benchmarks on a .soul checkpoint")
    bench_parser.add_argument("--model", "-m", default="models/sloughgpt.soul", help="Path to .soul checkpoint")
    bench_parser.add_argument("--test", "-t", default="all", choices=["all", "latency", "throughput"], help="Test type")
    bench_parser.add_argument("--runs", "-r", type=int, default=10, help="Number of runs")
    bench_parser.add_argument("--tokens", "-k", type=int, default=50, help="Max new tokens")
    bench_parser.add_argument("--prompt", "-p", default="The quick brown fox jumps over the lazy dog", help="Test prompt")
    bench_parser.set_defaults(func=cmd_benchmark)

    # Standalone aliases for backward compat (forward to models subcommands)
    hf_download_parser = subparsers.add_parser("hf-download", help="Download model from HuggingFace")
    hf_download_parser.add_argument("model_id", help="HuggingFace model ID")
    hf_download_parser.add_argument("--yes", "-y", action="store_true", help="Override: skip confirmation for this download (default from config: confirm on/off)")
    hf_download_parser.set_defaults(func=_cmd_models_download)

    info_parser = subparsers.add_parser("info", help="Show model checkpoint info")
    info_parser.add_argument("model", nargs="?", default="models/sloughgpt.soul", help="Checkpoint path")
    info_parser.set_defaults(func=_cmd_models_info)

    personalities_parser = subparsers.add_parser("personalities", help="List built-in personalities")
    personalities_parser.set_defaults(func=_cmd_models_personalities)

    compare_parser = subparsers.add_parser("compare", help="Compare models or benchmarks")
    compare_parser.set_defaults(func=_cmd_models_compare)
