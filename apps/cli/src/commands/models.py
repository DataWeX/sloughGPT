"""
Model commands - Model listing, export, and soul management.
"""
import sys
import os
import json
from pathlib import Path
from typing import Optional

from domains.logging import get_global

log = get_global()
from utils.formatting import format_size, format_number, truncate


def _fmt_model_option(model_id: str, display: str) -> str:
    """Format a model option for InteractivePrompt.select()."""
    return f"{model_id}  ({display})"


def _parse_model_option(option: str) -> str:
    """Extract model_id from a formatted option string."""
    return option.split("  (")[0]


def cmd_models(args):
    """List available models. Use subcommands for info, download, compare, personalities."""
    from utils.helpers import local_soul_candidate_paths

    models_dir = Path("models")

    log.header("Available Models")

    # Slo files
    log.section("Soul Files (.soul)")
    soul_files = local_soul_candidate_paths(models_dir)
    if soul_files:
        rows = []
        for f in soul_files:
            size = f.stat().st_size
            rows.append([f.name, format_size(size)])
        log.table(["Name", "Size"], rows)
    else:
        log.info("No soul files found")

    # Compiled models (.slnc mmap format)
    log.section("Compiled Models (.slnc)")
    slnc_files = sorted(models_dir.rglob("*.slnc")) if models_dir.is_dir() else []
    if slnc_files:
        rows = []
        for f in slnc_files:
            size = f.stat().st_size
            rows.append([f.name, format_size(size)])
        log.table(["Name", "Size"], rows)
    else:
        log.info("No .slnc files found")

    # SafeTensors
    log.section("SafeTensors (.safetensors)")
    st_files = list(models_dir.glob("*.safetensors"))
    if st_files:
        rows = []
        for f in sorted(st_files):
            size = f.stat().st_size
            rows.append([f.name, format_size(size)])
        log.table(["Name", "Size"], rows)
    else:
        log.info("No .safetensors files found")

    log.blank()
    log.section("Available Architectures")
    architectures = [
        ("gpt2", "GPT-2", "124M params"),
        ("gpt2-medium", "GPT-2 Medium", "355M params"),
        ("gpt2-large", "GPT-2 Large", "774M params"),
        ("llama", "LLaMA", "Meta model"),
        ("phi", "Phi", "Microsoft model"),
    ]
    log.table(["ID", "Name", "Info"], architectures)


def _cmd_models_info(args):
    """Show .soul checkpoint info."""
    import numpy as np
    from domains.training.slonet import import_from_sou

    model_path = Path(args.model)
    if not model_path.exists():
        log.error(f"Model not found: {model_path}")
        return

    log.header(f"Model: {model_path}")

    try:
        net = import_from_sou(str(model_path))
    except Exception as e:
        log.error(f"Failed to load: {e}")
        return

    log.key_value("Soul Name", getattr(net, "soul_name", "?"))
    if getattr(net, "soul_traits", None):
        log.key_value("Traits", str(net.soul_traits))
    params = list(net.parameters())
    total_params = sum(int(np.prod(p.shape)) for p in params)
    log.key_value("Parameters", f"{total_params:,}")

    meta = getattr(net, "metadata", None) or {}
    for k in ("vocab_size", "n_embed", "n_layer", "n_head", "block_size"):
        if meta.get(k) is not None:
            log.key_value(k.replace("n_", "Num "), str(meta[k]))
    if meta.get("tokenizer"):
        log.key_value("Tokenizer", str(meta["tokenizer"].get("type", "?")))
    training = meta.get("training") or {}
    for k, v in training.items():
        log.key_value(str(k), str(v))


def _interactive_download_select():
    """Show an interactive fuzzy-searchable list of popular HuggingFace models.

    Queries the HuggingFace Hub API for trending text-generation models,
    displays them in an interactive selector with arrow keys and type-to-filter,
    and returns the selected model ID. Returns None if the user cancels.

    Returns:
        Selected model ID string, or None if cancelled.
    """
    import requests

    log.step("Fetching popular models from HuggingFace Hub...")

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
            log.error(f"Failed to fetch models: HTTP {resp.status_code}")
            return None
        raw_models = resp.json()
    except Exception as e:
        log.error(f"Failed to fetch models: {e}")
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
        log.info("No models found")
        return None

    # Sort by downloads descending
    model_list.sort(key=lambda x: x[2], reverse=True)

    # Build formatted options for InteractivePrompt
    from domains.shell.interactive import InteractivePrompt
    from domains.shell.io import ConsoleIO

    io = ConsoleIO()
    prompt = InteractivePrompt(io)

    options = [_fmt_model_option(mid, display) for display, mid, _ in model_list]
    result = prompt.select("Download Model from HuggingFace", options)

    if not result:
        log.info("Selection cancelled")
        return None

    model_id = _parse_model_option(result)
    log.success(f"Selected: {model_id}")
    return model_id


def _cmd_models_download(args):
    """Download a HuggingFace model with size confirmation and live progress.

    Enforces the bandwidth policy: queries HuggingFace Hub for model size,
    shows the estimate, and requires user confirmation for downloads over
    50 MB. Use ``--yes`` or ``SLO_AUTO_DOWNLOAD=1`` to skip the prompt.

    Shows a live progress bar with percentage, speed, ETA, and current
    filename. Ctrl+C cancels gracefully.

    If no model_id is provided, shows an interactive fuzzy-searchable list
    of popular text-generation models from HuggingFace Hub.

    Side effects:
        - May download model files to HF cache directory
        - Prints to stdout via ANSI progress bars
    """
    from core.permissions import PermissionsManager
    from utils.progress import ProgressBar

    # ── Interactive model selection if no model_id ──────────
    if not getattr(args, "model_id", None):
        model_id = _interactive_download_select()
        if not model_id:
            return
        args.model_id = model_id

    log.header("Download Model")
    log.key_value("Model ID", args.model_id)
    log.blank()

    try:
        from domains.infrastructure.download_manager import get_download_manager
        import asyncio

        mgr = get_download_manager()

        if mgr.is_cached(args.model_id):
            log.success(f"Model already cached: {args.model_id}")
            return

        # ── Confirmation gate ──────────────────────────────
        pm = PermissionsManager(auto_yes=getattr(args, "yes", False))
        if not pm.confirm_download(args.model_id):
            log.info("Download cancelled by user")
            return

        # ── Live progress bar ──────────────────────────────
        bar = ProgressBar(total=100, desc="Downloading", width=30, show_eta=True)
        current_file = ""

        def _render_progress(progress_dict):
            nonlocal current_file
            pct = progress_dict.get("percentage", 0)
            fname = progress_dict.get("current_file", "")
            if fname:
                current_file = fname.split("/")[-1][:40]
            bar.desc = current_file or "Downloading"
            bar.set_progress(int(pct))

        async def _do_download():
            mgr.on_progress(args.model_id, _render_progress)
            bar.start()
            try:
                result = await mgr.download(args.model_id)
                return result
            finally:
                bar.finish()

        result = asyncio.run(_do_download())

        if result.get("status") == "complete":
            log.success(
                f"Downloaded in {result.get('elapsed_seconds', '?')}s "
                f"-> {result.get('cache_dir', '')}"
            )
        elif result.get("status") == "failed":
            log.error(f"Download failed: {result.get('error', 'unknown error')}")
        elif result.get("status") == "cancelled":
            log.warning("Download cancelled")
    except KeyboardInterrupt:
        log.warning("Download interrupted by user")
        try:
            mgr = get_download_manager()
            mgr.cancel(args.model_id)
        except (OSError, AttributeError):
            pass
    except Exception as e:
        log.error(f"Download failed: {e}")


def _cmd_models_status(args):
    """Show status of all cached/downloaded HuggingFace models.

    Scans the HuggingFace cache directory for downloaded models, shows
    their sizes, and indicates whether each has been converted to .slnc
    format. Useful for managing disk space and verifying downloads.
    """
    import sys

    hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
    if not hf_cache.exists():
        log.info("No HuggingFace cache found")
        log.key_value("Cache path", str(hf_cache))
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
            status = ".slnc"
        elif has_safetensors:
            status = "safetensors"
        else:
            status = "other"

        models.append({
            "id": model_id,
            "size": total_bytes,
            "files": file_count,
            "status": status,
        })

    if not models:
        log.info("No cached models found")
        log.key_value("Cache path", str(hf_cache))
        return

    # Sort by size descending
    models.sort(key=lambda m: m["size"], reverse=True)

    total_cache = sum(m["size"] for m in models)

    log.header(f"Cached Models ({len(models)} models, {format_size(total_cache)} total)")

    log.table(
        ["Model", "Size", "Files", "Format"],
        [[m["id"], format_size(m["size"]), str(m["files"]), m["status"]] for m in models],
        align=["l", "r", "r", "l"],
    )

    log.blank()
    log.key_value("Cache", str(hf_cache))


def _cmd_models_compare(args):
    """Compare benchmark results or models."""
    log.header("Model Comparison")

    # Compare benchmark results
    benchmarks_dir = Path("data/experiments/benchmarks")
    if benchmarks_dir.exists():
        benchmarks = list(benchmarks_dir.glob("*.json"))
        if benchmarks:
            log.section("Benchmark Results")
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
            log.table(["Model", "Tokens/s", "Latency (ms)", "Memory (MB)"], rows, align=["l", "r", "r", "r"])

    # Compare models
    log.section("Model Specifications")
    model_specs = [
        ("gpt2", "124M", "~250MB", "Fast"),
        ("gpt2-medium", "355M", "~700MB", "Medium"),
        ("gpt2-large", "774M", "~1.5GB", "Slow"),
        ("phi-2", "2.7B", "~5.4GB", "Medium"),
        ("mistral-7b", "7.3B", "~14GB", "Slow"),
        ("llama-2-7b", "7B", "~13GB", "Slow"),
    ]
    log.table(["Model", "Params", "Size", "Speed"], model_specs)

    log.blank()
    log.info("Run benchmarks: cli.py eval --checkpoint <path> --benchmark")


def _cmd_models_personalities(args):
    """List available personalities."""
    try:
        from domains.ai_personality import PERSONALITIES
    except ImportError:
        log.error("Personalities module not found")
        return

    log.header("Available Personalities")
    rows = []
    for ptype, personality in PERSONALITIES.items():
        rows.append([ptype.value.upper(), personality.name, personality.description[:50], ", ".join(personality.traits)])
    log.table(["Type", "Name", "Description", "Traits"], rows)


def cmd_export_cli(args):
    """Export a .soul model to different formats."""
    import numpy as np
    from domains.training.export import export_model, list_export_formats, ExportConfig

    log.header("Model Export")

    # List formats
    log.section("Supported Formats")
    formats = list_export_formats()
    for fmt, desc in formats.items():
        log.key_value(fmt, desc)

    model_path = Path(args.model)
    if not model_path.exists():
        log.error(f"Model not found: {args.model}")
        return

    log.blank()
    log.step(f"Loading: {args.model}")
    from domains.training.slonet import import_from_sou
    net = import_from_sou(str(model_path))
    metadata = dict(getattr(net, "metadata", None) or {})
    metadata.setdefault("name", getattr(net, "soul_name", "SloughGPT"))

    total_params = sum(int(np.prod(p.shape)) for p in net.parameters())
    log.success(f"Loaded: {format_number(total_params)} parameters")

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

    log.blank()
    log.section("Export Configuration")
    log.key_value("Format", args.format)
    log.key_value("Quantization", args.quantization or "N/A")
    log.key_value("Sequence Length", str(args.seq_len))
    log.key_value("Output", output_path)

    log.blank()
    log.step("Exporting...")
    results = export_model(config, model=net)

    if results:
        log.blank()
        log.success("Export successful!")
        for fmt, path in results.items():
            file_size = Path(path).stat().st_size if Path(path).exists() else 0
            log.key_value(fmt, f"{path} ({format_size(file_size)})")
    else:
        log.error("Export failed")


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
                log.header("Slo Loaded")
                log.key_value("Name", data.get("soul_name", "unknown"))
                log.key_value("Lineage", data.get("lineage", "unknown"))
                log.key_value("Born", data.get("born_at", ""))
                log.blank()
                log.section("Generation Params")
                for k, v in data.get("generation_params", {}).items():
                    log.key_value(k, str(v))
                log.section("Personality")
                for k, v in data.get("personality", {}).items():
                    log.key_value(k, str(v))
            else:
                log.error(f"Failed: {resp.json()}")
        except Exception as e:
            log.error(str(e))
        return

    if args.info:
        from domains.inference.slo_format import SouParser

        try:
            soul = SouParser.load(args.info)
            log.header(f"Slo: {soul.name}")
            log.key_value("Version", soul.version)
            log.key_value("Lineage", soul.lineage)
            log.key_value("Born", soul.born_at)
            log.key_value("Tags", ", ".join(soul.tags))
            log.blank()
            log.section("Personality")
            if soul.personality:
                for k, v in soul.personality.to_dict().items():
                    log.key_value(k, str(v))
            log.section("Behavior")
            if soul.behavior:
                for k, v in soul.behavior.to_dict().items():
                    log.key_value(k, str(v))
        except Exception as e:
            log.error(str(e))
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
            log.success(f"Created: {args.create}")
        else:
            SouParser.save(soul, args.create)
            log.success(f"Created: {args.create}")


def cmd_benchmark(args):
    """Benchmark a .soul checkpoint using pure-numpy SloNet inference."""
    import time
    import statistics
    import numpy as np
    from domains.training.slonet import _get_accelerator
    from domains.inference.slonet_provider import SloNetChatProvider

    acc = _get_accelerator()
    backend = acc.name if acc is not None else "cpu"

    log.header(f"Benchmark - {args.model}")
    log.key_value("Backend", backend)
    log.key_value("Device", getattr(acc, "device_name", "CPU") if acc is not None else "CPU")

    if not Path(args.model).exists():
        log.error(f"Checkpoint not found: {args.model}")
        return

    log.step("Loading checkpoint...")
    start_time = time.time()
    provider = SloNetChatProvider.from_soul(args.model, model_id="bench")
    load_time = time.time() - start_time
    net = provider._get_model()
    params = sum(int(np.prod(p.shape)) for p in net.parameters())
    log.key_value("Load Time", f"{load_time:.1f}s")
    log.key_value("Parameters", f"{params:,}")

    log.step("Warming up...")
    provider.generate(args.prompt, max_new_tokens=10)

    if args.test in ("all", "latency"):
        log.section("Latency Test")
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
        log.key_value("P50", f"{p50:.1f}ms")
        log.key_value("P95", f"{p95:.1f}ms")
        log.key_value("Mean", f"{avg:.1f}ms")

    if args.test in ("all", "throughput"):
        log.section("Throughput Test")
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
            log.key_value("Average", f"{statistics.mean(throughputs):.1f} tok/s")

    log.blank()
    log.success("Benchmark complete!")


def _cmd_models_select(args):
    """Interactive model selector with fuzzy search."""
    import curses
    import requests

    base_url = f"http://{args.host}:{args.port}"

    # Fetch available models
    log.step("Fetching available models...")
    try:
        resp = requests.get(f"{base_url}/models/hf", timeout=10)
        hf_models = resp.json() if resp.status_code == 200 else []
    except Exception as e:
        log.warning(f"HuggingFace models: {e}")
        hf_models = []

    try:
        resp = requests.get(f"{base_url}/models", timeout=10)
        local_models = resp.json() if resp.status_code == 200 else []
    except Exception as e:
        log.warning(f"Local models: {e}")
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
        log.info("No models available. Use 'model download <id>' to add one.")
        return

    model_list.sort(key=lambda x: x[0].lower())
    log.success(f"Found {len(model_list)} models")

    # Build formatted options for InteractivePrompt
    from domains.shell.interactive import InteractivePrompt
    from domains.shell.io import ConsoleIO

    io = ConsoleIO()
    prompt = InteractivePrompt(io)

    options = [_fmt_model_option(mid, f"{name} [{'HF' if src == 'hf' else 'LOCAL'}]") for name, mid, src in model_list]
    result = prompt.select("SloughGPT Model Selector", options)

    if not result:
        log.info("Selection cancelled")
        return

    model_id = _parse_model_option(result)
    log.success(f"Selected: {model_id}")

    # Load the model
    log.step(f"Loading {model_id}...")
    try:
        resp = requests.post(f"{base_url}/models/load", json={"model_id": model_id}, timeout=120)
        if resp.status_code == 200:
            log.success(f"Loaded {model_id}")
        else:
            log.error(f"Load failed: {resp.json().get('detail', resp.text)}")
    except Exception as e:
        log.error(f"Load error: {e}")


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
