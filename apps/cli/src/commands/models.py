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
    from ..cli import _local_soul_candidate_paths

    models_dir = Path("models")

    printer.header("Available Models")

    # Slo files
    printer.section("Soul Files (.soul)")
    soul_files = _local_soul_candidate_paths(models_dir)
    if soul_files:
        rows = []
        for f in soul_files:
            size = f.stat().st_size
            rows.append([f.name, format_size(size)])
        printer.table(["Name", "Size"], rows)
    else:
        printer.info("No soul files found")

    # PyTorch checkpoints
    printer.section("PyTorch Checkpoints (.pt)")
    pt_files = list(models_dir.glob("*.pt"))
    if pt_files:
        rows = []
        for f in sorted(pt_files):
            size = f.stat().st_size
            rows.append([f.name, format_size(size)])
        printer.table(["Name", "Size"], rows)
    else:
        printer.info("No .pt files found")

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
    """Show model checkpoint info."""
    import torch

    model_path = Path(args.model)
    if not model_path.exists():
        printer.error(f"Model not found: {model_path}")
        return

    printer.header(f"Model: {model_path}")

    checkpoint = torch.load(str(model_path), weights_only=False, map_location="cpu")

    if "model" in checkpoint:
        model = checkpoint["model"]
        if hasattr(model, "state_dict"):
            state = model.state_dict()
            printer.key_value("State Dict Keys", str(len(state)))
            total_params = sum(p.numel() for p in state.values() if isinstance(p, torch.Tensor))
            printer.key_value("Parameters", f"{total_params:,}")
        elif isinstance(model, dict):
            printer.key_value("Model Dict Keys", str(len(model)))
            total_params = sum(p.numel() for p in model.values() if isinstance(p, torch.Tensor))
            printer.key_value("Parameters", f"{total_params:,}")

    if "chars" in checkpoint:
        printer.key_value("Vocab Size", str(len(checkpoint["chars"])))
    if "stoi" in checkpoint:
        printer.key_value("Char-to-int Map", str(len(checkpoint["stoi"])))
    if "itos" in checkpoint:
        printer.key_value("Int-to-char Map", str(len(checkpoint["itos"])))
    if "training_info" in checkpoint:
        info = checkpoint["training_info"]
        if isinstance(info, dict):
            for k, v in info.items():
                printer.key_value(str(k), str(v))


def _cmd_models_download(args):
    """Download a HuggingFace model with progress tracking."""
    import sys
    import time
    from pathlib import Path

    printer.header("Download Model")
    printer.key_value("Model ID", args.model_id)
    printer.blank()

    try:
        from domains.training.huggingface.model_map import get_model_info
        from domains.infrastructure.download_manager import get_download_manager
        import asyncio

        info = get_model_info(args.model_id)
        if info:
            printer.key_value("Name", info.name)
            printer.key_value("Description", info.description)
            printer.key_value("Parameters", f"{info.params:,}")
            printer.key_value("Context", str(info.context_length))
            printer.blank()

        mgr = get_download_manager()

        if mgr.is_cached(args.model_id):
            printer.success(f"Model already cached: {args.model_id}")
            return

        printer.step("Starting download...")

        def _render_progress(progress_dict):
            status = progress_dict.get("status", "")
            pct = progress_dict.get("percentage", 0)
            speed = progress_dict.get("speed_mb_per_sec", 0)
            eta = progress_dict.get("eta_seconds", 0)
            current_file = progress_dict.get("current_file", "")
            downloaded_gb = progress_dict.get("bytes_downloaded", 0) / (1024 ** 3)
            total_gb = progress_dict.get("total_bytes", 0) / (1024 ** 3)

            bar_width = 40
            filled = int(bar_width * min(pct, 100) / 100)
            bar = "█" * filled + "░" * (bar_width - filled)

            sys.stdout.write("\r")
            sys.stdout.write(f"  [{bar}] {pct:5.1f}%  {downloaded_gb:.2f}/{total_gb:.2f} GB  {speed:.1f} MB/s")
            if eta > 0:
                sys.stdout.write(f"  ETA: {int(eta)}s")
            if current_file:
                sys.stdout.write(f"\n  ↳ {current_file}")
            sys.stdout.flush()

        async def _do_download():
            mgr.on_progress(args.model_id, _render_progress)
            result = await mgr.download(args.model_id)
            return result

        result = asyncio.run(_do_download())
        sys.stdout.write("\n\n")

        if result.get("status") == "complete":
            printer.success(f"Downloaded in {result.get('elapsed_seconds', '?')}s → {result.get('cache_dir', '')}")
        elif result.get("status") == "failed":
            printer.error(f"Download failed: {result.get('error', 'unknown error')}")
        elif result.get("status") == "cancelled":
            printer.warn("Download cancelled")
    except KeyboardInterrupt:
        printer.warn("Download interrupted by user")
        try:
            from domains.infrastructure.download_manager import get_download_manager
            get_download_manager().cancel(args.model_id)
        except Exception:
            pass
    except Exception as e:
        printer.error(f"Download failed: {e}")


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
    """Export model to different formats."""
    import torch
    from domains.training.export import export_model, list_export_formats, ExportConfig
    from domains.models import SloughGPTModel

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
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
        metadata = checkpoint.get("metadata", {})
    else:
        state_dict = checkpoint
        metadata = {}

    vocab_size = metadata.get("vocab_size", 256)
    n_embed = metadata.get("n_embed", 256)
    n_layer = metadata.get("n_layer", 6)
    n_head = metadata.get("n_head", 8)
    block_size = metadata.get("block_size", 128)

    model = SloughGPTModel(
        vocab_size=vocab_size,
        n_embed=n_embed,
        n_layer=n_layer,
        n_head=n_head,
        block_size=block_size,
    )
    model.load_state_dict(state_dict)

    printer.success(f"Loaded: {format_number(model.num_parameters())} parameters")

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
    results = export_model(config, model=model)

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
        from domains.inference.slo_format import create_soul_profile, export_to_sou, SouParser
        from domains.models import SloughGPTModel
        import torch

        soul = create_soul_profile(
            name=args.name or "SloughGPT-Slo",
            base_model="nanogpt",
            training_dataset=args.dataset or "",
            epochs_trained=args.epochs or 0,
            lineage=args.lineage or "nanogpt",
            tags=args.tags.split(",") if args.tags else ["sloughgpt", "soul"],
        )

        if args.model:
            checkpoint = torch.load(args.model, weights_only=False, map_location="cpu")
            state_dict = checkpoint.get("model_state_dict") or checkpoint.get("model") or checkpoint
            cfg = checkpoint.get("config") or {}
            n_embed = cfg.get("n_embed", 256)
            n_layer = cfg.get("n_layer", 6)
            n_head = cfg.get("n_head", 8)
            block_size = cfg.get("block_size", 128)
            vocab_size = cfg.get("vocab_size", 256)

            model = SloughGPTModel(
                vocab_size=vocab_size,
                n_embed=n_embed,
                n_layer=n_layer,
                n_head=n_head,
                block_size=block_size,
            )
            model.load_state_dict(state_dict, strict=False)
            export_to_sou(model, args.create, soul_profile=soul)
            printer.success(f"Created: {args.create}")
        else:
            SouParser.save(soul, args.create)
            printer.success(f"Created: {args.create}")


def cmd_benchmark(args):
    """Run performance benchmarks on models."""
    import torch
    import time
    import statistics
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = args.device
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    printer.header(f"Benchmark - {args.model}")
    printer.key_value("Device", device)

    if device == "mps" and not torch.backends.mps.is_available():
        printer.warning("MPS not available, falling back to CPU")
        device = "cpu"

    printer.step("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    printer.step("Loading model...")
    start_time = time.time()
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model = model.to(device)
    model.eval()
    load_time = time.time() - start_time
    params = sum(p.numel() for p in model.parameters())
    printer.key_value("Load Time", f"{load_time:.1f}s")
    printer.key_value("Parameters", f"{params:,}")

    input_ids = tokenizer.encode(args.prompt, return_tensors="pt").to(device)
    prompt_length = input_ids.shape[1]

    printer.step("Warming up...")
    with torch.no_grad():
        _ = model.generate(input_ids, max_new_tokens=10, do_sample=False)

    if args.test in ("all", "latency"):
        printer.section("Latency Test")
        latencies = []
        for i in range(args.runs):
            start = time.perf_counter()
            with torch.no_grad():
                _ = model.generate(input_ids, max_new_tokens=args.tokens, do_sample=False)
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
        for i in range(min(args.runs, 5)):
            start = time.perf_counter()
            with torch.no_grad():
                output = model.generate(input_ids, max_new_tokens=args.tokens, do_sample=False)
            elapsed = time.perf_counter() - start
            tokens = output.shape[1]
            tps = tokens / elapsed
            throughputs.append(tps)
        if throughputs:
            printer.key_value("Average", f"{statistics.mean(throughputs):.1f} tok/s")

    printer.blank()
    printer.success("Benchmark complete!")


def register(subparsers):
    """Register model commands with argparse."""
    # Models (with subcommands)
    models_parser = subparsers.add_parser(
        "models",
        help="List models. Subcommands: info, download, compare, personalities",
    )
    models_sub = models_parser.add_subparsers(dest="models_cmd", metavar="SUBCOMMAND")

    # List (default)
    models_list = models_sub.add_parser("list", help="List available models")
    models_list.set_defaults(func=cmd_models)

    # Info
    info_parser = models_sub.add_parser("info", help="Show checkpoint info")
    info_parser.add_argument("model", help="Path to model checkpoint")
    info_parser.set_defaults(func=_cmd_models_info)

    # Download
    download_parser = models_sub.add_parser("download", help="Download model from HuggingFace")
    download_parser.add_argument("model_id", help="HuggingFace model ID (e.g., gpt2)")
    download_parser.set_defaults(func=_cmd_models_download)

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
    export_parser.add_argument("model", nargs="?", default="models/sloughgpt.pt", help="Input model")
    export_parser.add_argument("--output", "-o", help="Output path")
    export_parser.add_argument(
        "--format", "-f",
        default="safetensors",
        choices=["safetensors", "safetensors_bf16", "onnx", "gguf_q4_k_m", "gguf_fp16", "gguf_q5_k_m", "gguf_q8_0", "torch", "torchscript", "sou", "all"],
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
    bench_parser = subparsers.add_parser("benchmark", help="Run performance benchmarks")
    bench_parser.add_argument("--model", "-m", default="gpt2", help="Model to benchmark")
    bench_parser.add_argument("--device", "-d", default="auto", choices=["auto", "cpu", "cuda", "mps"], help="Device")
    bench_parser.add_argument("--test", "-t", default="all", choices=["all", "latency", "throughput"], help="Test type")
    bench_parser.add_argument("--runs", "-r", type=int, default=10, help="Number of runs")
    bench_parser.add_argument("--tokens", "-k", type=int, default=50, help="Max new tokens")
    bench_parser.add_argument("--prompt", "-p", default="The quick brown fox jumps over the lazy dog", help="Test prompt")
    bench_parser.set_defaults(func=cmd_benchmark)

    # Standalone aliases for backward compat (forward to models subcommands)
    hf_download_parser = subparsers.add_parser("hf-download", help="Download model from HuggingFace")
    hf_download_parser.add_argument("model_id", help="HuggingFace model ID")
    hf_download_parser.set_defaults(func=_cmd_models_download)

    info_parser = subparsers.add_parser("info", help="Show model checkpoint info")
    info_parser.add_argument("model", nargs="?", default="models/sloughgpt.pt", help="Checkpoint path")
    info_parser.set_defaults(func=_cmd_models_info)

    personalities_parser = subparsers.add_parser("personalities", help="List built-in personalities")
    personalities_parser.set_defaults(func=_cmd_models_personalities)

    compare_parser = subparsers.add_parser("compare", help="Compare models or benchmarks")
    compare_parser.set_defaults(func=_cmd_models_compare)
