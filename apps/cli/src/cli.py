"""
SloughGPT CLI — Click-powered entry point with Rich output.

All commands delegate to the existing cmd_* functions in commands/
modules, keeping them fully compatible.
"""

import sys
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import click

from core.version import format_version_display
from core.printer import printer

# ── Helpers (from original cli.py, unchanged) ────────────────────────


def _chat_repository_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent.parent


def _chat_uvicorn_bind_host(client_host: str) -> str:
    if client_host in ("localhost", "127.0.0.1"):
        return "127.0.0.1"
    return client_host


def _chat_find_available_port(bind_host: str, start_port: int, max_attempts: int = 10) -> int:
    from domains.shared import find_available_port
    return find_available_port(host=bind_host, start_port=start_port, max_attempts=max_attempts)


def _chat_wait_for_health(base_url: str, timeout_sec: float = 45.0) -> bool:
    import time
    import requests
    deadline = time.monotonic() + timeout_sec
    url = f"{base_url.rstrip('/')}/health"
    while time.monotonic() < deadline:
        try:
            r = requests.get(url, timeout=2)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(0.25)
    return False


def _train_export_stem_slug(part: str, fallback: str) -> str:
    import re
    s = re.sub(r"[^a-zA-Z0-9._-]+", "-", (part or "").strip()).strip("-")
    return s[:64] or fallback


def _train_export_default_stem(model_name: str, dataset_label: str) -> str:
    from datetime import datetime
    stamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    return f"{_train_export_stem_slug(model_name, 'model')}-{_train_export_stem_slug(dataset_label, 'data')}-{stamp}"


def _local_soul_candidate_paths(models_dir: Path, *, default_name: str = "sloughgpt.slo") -> list[Path]:
    default = models_dir / default_name
    out: list[Path] = []
    if default.exists():
        out.append(default)
    if models_dir.is_dir():
        others = sorted(
            (p for p in models_dir.glob("*.slo") if p != default),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        out.extend(others)
    return out


def _apply_optimized_train_preset(config, args) -> bool:
    if not getattr(args, "optimized", False):
        return False
    config.training.use_mixed_precision = True
    config.training.mixed_precision_dtype = "fp16"
    return True


# ── Simple namespace marshaller ──────────────────────────────────────


def _ns(**kwargs) -> SimpleNamespace:
    """Build a SimpleNamespace, dropping None values."""
    return SimpleNamespace(**{k: v for k, v in kwargs.items() if v is not None})


# ── Top-level CLI ────────────────────────────────────────────────────


@click.group(invoke_without_command=True)
@click.option("--host", default="localhost", help="API hostname", show_default=True)
@click.option("--port", default=8000, type=int, help="API port", show_default=True)
@click.option("-c", "--config", default="config.yaml", help="Config path", show_default=True)
@click.pass_context
def cli(ctx, host: str, port: int, config: str):
    """SloughGPT CLI — train, chat, serve, and manage models."""
    ctx.ensure_object(dict)
    ctx.obj["host"] = host
    ctx.obj["port"] = port
    ctx.obj["config"] = config

    if ctx.invoked_subcommand is None:
        version_line = f"SloughGPT CLI — {format_version_display()}"
        click.echo()
        click.echo(f"  {version_line}")
        click.echo(f"  {'─' * len(version_line)}")
        click.echo()
        click.echo(ctx.get_help())
        click.echo()


# ── Getting Started ──────────────────────────────────────────────────


@cli.command(help="Welcome guide with next steps")
def start():
    from commands import dev
    root = _chat_repository_root()
    click.echo(f"""
SloughGPT — getting started
===========================

  1. Install Python package:
       python3 -m pip install -e ".[dev]"

  2. Verify environment:
       python3 cli.py config check

  3. First training run:
       make train-demo

  4. HTTP API:
       python3 cli.py dev

  5. Terminal UI:
       python3 cli.py tui

  6. Web UI (separate terminal):
       cd apps/web && npm install && npm run dev

  7. Colab: sloughgpt_colab.ipynb

Repository: {root}

Version: {format_version_display()}
""")


@cli.command(help="Launch interactive terminal UI")
@click.pass_context
def tui(ctx):
    from apps.tui.interactive import main as tui_main
    args = _ns(host=ctx.obj["host"], port=ctx.obj["port"])
    tui_main(["--host", args.host, "--port", str(args.port)])


# ── Serve & Chat ─────────────────────────────────────────────────────


@cli.command(help="Start API + Web servers (uvicorn + npm)")
@click.option("--model", default=None, help="Model path (SLOUGHGPT_MODEL_PATH)")
@click.option("--web-port", default=3000, type=int, help="Web dev server port")
@click.option("--watch-web", is_flag=True, help="Watch web files for changes")
@click.pass_context
def dev(ctx, model, web_port, watch_web):
    from commands.dev import cmd_dev
    args = _ns(
        model=model, web_port=web_port, watch_web=watch_web,
        port=ctx.obj["port"], host=ctx.obj["host"],
    )
    cmd_dev(args)


@cli.command(help="Lightweight HTTP inference server")
@click.option("--host", default="localhost", help="Bind address", show_default=True)
@click.option("--port", default=8080, type=int, help="Listen port", show_default=True)
@click.option("--model", metavar="PATH", help="Model to preload")
def serve(host, port, model):
    from commands.dev import cmd_serve
    args = _ns(host=host, port=port, model=model)
    cmd_serve(args)


@cli.command(help="Interactive chat with API")
@click.option("--no-serve", is_flag=True, help="Don't auto-start server")
@click.option("--auto-model", default=None, help="Auto-load model")
@click.option("--model", default=None, help="Legacy model alias")
@click.option("--load-mode", type=click.Choice(["local", "api"]), default="local")
@click.option("--device", default="auto", help="Device hint")
@click.option("--max-tokens", default=100, type=int, help="Max tokens per reply")
@click.option("--temperature", default=0.8, type=float, help="Temperature")
@click.pass_context
def chat(ctx, no_serve, auto_model, model, load_mode, device, max_tokens, temperature):
    from commands.chat import cmd_chat
    args = _ns(
        no_serve=no_serve, auto_model=auto_model, model=model,
        load_mode=load_mode, device=device, max_tokens=max_tokens,
        temperature=temperature, host=ctx.obj["host"], port=ctx.obj["port"],
    )
    cmd_chat(args)


@cli.command(help="One-shot text generation")
@click.argument("prompt")
@click.option("--model", metavar="NAME_OR_PATH", help="Model override")
@click.option("--max-tokens", default=100, type=int, help="Max tokens", show_default=True)
@click.option("--temperature", default=0.8, type=float, help="Temperature", show_default=True)
@click.pass_context
def generate(ctx, prompt, model, max_tokens, temperature):
    from commands.chat import cmd_generate
    args = _ns(
        prompt=prompt, model=model, max_tokens=max_tokens,
        temperature=temperature, host=ctx.obj["host"], port=ctx.obj["port"],
    )
    cmd_generate(args)


# ── Training ─────────────────────────────────────────────────────────


@cli.command(help="Full training pipeline")
@click.option("--dataset", default="shakespeare", help="Dataset name")
@click.option("--epochs", default=3, type=int, help="Training epochs")
@click.option("--batch-size", default=32, type=int, help="Batch size")
@click.option("--lr", default=0.01, type=float, help="Learning rate")
@click.option("--api", is_flag=True, help="Use API training")
@click.option("--resume", default=None, help="Resume from checkpoint")
@click.option("--resume-latest", is_flag=True, help="Resume latest")
@click.option("--save-stem", default=None, help="Output filename stem")
@click.option("--optimized", is_flag=True, help="Apply fp16 optimizations")
# Self-train
@click.option("--self", "self_train", is_flag=True, help="Self-training loop")
@click.option("--self-steps", default=1000, type=int, help="Self-train steps")
@click.option("--self-model", default="gpt2", help="Model for self-train")
@click.option("--self-max-tokens", default=50, type=int, help="Self-train max tokens per gen")
@click.option("--self-seed", default="Hello", help="Self-train starting text")
@click.option("--self-forever", is_flag=True, help="Self-train until Ctrl+C")
# Auto-train
@click.option("--auto", "auto_train_action", type=click.Choice(["start", "stop", "status"]), help="Auto-training via API")
@click.option("--auto-teacher", default="gpt2", help="Auto-train teacher model")
@click.option("--auto-steps", default=1000, type=int, help="Auto-train max steps")
# Watch
@click.option("--watch", is_flag=True, help="Monitor training jobs")
@click.option("--interval", default=5, type=int, help="Watch refresh interval (s)")
# Adapters
@click.option("--adapters", "adapters_action", type=click.Choice(["list", "info", "delete", "merge"]), help="Manage LoRA adapters")
@click.option("--adapters-user", help="User ID for adapter info/delete")
@click.option("--adapters-users", help="Comma-separated user IDs for adapter merge")
# Feedback
@click.option("--from-feedback", "feedback_train", is_flag=True, help="Prepare training data from feedback")
@click.option("--feedback-format", type=click.Choice(["all", "dpo", "sft", "reward"]), default="all")
@click.option("--feedback-output", help="Output directory for feedback training data")
@click.option("--feedback-stats-only", is_flag=True, help="Show feedback stats only")
@click.option("--export-feedback", is_flag=True, help="Export feedback data")
@click.option("--export-feedback-output", default="data/training_feedback.jsonl", help="Export path")
@click.option("--export-feedback-format", type=click.Choice(["jsonl", "dpo"]), default="jsonl", help="Export format")
@click.pass_context
def train(ctx, **kwargs):
    from commands.train import cmd_train
    kwargs["host"] = ctx.obj["host"]
    kwargs["port"] = ctx.obj["port"]
    kwargs["config"] = ctx.obj["config"]
    args = _ns(**kwargs)
    cmd_train(args)


# ── Quick smoke test ─────────────────────────────────────────────────


@cli.command(help="Smoke test: train briefly and generate")
@click.option("--dataset", "-d", default="datasets/shakespeare/input.txt", help="Corpus file")
@click.option("--prompt", default="The king", help="Generation prompt")
@click.option("--epochs", default=1, type=int, help="Training epochs")
@click.option("--steps", default=100, type=int, help="Max steps")
@click.option("--embed", default=128, type=int, help="Embedding size")
@click.option("--layers", default=4, type=int, help="Transformer layers")
@click.option("--heads", default=4, type=int, help="Attention heads")
@click.option("--block", default=128, type=int, help="Context length")
@click.option("--batch", default=16, type=int, help="Batch size")
@click.option("--lr", default=1e-3, type=float, help="Learning rate")
@click.option("--max-tokens", default=100, type=int, help="Generated tokens")
@click.option("--temperature", default=0.8, type=float, help="Temperature")
@click.option("--output", default="models/quick.pt", help="Output path")
@click.option("--no-optimize", is_flag=True, help="Disable optimizations")
@click.option("--soul-name", default="SloughGPT-Quick", help="Slo name")
@click.option("--datasets", help="Comma-separated datasets (overrides --dataset)")
@click.option("--ratios", help="Comma-separated dataset ratios")
@click.option("--preset", type=click.Choice(["tiny", "small", "medium", "large"]), help="Model preset")
@click.pass_context
def quick(ctx, **kwargs):
    from commands.train import cmd_quick
    kwargs["host"] = ctx.obj["host"]
    kwargs["port"] = ctx.obj["port"]
    args = _ns(**kwargs)
    cmd_quick(args)


# ── Eval & Demo ──────────────────────────────────────────────────────


@cli.command(help="Evaluate model perplexity")
@click.option("--checkpoint", default="models/sloughgpt.pt", help="Checkpoint path")
@click.option("--data", default="datasets/shakespeare/input.txt", help="Eval text")
@click.option("--device", default="cpu", help="Device for scoring")
@click.option("--no-strict", is_flag=True, help="Allow partial load")
@click.option("--benchmark", is_flag=True, help="Run benchmark")
@click.pass_context
def eval(ctx, checkpoint, data, device, no_strict, benchmark):
    from commands.train import cmd_eval
    args = _ns(checkpoint=checkpoint, data=data, device=device, no_strict=no_strict, benchmark=benchmark)
    cmd_eval(args)


@cli.command(help="Run system demos (RAG, KG, EWC)")
@click.option("--component", type=click.Choice(["all", "rag", "kg", "ewc", "inference"]), default="all")
def demo(component):
    from commands.train import cmd_demo
    args = _ns(component=component)
    cmd_demo(args)


@cli.command(help="Run RLHF demo")
@click.option("--steps", default=20, type=int, help="PPO steps")
def rlhf(steps):
    from commands.train import cmd_rlhf
    args = _ns(steps=steps)
    cmd_rlhf(args)


@cli.command(help="Setup Pinecone vector store")
@click.option("--api-key", help="Pinecone API key")
@click.option("--index", default="sloughgpt", help="Index name")
@click.option("--dimension", default=768, type=int, help="Vector dimension")
@click.option("--environment", default="us-east-1", help="Pinecone environment")
def cloud(api_key, index, dimension, environment):
    from commands.train import cmd_cloud_setup
    args = _ns(api_key=api_key, index=index, dimension=dimension, environment=environment)
    cmd_cloud_setup(args)


@cli.command(help="Monitor training jobs")
@click.option("--watch", is_flag=True, help="Continuous watch")
@click.option("--interval", default=5, type=int, help="Refresh interval (s)")
@click.pass_context
def monitor(ctx, watch, interval):
    from commands.train import _cmd_monitor
    args = _ns(watch=watch, interval=interval, host=ctx.obj["host"], port=ctx.obj["port"])
    _cmd_monitor(args)


# ── Model subcommands ────────────────────────────────────────────────


@cli.group(help="List, inspect, download, compare models")
@click.pass_context
def models(ctx):
    pass


@models.command("list", help="List available models")
@click.pass_context
def models_list(ctx):
    from commands.models import cmd_models
    cmd_models(_ns())


@models.command("info", help="Show checkpoint info")
@click.argument("model", default="models/sloughgpt.pt")
def models_info(model):
    from commands.models import _cmd_models_info
    _cmd_models_info(_ns(model=model))


@models.command("download", help="Download model from HuggingFace")
@click.argument("model_id")
def models_download(model_id):
    from commands.models import _cmd_models_download
    _cmd_models_download(_ns(model_id=model_id))


@models.command("compare", help="Compare models or benchmarks")
def models_compare():
    from commands.models import _cmd_models_compare
    _cmd_models_compare(_ns())


@models.command("personalities", help="List built-in personalities")
def models_personalities():
    from commands.models import _cmd_models_personalities
    _cmd_models_personalities(_ns())


# ── Export ───────────────────────────────────────────────────────────


@cli.command(help="Export model to different formats")
@click.argument("model", default="models/sloughgpt.pt")
@click.option("--output", "-o", help="Output path")
@click.option("--format", "-f", "fmt",
    type=click.Choice(["safetensors", "safetensors_bf16", "onnx", "gguf_q4_k_m",
                       "gguf_fp16", "gguf_q5_k_m", "gguf_q8_0", "torch",
                       "torchscript", "sou", "all"]),
    default="safetensors", help="Export format")
@click.option("--quantize", type=click.Choice(["Q4_K_M", "Q5_K_M", "Q8_0", "F16", "F32"]))
@click.option("--seq-len", default=128, type=int, help="Sequence length for ONNX")
@click.option("--opset", default=17, type=int, help="ONNX opset")
@click.option("--ctx", "n_ctx", default=2048, type=int, help="Context length for GGUF")
@click.option("--soul-name", default=None, help="Slo name")
@click.option("--metadata", multiple=True, help="Metadata KEY=VALUE")
def export(model, output, fmt, quantize, seq_len, opset, n_ctx, soul_name, metadata):
    from commands.models import cmd_export_cli
    args = _ns(
        model=model, output=output, format=fmt, quantization=quantize,
        seq_len=seq_len, opset=opset, n_ctx=n_ctx, soul_name=soul_name,
        metadata=list(metadata) or None,
    )
    cmd_export_cli(args)


# ── Soul ─────────────────────────────────────────────────────────────


@cli.command(help="Manage .slo personality files")
@click.option("--load", "-l", "load_path", metavar="PATH", help="Load soul via API")
@click.option("--info", "-i", "info_path", metavar="PATH", help="Inspect soul file")
@click.option("--create", "-c", metavar="PATH", help="Create new soul")
@click.option("--model", "-m", metavar="PATH", help="Weights for --create")
@click.option("--name", "-n", metavar="NAME", help="Slo name")
@click.option("--dataset", "-d", metavar="PATH", help="Dataset citation")
@click.option("--epochs", "-e", default=0, type=int, help="Epoch count")
@click.option("--lineage", default="nanogpt", help="Architecture label")
@click.option("--tags", default="", help="Comma-separated tags")
@click.pass_context
def soul(ctx, load_path, info_path, create, model, name, dataset, epochs, lineage, tags):
    from commands.models import cmd_soul
    args = _ns(
        load=load_path, info=info_path, create=create, model=model,
        name=name, dataset=dataset, epochs=epochs, lineage=lineage,
        tags=tags, host=ctx.obj["host"], port=ctx.obj["port"],
    )
    cmd_soul(args)


# ── Benchmark ────────────────────────────────────────────────────────


@cli.command(help="Run performance benchmarks")
@click.option("--model", "-m", default="gpt2", help="Model to benchmark")
@click.option("--device", "-d", type=click.Choice(["auto", "cpu", "cuda", "mps"]), default="auto")
@click.option("--test", "-t", type=click.Choice(["all", "latency", "throughput"]), default="all")
@click.option("--runs", "-r", default=10, type=int, help="Number of runs")
@click.option("--tokens", "-k", default=50, type=int, help="Max new tokens")
@click.option("--prompt", "-p", default="The quick brown fox jumps over the lazy dog", help="Test prompt")
def benchmark(model, device, test, runs, tokens, prompt):
    from commands.models import cmd_benchmark
    args = _ns(model=model, device=device, test=test, runs=runs, tokens=tokens, prompt=prompt)
    cmd_benchmark(args)


# ── Legacy aliases ───────────────────────────────────────────────────


@cli.command("hf-download", help="Download model from HuggingFace")
@click.argument("model_id")
def hf_download(model_id):
    from commands.models import _cmd_models_download
    _cmd_models_download(_ns(model_id=model_id))


@cli.command(help="Show model checkpoint info")
@click.argument("model", default="models/sloughgpt.pt")
def info(model):
    from commands.models import _cmd_models_info
    _cmd_models_info(_ns(model=model))


@cli.command(help="List built-in personalities")
def personalities():
    from commands.models import _cmd_models_personalities
    _cmd_models_personalities(_ns())


@cli.command(help="Compare models or benchmarks")
def compare():
    from commands.models import _cmd_models_compare
    _cmd_models_compare(_ns())


# ── Legacy training aliases ──────────────────────────────────────────


@cli.command("self-train", help="Model talks to itself")
@click.option("--steps", default=1000, type=int, help="Training steps")
@click.option("--model", default="gpt2", help="Teacher model")
@click.option("--temperature", default=0.8, type=float, help="Temperature")
@click.option("--max-tokens", default=50, type=int, help="Max tokens per generation")
@click.option("--seed", default="Hello", help="Starting text")
@click.option("--forever", is_flag=True, help="Run until Ctrl+C")
def self_train(steps, model, temperature, max_tokens, seed, forever):
    from commands.train import _cmd_self_train
    args = _ns(
        steps=steps, model=model, temperature=temperature,
        max_tokens=max_tokens, seed=seed, forever=forever,
    )
    _cmd_self_train(args)


@cli.command(help="Control auto-training via API")
@click.argument("action", type=click.Choice(["start", "stop", "status"]))
@click.option("--teacher", default="gpt2", help="Teacher model")
@click.option("--temperature", default=0.8, type=float, help="Temperature")
@click.option("--steps", default=1000, type=int, help="Max steps")
@click.pass_context
def autotrain(ctx, action, teacher, temperature, steps):
    from commands.train import _cmd_autotrain
    args = _ns(
        action=action, teacher=teacher, temperature=temperature,
        steps=steps, host=ctx.obj["host"], port=ctx.obj["port"],
    )
    _cmd_autotrain(args)


@cli.command("user-adapters", help="Manage per-user LoRA adapters")
@click.argument("action", type=click.Choice(["list", "info", "delete", "merge"]))
@click.argument("user", required=False)
@click.option("--users", help="Comma-separated user IDs for adapter merge")
def user_adapters(action, user, users):
    from commands.train import _cmd_user_adapters
    args = _ns(action=action, user=user, users=users)
    _cmd_user_adapters(args)


@cli.command("feedback-export", help="Export feedback data for training")
@click.option("--format", type=click.Choice(["jsonl", "dpo"]), default="jsonl")
@click.option("--output", default="data/training_feedback.jsonl")
def feedback_export(fmt, output):
    from commands.train import _cmd_feedback_export
    args = _ns(format=fmt, output=output)
    _cmd_feedback_export(args)


@cli.command("feedback-train", help="Prepare training data from feedback")
@click.option("--format", type=click.Choice(["all", "dpo", "sft", "reward"]), default="all")
@click.option("--output")
@click.option("--stats-only", is_flag=True)
def feedback_train(fmt, output, stats_only):
    from commands.train import _cmd_feedback_train
    args = _ns(format=fmt, output=output, stats_only=stats_only)
    _cmd_feedback_train(args)


# ── Datasets ─────────────────────────────────────────────────────────


@cli.group(help="Dataset utilities")
@click.pass_context
def datasets(ctx):
    pass


@datasets.command("list", help="List available datasets")
def datasets_list():
    from commands.data import cmd_datasets
    cmd_datasets(_ns())


@datasets.command("search", help="Search online datasets")
@click.argument("query")
@click.option("--limit", "-n", default=10, type=int, help="Max results")
@click.option("--source", type=click.Choice(["hf", "github"]), default="hf")
def datasets_search(query, limit, source):
    from commands.data import cmd_dataset_search
    args = _ns(query=query, limit=limit, source=source)
    cmd_dataset_search(args)


@datasets.command("github", help="Import from GitHub")
@click.argument("url")
@click.argument("name", required=False)
def datasets_github(url, name):
    from commands.data import cmd_dataset_import
    args = _ns(url=url, name=name)
    cmd_dataset_import(args, "github")


@datasets.command("hf", help="Import from HuggingFace")
@click.argument("dataset_id")
@click.argument("name", required=False)
def datasets_hf(dataset_id, name):
    from commands.data import cmd_dataset_import
    args = _ns(dataset_id=dataset_id, name=name)
    cmd_dataset_import(args, "hf")


@datasets.command("url", help="Import from URL")
@click.argument("url")
@click.argument("name")
def datasets_url(url, name):
    from commands.data import cmd_dataset_import
    args = _ns(url=url, name=name)
    cmd_dataset_import(args, "url")


@datasets.command("stats", help="Dataset statistics")
@click.argument("name")
def datasets_stats(name):
    from commands.data import cmd_dataset_stats
    args = _ns(name=name)
    cmd_dataset_stats(args)


@datasets.command("export", help="Export dataset to zip")
@click.argument("name")
@click.option("--output", "-o", help="Output zip file")
def datasets_export(name, output):
    from commands.data import cmd_dataset_export
    args = _ns(name=name, output=output)
    cmd_dataset_export(args)


# ── Data tools ───────────────────────────────────────────────────────


@cli.group(help="Dataset file tools")
def data():
    pass


@data.command("stats", help="Show file or directory statistics")
@click.argument("path")
def data_stats(path):
    from commands.data import cmd_data_tool
    cmd_data_tool(_ns(path=path), "stats")


@data.command("validate", help="Validate dataset file")
@click.argument("path")
def data_validate(path):
    from commands.data import cmd_data_tool
    cmd_data_tool(_ns(path=path), "validate")


# ── System ───────────────────────────────────────────────────────────


@cli.command(help="Show system information")
def system():
    from commands.system import cmd_system
    cmd_system(_ns())


@cli.command(help="Show system status")
@click.option("--watch", is_flag=True, help="Auto-refresh")
@click.option("--interval", default=3, type=int, help="Refresh interval")
def status(watch, interval):
    from commands.system import cmd_status
    cmd_status(_ns(watch=watch, interval=interval))


@cli.command(help="Show optimization settings")
@click.option("--optimize", is_flag=True, help="Apply optimizations")
def optimize(optimize):
    from commands.system import cmd_optimize
    cmd_optimize(_ns(optimize=optimize))


@cli.group(help="Configuration utilities")
def config():
    pass


@config.command("check", help="Environment check")
def config_check():
    from commands.system import cmd_config_check
    cmd_config_check(_ns())


@config.command("validate", help="Validate .env file")
@click.option("--env", default=".env", help="Dotenv file")
def config_validate(env):
    from commands.system import cmd_config_validate
    cmd_config_validate(_ns(env=env))


@config.command("generate", help="Generate secrets")
@click.option("--type", "secret_type", type=click.Choice(["api-key", "jwt-secret", "all"]), default="all")
def config_generate(secret_type):
    from commands.system import cmd_config_generate
    cmd_config_generate(_ns(type=secret_type))


@cli.command(help="Show models/datasets statistics")
def stats():
    from commands.system import cmd_stats
    cmd_stats(_ns())


@cli.command(help="Bootstrap environment")
@click.option("--gpu", is_flag=True, help="GPU support")
@click.option("--docker-only", is_flag=True, help="Docker only")
@click.option("--local-only", is_flag=True, help="Local only")
@click.option("--venv", default=".venv", help="Virtual env directory")
def setup(gpu, docker_only, local_only, venv):
    from commands.system import cmd_setup
    args = _ns(gpu=gpu, docker_only=docker_only, local_only=local_only, venv=venv)
    cmd_setup(args)


# ── Health / API / HF commands from dev module ───────────────────────


@cli.command(help="Quick API health check")
@click.pass_context
def health(ctx):
    from commands.dev import cmd_health
    args = _ns(host=ctx.obj["host"], port=ctx.obj["port"])
    cmd_health(args)


@cli.command("api-status", help="Show detailed API status")
@click.pass_context
def api_status(ctx):
    from commands.dev import cmd_api_status
    args = _ns(host=ctx.obj["host"], port=ctx.obj["port"])
    cmd_api_status(args)


@cli.command("api-test", help="Test API endpoints")
@click.pass_context
def api_test(ctx):
    from commands.dev import cmd_api_test
    args = _ns(host=ctx.obj["host"], port=ctx.obj["port"])
    cmd_api_test(args)


@cli.command("api-auth", help="Test API authentication")
@click.pass_context
def api_auth(ctx):
    from commands.dev import cmd_api_auth
    args = _ns(host=ctx.obj["host"], port=ctx.obj["port"])
    cmd_api_auth(args)


@cli.command("hf-serve", help="Serve a HuggingFace model via API")
@click.argument("model_name")
@click.option("--mode", type=click.Choice(["api", "local"]), default="local")
@click.option("--device", default="auto", help="Device (auto, cuda, cpu, mps)")
@click.pass_context
def hf_serve(ctx, model_name, mode, device):
    from commands.dev import cmd_hf_serve
    args = _ns(
        model=model_name, mode=mode, device=device,
        host=ctx.obj["host"], port=ctx.obj["port"],
    )
    cmd_hf_serve(args)


# ── Docker subcommands ───────────────────────────────────────────────


@cli.group(help="Docker compose workflows")
def docker():
    pass


@docker.command("start", help="Start Docker services")
@click.option("--gpu", is_flag=True, help="Use GPU profile")
@click.option("--dev", is_flag=True, help="Use dev profile")
def docker_start(gpu, dev):
    from commands.dev import _chat_repository_root, _docker_compose_file
    _docker_action("start", _ns(gpu=gpu, dev=dev))


@docker.command("stop", help="Stop Docker services")
def docker_stop():
    _docker_action("stop", _ns())


@docker.command("status", help="Show Docker status")
def docker_status():
    _docker_action("status", _ns())


@docker.command("logs", help="Show Docker logs")
@click.argument("service", required=False)
def docker_logs(service):
    _docker_action("logs", _ns(service=service))


@docker.command("build", help="Build Docker images")
@click.option("--no-cache", is_flag=True, help="Build without cache")
def docker_build(no_cache):
    _docker_action("build", _ns(no_cache=no_cache))


@docker.command("shell", help="Shell into container")
@click.argument("service", default="api")
def docker_shell(service):
    _docker_action("shell", _ns(service=service))


def _docker_compose_file():
    return _chat_repository_root() / "infra" / "docker" / "docker-compose.yml"


def _docker_action(action: str, a):
    import subprocess
    compose = _docker_compose_file()
    if not compose.is_file():
        printer.error(f"Compose file not found: {compose}")
        return

    if action == "start":
        profile = []
        if getattr(a, "dev", False):
            profile = ["--profile", "dev"]
        elif getattr(a, "gpu", False):
            profile = ["--profile", "gpu"]
        printer.step("Starting Docker services...")
        subprocess.run(["docker", "compose", "-f", str(compose), "up", "-d", *profile])
        printer.success("Services started")
        subprocess.run(["docker", "compose", "-f", str(compose), "ps"])

    elif action == "stop":
        printer.step("Stopping Docker services...")
        subprocess.run(["docker", "compose", "-f", str(compose), "down"])
        printer.success("Services stopped")

    elif action == "status":
        subprocess.run(["docker", "compose", "-f", str(compose), "ps"])

    elif action == "logs":
        cmd = ["docker", "compose", "-f", str(compose), "logs", "-f"]
        if getattr(a, "service", None):
            cmd.append(a.service)
        subprocess.run(cmd)

    elif action == "build":
        cmd = ["docker", "compose", "-f", str(compose), "build"]
        if getattr(a, "no_cache", False):
            cmd.append("--no-cache")
        printer.step("Building Docker images...")
        subprocess.run(cmd)
        printer.success("Build complete")

    elif action == "shell":
        service = getattr(a, "service", "api")
        subprocess.run(["docker", "compose", "-f", str(compose), "exec", service, "/bin/bash"])


@cli.command(help="Generate shell completion script")
@click.argument("shell", type=click.Choice(["bash", "zsh", "fish"]), default="bash")
def completion(shell):
    """Print a shell completion script. Source it to enable tab-completion.

    \b
    Examples:
      eval "$(sloughgpt completion bash)"   # bash
      eval "$(sloughgpt completion zsh)"    # zsh
      sloughgpt completion fish | source    # fish
    """
    import click as _click
    import os as _os

    _shell = shell.lower()
    if _shell == "bash":
        script = _click.format_help_text(
            _cli_group=_click.Context(cli).command.get_complete_prog_name()
        )
        # Click 8.x provides shell completion via _COMPLETE variable
        click.echo(f'eval "$(_{{COMPLETE}}={_shell}_complete {{prog}})"'.replace("{{COMPLETE}}", "_COMPLETE").replace("{{prog}}", "sloughgpt"))
    elif _shell == "zsh":
        click.echo(f'eval "$(_{{COMPLETE}}={_shell}_complete {{prog}})"'.replace("{{COMPLETE}}", "_COMPLETE").replace("{{prog}}", "sloughgpt"))
    elif _shell == "fish":
        click.echo(f"source (_{{COMPLETE}}={_shell}_complete sloughgpt | psub)")


# ── Entry point ──────────────────────────────────────────────────────


def main():
    cli(obj={})


if __name__ == "__main__":
    main()
