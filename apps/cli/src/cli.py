"""
SloughGPT CLI — Click-powered entry point with Rich output.

Commands organized into logical groups. All delegate to existing
cmd_* functions in commands/ modules.
"""

import sys
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import click

# Ensure both CLI core and core-py domains are on the path
_CLI_DIR = Path(__file__).resolve().parent
_CORE_PY_DIR = _CLI_DIR.parent.parent.parent / "packages" / "core-py"
for _p in [_CLI_DIR, str(_CORE_PY_DIR)]:
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from core.version import format_version_display
from core.printer import printer

# ── Helpers ───────────────────────────────────────────────────────────


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


def _local_soul_candidate_paths(models_dir: Path, *, default_name: str = "sloughgpt.soul") -> list[Path]:
    default = models_dir / default_name
    out: list[Path] = []
    if default.exists():
        out.append(default)
    if models_dir.is_dir():
        others = sorted(
            (p for p in models_dir.glob("*.soul") if p != default),
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


# ── Namespace marshaller ──────────────────────────────────────────────


def _ns(**kwargs) -> SimpleNamespace:
    """Build a SimpleNamespace from keyword arguments."""
    return SimpleNamespace(**kwargs)


# ── Docker helpers ────────────────────────────────────────────────────


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


# ── Top-level CLI ─────────────────────────────────────────────────────


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


# ═══════════════════════════════════════════════════════════════════════
# Welcome & Shell
# ═══════════════════════════════════════════════════════════════════════


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
       sloughgpt system doctor

  3. First training run:
       sloughgpt train quick

  4. HTTP API:
       sloughgpt dev

  5. Terminal UI:
       sloughgpt tui

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


@cli.command(help="Launch interactive shell REPL")
@click.option("--command", "-c", help="Run a single command and exit")
@click.pass_context
def shell(ctx, command):
    """Launch the SloughGPT interactive shell REPL."""
    from domains.shell.repl import ShellREPL
    from domains.shell.kernel import DaitRuntime

    os = DaitRuntime()
    repl = ShellREPL(os)
    if command:
        commands, is_bg, should_time = repl._parse_pipeline(command)
        if is_bg:
            repl._execute_background(command.rstrip("& ").strip())
        elif len(commands) > 1:
            repl._execute_pipeline(commands, should_time=should_time)
        else:
            expanded = repl._expand_alias(command)
            out = repl._execute_single(expanded, "")
            if out:
                click.echo(out, nl=False)
    else:
        repl.run()


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
    _shell = shell.lower()
    if _shell == "bash":
        click.echo(f'eval "$(_{{COMPLETE}}={_shell}_complete {{prog}})"'.replace("{{COMPLETE}}", "_COMPLETE").replace("{{prog}}", "sloughgpt"))
    elif _shell == "zsh":
        click.echo(f'eval "$(_{{COMPLETE}}={_shell}_complete {{prog}})"'.replace("{{COMPLETE}}", "_COMPLETE").replace("{{prog}}", "sloughgpt"))
    elif _shell == "fish":
        click.echo(f"source (_{{COMPLETE}}={_shell}_complete sloughgpt | psub)")


# ═══════════════════════════════════════════════════════════════════════
# Serve & Chat
# ═══════════════════════════════════════════════════════════════════════


@cli.command(help="Interactive chat with API")
@click.option("--no-serve", is_flag=True, help="Don't auto-start server")
@click.option("--auto-model", default=None, help="Auto-load model")
@click.option("--load-mode", type=click.Choice(["local", "api"]), default="local")
@click.option("--device", default="auto", help="Device hint")
@click.option("--max-tokens", default=100, type=int, help="Max tokens per reply")
@click.option("--temperature", default=0.8, type=float, help="Temperature")
@click.pass_context
def chat(ctx, no_serve, auto_model, load_mode, device, max_tokens, temperature):
    from commands.chat import cmd_chat
    args = _ns(
        no_serve=no_serve, auto_model=auto_model,
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


@cli.command(help="Start API + Web dev servers")
@click.option("--model", default=None, help="Model path")
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


@cli.command(help="Start HTTP inference server (with --web: full FastAPI + frontend)")
@click.option("--host", default="localhost", help="Bind address", show_default=True)
@click.option("--port", default=8000, type=int, help="API port", show_default=True)
@click.option("--model", metavar="PATH", help="Model to preload")
@click.option("--web", is_flag=True, help="Start full FastAPI server + Next.js web UI and open browser")
@click.option("--web-port", default=3000, type=int, help="Web UI port", show_default=True)
def serve(host, port, model, web, web_port):
    from commands.dev import cmd_serve
    args = _ns(host=host, port=port, model=model, web=web, web_port=web_port)
    cmd_serve(args)


@cli.command("hf-serve", hidden=True, help="Serve a HuggingFace model via API")
@click.argument("model_name")
@click.option("--mode", type=click.Choice(["api", "local"]), default="local")
@click.option("--device", default="auto")
@click.pass_context
def hf_serve(ctx, model_name, mode, device):
    from commands.dev import cmd_hf_serve
    args = _ns(
        model=model_name, mode=mode, device=device,
        host=ctx.obj["host"], port=ctx.obj["port"],
    )
    cmd_hf_serve(args)


# ═══════════════════════════════════════════════════════════════════════
# model  — list, info, download, export, benchmark, compare
# ═══════════════════════════════════════════════════════════════════════


@cli.group(help="List, inspect, download, export, and benchmark models")
@click.pass_context
def model(ctx):
    pass


@model.command("list", help="List available models")
@click.pass_context
def model_list(ctx):
    from commands.models import cmd_models
    cmd_models(_ns())


@model.command("info", help="Show checkpoint info")
@click.argument("checkpoint", default="models/sloughgpt.pt")
def model_info(checkpoint):
    from commands.models import _cmd_models_info
    _cmd_models_info(_ns(model=checkpoint))


@model.command("download", help="Download model from HuggingFace")
@click.argument("model_id")
def model_download(model_id):
    from commands.models import _cmd_models_download
    _cmd_models_download(_ns(model_id=model_id))


@model.command("export", help="Export model to different formats")
@click.argument("checkpoint", default="models/sloughgpt.pt")
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
def model_export(checkpoint, output, fmt, quantize, seq_len, opset, n_ctx, soul_name, metadata):
    from commands.models import cmd_export_cli
    args = _ns(
        model=checkpoint, output=output, format=fmt, quantization=quantize,
        seq_len=seq_len, opset=opset, n_ctx=n_ctx, soul_name=soul_name,
        metadata=list(metadata) or None,
    )
    cmd_export_cli(args)


@model.command("benchmark", help="Run performance benchmarks")
@click.option("--checkpoint", "-m", default="gpt2", help="Model to benchmark")
@click.option("--device", "-d", type=click.Choice(["auto", "cpu", "cuda", "mps"]), default="auto")
@click.option("--test", "-t", type=click.Choice(["all", "latency", "throughput"]), default="all")
@click.option("--runs", "-r", default=10, type=int, help="Number of runs")
@click.option("--tokens", "-k", default=50, type=int, help="Max new tokens")
@click.option("--prompt", "-p", default="The quick brown fox jumps over the lazy dog", help="Test prompt")
def model_benchmark(checkpoint, device, test, runs, tokens, prompt):
    from commands.models import cmd_benchmark
    args = _ns(model=checkpoint, device=device, test=test, runs=runs, tokens=tokens, prompt=prompt)
    cmd_benchmark(args)


@model.command("compare", help="Compare models or benchmarks")
def model_compare():
    from commands.models import _cmd_models_compare
    _cmd_models_compare(_ns())


# ═══════════════════════════════════════════════════════════════════════
# dataset  — list, stats, search, import, export, validate
# ═══════════════════════════════════════════════════════════════════════


@cli.group(help="List, import, export, and validate datasets")
@click.pass_context
def dataset(ctx):
    pass


@dataset.command("list", help="List available datasets")
def dataset_list():
    from commands.data import cmd_datasets
    cmd_datasets(_ns())


@dataset.command("stats", help="Show dataset statistics")
@click.argument("name")
def dataset_stats(name):
    from commands.data import cmd_dataset_stats
    args = _ns(name=name)
    cmd_dataset_stats(args)


@dataset.command("search", help="Search online datasets")
@click.argument("query")
@click.option("--limit", "-n", default=10, type=int, help="Max results")
@click.option("--source", type=click.Choice(["hf", "github"]), default="hf")
def dataset_search(query, limit, source):
    from commands.data import cmd_dataset_search
    args = _ns(query=query, limit=limit, source=source)
    cmd_dataset_search(args)


@dataset.command("import", help="Import dataset from various sources")
@click.argument("source", type=click.Choice(["github", "hf", "url", "local"]))
@click.argument("identifier")
@click.argument("name", required=False)
def dataset_import(source, identifier, name):
    from commands.data import cmd_dataset_import
    args = _ns(**({"url": identifier} if source in ("github", "url") else {"dataset_id": identifier}), name=name)
    cmd_dataset_import(args, source)


@dataset.command("export", help="Export dataset to zip")
@click.argument("name")
@click.option("--output", "-o", help="Output zip file")
def dataset_export(name, output):
    from commands.data import cmd_dataset_export
    args = _ns(name=name, output=output)
    cmd_dataset_export(args)


@dataset.command("validate", help="Validate dataset file")
@click.argument("path")
def dataset_validate(path):
    from commands.data import cmd_data_tool
    cmd_data_tool(_ns(path=path), "validate")


@dataset.command("info", help="Show file or directory statistics")
@click.argument("path")
def dataset_info(path):
    from commands.data import cmd_data_tool
    cmd_data_tool(_ns(path=path), "stats")


# ═══════════════════════════════════════════════════════════════════════
# train  — start, quick, auto, self, eval, monitor, rlhf, demo, cloud
# ═══════════════════════════════════════════════════════════════════════


@cli.group(help="Train, evaluate, and monitor models")
@click.pass_context
def train(ctx):
    pass


@train.command("start", help="Full training pipeline")
@click.option("--dataset", default="shakespeare", help="Dataset name")
@click.option("--epochs", default=3, type=int, help="Training epochs")
@click.option("--batch-size", default=32, type=int, help="Batch size")
@click.option("--lr", default=0.01, type=float, help="Learning rate")
@click.option("--api", is_flag=True, help="Use API training")
@click.option("--resume", default=None, help="Resume from checkpoint")
@click.option("--resume-latest", is_flag=True, help="Resume latest")
@click.option("--save-stem", default=None, help="Output filename stem")
@click.option("--optimized", is_flag=True, help="Apply fp16 optimizations")
@click.pass_context
def train_start(ctx, dataset, epochs, batch_size, lr, api, resume, resume_latest, save_stem, optimized):
    from commands.train import cmd_train
    kwargs = dict(
        dataset=dataset, epochs=epochs, batch_size=batch_size, lr=lr,
        api=api, resume=resume, resume_latest=resume_latest,
        save_stem=save_stem, optimized=optimized,
        host=ctx.obj["host"], port=ctx.obj["port"], config=ctx.obj["config"],
    )
    cmd_train(_ns(**kwargs))


@train.command("quick", help="Smoke test: train briefly and generate")
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
def train_quick(ctx, **kwargs):
    from commands.train import cmd_quick
    kwargs["host"] = ctx.obj["host"]
    kwargs["port"] = ctx.obj["port"]
    cmd_quick(_ns(**kwargs))


@train.command("auto", help="Control auto-training via API")
@click.argument("action", type=click.Choice(["start", "stop", "status"]))
@click.option("--teacher", default="gpt2", help="Teacher model")
@click.option("--temperature", default=0.8, type=float, help="Temperature")
@click.option("--steps", default=1000, type=int, help="Max steps")
@click.pass_context
def train_auto(ctx, action, teacher, temperature, steps):
    from commands.train import _cmd_autotrain
    args = _ns(
        action=action, teacher=teacher, temperature=temperature,
        steps=steps, host=ctx.obj["host"], port=ctx.obj["port"],
    )
    _cmd_autotrain(args)


@train.command(name="self", help="Model talks to itself")
@click.option("--steps", default=1000, type=int, help="Training steps")
@click.option("--model", default="gpt2", help="Teacher model")
@click.option("--temperature", default=0.8, type=float, help="Temperature")
@click.option("--max-tokens", default=50, type=int, help="Max tokens per generation")
@click.option("--seed", default="Hello", help="Starting text")
@click.option("--forever", is_flag=True, help="Run until Ctrl+C")
def train_self(steps, model, temperature, max_tokens, seed, forever):
    from commands.train import _cmd_self_train
    args = _ns(
        steps=steps, model=model, temperature=temperature,
        max_tokens=max_tokens, seed=seed, forever=forever,
    )
    _cmd_self_train(args)


@train.command("eval", help="Evaluate model perplexity")
@click.option("--checkpoint", default="models/sloughgpt.pt", help="Checkpoint path")
@click.option("--data", default="datasets/shakespeare/input.txt", help="Eval text")
@click.option("--device", default="cpu", help="Device for scoring")
@click.option("--no-strict", is_flag=True, help="Allow partial load")
@click.option("--benchmark", is_flag=True, help="Run benchmark")
def train_eval(checkpoint, data, device, no_strict, benchmark):
    from commands.train import cmd_eval
    args = _ns(checkpoint=checkpoint, data=data, device=device, no_strict=no_strict, benchmark=benchmark)
    cmd_eval(args)


@train.command("monitor", help="Monitor training jobs")
@click.option("--watch", is_flag=True, help="Continuous watch")
@click.option("--interval", default=5, type=int, help="Refresh interval (s)")
@click.pass_context
def train_monitor(ctx, watch, interval):
    from commands.train import _cmd_monitor
    args = _ns(watch=watch, interval=interval, host=ctx.obj["host"], port=ctx.obj["port"])
    _cmd_monitor(args)


@train.command("rlhf", help="Run RLHF demo")
@click.option("--steps", default=20, type=int, help="PPO steps")
def train_rlhf(steps):
    from commands.train import cmd_rlhf
    args = _ns(steps=steps)
    cmd_rlhf(args)


@train.command("demo", help="Run system demos (RAG, KG, EWC)")
@click.option("--component", type=click.Choice(["all", "rag", "kg", "ewc", "inference"]), default="all")
def train_demo(component):
    from commands.train import cmd_demo
    args = _ns(component=component)
    cmd_demo(args)


@train.command("cloud", help="Setup Pinecone vector store")
@click.option("--api-key", help="Pinecone API key")
@click.option("--index", default="sloughgpt", help="Index name")
@click.option("--dimension", default=768, type=int, help="Vector dimension")
@click.option("--environment", default="us-east-1", help="Pinecone environment")
def train_cloud(api_key, index, dimension, environment):
    from commands.train import cmd_cloud_setup
    args = _ns(api_key=api_key, index=index, dimension=dimension, environment=environment)
    cmd_cloud_setup(args)


@train.command("embed", help="Train a text embedder on your corpus (no downloads)")
@click.option("--corpus", default=None, help="Text file or directory to train on (default: knowledge + chat history)")
@click.option("--epochs", default=20, type=int, help="Training epochs")
@click.option("--lr", default=3e-4, type=float, help="Learning rate")
@click.option("--batch-size", default=32, type=int, help="Batch size")
@click.option("--embed-dim", default=384, type=int, help="Embedding dimension")
@click.option("--vocab-size", default=4096, type=int, help="Max vocabulary size")
@click.option("--output", default=None, help="Output checkpoint path")
@click.option("--test", default=None, help="Test: embed a query string and print top matches")
def train_embed(corpus, epochs, lr, batch_size, embed_dim, vocab_size, output, test):
    """Train a text embedder on your own data using contrastive learning.

    \b
    Examples:
      sloughgpt train embed                          # train on knowledge + chat history
      sloughgpt train embed --corpus datasets/       # train on a directory of text files
      sloughgpt train embed --corpus my_corpus.txt   # train on a single file
      sloughgpt train embed --test "neural networks" # embed a test query
    """
    from commands.train import cmd_train_embed
    args = _ns(
        corpus=corpus, epochs=epochs, lr=lr, batch_size=batch_size,
        embed_dim=embed_dim, vocab_size=vocab_size, output=output, test=test,
    )
    cmd_train_embed(args)


# ═══════════════════════════════════════════════════════════════════════
# distill — knowledge distillation from teacher → student
# ═══════════════════════════════════════════════════════════════════════


@train.command("distill", help="Distill a teacher model into a smaller student")
@click.argument("text_source", required=False, default=None)
@click.option("--file", "-f", default=None, help="Text file to train on")
@click.option("--epochs", default=10, type=int, help="Training epochs")
@click.option("--lr", default=3e-4, type=float, help="Learning rate")
@click.option("--batch-size", default=8, type=int, help="Batch size")
@click.option("--n-embed", default=128, type=int, help="Student embedding size")
@click.option("--n-layer", default=4, type=int, help="Student layers")
@click.option("--n-head", default=4, type=int, help="Student attention heads")
@click.option("--block-size", default=128, type=int, help="Context length")
@click.option("--temperature", default=4.0, type=float, help="Distillation temperature")
@click.option("--dropout", default=0.1, type=float, help="Dropout rate")
@click.option("--checkpoint-dir", default="models/auto-training", help="Save directory")
@click.option("--log-interval", default=10, type=int, help="Log every N steps")
@click.option("--preset", type=click.Choice(["tiny", "small", "medium"]), help="Architecture preset")
@click.option("--api", is_flag=True, help="Use server API instead of local")
@click.option("--json", "json_output", is_flag=True, help="JSON output")
@click.option("--resume", default=None, help="Resume from checkpoint path (.soul file)")
@click.pass_context
def train_distill(ctx, text_source, file, epochs, lr, batch_size, n_embed, n_layer,
                  n_head, block_size, temperature, dropout, checkpoint_dir,
                  log_interval, preset, api, json_output, resume):
    """Distill GPT-2 into a smaller, faster student model.

    \b
    Examples:
      sloughgpt train distill datasets/shakespeare/input.txt
      sloughgpt train distill -f my_book.txt --epochs 20 --preset small
      sloughgpt train distill datasets/shakespeare/input.txt --api
      sloughgpt train distill datasets/shakespeare/input.txt --n-embed 64 --n-layer 2
      sloughgpt train distill datasets/shakespeare/input.txt --resume models/auto-training/checkpoint.soul
    """
    from commands.train import cmd_distill
    args = _ns(
        text_source=text_source, file=file, epochs=epochs, lr=lr,
        batch_size=batch_size, n_embed=n_embed, n_layer=n_layer,
        n_head=n_head, block_size=block_size, temperature=temperature,
        dropout=dropout, checkpoint_dir=checkpoint_dir,
        log_interval=log_interval, preset=preset, api=api,
        json_output=json_output, host=ctx.obj["host"], port=ctx.obj["port"],
        resume=resume,
    )
    cmd_distill(args)


# ═══════════════════════════════════════════════════════════════════════
# checkpoint — list, load, delete training checkpoints
# ═══════════════════════════════════════════════════════════════════════


@cli.group(help="List, load, and delete training checkpoints")
def checkpoint():
    pass


@checkpoint.command("list", help="List all training checkpoints")
@click.option("--sort", type=click.Choice(["date", "size", "name"]), default="date", help="Sort order")
@click.option("--json", "json_output", is_flag=True, help="JSON output")
@click.pass_context
def checkpoint_list(ctx, sort, json_output):
    """List all saved training checkpoints.

    \b
    Examples:
      sloughgpt checkpoint list
      sloughgpt checkpoint list --sort size
      sloughgpt checkpoint list --json
    """
    import requests
    base_url = f"http://{ctx.obj['host']}:{ctx.obj['port']}"
    resp = requests.get(f"{base_url}/auto-train/checkpoints", timeout=10)
    if resp.status_code != 200:
        printer.error(f"Failed to list checkpoints: {resp.text}")
        return
    checkpoints = resp.json()
    if not checkpoints:
        printer.info("No checkpoints found")
        return

    if json_output:
        printer.print_json(checkpoints)
        return

    printer.header(f"Training Checkpoints ({len(checkpoints)})")
    rows = []
    for cp in checkpoints:
        name = cp.get("name", "unknown")
        size = cp.get("size_mb", 0)
        traits = cp.get("traits", {})
        trait_str = ", ".join(f"{k}={v:.2f}" for k, v in traits.items() if v != 0.5) if traits else ""
        rows.append([name, f"{size:.1f} MB", trait_str or "-"])
    printer.table(["Name", "Size", "Traits"], rows)


@checkpoint.command("load", help="Load a checkpoint into the model")
@click.argument("name")
@click.pass_context
def checkpoint_load(ctx, name):
    """Load a training checkpoint into the active model.

    \b
    Example:
      sloughgpt checkpoint load my-checkpoint.soul
    """
    import requests
    base_url = f"http://{ctx.obj['host']}:{ctx.obj['port']}"
    resp = requests.post(f"{base_url}/auto-train/checkpoints/{name}/load", timeout=30)
    if resp.status_code == 200:
        data = resp.json()
        printer.success(f"Loaded checkpoint: {name}")
        for k, v in data.items():
            if k not in ("status",):
                printer.key_value(k, str(v))
    else:
        printer.error(f"Failed to load: {resp.text}")


@checkpoint.command("delete", help="Delete a training checkpoint")
@click.argument("name")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation")
@click.pass_context
def checkpoint_delete(ctx, name, yes):
    """Delete a training checkpoint.

    \b
    Example:
      sloughgpt checkpoint delete my-checkpoint.soul
    """
    if not yes:
        click.confirm(f"Delete checkpoint '{name}'?", abort=True)
    import requests
    base_url = f"http://{ctx.obj['host']}:{ctx.obj['port']}"
    resp = requests.delete(f"{base_url}/auto-train/checkpoints/{name}", timeout=10)
    if resp.status_code == 200:
        printer.success(f"Deleted: {name}")
    else:
        printer.error(f"Failed to delete: {resp.text}")


# ═══════════════════════════════════════════════════════════════════════
# personality  — list, load, info, create, export
# ═══════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════
# knowledge  — search, duplicates, categorize, gaps, ingest
# ═══════════════════════════════════════════════════════════════════════


@cli.group(help="Semantic knowledge operations — search, dedup, categorize, gaps")
def knowledge():
    pass


@knowledge.command("search", help="Search codebase with natural language")
@click.argument("query")
@click.option("--path", default=".", help="Directory to search")
@click.option("--top-k", default=10, type=int, help="Max results")
@click.option("--extensions", default=None, help="Comma-separated file extensions")
@click.pass_context
def knowledge_search(ctx, query, path, top_k, extensions):
    """Search your codebase using natural language.

    \b
    Examples:
      sloughgpt knowledge search "how does embedding work"
      sloughgpt knowledge search "training loop" --path packages/core-py
      sloughgpt knowledge search "error handling" --extensions py,ts
    """
    import requests
    exts = extensions.split(",") if extensions else None
    r = requests.post(f"http://{ctx.obj['host']}:{ctx.obj['port']}/knowledge/search-files",
                      json={"query": query, "path": path, "top_k": top_k, "extensions": exts})
    if r.status_code != 200:
        printer.error(f"Search failed: {r.text}")
        return
    data = r.json()
    printer.header(f"Found {len(data['results'])} results (indexed {data['indexed_files']} files)")
    for i, res in enumerate(data["results"], 1):
        printer.info(f"[{res['score']:.3f}] {res['path']}:{res['line']}")
        snippet = res['snippet'].replace('\n', ' ')[:100]
        printer.info(f"  {snippet}")
        print()


@knowledge.command("dedup", help="Check for duplicate knowledge")
@click.argument("content")
@click.option("--threshold", default=0.85, type=float, help="Similarity threshold")
@click.pass_context
def knowledge_dedup(ctx, content, threshold):
    """Check if content already exists in the knowledge base.

    \b
    Example:
      sloughgpt knowledge dedup "neural networks learn from data"
    """
    import requests
    r = requests.post(f"http://{ctx.obj['host']}:{ctx.obj['port']}/knowledge/check-duplicate",
                      json={"content": content, "threshold": threshold})
    if r.status_code != 200:
        printer.error(f"Check failed: {r.text}")
        return
    data = r.json()
    if data["is_duplicate"]:
        printer.warning(f"DUPLICATE (score: {data['score']:.3f})")
        printer.info(f"  Existing: {data['best_match'][:100]}")
    else:
        printer.success(f"Unique (best match score: {data['score']:.3f})")


@knowledge.command("categorize", help="Auto-categorize content")
@click.argument("content")
@click.pass_context
def knowledge_categorize(ctx, content):
    """Auto-assign a topic to content based on existing categories.

    \b
    Example:
      sloughgpt knowledge categorize "gradient descent optimizes loss"
    """
    import requests
    r = requests.post(f"http://{ctx.obj['host']}:{ctx.obj['port']}/knowledge/categorize",
                      json={"content": content})
    if r.status_code != 200:
        printer.error(f"Categorize failed: {r.text}")
        return
    data = r.json()
    printer.success(f"Topic: {data['topic']}")
    if data["suggestions"]:
        printer.info("Suggestions:")
        for s in data["suggestions"]:
            printer.info(f"  {s['topic']} ({s['score']:.3f})")


@knowledge.command("gaps", help="Find knowledge gaps")
@click.pass_context
def knowledge_gaps(ctx):
    """Show under-represented topics in your knowledge base."""
    import requests
    r = requests.get(f"http://{ctx.obj['host']}:{ctx.obj['port']}/knowledge/gaps")
    if r.status_code != 200:
        printer.error(f"Gaps failed: {r.text}")
        return
    data = r.json()
    printer.header(f"Knowledge gaps ({data['total_facts']} facts, {len(data['topics'])} topics)")
    if data["gaps"]:
        for g in data["gaps"]:
            printer.info(f"  {g['topic']}: {g['suggestion']}")
    else:
        printer.success("No significant gaps found")


@knowledge.command("ingest", help="Bulk ingest texts with dedup")
@click.argument("texts", nargs=-1)
@click.option("--topic", default="imported", help="Topic tag")
@click.option("--file", "file_path", default=None, help="Read texts from file (one per line)")
@click.pass_context
def knowledge_ingest(ctx, texts, topic, file_path):
    """Bulk ingest texts with automatic deduplication.

    \b
    Examples:
      sloughgpt knowledge ingest "fact 1" "fact 2" "fact 3"
      sloughgpt knowledge ingest --file facts.txt --topic ml
    """
    import requests
    items = list(texts)
    if file_path:
        with open(file_path) as f:
            items.extend(line.strip() for line in f if line.strip())
    if not items:
        printer.error("No texts to ingest")
        return
    r = requests.post(f"http://{ctx.obj['host']}:{ctx.obj['port']}/knowledge/bulk-ingest",
                      json={"items": items, "topic": topic})
    if r.status_code != 200:
        printer.error(f"Ingest failed: {r.text}")
        return
    data = r.json()
    printer.success(f"Bulk ingest: {data['added']} added, {data['skipped']} skipped, {data['errors']} errors")


# ═══════════════════════════════════════════════════════════════════════
# personality  — list, load, info, create, export
# ═══════════════════════════════════════════════════════════════════════


@cli.group(help="List, load, and manage .soul personality files")
def personality():
    pass


@personality.command("list", help="List built-in personalities")
def personality_list():
    from commands.models import _cmd_models_personalities
    _cmd_models_personalities(_ns())


@personality.command("load", help="Load soul via API")
@click.argument("path")
@click.pass_context
def personality_load(ctx, path):
    from commands.models import cmd_soul
    cmd_soul(_ns(load=path, host=ctx.obj["host"], port=ctx.obj["port"]))


@personality.command("info", help="Inspect soul file")
@click.argument("path")
def personality_info(path):
    from commands.models import cmd_soul
    cmd_soul(_ns(info=path))


@personality.command("create", help="Create new soul from checkpoint")
@click.option("--checkpoint", "-m", required=True, help="Weights path")
@click.option("--name", "-n", required=True, help="Soul name")
@click.option("--dataset", "-d", help="Dataset citation")
@click.option("--epochs", "-e", default=0, type=int, help="Epoch count")
@click.option("--lineage", default="nanogpt", help="Architecture label")
@click.option("--tags", default="", help="Comma-separated tags")
@click.option("--output", "-o", help="Output .soul path")
def personality_create(checkpoint, name, dataset, epochs, lineage, tags, output):
    from commands.models import cmd_soul
    args = _ns(
        create=output or f"models/{name}.soul", model=checkpoint,
        name=name, dataset=dataset, epochs=epochs, lineage=lineage, tags=tags,
    )
    cmd_soul(args)


# ═══════════════════════════════════════════════════════════════════════
# adapter  — list, info, merge, delete
# ═══════════════════════════════════════════════════════════════════════


@cli.group(help="Manage per-user LoRA adapters")
def adapter():
    pass


@adapter.command("list", help="List LoRA adapters")
def adapter_list():
    from commands.train import _cmd_user_adapters
    _cmd_user_adapters(_ns(action="list"))


@adapter.command("info", help="Show adapter info")
@click.argument("user")
def adapter_info(user):
    from commands.train import _cmd_user_adapters
    _cmd_user_adapters(_ns(action="info", user=user))


@adapter.command("merge", help="Merge adapters")
@click.option("--users", required=True, help="Comma-separated user IDs")
def adapter_merge(users):
    from commands.train import _cmd_user_adapters
    _cmd_user_adapters(_ns(action="merge", users=users))


@adapter.command("delete", help="Delete adapter")
@click.argument("user")
def adapter_delete(user):
    from commands.train import _cmd_user_adapters
    _cmd_user_adapters(_ns(action="delete", user=user))


# ═══════════════════════════════════════════════════════════════════════
# feedback  — export, prepare
# ═══════════════════════════════════════════════════════════════════════


@cli.group(help="Export and prepare feedback data")
def feedback():
    pass


@feedback.command("export", help="Export feedback data")
@click.option("--format", type=click.Choice(["jsonl", "dpo"]), default="jsonl")
@click.option("--output", default="data/training_feedback.jsonl")
def feedback_export(fmt, output):
    from commands.train import _cmd_feedback_export
    args = _ns(format=fmt, output=output)
    _cmd_feedback_export(args)


@feedback.command("prepare", help="Prepare training data from feedback")
@click.option("--format", type=click.Choice(["all", "dpo", "sft", "reward"]), default="all")
@click.option("--output")
@click.option("--stats-only", is_flag=True)
def feedback_prepare(fmt, output, stats_only):
    from commands.train import _cmd_feedback_train
    args = _ns(format=fmt, output=output, stats_only=stats_only)
    _cmd_feedback_train(args)


# ═══════════════════════════════════════════════════════════════════════
# system  — status, info, health, stats, doctor, optimize, setup
# ═══════════════════════════════════════════════════════════════════════


@cli.group(help="System information, health, and environment tools")
def system():
    pass


@system.command("status", help="Show live system status")
@click.option("--watch", is_flag=True, help="Auto-refresh")
@click.option("--interval", default=3, type=int, help="Refresh interval")
def system_status(watch, interval):
    from commands.system import cmd_status
    cmd_status(_ns(watch=watch, interval=interval))


@system.command("info", help="Show system information")
def system_info():
    from commands.system import cmd_system
    cmd_system(_ns())


@system.command("health", help="Quick API health check")
@click.pass_context
def system_health(ctx):
    from commands.dev import cmd_health
    args = _ns(host=ctx.obj["host"], port=ctx.obj["port"])
    cmd_health(args)


@system.command("stats", help="Show models/datasets statistics")
def system_stats():
    from commands.system import cmd_stats
    cmd_stats(_ns())


@system.command("doctor", help="Run environment checks")
def system_doctor():
    from commands.system import cmd_config_check
    cmd_config_check(_ns())


@system.command("config", help="Show or validate configuration")
@click.option("--validate", "do_validate", is_flag=True, help="Validate .env file")
@click.option("--env", default=".env", help="Dotenv file")
@click.option("--generate", "do_generate", is_flag=True, help="Generate secrets")
@click.option("--type", "secret_type", type=click.Choice(["api-key", "jwt-secret", "all"]), default="all")
def system_config(do_validate, env, do_generate, secret_type):
    if do_generate:
        from commands.system import cmd_config_generate
        cmd_config_generate(_ns(type=secret_type))
    elif do_validate:
        from commands.system import cmd_config_validate
        cmd_config_validate(_ns(env=env))
    else:
        from commands.system import cmd_config_check
        cmd_config_check(_ns())


@system.command("optimize", help="Show or apply optimization settings")
@click.option("--apply", "do_apply", is_flag=True, help="Apply optimizations")
def system_optimize(do_apply):
    from commands.system import cmd_optimize
    cmd_optimize(_ns(optimize=do_apply))


@system.command("setup", help="Bootstrap environment")
@click.option("--gpu", is_flag=True, help="GPU support")
@click.option("--docker-only", is_flag=True, help="Docker only")
@click.option("--local-only", is_flag=True, help="Local only")
@click.option("--venv", default=".venv", help="Virtual env directory")
def system_setup(gpu, docker_only, local_only, venv):
    from commands.system import cmd_setup
    args = _ns(gpu=gpu, docker_only=docker_only, local_only=local_only, venv=venv)
    cmd_setup(args)


@system.command("api", help="Test API endpoints or authentication")
@click.argument("action", type=click.Choice(["status", "test", "auth"]), default="status")
@click.pass_context
def system_api(ctx, action):
    from commands.dev import cmd_api_status, cmd_api_test, cmd_api_auth
    args = _ns(host=ctx.obj["host"], port=ctx.obj["port"])
    {
        "status": cmd_api_status,
        "test": cmd_api_test,
        "auth": cmd_api_auth,
    }[action](args)


# ═══════════════════════════════════════════════════════════════════════
# docker  — start, stop, status, logs, build, shell
# ═══════════════════════════════════════════════════════════════════════


@cli.group(help="Docker compose workflows")
def docker():
    pass


@docker.command("start", help="Start Docker services")
@click.option("--gpu", is_flag=True, help="Use GPU profile")
@click.option("--dev", is_flag=True, help="Use dev profile")
def docker_start(gpu, dev):
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


# ═══════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════


def main():
    cli(obj={})


if __name__ == "__main__":
    main()
