"""
Chat commands - Interactive chat and one-shot generation.
"""
import sys
import os
import tempfile
from pathlib import Path
from typing import Optional

from core.printer import printer
from utils.formatting import truncate


def cmd_chat(args):
    """Interactive chat against the API."""
    import subprocess
    import requests
    from requests.exceptions import ConnectionError as RequestsConnectionError

    from utils.helpers import chat_repository_root, chat_uvicorn_bind_host, chat_find_available_port, chat_wait_for_health

    base_url = f"http://{args.host}:{args.port}".rstrip("/")
    server_proc = None
    started_server_here = False
    log_path = None
    printed_no_model_hint = False

    def api_reachable() -> bool:
        try:
            r = requests.get(f"{base_url}/health", timeout=3)
            return r.status_code == 200
        except RequestsConnectionError:
            return False
        except Exception:
            return False

    def try_load_model(model_id: str) -> bool:
        load_mode = getattr(args, "load_mode", "local")
        device = getattr(args, "device", "auto")
        try:
            r = requests.post(
                f"{base_url}/models/load",
                json={"model_id": model_id, "mode": load_mode, "device": device},
                timeout=120,
            )
        except Exception as e:
            printer.warning(f"Model load failed: {e}")
            return False

        if r.ok:
            printer.success(f"Model ready: {model_id}")
            return True

        printer.warning(f"Could not load '{model_id}' ({r.status_code})")
        return False

    if not api_reachable():
        if getattr(args, "no_serve", False):
            printer.error("API not reachable and --no-serve set")
            printer.info("Start server: python3 cli.py dev")
            return

        repo = chat_repository_root()
        marker = repo / "apps" / "api" / "server" / "main.py"
        if not marker.is_file():
            printer.error("Not inside SloughGPT repo")
            return

        bind_host = chat_uvicorn_bind_host(args.host)
        try:
            listen_port = chat_find_available_port(bind_host, args.port)
        except RuntimeError as e:
            printer.error(str(e))
            return

        if listen_port != args.port:
            printer.warning(f"Port {args.port} busy, using {listen_port}")

        base_url = f"http://{args.host}:{listen_port}".rstrip("/")

        printer.info("API not reachable, starting server...")
        log_f = tempfile.NamedTemporaryFile(prefix="sloughgpt-chat-", suffix=".log", delete=False)
        log_path = log_f.name
        log_f.close()

        server_dir = repo / "apps" / "api" / "server"
        cmd = [
            sys.executable,
            "-m",
            "uvicorn",
            "main:app",
            "--app-dir",
            str(server_dir),
            "--host",
            bind_host,
            "--port",
            str(listen_port),
        ]

        try:
            with open(log_path, "wb") as out:
                server_proc = subprocess.Popen(
                    cmd,
                    cwd=str(repo),
                    stdout=out,
                    stderr=subprocess.STDOUT,
                )
            started_server_here = True
        except OSError as e:
            printer.error(f"Failed to start server: {e}")
            try:
                os.unlink(log_path)
            except OSError:
                pass
            return

        if not chat_wait_for_health(base_url):
            if server_proc.poll() is None:
                server_proc.terminate()
                try:
                    server_proc.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    server_proc.kill()
            printer.error("Server did not respond in time")
            try:
                with open(log_path, encoding="utf-8", errors="replace") as lf:
                    tail = lf.read()[-4000:]
                if tail.strip():
                    print(tail)
            except OSError:
                pass
            try:
                os.unlink(log_path)
            except OSError:
                pass
            return

        printer.success("Server ready")
        print()

    printer.header("SloughGPT Chat")
    printer.info(f"Connected to {base_url}")
    printer.info("Type 'quit' to exit")
    printer.info("Tip: --auto-model gpt2 to preload a model")

    auto_model = getattr(args, "auto_model", None)
    legacy_model = getattr(args, "model", None)
    if auto_model and legacy_model:
        printer.warning("Both --auto-model and --model provided, using --auto-model")
    model_to_autoload = auto_model or legacy_model
    if model_to_autoload:
        printer.step(f"Loading: {model_to_autoload}")
        try_load_model(model_to_autoload)

    printer.blank()

    try:
        while True:
            user_input = input("You: ")
            if user_input.lower() in ["quit", "exit", "q"]:
                break

            try:
                response = requests.post(
                    f"{base_url}/generate",
                    json={
                        "prompt": user_input,
                        "max_new_tokens": args.max_tokens,
                        "temperature": args.temperature,
                    },
                    timeout=120,
                )

                if response.status_code == 200:
                    data = response.json()
                    text = data.get("text", data)
                    print(f"\nSloughGPT: {text}\n")
                    if isinstance(text, str) and "No model loaded" in text and not printed_no_model_hint:
                        printer.info("Load a model first: --auto-model gpt2")
                        printed_no_model_hint = True
                else:
                    printer.error(f"HTTP {response.status_code}: {response.text}")

            except RequestsConnectionError:
                printer.error("Lost connection to API")
            except Exception as e:
                printer.error(str(e))
    finally:
        if started_server_here and server_proc is not None and server_proc.poll() is None:
            print()
            printer.info("Stopping server...")
            server_proc.terminate()
            try:
                server_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server_proc.kill()
        if log_path:
            try:
                os.unlink(log_path)
            except OSError:
                pass


def cmd_generate(args):
    """One-shot text generation."""
    from pathlib import Path
    from domains.core import SloEngine
    from utils.helpers import local_soul_candidate_paths

    models_dir = Path("models")
    pt_path = models_dir / "sloughgpt_finetuned.pt"

    printer.header("Text Generation")
    printer.key_value("Prompt", args.prompt)
    printer.key_value("Max Tokens", str(args.max_tokens))
    printer.key_value("Temperature", str(args.temperature))
    printer.blank()

    engine = SloEngine(device="cpu")
    loaded = False

    def _try_load_sou(path: Path) -> bool:
        nonlocal loaded
        try:
            soul = engine.load_soul(str(path))
            printer.success(f"Loaded soul: {soul.name} from {path.name}")
            loaded = True
            return True
        except Exception as e:
            return False

    for sou_path in local_soul_candidate_paths(models_dir):
        if _try_load_sou(sou_path):
            break

    if not loaded and pt_path.exists():
        try:
            import torch
            from domains.training.checkpoint_utils import (
                load_sloughgpt_from_checkpoint,
                normalize_raw_checkpoint,
                torch_load_checkpoint,
                tokenizer_maps_from_bundle,
            )

            raw = torch_load_checkpoint(str(pt_path), map_location="cpu")
            bundle = normalize_raw_checkpoint(raw)
            stoi_len = len(bundle.get("stoi", {})) or 65
            model, _ = load_sloughgpt_from_checkpoint(
                bundle,
                device="cpu",
                strict=True,
                fallback_vocab_size=stoi_len,
                fallback_n_embed=128,
                fallback_n_layer=4,
                fallback_n_head=4,
                fallback_block_size=64,
            )
            engine._model = model
            stoi, itos = tokenizer_maps_from_bundle(bundle)
            if stoi is not None and itos is not None:
                engine.set_vocab(stoi, itos)
            loaded = True
            printer.success(f"Loaded checkpoint: {pt_path.name}")
        except Exception as e:
            printer.warning(f"Failed to load .pt: {e}")

    if not loaded:
        printer.warning("No model found, using demo mode")

    printer.step("Generating...")
    result = engine.generate(
        args.prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
    )
    printer.blank()
    printer.key_value("Generated", truncate(result, 500))


def register(subparsers):
    """Register chat commands with argparse."""
    # Chat
    chat_parser = subparsers.add_parser(
        "chat",
        help="Interactive chat with API",
    )
    chat_parser.add_argument("--no-serve", action="store_true", help="Don't auto-start server")
    chat_parser.add_argument("--auto-model", default=None, help="Auto-load model")
    chat_parser.add_argument("--model", default=None, help="Legacy model alias")
    chat_parser.add_argument("--load-mode", choices=["local", "api"], default="local")
    chat_parser.add_argument("--device", default="auto", help="Device hint")
    chat_parser.add_argument("--max-tokens", type=int, default=100, help="Max tokens per reply")
    chat_parser.add_argument("--temperature", type=float, default=0.8, help="Temperature")
    chat_parser.set_defaults(func=cmd_chat)

    # Generate
    gen_parser = subparsers.add_parser(
        "generate",
        aliases=["gen"],
        help="One-shot text generation",
    )
    gen_parser.add_argument("prompt", help="Starter text")
    gen_parser.add_argument("--model", metavar="NAME_OR_PATH", help="Model override")
    gen_parser.add_argument("--max-tokens", type=int, default=100, help="Max tokens")
    gen_parser.add_argument("--temperature", type=float, default=0.8, help="Temperature")
    gen_parser.set_defaults(func=cmd_generate)
