"""
System commands - System info, status, optimization, and configuration.
"""
import sys
import os
import json
import platform
import time
import secrets
from pathlib import Path
from typing import Optional

from core.printer import printer
from core.validator import Doctor
from utils.formatting import format_size, format_time, format_number


def cmd_system(args):
    """Show system information."""
    try:
        import psutil
    except ImportError:
        psutil = None

    printer.header("System Information")

    printer.section("Platform")
    printer.key_value("Platform", platform.platform())
    printer.key_value("Python", platform.python_version())
    printer.key_value("Machine", platform.machine())

    if psutil:
        printer.section("CPU")
        printer.key_value("Cores", str(psutil.cpu_count()))
        printer.key_value("Usage", f"{psutil.cpu_percent()}%")

        printer.section("Memory")
        mem = psutil.virtual_memory()
        printer.key_value("Total", format_size(mem.total))
        printer.key_value("Used", format_size(mem.used))
        printer.key_value("Available", format_size(mem.available))
        printer.key_value("Usage", f"{mem.percent}%")

        printer.section("Disk")
        disk = psutil.disk_usage("/")
        printer.key_value("Total", format_size(disk.total))
        printer.key_value("Used", format_size(disk.used))
        printer.key_value("Free", format_size(disk.free))
        printer.key_value("Usage", f"{disk.percent}%")


def cmd_status(args):
    """Show system status with optional watch mode."""
    import requests

    def print_status():
        print("\033[2J\033[H")
        printer.header("SloughGPT Status")

        try:
            r = requests.get("http://localhost:8000/health", timeout=2)
            printer.status("API", "Online" if r.status_code == 200 else "Offline", "ok" if r.status_code == 200 else "error")
        except (requests.RequestException, ConnectionError):
            printer.status("API", "Not running", "error")

        models_dir = Path("models")
        if models_dir.exists():
            models = list(models_dir.rglob("*.pt")) + list(models_dir.rglob("*.pth")) + list(models_dir.rglob("*.soul"))
            printer.status("Models", f"{len(models)} found", "ok")
        else:
            printer.status("Models", "Directory not found", "error")

        datasets_dir = Path("datasets")
        if datasets_dir.exists():
            datasets = list(datasets_dir.iterdir())
            printer.status("Datasets", f"{len([d for d in datasets if d.is_dir()])} found", "ok")
        else:
            printer.status("Datasets", "Directory not found", "warn")

    if args.watch:
        try:
            while True:
                print_status()
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print()
            printer.info("Stopped watching")
    else:
        print_status()
        printer.info("Use --watch to auto-refresh")


def cmd_optimize(args):
    """Show and configure optimization settings."""
    try:
        import torch
    except ImportError:
        printer.error("PyTorch not installed")
        return

    printer.header("Optimization System")

    printer.section("PyTorch")
    printer.key_value("Version", torch.__version__)

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    printer.key_value("Device", device)

    printer.section("Available Optimizations")
    printer.status("torch.compile", "Yes" if hasattr(torch, "compile") else "No (PyTorch 2.0+)", "ok" if hasattr(torch, "compile") else "warn")
    printer.status("CUDA", "Yes" if torch.cuda.is_available() else "No", "ok" if torch.cuda.is_available() else "info")
    printer.status("MPS", "Yes" if torch.backends.mps.is_available() else "No", "ok" if torch.backends.mps.is_available() else "info")

    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability()
        printer.key_value("CUDA Compute", f"{cap}")
        printer.key_value("BF16 Support", "Yes" if cap[0] >= 8 else "No (use FP16)")

    try:
        from flash_attn import flash_attn_func
        printer.status("Flash Attention", "Yes", "ok")
    except ImportError:
        printer.status("Flash Attention", "No (pip install flash-attn)", "warn")

    printer.section("Training Optimizations")
    printer.info("Mixed Precision (FP16/BF16):  2-3x speedup, 50% memory")
    printer.info("Gradient Checkpointing:        50% memory savings")
    printer.info("Flash Attention:               2-4x speedup")
    printer.info("torch.compile:                 1.5-2x speedup")

    printer.section("Inference Optimizations")
    printer.info("Dynamic Batching:               Maximize GPU utilization")
    printer.info("KV Cache:                      Skip recomputation")
    printer.info("Prompt Caching:                Reuse computed states")

    printer.section("Recommended Configurations")
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability()
        if cap[0] >= 8:
            printer.command("High-End GPU", "config = TrainingConfig(dtype='bf16', use_compile=True, batch_size=32)")
        else:
            printer.command("Mid-Range GPU", "config = TrainingConfig(dtype='fp16', use_compile=True, batch_size=16)")
    elif torch.backends.mps.is_available():
        printer.command("Apple Silicon", "config = TrainingConfig(dtype='fp16', batch_size=8)")
    else:
        printer.command("CPU Only", "config = TrainingConfig(dtype='fp32', batch_size=4)")

    if args.optimize:
        printer.blank()
        printer.step("Applying optimizations...")
        torch.set_num_threads(min(8, torch.get_num_threads()))
        printer.success("Thread count optimized")
        printer.info("Memory format set to channels_last (where applicable)")


def cmd_config_check(args):
    """Check environment setup."""
    printer.header("Environment Check")

    doctor = Doctor()
    result = doctor.run_all()

    printer.blank()
    for check in result.checks:
        printer.status(check.name, check.message, "ok" if check.passed else "error")

    printer.blank()
    if result.passed:
        printer.success(f"All {len(result.checks)} checks passed")
    else:
        printer.error(f"{result.failed_count}/{len(result.checks)} checks failed")
        for check in result.checks:
            if not check.passed and check.suggestion:
                printer.info(f"  → {check.suggestion}")


def cmd_config_validate(args):
    """Validate environment configuration."""
    env_file = args.env
    issues = []
    warnings = []

    printer.header("Configuration Validator")
    printer.key_value("File", env_file)

    if not os.path.exists(env_file):
        printer.warning(f"{env_file} not found")
        return

    with open(env_file, "r") as f:
        content = f.read()

    required_vars = ["SLO_API_KEY", "SLO_JWT_SECRET"]
    security_vars = ["SLO_API_KEY", "SLO_JWT_SECRET", "JWT_SECRET_KEY"]

    for var in required_vars:
        if var not in content:
            issues.append(f"Missing required: {var}")
        else:
            val = content.split(f"{var}=")[1].split("\n")[0] if f"{var}=" in content else ""
            if "hash21" in val.lower() or "change" in val.lower() or len(val) < 32:
                warnings.append(f"Weak {var}: should be >32 random chars")

    for var in security_vars:
        if var in content:
            val = content.split(f"{var}=")[1].split("\n")[0] if f"{var}=" in content else ""
            if "change-this" in val.lower() or "your-" in val.lower():
                warnings.append(f"Default {var}: change to secure value")

    if "SSL_ENABLED=true" in content and "SSL_CERT_PATH" not in content:
        issues.append("SSL enabled but no cert path")

    printer.blank()
    printer.section("Results")

    if issues:
        for issue in issues:
            printer.error(issue)

    if warnings:
        for warn in warnings:
            printer.warning(warn)

    if not issues and not warnings:
        printer.success("Configuration looks good")

    printer.info(f"{len(issues)} issues, {len(warnings)} warnings")


def cmd_config_generate(args):
    """Generate new secrets."""
    printer.header("Secret Generator")

    if args.type in ["api-key", "all"]:
        api_key = secrets.token_urlsafe(32)
        printer.command(f"SLO_API_KEY={api_key}")

    if args.type in ["jwt-secret", "all"]:
        jwt_secret = secrets.token_urlsafe(64)
        printer.command(f"SLO_JWT_SECRET={jwt_secret}")

    if args.type == "all":
        encryption_key = secrets.token_hex(32)
        printer.command(f"ENCRYPTION_KEY={encryption_key}")

    printer.blank()
    printer.info("Copy these to your .env file and restart the server")


def cmd_setup(args):
    """Setup SloughGPT environment."""
    import subprocess

    printer.header("SloughGPT Setup")

    if not os.path.exists("cli.py"):
        printer.error("Run from SloughGPT root directory")
        return

    printer.section("Creating Directories")
    dirs = ["models", "datasets", "data", "checkpoints", "experiments", "logs", "cache"]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        printer.success(d)

    if not os.path.exists(".env") and os.path.exists(".env.example"):
        subprocess.run(["cp", ".env.example", ".env"])
        printer.success(".env created")

    printer.section("Environment")
    printer.key_value("Python", platform.python_version())

    try:
        import torch
        printer.key_value("PyTorch", torch.__version__)
        printer.key_value("CUDA", str(torch.cuda.is_available()))
        printer.key_value("MPS", str(torch.backends.mps.is_available()))
    except ImportError:
        printer.warning("PyTorch not installed")

    if not args.docker_only:
        printer.section("Virtual Environment")
        venv_dir = args.venv
        if not os.path.exists(venv_dir):
            subprocess.run([sys.executable, "-m", "venv", venv_dir])
            printer.success(f"Created {venv_dir}")

        pip_exe = os.path.join(venv_dir, "bin", "pip")
        printer.info("Installing dependencies...")
        subprocess.run([pip_exe, "install", "--upgrade", "pip"])
        subprocess.run([pip_exe, "install", "torch", "transformers", "fastapi", "uvicorn", "pydantic"])
        printer.success("Dependencies installed")

    if not args.local_only:
        printer.section("Docker")
        compose = Path("infra/docker/docker-compose.yml")
        if compose.is_file():
            printer.success("docker-compose.yml found")
            printer.info("Run: docker compose -f infra/docker/docker-compose.yml up -d")
        else:
            printer.warning("docker-compose.yml not found")

    printer.blank()
    printer.success("Setup complete!")
    printer.info("Next: source .venv/bin/activate")
    printer.info("Then: python3 cli.py dev")


def cmd_stats(args):
    """Show training and model statistics."""
    printer.header("SloughGPT Statistics")

    models_dir = Path("models")
    model_count = 0
    total_size = 0
    if models_dir.exists():
        for f in list(models_dir.glob("*.pt")) + list(models_dir.glob("*.safetensors")):
            model_count += 1
            total_size += f.stat().st_size
    printer.section("Models")
    printer.key_value("Count", str(model_count))
    printer.key_value("Total Size", format_size(total_size))

    datasets_dir = Path("datasets")
    ds_count = 0
    ds_size = 0
    if datasets_dir.exists():
        for f in datasets_dir.rglob("*"):
            if f.is_file():
                ds_count += 1
                ds_size += f.stat().st_size
    printer.section("Datasets")
    printer.key_value("Files", str(ds_count))
    printer.key_value("Total Size", format_size(ds_size))

    ckpt_dir = Path("checkpoints")
    ckpt_count = 0
    if ckpt_dir.exists():
        ckpt_count = len(list(ckpt_dir.glob("*.pt")))
    printer.section("Checkpoints")
    printer.key_value("Saved", str(ckpt_count))

    exp_file = Path("data/experiments/experiments.json")
    if exp_file.exists():
        with open(exp_file) as f:
            experiments = json.load(f)
        printer.section("Experiments")
        printer.key_value("Total", str(len(experiments)))


def register(subparsers):
    """Register system commands with argparse."""
    # System
    sys_parser = subparsers.add_parser("system", help="Show system information")
    sys_parser.set_defaults(func=cmd_system)

    # Status
    status_parser = subparsers.add_parser("status", help="Show system status")
    status_parser.add_argument("--watch", action="store_true", help="Auto-refresh")
    status_parser.add_argument("--interval", type=int, default=3, help="Refresh interval")
    status_parser.set_defaults(func=cmd_status)

    # Optimize
    opt_parser = subparsers.add_parser("optimize", help="Show optimization settings")
    opt_parser.add_argument("--optimize", action="store_true", help="Apply optimizations")
    opt_parser.set_defaults(func=cmd_optimize)

    # Config
    config_parser = subparsers.add_parser("config", help="Configuration utilities")
    config_sub = config_parser.add_subparsers(dest="config_cmd", metavar="SUBCOMMAND")

    check_parser = config_sub.add_parser("check", help="Environment check")
    check_parser.set_defaults(func=cmd_config_check)

    validate_parser = config_sub.add_parser("validate", help="Validate .env")
    validate_parser.add_argument("--env", default=".env", help="Dotenv file")
    validate_parser.set_defaults(func=cmd_config_validate)

    generate_parser = config_sub.add_parser("generate", help="Generate secrets")
    generate_parser.add_argument("--type", choices=["api-key", "jwt-secret", "all"], default="all")
    generate_parser.set_defaults(func=cmd_config_generate)

    # Stats
    stats_parser = subparsers.add_parser("stats", help="Show models/datasets statistics")
    stats_parser.set_defaults(func=cmd_stats)

    # Setup
    setup_parser = subparsers.add_parser("setup", help="Bootstrap environment")
    setup_parser.add_argument("--gpu", action="store_true", help="GPU support")
    setup_parser.add_argument("--docker-only", action="store_true", help="Docker only")
    setup_parser.add_argument("--local-only", action="store_true", help="Local only")
    setup_parser.add_argument("--venv", default=".venv", help="Virtual env directory")
    setup_parser.set_defaults(func=cmd_setup)
