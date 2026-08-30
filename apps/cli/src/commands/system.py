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

from domains.logging import get_global

log = get_global()
from core.validator import Doctor
from utils.formatting import format_size, format_time, format_number


def cmd_system(args):
    """Show system information."""
    import json as _json

    try:
        import psutil
    except ImportError:
        psutil = None

    data = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "machine": platform.machine(),
    }

    if psutil:
        try:
            from domains.infrastructure.resource_manager import get_resource_manager
            rm = get_resource_manager()
            cores = f"{rm.topology.logical_cores} logical / {rm.topology.physical_cores} physical"
        except Exception:
            cores = str(psutil.cpu_count())
        data["cpu"] = {"cores": cores, "usage": f"{psutil.cpu_percent()}%"}
        mem = psutil.virtual_memory()
        data["memory"] = {
            "total": format_size(mem.total),
            "used": format_size(mem.used),
            "available": format_size(mem.available),
            "usage": f"{mem.percent}%",
        }
        disk = psutil.disk_usage("/")
        data["disk"] = {
            "total": format_size(disk.total),
            "used": format_size(disk.used),
            "free": format_size(disk.free),
            "usage": f"{disk.percent}%",
        }

    if getattr(args, "json_output", False):
        print(_json.dumps(data, indent=2))
        return

    log.header("System Information")
    log.section("Platform")
    log.key_value("Platform", data["platform"])
    log.key_value("Python", data["python"])
    log.key_value("Machine", data["machine"])

    if psutil and "cpu" in data:
        log.section("CPU")
        log.key_value("Cores", data["cpu"]["cores"])
        log.key_value("Usage", data["cpu"]["usage"])
        log.section("Memory")
        log.key_value("Total", data["memory"]["total"])
        log.key_value("Used", data["memory"]["used"])
        log.key_value("Available", data["memory"]["available"])
        log.key_value("Usage", data["memory"]["usage"])
        log.section("Disk")
        log.key_value("Total", data["disk"]["total"])
        log.key_value("Used", data["disk"]["used"])
        log.key_value("Free", data["disk"]["free"])
        log.key_value("Usage", data["disk"]["usage"])


def cmd_status(args):
    """Show system status with optional watch mode."""
    import json as _json
    import requests

    timeout = getattr(args, "timeout", 10)

    def _get_status():
        data = {}
        try:
            r = requests.get("http://localhost:8000/health", timeout=timeout)
            data["api"] = {"status": "online" if r.status_code == 200 else "offline", "code": r.status_code}
        except (requests.RequestException, ConnectionError):
            data["api"] = {"status": "not running", "code": 0}

        models_dir = Path("models")
        if models_dir.exists():
            models = list(models_dir.rglob("*.soul"))
            data["models"] = {"count": len(models), "status": "ok"}
        else:
            data["models"] = {"count": 0, "status": "not found"}

        datasets_dir = Path("data")
        if datasets_dir.exists():
            datasets = list(datasets_dir.iterdir())
            data["datasets"] = {"count": len([d for d in datasets if d.is_dir()]), "status": "ok"}
        else:
            data["datasets"] = {"count": 0, "status": "not found"}

        return data

    if args.watch:
        try:
            while True:
                data = _get_status()
                if getattr(args, "json_output", False):
                    print(_json.dumps(data, indent=2))
                else:
                    print("\033[2J\033[H")
                    log.header("SloughGPT Status")
                    log.status("API", data["api"]["status"], "ok" if data["api"]["code"] == 200 else "error")
                    log.status("Models", f"{data['models']['count']} found", data["models"]["status"])
                    log.status("Datasets", f"{data['datasets']['count']} found", data["datasets"]["status"])
                time.sleep(args.interval)
        except KeyboardInterrupt:
            log.blank()
            log.info("Stopped watching")
    else:
        data = _get_status()
        if getattr(args, "json_output", False):
            print(_json.dumps(data, indent=2))
        else:
            log.header("SloughGPT Status")
            log.status("API", data["api"]["status"], "ok" if data["api"]["code"] == 200 else "error")
            log.status("Models", f"{data['models']['count']} found", data["models"]["status"])
            log.status("Datasets", f"{data['datasets']['count']} found", data["datasets"]["status"])
            log.info("Use --watch to auto-refresh")


def cmd_optimize(args):
    """Show and configure optimization settings (SloNet accelerator + BLAS)."""
    log.header("Optimization System")

    log.section("Accelerator (SloNet)")
    try:
        from domains.training.slonet import _ACCEL_THRESHOLD, _get_accelerator
        acc = _get_accelerator()
        if acc is not None:
            log.key_value("Backend", acc.name)
            log.key_value("Device", getattr(acc, "device_name", str(acc)))
        else:
            log.key_value("Backend", "cpu (numpy)")
            log.key_value("Device", "CPU — no accelerator active")
    except Exception as e:
        log.key_value("Backend", "cpu (numpy)")
        log.warning(f"Accelerator probe failed: {e}")

    log.section("Available Optimizations")
    log.status("Accelerator Dispatch", "Yes (threshold-gated)", "ok")
    log.status("KV Cache", "Yes (greedy generation)", "ok")
    log.status("Accelerator Threshold", f"{_ACCEL_THRESHOLD} elements", "info")

    log.section("Training Optimizations")
    log.info("SloNet pure-numpy autograd:  no external dependencies")
    log.info("Metal/CUDA/OpenCL dispatch:   threshold-gated GEMM path")
    log.info("Gradient clipping:           SloAdam + SloReduceLROnPlateau")
    log.info("Deterministic inference:     accelerator disabled during generate()")

    log.section("Inference Optimizations")
    log.info("KV Cache:                      Skip recomputation (greedy path)")
    log.info("Prompt Caching:                Reuse computed states")
    log.info("MorphTokenizer:                Pure Python BPE, no Rust binary")

    if args.optimize:
        log.blank()
        log.step("Applying optimizations...")
        try:
            from domains.infrastructure.resource_manager import get_resource_manager
            rm = get_resource_manager()
            rm.apply_blas_env()
            rm.apply_compute_limits()
            log.success(f"Thread count optimized: compute={rm.compute_threads} io={rm.io_threads}")
            log.info(f"OMP={rm.omp_num_threads} MKL={rm.mkl_num_threads} NUMEXPR={rm.numexpr_num_threads}")
        except Exception as e:
            log.warning(f"Thread optimization failed: {e}")
        log.info("Accelerator dispatch is threshold-gated (no runtime change needed)")


def cmd_config_check(args):
    """Check environment setup."""
    import json as _json

    doctor = Doctor()
    result = doctor.run_all()

    checks = []
    for check in result.checks:
        checks.append({
            "name": check.name,
            "message": check.message,
            "passed": check.passed,
            "suggestion": check.suggestion,
        })

    data = {
        "checks": checks,
        "total": len(result.checks),
        "passed": result.passed,
        "failed_count": result.failed_count,
    }

    if getattr(args, "json_output", False):
        print(_json.dumps(data, indent=2))
        return

    log.header("Environment Check")
    log.blank()
    for check in result.checks:
        log.status(check.name, check.message, "ok" if check.passed else "error")
    log.blank()
    if result.passed:
        log.success(f"All {len(result.checks)} checks passed")
    else:
        log.error(f"{result.failed_count}/{len(result.checks)} checks failed")
        for check in result.checks:
            if not check.passed and check.suggestion:
                log.info(f"  -> {check.suggestion}")


def cmd_config_validate(args):
    """Validate environment configuration."""
    env_file = args.env
    issues = []
    warnings = []

    log.header("Configuration Validator")
    log.key_value("File", env_file)

    if not os.path.exists(env_file):
        log.warning(f"{env_file} not found")
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

    log.blank()
    log.section("Results")

    if issues:
        for issue in issues:
            log.error(issue)

    if warnings:
        for warn in warnings:
            log.warning(warn)

    if not issues and not warnings:
        log.success("Configuration looks good")

    log.info(f"{len(issues)} issues, {len(warnings)} warnings")


def cmd_config_generate(args):
    """Generate new secrets."""
    log.header("Secret Generator")

    if args.type in ["api-key", "all"]:
        api_key = secrets.token_urlsafe(32)
        log.command(f"SLO_API_KEY={api_key}")

    if args.type in ["jwt-secret", "all"]:
        jwt_secret = secrets.token_urlsafe(64)
        log.command(f"SLO_JWT_SECRET={jwt_secret}")

    if args.type == "all":
        encryption_key = secrets.token_hex(32)
        log.command(f"ENCRYPTION_KEY={encryption_key}")

    log.blank()
    log.info("Copy these to your .env file and restart the server")


def cmd_setup(args):
    """Setup SloughGPT environment."""
    import subprocess

    log.header("SloughGPT Setup")

    if not os.path.exists("cli.py"):
        log.error("Run from SloughGPT root directory")
        return

    log.section("Creating Directories")
    dirs = ["models", "data", "checkpoints", "experiments", "logs", "cache"]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        log.success(d)

    if not os.path.exists(".env") and os.path.exists(".env.example"):
        subprocess.run(["cp", ".env.example", ".env"])
        log.success(".env created")

    log.section("Environment")
    log.key_value("Python", platform.python_version())

    try:
        from domains.training.slonet import _get_accelerator
        acc = _get_accelerator()
        backend = acc.name if acc is not None else "cpu"
        log.key_value("SloNet Accelerator", backend)
        log.key_value("Device", getattr(acc, "device_name", "CPU") if acc is not None else "CPU")
    except Exception as e:
        log.warning(f"Accelerator probe failed: {e}")

    if not args.docker_only:
        log.section("Virtual Environment")
        venv_dir = args.venv
        if not os.path.exists(venv_dir):
            subprocess.run([sys.executable, "-m", "venv", venv_dir])
            log.success(f"Created {venv_dir}")

        pip_exe = os.path.join(venv_dir, "bin", "pip")
        log.info("Installing dependencies...")
        subprocess.run([pip_exe, "install", "--upgrade", "pip"])
        subprocess.run([pip_exe, "install", "transformers", "fastapi", "uvicorn", "pydantic"])
        log.success("Dependencies installed")

    if not args.local_only:
        log.section("Docker")
        compose = Path("infra/docker/docker-compose.yml")
        if compose.is_file():
            log.success("docker-compose.yml found")
            log.info("Run: docker compose -f infra/docker/docker-compose.yml up -d")
        else:
            log.warning("docker-compose.yml not found")

    log.blank()
    log.success("Setup complete!")
    log.info("Next: source .venv/bin/activate")
    log.info("Then: python3 cli.py dev")


def cmd_stats(args):
    """Show training and model statistics."""
    import json as _json

    models_dir = Path("models")
    model_count = 0
    total_size = 0
    if models_dir.exists():
        for f in list(models_dir.glob("*.soul")) + list(models_dir.glob("*.safetensors")):
            model_count += 1
            total_size += f.stat().st_size

    datasets_dir = Path("datasets")
    ds_count = 0
    ds_size = 0
    if datasets_dir.exists():
        for f in datasets_dir.rglob("*"):
            if f.is_file():
                ds_count += 1
                ds_size += f.stat().st_size

    ckpt_dir = Path("checkpoints")
    ckpt_count = 0
    if ckpt_dir.exists():
        ckpt_count = len(list(ckpt_dir.glob("*.soul")))

    exp_count = 0
    exp_file = Path("data/experiments/experiments.json")
    if exp_file.exists():
        with open(exp_file) as f:
            experiments = json.load(f)
        exp_count = len(experiments)

    data = {
        "models": {"count": model_count, "total_size": format_size(total_size)},
        "datasets": {"files": ds_count, "total_size": format_size(ds_size)},
        "checkpoints": {"saved": ckpt_count},
        "experiments": {"total": exp_count},
    }

    if getattr(args, "json_output", False):
        print(_json.dumps(data, indent=2))
        return

    log.header("SloughGPT Statistics")
    log.section("Models")
    log.key_value("Count", str(model_count))
    log.key_value("Total Size", format_size(total_size))
    log.section("Datasets")
    log.key_value("Files", str(ds_count))
    log.key_value("Total Size", format_size(ds_size))
    log.section("Checkpoints")
    log.key_value("Saved", str(ckpt_count))
    if exp_count > 0:
        log.section("Experiments")
        log.key_value("Total", str(exp_count))


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
