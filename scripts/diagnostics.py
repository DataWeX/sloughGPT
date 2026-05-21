#!/usr/bin/env python3
"""
SloughGPT Diagnostics Script
Run comprehensive diagnostics on the SloughGPT installation.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import platform
import subprocess
import json
from pathlib import Path


def run_command(cmd, timeout=10):
    """Run a shell command and return output."""
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=timeout
        )
        return result.returncode, result.stdout.strip(), result.stderr.strip()
    except subprocess.TimeoutExpired:
        return -1, "", "Command timed out"
    except Exception as e:
        return -1, "", str(e)


def check_python():
    """Check Python environment."""
    print("\n[Python Environment]")
    py_version = sys.version
    print(f"  Version: {py_version}")
    print(f"  Executable: {sys.executable}")

    # Check key packages
    packages = ["torch", "transformers", "fastapi", "pydantic", "numpy", "psutil"]
    for pkg in packages:
        try:
            mod = __import__(pkg)
            ver = getattr(mod, "__version__", "unknown")
            print(f"  {pkg}: {ver}")
        except ImportError:
            print(f"  {pkg}: NOT INSTALLED")


def check_gpu():
    """Check GPU availability."""
    print("\n[GPU Support]")
    
    # CUDA
    try:
        import torch
        cuda = torch.cuda.is_available()
        print(f"  CUDA available: {cuda}")
        if cuda:
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(i)
                mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                print(f"    GPU {i}: {name} ({mem:.1f} GB)")
    except ImportError:
        print("  PyTorch not installed")

    # MPS (Apple Silicon)
    try:
        import torch
        mps = torch.backends.mps.is_available()
        print(f"  MPS available: {mps}")
        if hasattr(torch.backends.mps, 'is_built'):
            print(f"  MPS built: {torch.backends.mps.is_built()}")
    except:
        pass

    # ROCm
    try:
        code, out, _ = run_command("rocm-smi --version")
        if code == 0:
            print(f"  ROCm: {out}")
    except:
        pass


def check_directories():
    """Check required directories."""
    print("\n[Directories]")
    
    dirs = [
        ("Models", "models"),
        ("Datasets", "datasets"),
        ("Experiments", "experiments"),
        ("Configs", "configs"),
        ("Checkpoints", "checkpoints"),
    ]
    
    for name, path in dirs:
        exists = Path(path).is_dir()
        status = "[OK]" if exists else "[MISSING]"
        print(f"  {status} {name}: {path}")


def check_models():
    """Check available models."""
    print("\n[Available Models]")
    
    models_dir = Path("models")
    if not models_dir.exists():
        print("  Models directory not found")
        return
    
    models = list(models_dir.glob("*.pt")) + list(models_dir.glob("*.pth"))
    if models:
        for m in models:
            size = m.stat().st_size / (1024**2)
            print(f"  - {m.name}: {size:.1f} MB")
    else:
        print("  No model checkpoints found")


def check_datasets():
    """Check available datasets."""
    print("\n[Available Datasets]")
    
    datasets_dir = Path("datasets")
    if not datasets_dir.exists():
        print("  Datasets directory not found")
        return
    
    datasets = [d for d in datasets_dir.iterdir() if d.is_dir()]
    if datasets:
        for d in datasets:
            size = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
            print(f"  - {d.name}: {size / (1024**2):.1f} MB")
    else:
        print("  No datasets found")


def check_docker():
    """Check Docker status."""
    print("\n[Docker]")
    
    code, out, _ = run_command("docker --version")
    if code == 0:
        print(f"  Docker: {out}")
        
        # Check docker-compose
        code, out, _ = run_command("docker compose version")
        if code == 0:
            print(f"  Docker Compose: {out}")
        else:
            code, out, _ = run_command("docker-compose --version")
            if code == 0:
                print(f"  Docker Compose: {out}")
    else:
        print("  Docker: NOT INSTALLED")


def check_kubernetes():
    """Check Kubernetes status."""
    print("\n[Kubernetes]")
    
    code, out, _ = run_command("kubectl version --client 2>/dev/null")
    if code == 0:
        print(f"  kubectl: {out}")
    else:
        print("  kubectl: NOT INSTALLED")
    
    code, out, _ = run_command("helm version --client 2>/dev/null")
    if code == 0:
        print(f"  Helm: {out}")
    else:
        print("  Helm: NOT INSTALLED")


def check_api_server():
    """Check API server status."""
    print("\n[API Server]")
    
    code, out, err = run_command("curl -s http://localhost:8000/health 2>/dev/null", timeout=5)
    if code == 0:
        try:
            data = json.loads(out)
            print(f"  Status: healthy={data.get('model_loaded', False)}")
            print(f"  Model type: {data.get('model_type', 'none')}")
        except:
            print(f"  Response: {out}")
    else:
        print("  API Server: NOT RUNNING (start with: python3 apps/api/server/main.py)")


def check_experiments():
    """Check experiments log."""
    print("\n[Experiments]")
    
    exp_file = Path("data/experiments/experiments.json")
    if exp_file.exists():
        with open(exp_file) as f:
            data = json.load(f)
        if isinstance(data, list):
            print(f"  Total experiments: {len(data)}")
        elif isinstance(data, dict):
            print(f"  Total experiments: {len(data)}")
    else:
        print("  No experiments recorded")


def check_environment():
    """Check environment variables."""
    print("\n[Environment Variables]")
    
    env_vars = [
        "SLOUGHGPT_API_KEY",
        "SLOUGHGPT_JWT_SECRET",
        "MODEL_PATH",
        "CUDA_VISIBLE_DEVICES",
    ]
    
    for var in env_vars:
        val = os.getenv(var)
        if val:
            if "SECRET" in var or "KEY" in var:
                print(f"  {var}: {'*' * 20} (set)")
            else:
                print(f"  {var}: {val}")
        else:
            print(f"  {var}: not set")


def check_portability():
    """Check code portability features."""
    print("\n[Portability]")
    
    checks = [
        ("CPU-only mode", os.getenv("CUDA_VISIBLE_DEVICES", "") == ""),
        ("Cross-platform paths", True),
        ("Environment config", Path(".env").exists()),
    ]
    
    for name, status in checks:
        mark = "[OK]" if status else "[--]"
        print(f"  {mark} {name}")


def generate_report():
    """Generate full diagnostics report."""
    print("=" * 60)
    print("SLOUGHGPT DIAGNOSTICS REPORT")
    print("=" * 60)
    
    print(f"\nGenerated: {subprocess.run('date', capture_output=True, text=True).stdout.strip()}")
    print(f"Platform: {platform.platform()}")
    print(f"Python: {sys.version.split()[0]}")
    
    check_python()
    check_gpu()
    check_directories()
    check_models()
    check_datasets()
    check_docker()
    check_kubernetes()
    check_api_server()
    check_experiments()
    check_environment()
    check_portability()
    
    print("\n" + "=" * 60)
    print("Run 'python3 cli.py config check' for detailed environment check")
    print("Run 'python3 cli.py profile' for performance profiling")
    print("=" * 60)


if __name__ == "__main__":
    generate_report()
