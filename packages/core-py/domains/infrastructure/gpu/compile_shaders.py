#!/usr/bin/env python3
"""
compile_shaders.py — Compile WGSL shaders to platform binaries.

Outputs:
  - SPIR-V (.spv) for Vulkan
  - HLSL (.hlsl) for DX12 (via naga)
  - Metal shading language (.metal) for Metal (via naga)

Uses naga (https://github.com/gfx-rs/naga) as build-time tool.
Install: cargo install naga-cli

Usage:
    python3 compile_shaders.py [--naga PATH]
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

SHADERS_DIR = Path(__file__).parent / "shaders"
OUTPUT_DIR = Path(__file__).parent / "shaders"

# Shaders that need compilation
COMPUTE_SHADERS = [
    "matmul",
    "softmax",
    "rmsnorm",
    "rope",
    "silu",
    "gelu",
]


def find_naga() -> str:
    """Find naga CLI binary."""
    # Check PATH
    for p in os.environ.get("PATH", "").split(":"):
        naga = Path(p) / "naga"
        if naga.exists():
            return str(naga)
        naga = Path(p) / "naga-cli"
        if naga.exists():
            return str(naga)

    # Check cargo install location
    cargo_home = Path(os.environ.get("CARGO_HOME", Path.home() / ".cargo"))
    naga = cargo_home / "bin" / "naga"
    if naga.exists():
        return str(naga)

    return "naga"  # Hope it's in PATH


def compile_spirv(naga: str, wgsl_path: Path, out_path: Path) -> bool:
    """Compile WGSL → SPIR-V."""
    cmd = [naga, str(wgsl_path), str(out_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  SPIR-V FAILED: {wgsl_path.name}")
        print(f"  {result.stderr.strip()}")
        return False
    print(f"  SPIR-V: {out_path.name} ({out_path.stat().st_size} bytes)")
    return True


def compile_hlsl(naga: str, wgsl_path: Path, out_path: Path) -> bool:
    """Compile WGSL → HLSL (DX12)."""
    cmd = [naga, str(wgsl_path), str(out_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  HLSL FAILED: {wgsl_path.name}")
        print(f"  {result.stderr.strip()}")
        return False
    print(f"  HLSL: {out_path.name} ({out_path.stat().st_size} bytes)")
    return True


def compile_msl(naga: str, wgsl_path: Path, out_path: Path) -> bool:
    """Compile WGSL → Metal Shading Language."""
    cmd = [naga, str(wgsl_path), str(out_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  MSL FAILED: {wgsl_path.name}")
        print(f"  {result.stderr.strip()}")
        return False
    print(f"  MSL: {out_path.name} ({out_path.stat().st_size} bytes)")
    return True


def main():
    parser = argparse.ArgumentParser(description="Compile WGSL shaders")
    parser.add_argument("--naga", default=None, help="Path to naga binary")
    parser.add_argument("--spirv-only", action="store_true", help="Only compile SPIR-V")
    parser.add_argument("--hlsl-only", action="store_true", help="Only compile HLSL")
    parser.add_argument("--msl-only", action="store_true", help="Only compile MSL")
    args = parser.parse_args()

    naga = args.naga or find_naga()
    print(f"Using naga: {naga}")

    OUTPUT_DIR.mkdir(exist_ok=True)

    ok = True
    for name in COMPUTE_SHADERS:
        wgsl = SHADERS_DIR / f"{name}.wgsl"
        if not wgsl.exists():
            print(f"SKIP: {wgsl} not found")
            continue

        print(f"\nCompiling {name}.wgsl:")

        if not args.hlsl_only and not args.msl_only:
            spv_out = OUTPUT_DIR / f"{name}.spv"
            if not compile_spirv(naga, wgsl, spv_out):
                ok = False

        if not args.spirv_only and not args.msl_only:
            hlsl_out = OUTPUT_DIR / f"{name}.hlsl"
            if not compile_hlsl(naga, wgsl, hlsl_out):
                ok = False

        if not args.spirv_only and not args.hlsl_only:
            msl_out = OUTPUT_DIR / f"{name}.metal"
            if not compile_msl(naga, wgsl, msl_out):
                ok = False

    if ok:
        print(f"\n✓ All shaders compiled to {OUTPUT_DIR}")
    else:
        print(f"\n✗ Some shaders failed to compile")
        sys.exit(1)


if __name__ == "__main__":
    main()
