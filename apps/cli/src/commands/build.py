"""
Build commands - Buildroot image building and management.
"""
import sys
import os
import subprocess
import shutil
from pathlib import Path
from typing import Optional

from domains.logging import get_global

log = get_global()

# Paths relative to repo root
BUILDROOT_DIR = Path("buildroot")
BUILDROOT_REPO = "https://github.com/buildroot/buildroot.git"
BUILDROOT_BRANCH = "2024.02.x"
OUTPUT_IMAGES = BUILDROOT_DIR / "output" / "images"
V86_IMAGE = "buildroot.img"


def cmd_build(args):
    """Build a Buildroot image for v86 browser VM."""
    log.header("SloughGPT Buildroot Builder")

    # Check if Buildroot exists
    if not BUILDROOT_DIR.exists():
        log.warning("Buildroot directory not found")
        log.info("Initializing Buildroot...")
        _init_buildroot()

    # Check if defconfig exists
    defconfig = BUILDROOT_DIR / "configs" / "sloughgpt_defconfig"
    if not defconfig.exists():
        log.error("Defconfig not found at buildroot/configs/sloughgpt_defconfig")
        return

    log.section("Build Configuration")
    log.key_value("Buildroot", str(BUILDROOT_DIR))
    log.key_value("Defconfig", str(defconfig))
    log.key_value("Target", "x86_64")

    # Configure
    log.section("Configuring")
    _run_buildroot_cmd(["make", "sloughgpt_defconfig"])

    # Build
    log.section("Building")
    _run_buildroot_cmd(["make", "-j$(nproc)"])

    # Check output
    image = OUTPUT_IMAGES / "rootfs.ext4"
    if not image.exists():
        log.error(f"Build failed - image not found: {image}")
        return

    log.success(f"Build complete: {image}")

    # Create v86 compatible image
    v86_path = OUTPUT_IMAGES / V86_IMAGE
    log.section("Creating v86 Image")
    shutil.copy2(image, v86_path)
    log.success(f"v86 image: {v86_path}")

    # Copy to web public directory
    web_public = Path("apps/web/public/buildroot")
    web_public.mkdir(parents=True, exist_ok=True)
    shutil.copy2(v86_path, web_public / V86_IMAGE)
    log.success(f"Copied to {web_public / V86_IMAGE}")

    log.blank()
    log.success("Build complete!")
    log.info("To use custom image, update apps/web/hooks/useV86.ts")
    log.info("Or run: slooughgpt build --install")


def cmd_build_init(args):
    """Initialize the Buildroot build environment."""
    log.header("Initializing Buildroot")

    clean = getattr(args, "clean", False)

    build_script = BUILDROOT_DIR / "build.sh"
    if not build_script.exists():
        log.error(f"Build script not found: {build_script}")
        return

    if clean:
        log.section("Cleaning first")
        result = subprocess.run(
            ["bash", str(build_script), "clean"],
            capture_output=False,
        )
        if result.returncode != 0:
            log.warning("Clean returned non-zero (may be ok)")

    log.section("Running setup")
    result = subprocess.run(
        ["bash", str(build_script), "setup"],
        capture_output=False,
    )

    if result.returncode == 0:
        log.success("Build environment initialized")
    else:
        log.error(f"Setup failed with exit code {result.returncode}")


def cmd_build_clean(args):
    """Clean Buildroot build directory."""
    log.header("Cleaning Buildroot")

    output_dir = BUILDROOT_DIR / "output"
    if output_dir.exists():
        shutil.rmtree(output_dir)
        log.success("Cleaned output directory")
    else:
        log.info("Output directory not found, nothing to clean")


def cmd_build_status(args):
    """Show Buildroot build status."""
    log.header("Buildroot Status")

    if not BUILDROOT_DIR.exists():
        log.status("Buildroot", "Not initialized", "error")
        log.info("Run: slooughgpt build init")
        return

    log.status("Buildroot", "Initialized", "ok")

    # Check for image
    image = OUTPUT_IMAGES / "rootfs.ext4"
    if image.exists():
        size = image.stat().st_size
        log.status("Image", f"Built ({_format_size(size)})", "ok")
    else:
        log.status("Image", "Not built", "warning")

    # Check for v86 image
    v86_img = OUTPUT_IMAGES / V86_IMAGE
    if v86_img.exists():
        log.status("v86 Image", "Ready", "ok")
    else:
        log.status("v86 Image", "Not created", "warning")


def cmd_build_install(args):
    """Install built image to web public directory."""
    log.header("Installing v86 Image")

    v86_img = OUTPUT_IMAGES / V86_IMAGE
    if not v86_img.exists():
        log.error("v86 image not found. Run 'slooughgpt build' first.")
        return

    web_public = Path("apps/web/public/buildroot")
    web_public.mkdir(parents=True, exist_ok=True)

    dest = web_public / V86_IMAGE
    shutil.copy2(v86_img, dest)
    log.success(f"Installed to: {dest}")
    log.info("Update apps/web/hooks/useV86.ts to use custom image")


def _init_buildroot():
    """Initialize Buildroot by cloning the repository."""
    log.info(f"Cloning Buildroot {BUILDROOT_BRANCH}...")
    subprocess.run(
        ["git", "clone", "--depth", "1", "--branch", BUILDROOT_BRANCH,
         BUILDROOT_REPO, str(BUILDROOT_DIR)],
        check=True
    )
    log.success("Buildroot cloned")


def _run_buildroot_cmd(cmd):
    """Run a command in the Buildroot directory."""
    result = subprocess.run(
        cmd,
        cwd=BUILDROOT_DIR,
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        log.error(f"Command failed: {' '.join(cmd)}")
        log.error(result.stderr)
        raise subprocess.CalledProcessError(result.returncode, cmd)
    return result


def _format_size(size_bytes):
    """Format bytes to human readable size."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


# CLI registration
COMMANDS = {
    "build": {
        "func": cmd_build,
        "help": "Build a Buildroot image for v86 browser VM",
        "subcommands": {
            "init": cmd_build_init,
            "clean": cmd_build_clean,
            "status": cmd_build_status,
            "install": cmd_build_install,
        }
    }
}
