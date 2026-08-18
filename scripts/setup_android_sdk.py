#!/usr/bin/env python3
"""Set up Android SDK for building React Native apps.

Downloads Android command-line tools via downcraft (resumable),
installs required SDK packages, and writes an envrc snippet.

Usage:
    python3 scripts/setup_android_sdk.py
    source ~/.android-sdk-env   # after first run
"""

import os
import platform
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "packages" / "core-py"))
sys.path.insert(0, str(REPO_ROOT / "packages" / "downcraft"))
SDK_ROOT = Path.home() / "android-sdk"
CMD_TOOLS_DIR = SDK_ROOT / "cmdline-tools" / "latest"
PLATFORM_DIR = SDK_ROOT / "platforms" / "android-35"
BUILD_TOOLS_DIR = SDK_ROOT / "build-tools" / "35.0.0"
ENV_FILE = Path.home() / ".android-sdk-env"

# Linux x86_64 — the only target on this machine
CMD_TOOLS_URL = (
    "https://dl.google.com/android/repository/commandlinetools-linux-11076708_latest.zip"
)
CMD_TOOLS_SHA = "1f28a6a41e2f5e3e4e9b3b8c6a5b3b8c"


def _progress(downloaded: int, total: int, speed: float) -> None:
    pct = downloaded / total * 100 if total else 0
    mb = downloaded / 1024 / 1024
    total_mb = total / 1024 / 1024 if total else 0
    speed_mb = speed / 1024 / 1024
    sys.stdout.write(f"\r  {mb:.1f}/{total_mb:.1f} MB ({pct:.0f}%) @ {speed_mb:.1f} MB/s")
    sys.stdout.flush()


def download_cmdtools() -> Path:
    """Download and extract Android command-line tools."""
    if CMD_TOOLS_DIR.exists():
        print("  Command-line tools already installed")
        return CMD_TOOLS_DIR

    zip_path = Path("/tmp/android-cmdtools.zip")

    print("Downloading Android command-line tools...")
    try:
        from downcraft import download
        download(
            url=CMD_TOOLS_URL,
            dest=zip_path,
            label="android-cmdtools",
            on_progress=_progress,
        )
        print()
    except ImportError:
        print("  downcraft not available, falling back to curl")
        subprocess.run(
            ["curl", "-L", "-o", str(zip_path), CMD_TOOLS_URL],
            check=True,
        )

    print("Extracting...")
    SDK_ROOT.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(SDK_ROOT / "cmdline-tools")
    # Rename to 'latest' as SDK expects
    extracted = SDK_ROOT / "cmdline-tools" / "cmdline-tools"
    if extracted.exists():
        extracted.rename(CMD_TOOLS_DIR)
    # Ensure executables are runnable
    for f in (CMD_TOOLS_DIR / "bin").iterdir():
        f.chmod(f.stat().st_mode | 0o111)
    zip_path.unlink(missing_ok=True)
    return CMD_TOOLS_DIR


def install_sdk_packages() -> None:
    """Install platform, build-tools, and platform-tools."""
    sdkmanager = CMD_TOOLS_DIR / "bin" / "sdkmanager"
    if not sdkmanager.exists():
        print(f"  sdkmanager not found at {sdkmanager}")
        return

    # Accept licenses first
    print("Accepting SDK licenses...")
    subprocess.run(
        [str(sdkmanager), "--licenses"],
        input=b"y\ny\ny\ny\ny\ny\ny\ny\n",
        capture_output=True,
    )

    packages = [
        "platform-tools",
        "platforms;android-35",
        "build-tools;35.0.0",
    ]
    for pkg in packages:
        target = pkg.split(";")[-1]
        print(f"Installing {target}...")
        subprocess.run(
            [str(sdkmanager), pkg],
            capture_output=True,
        )


def write_env_file() -> None:
    """Write shell environment file."""
    env_content = f"""# Android SDK environment — source this file
export ANDROID_HOME="{SDK_ROOT}"
export ANDROID_SDK_ROOT="{SDK_ROOT}"
export PATH="$ANDROID_HOME/cmdline-tools/latest/bin:$ANDROID_HOME/platform-tools:$ANDROID_HOME/build-tools/35.0.0:$PATH"
"""
    ENV_FILE.write_text(env_content)
    print(f"  Wrote {ENV_FILE}")


def write_envrc() -> None:
    """Write .envrc in repo root for direnv users."""
    envrc = REPO_ROOT / ".envrc"
    snippet = f"""
# Android SDK
export ANDROID_HOME="{SDK_ROOT}"
export ANDROID_SDK_ROOT="{SDK_ROOT}"
export PATH="$ANDROID_HOME/cmdline-tools/latest/bin:$ANDROID_HOME/platform-tools:$ANDROID_HOME/build-tools/35.0.0:$PATH"
""".strip()
    if envrc.exists():
        existing = envrc.read_text()
        if "ANDROID_HOME" not in existing:
            envrc.write_text(existing.rstrip() + "\n" + snippet + "\n")
            print(f"  Appended ANDROID_HOME to {envrc}")
    else:
        envrc.write_text(snippet + "\n")
        print(f"  Wrote {envrc}")


def verify() -> bool:
    """Verify SDK installation."""
    adb = SDK_ROOT / "platform-tools" / "adb"
    sdkmanager = CMD_TOOLS_DIR / "bin" / "sdkmanager"
    ok = True
    if not adb.exists():
        print(f"  Missing: {adb}")
        ok = False
    if not sdkmanager.exists():
        print(f"  Missing: {sdkmanager}")
        ok = False
    return ok


def main() -> None:
    print(f"=== Android SDK Setup ===")
    print(f"  SDK root: {SDK_ROOT}")
    print(f"  Java: {subprocess.getoutput('java -version 2>&1 | head -1')}")

    download_cmdtools()
    install_sdk_packages()
    write_env_file()
    write_envrc()

    if verify():
        print("\n  Android SDK ready.")
        print(f"  Run: source {ENV_FILE}")
    else:
        print("\n  Some packages failed to install.")
        sys.exit(1)


if __name__ == "__main__":
    main()
