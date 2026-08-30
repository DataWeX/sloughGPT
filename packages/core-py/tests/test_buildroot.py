"""
Tests for Buildroot infrastructure - validates configuration files and structure.
"""
import pytest
from pathlib import Path
import os


# Paths — test is at packages/core-py/tests/, repo root is 3 levels up
REPO_ROOT = Path(__file__).resolve().parents[3]
BUILDROOT_DIR = REPO_ROOT / "buildroot"
CONFIGS_DIR = BUILDROOT_DIR / "configs"
OVERLAYS_DIR = BUILDROOT_DIR / "overlays"
PACKAGES_DIR = BUILDROOT_DIR / "packages"


class TestBuildrootStructure:
    """Test that the buildroot directory structure is correct."""

    def test_buildroot_dir_exists(self):
        assert BUILDROOT_DIR.exists(), "buildroot/ directory must exist"

    def test_configs_dir_exists(self):
        assert CONFIGS_DIR.exists(), "buildroot/configs/ directory must exist"

    def test_overlays_dir_exists(self):
        assert OVERLAYS_DIR.exists(), "buildroot/overlays/ directory must exist"

    def test_packages_dir_exists(self):
        assert PACKAGES_DIR.exists(), "buildroot/packages/ directory must exist"

    def test_readme_exists(self):
        readme = BUILDROOT_DIR / "README.md"
        assert readme.exists(), "buildroot/README.md must exist"

    def test_post_build_script_exists(self):
        script = BUILDROOT_DIR / "post-build.sh"
        assert script.exists(), "buildroot/post-build.sh must exist"
        assert os.access(script, os.X_OK), "post-build.sh must be executable"

    def test_post_image_script_exists(self):
        script = BUILDROOT_DIR / "post-image.sh"
        assert script.exists(), "buildroot/post-image.sh must exist"
        assert os.access(script, os.X_OK), "post-image.sh must be executable"


class TestBuildrootConfigs:
    """Test Buildroot configuration files."""

    def test_defconfig_exists(self):
        defconfig = CONFIGS_DIR / "sloughgpt_defconfig"
        assert defconfig.exists(), "sloughgpt_defconfig must exist"

    def test_defconfig_content(self):
        defconfig = CONFIGS_DIR / "sloughgpt_defconfig"
        content = defconfig.read_text()
        assert "BR2_x86_64=y" in content, "Must target x86_64"
        assert "BR2_LINUX_KERNEL=y" in content, "Must include Linux kernel"
        assert "BR2_PACKAGE_BUSYBOX=y" in content, "Must include BusyBox"

    def test_linux_defconfig_exists(self):
        defconfig = CONFIGS_DIR / "linux-soughgpt.defconfig"
        assert defconfig.exists(), "linux-soughgpt.defconfig must exist"

    def test_linux_defconfig_content(self):
        defconfig = CONFIGS_DIR / "linux-soughgpt.defconfig"
        content = defconfig.read_text()
        assert "CONFIG_SYSVIPC=y" in content, "Must enable IPC"
        assert "CONFIG_EXT2_FS=y" in content, "Must enable ext2 filesystem"
        assert "CONFIG_VT=y" in content, "Must enable virtual terminal"

    def test_busybox_defconfig_exists(self):
        defconfig = CONFIGS_DIR / "busybox-soughgpt.defconfig"
        assert defconfig.exists(), "busybox-soughgpt.defconfig must exist"

    def test_busybox_defconfig_content(self):
        defconfig = CONFIGS_DIR / "busybox-soughgpt.defconfig"
        content = defconfig.read_text()
        assert "CONFIG_ASH=y" in content, "Must enable ash shell"
        assert "CONFIG_CAT=y" in content, "Must include cat"
        assert "CONFIG_LS=y" in content, "Must include ls"


class TestOverlays:
    """Test rootfs overlay structure."""

    def test_etc_dir_exists(self):
        etc_dir = OVERLAYS_DIR / "etc"
        assert etc_dir.exists(), "overlays/etc/ must exist"

    def test_init_d_dir_exists(self):
        init_d = OVERLAYS_DIR / "etc" / "init.d"
        assert init_d.exists(), "overlays/etc/init.d/ must exist"

    def test_init_script_exists(self):
        init_script = OVERLAYS_DIR / "etc" / "init.d" / "S00setup"
        assert init_script.exists(), "S00setup init script must exist"
        assert os.access(init_script, os.X_OK), "S00setup must be executable"

    def test_fstab_exists(self):
        fstab = OVERLAYS_DIR / "etc" / "fstab"
        assert fstab.exists(), "fstab must exist"

    def test_fstab_content(self):
        fstab = OVERLAYS_DIR / "etc" / "fstab"
        content = fstab.read_text()
        assert "proc" in content, "Must mount proc"
        assert "sysfs" in content, "Must mount sysfs"
        assert "devpts" in content, "Must mount devpts"

    def test_profile_exists(self):
        profile = OVERLAYS_DIR / "etc" / "profile"
        assert profile.exists(), "profile must exist"

    def test_profile_content(self):
        profile = OVERLAYS_DIR / "etc" / "profile"
        content = profile.read_text()
        assert "PATH=" in content, "Must set PATH"
        assert "PS1=" in content, "Must set prompt"
        assert "sloughgpt" in content.lower(), "Must reference sloughgpt"


class TestPackages:
    """Test custom package structure."""

    def test_sloughgpt_package_dir_exists(self):
        pkg_dir = PACKAGES_DIR / "sloughgpt"
        assert pkg_dir.exists(), "packages/sloughgpt/ must exist"

    def test_makefile_exists(self):
        makefile = PACKAGES_DIR / "sloughgpt" / "sloughgpt.mk"
        assert makefile.exists(), "sloughgpt.mk must exist"

    def test_makefile_content(self):
        makefile = PACKAGES_DIR / "sloughgpt" / "sloughgpt.mk"
        content = makefile.read_text()
        assert "SLOUGHPGT_VERSION" in content, "Must define version"
        assert "SLOUGHPGT_LICENSE" in content, "Must define license"
        assert "python3" in content.lower() or "PYTHON3" in content, "Must depend on Python3"

    def test_config_in_exists(self):
        config_in = PACKAGES_DIR / "sloughgpt" / "Config.in"
        assert config_in.exists(), "Config.in must exist"

    def test_config_in_content(self):
        config_in = PACKAGES_DIR / "sloughgpt" / "Config.in"
        content = config_in.read_text()
        assert "BR2_PACKAGE_SLOUGHPGT" in content, "Must define package config"
        assert "help" in content, "Must provide help text"


class TestV86Integration:
    """Test v86 integration files."""

    def test_v86_controller_exists(self):
        controller = REPO_ROOT / "apps" / "web" / "lib" / "v86-controller.ts"
        assert controller.exists(), "v86-controller.ts must exist"

    def test_v86_hook_exists(self):
        hook = REPO_ROOT / "apps" / "web" / "hooks" / "useV86.ts"
        assert hook.exists(), "useV86.ts must exist"

    def test_v86_hook_has_options(self):
        hook = REPO_ROOT / "apps" / "web" / "hooks" / "useV86.ts"
        content = hook.read_text()
        assert "UseV86Options" in content, "Must define UseV86Options interface"
        assert "imageUrl" in content, "Must support custom imageUrl"
        assert "imageSize" in content, "Must support custom imageSize"


class TestCLICommand:
    """Test CLI build command."""

    def test_build_command_exists(self):
        build_cmd = REPO_ROOT / "apps" / "cli" / "src" / "commands" / "build.py"
        assert build_cmd.exists(), "build.py CLI command must exist"

    def test_build_command_has_functions(self):
        build_cmd = REPO_ROOT / "apps" / "cli" / "src" / "commands" / "build.py"
        content = build_cmd.read_text()
        assert "def cmd_build" in content, "Must have cmd_build function"
        assert "def cmd_build_clean" in content, "Must have cmd_build_clean function"
        assert "def cmd_build_status" in content, "Must have cmd_build_status function"
        assert "def cmd_build_install" in content, "Must have cmd_build_install function"

    def test_build_command_has_commands_dict(self):
        build_cmd = REPO_ROOT / "apps" / "cli" / "src" / "commands" / "build.py"
        content = build_cmd.read_text()
        assert "COMMANDS = {" in content, "Must have COMMANDS dict"


class TestInitScript:
    """Test init script functionality."""

    def test_init_script_mounts_proc(self):
        init_script = OVERLAYS_DIR / "etc" / "init.d" / "S00setup"
        content = init_script.read_text()
        assert "mount -t proc" in content, "Must mount proc filesystem"

    def test_init_script_mounts_sysfs(self):
        init_script = OVERLAYS_DIR / "etc" / "init.d" / "S00setup"
        content = init_script.read_text()
        assert "mount -t sysfs" in content, "Must mount sysfs"

    def test_init_script_sets_hostname(self):
        init_script = OVERLAYS_DIR / "etc" / "init.d" / "S00setup"
        content = init_script.read_text()
        assert "hostname" in content, "Must set hostname"

    def test_init_script_configures_network(self):
        init_script = OVERLAYS_DIR / "etc" / "init.d" / "S00setup"
        content = init_script.read_text()
        assert "ifconfig" in content or "ip" in content, "Must configure network"
