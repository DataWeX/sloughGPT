"""
Tests for new CLI commands - build and vm.
"""
import pytest
from pathlib import Path


# Paths
REPO_ROOT = Path(__file__).resolve().parents[3]


class TestBuildCommand:
    """Test build CLI command."""

    def test_build_command_exists(self):
        build_cmd = REPO_ROOT / "apps" / "cli" / "src" / "commands" / "build.py"
        assert build_cmd.exists(), "build.py must exist"

    def test_build_command_imports(self):
        build_cmd = REPO_ROOT / "apps" / "cli" / "src" / "commands" / "build.py"
        content = build_cmd.read_text()
        assert "from domains.logging import get_global" in content

    def test_build_command_has_main_function(self):
        build_cmd = REPO_ROOT / "apps" / "cli" / "src" / "commands" / "build.py"
        content = build_cmd.read_text()
        assert "def cmd_build(args):" in content
        assert "def cmd_build_clean(args):" in content
        assert "def cmd_build_status(args):" in content
        assert "def cmd_build_install(args):" in content

    def test_build_command_has_commands_dict(self):
        build_cmd = REPO_ROOT / "apps" / "cli" / "src" / "commands" / "build.py"
        content = build_cmd.read_text()
        assert 'COMMANDS = {' in content
        assert '"build"' in content

    def test_build_command_has_subcommands(self):
        build_cmd = REPO_ROOT / "apps" / "cli" / "src" / "commands" / "build.py"
        content = build_cmd.read_text()
        assert '"clean"' in content
        assert '"status"' in content
        assert '"install"' in content


class TestVMCommand:
    """Test vm CLI command."""

    def test_vm_command_exists(self):
        vm_cmd = REPO_ROOT / "apps" / "cli" / "src" / "commands" / "vm.py"
        assert vm_cmd.exists(), "vm.py must exist"

    def test_vm_command_imports(self):
        vm_cmd = REPO_ROOT / "apps" / "cli" / "src" / "commands" / "vm.py"
        content = vm_cmd.read_text()
        assert "from domains.logging import get_global" in content

    def test_vm_command_has_main_function(self):
        vm_cmd = REPO_ROOT / "apps" / "cli" / "src" / "commands" / "vm.py"
        content = vm_cmd.read_text()
        assert "def cmd_vm(args):" in content
        assert "def cmd_vm_run(args):" in content
        assert "def cmd_vm_list(args):" in content
        assert "def cmd_vm_info(args):" in content

    def test_vm_command_has_commands_dict(self):
        vm_cmd = REPO_ROOT / "apps" / "cli" / "src" / "commands" / "vm.py"
        content = vm_cmd.read_text()
        assert 'COMMANDS = {' in content
        assert '"vm"' in content

    def test_vm_command_has_subcommands(self):
        vm_cmd = REPO_ROOT / "apps" / "cli" / "src" / "commands" / "vm.py"
        content = vm_cmd.read_text()
        assert '"run"' in content
        assert '"list"' in content
        assert '"info"' in content


class TestBuildrootDockerfile:
    """Test Buildroot Dockerfile."""

    def test_dockerfile_exists(self):
        dockerfile = REPO_ROOT / "buildroot" / "Dockerfile"
        assert dockerfile.exists(), "Dockerfile must exist"

    def test_dockerfile_has_base_image(self):
        dockerfile = REPO_ROOT / "buildroot" / "Dockerfile"
        content = dockerfile.read_text()
        assert "FROM ubuntu:22.04" in content

    def test_dockerfile_has_build_deps(self):
        dockerfile = REPO_ROOT / "buildroot" / "Dockerfile"
        content = dockerfile.read_text()
        assert "build-essential" in content
        assert "gcc" in content
        assert "make" in content

    def test_dockerfile_has_build_command(self):
        dockerfile = REPO_ROOT / "buildroot" / "Dockerfile"
        content = dockerfile.read_text()
        assert "sloughgpt_defconfig" in content


class TestBuildrootBuildScript:
    """Test buildroot build script."""

    def test_build_script_exists(self):
        script = REPO_ROOT / "buildroot" / "build.sh"
        assert script.exists(), "build.sh must exist"

    def test_build_script_executable(self):
        import os
        script = REPO_ROOT / "buildroot" / "build.sh"
        assert os.access(script, os.X_OK), "build.sh must be executable"

    def test_build_script_has_commands(self):
        script = REPO_ROOT / "buildroot" / "build.sh"
        content = script.read_text()
        assert "build_image()" in content
        assert "clean_build()" in content
        assert "build_shell()" in content
        assert "build_status()" in content

    def test_build_script_has_docker_commands(self):
        script = REPO_ROOT / "buildroot" / "build.sh"
        content = script.read_text()
        assert "docker build" in content
        assert "docker run" in content


class TestMakefileBuildrootTargets:
    """Test Makefile buildroot targets."""

    def test_makefile_has_buildroot_target(self):
        makefile = REPO_ROOT / "Makefile"
        content = makefile.read_text()
        assert "buildroot:" in content

    def test_makefile_has_buildroot_clean(self):
        makefile = REPO_ROOT / "Makefile"
        content = makefile.read_text()
        assert "buildroot-clean:" in content

    def test_makefile_has_buildroot_status(self):
        makefile = REPO_ROOT / "Makefile"
        content = makefile.read_text()
        assert "buildroot-status:" in content


class TestDaitInitScript:
    """Test Dait init script."""

    def test_init_script_exists(self):
        init_script = REPO_ROOT / "buildroot" / "overlays" / "etc" / "init.d" / "S99dait"
        assert init_script.exists(), "S99dait must exist"

    def test_init_script_executable(self):
        import os
        init_script = REPO_ROOT / "buildroot" / "overlays" / "etc" / "init.d" / "S99dait"
        assert os.access(init_script, os.X_OK), "S99dait must be executable"

    def test_init_script_has_runlevels(self):
        init_script = REPO_ROOT / "buildroot" / "overlays" / "etc" / "init.d" / "S99dait"
        content = init_script.read_text()
        assert "case" in content
        assert "0|1)" in content
        assert "2)" in content
        assert "3|4|5)" in content
        assert "6)" in content

    def test_init_script_starts_services(self):
        init_script = REPO_ROOT / "buildroot" / "overlays" / "etc" / "init.d" / "S99dait"
        content = init_script.read_text()
        assert "_start_kernel" in content
        assert "_start_shell" in content
        assert "_start_network" in content


class TestDaitConfig:
    """Test Dait configuration file."""

    def test_config_exists(self):
        config = REPO_ROOT / "buildroot" / "overlays" / "opt" / "sloughgpt" / "etc" / "dait.conf"
        assert config.exists(), "dait.conf must exist"

    def test_config_has_sections(self):
        config = REPO_ROOT / "buildroot" / "overlays" / "opt" / "sloughgpt" / "etc" / "dait.conf"
        content = config.read_text()
        assert "[general]" in content
        assert "[shell]" in content
        assert "[kernel]" in content
        assert "[vm]" in content
        assert "[devices]" in content

    def test_config_has_version(self):
        config = REPO_ROOT / "buildroot" / "overlays" / "opt" / "sloughgpt" / "etc" / "dait.conf"
        content = config.read_text()
        assert "version = 0.1.0" in content
