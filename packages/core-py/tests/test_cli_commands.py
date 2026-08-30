"""
Tests for new CLI commands - build and vm.
"""
import pytest
import subprocess
import sys
from pathlib import Path


# Paths
REPO_ROOT = Path(__file__).resolve().parents[3]
CLI_ENTRY = REPO_ROOT / "apps" / "cli" / "src" / "cli.py"


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

    def test_build_command_has_init(self):
        build_cmd = REPO_ROOT / "apps" / "cli" / "src" / "commands" / "build.py"
        content = build_cmd.read_text()
        assert "def cmd_build_init(args):" in content
        assert '"init"' in content


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

    def test_build_script_has_setup(self):
        script = REPO_ROOT / "buildroot" / "build.sh"
        content = script.read_text()
        assert "setup_build()" in content
        assert 'setup)' in content


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


# ═══════════════════════════════════════════════════════════════════════════════
# Integration Tests — actually run CLI commands via subprocess
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.integration
class TestVMRunFile:
    """vm run --file executes assembly correctly."""

    def test_vm_run_file(self, tmp_path):
        """vm run --file executes assembly and produces output."""
        asm_file = tmp_path / "test.asm"
        asm_file.write_text(
            'mov eax, 3\n'
            'mov ebx, 1\n'
            'push msg\n'
            'mov ecx, esp\n'
            'mov edx, 13\n'
            'int 0x80\n'
            'pop eax\n'
            'mov eax, 1\n'
            'mov ebx, 0\n'
            'int 0x80\n'
            'msg: db "Hello, World!", 10, 0\n'
        )

        result = subprocess.run(
            [sys.executable, str(CLI_ENTRY), "vm", "run", "--file", str(asm_file)],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
            timeout=30,
        )

        assert result.returncode == 0, f"CLI failed: {result.stderr}"
        assert "VM Execution" in result.stdout or "Hello" in result.stdout


@pytest.mark.integration
class TestVMRunProgramName:
    """vm run <name> executes built-in program."""

    def test_vm_run_program_name_hello(self):
        """vm run test_hello executes built-in test_hello program."""
        result = subprocess.run(
            [sys.executable, str(CLI_ENTRY), "vm", "run", "test_hello"],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
            timeout=30,
        )

        assert result.returncode == 0, f"CLI failed: {result.stderr}"
        assert "VM Execution" in result.stdout or "Running built-in" in result.stdout

    def test_vm_run_program_name_empty(self):
        """vm run empty executes built-in empty program."""
        result = subprocess.run(
            [sys.executable, str(CLI_ENTRY), "vm", "run", "empty"],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
            timeout=30,
        )

        assert result.returncode == 0, f"CLI failed: {result.stderr}"
        assert "VM Execution" in result.stdout or "Running built-in" in result.stdout

    def test_vm_run_program_list_shows_names(self):
        """vm run with no args shows available program names."""
        result = subprocess.run(
            [sys.executable, str(CLI_ENTRY), "vm", "run"],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
            timeout=30,
        )

        assert result.returncode == 0, f"CLI failed: {result.stderr}"
        assert "test_hello" in result.stdout
        assert "hello" in result.stdout


@pytest.mark.integration
class TestBuildStatus:
    """build status shows current state."""

    def test_build_status(self):
        """build status outputs build state information."""
        result = subprocess.run(
            [sys.executable, str(CLI_ENTRY), "build", "status"],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
            timeout=30,
        )

        assert result.returncode == 0, f"CLI failed: {result.stderr}"
        assert "Buildroot" in result.stdout or "Status" in result.stdout


@pytest.mark.integration
class TestBuildInit:
    """build init sets up build environment."""

    def test_build_init_has_help(self):
        """build init --help shows usage information."""
        result = subprocess.run(
            [sys.executable, str(CLI_ENTRY), "build", "init", "--help"],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
            timeout=30,
        )

        assert result.returncode == 0, f"CLI failed: {result.stderr}"
        assert "clean" in result.stdout.lower() or "Initialize" in result.stdout

    def test_build_init_function_exists(self):
        """cmd_build_init function is defined and importable."""
        result = subprocess.run(
            [sys.executable, "-c",
             "import sys; sys.path.insert(0, 'apps/cli/src'); "
             "from commands.build import cmd_build_init; "
             "print('OK')"],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
            timeout=30,
        )

        assert result.returncode == 0, f"Import failed: {result.stderr}"
        assert "OK" in result.stdout


@pytest.mark.integration
class TestVMPrograms:
    """Test that PROGRAMS dict is accessible and complete."""

    def test_programs_dict_exists(self):
        """PROGRAMS dict is importable from vm_programs."""
        result = subprocess.run(
            [sys.executable, "-c",
             "import sys; sys.path.insert(0, 'packages/core-py'); "
             "from domains.shell.vm_programs import PROGRAMS; "
             "print(f'Found {len(PROGRAMS)} programs'); "
             "assert 'test_hello' in PROGRAMS; "
             "assert 'hello' in PROGRAMS; "
             "assert 'empty' in PROGRAMS; "
             "print('OK')"],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
            timeout=30,
        )

        assert result.returncode == 0, f"Import failed: {result.stderr}"
        assert "OK" in result.stdout

    def test_programs_dict_has_all_required(self):
        """PROGRAMS dict contains all required program names."""
        required = [
            "test_hello", "hello", "test_syscalls", "test_privilege",
            "test_multiprocess", "test_fork", "test_pipe", "test_mmap",
            "test_signal", "test_usermode", "test_ebx_ecx", "test_ergonomics",
            "test_singlestep", "test_v86_dos", "empty", "hello_linux",
        ]

        result = subprocess.run(
            [sys.executable, "-c",
             "import sys; sys.path.insert(0, 'packages/core-py'); "
             "from domains.shell.vm_programs import PROGRAMS; "
             + "; ".join(
                 f"assert '{name}' in PROGRAMS, 'Missing {name}'"
                 for name in required
             )
             + "; print('OK')"],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
            timeout=30,
        )

        assert result.returncode == 0, f"Missing programs: {result.stderr}"
        assert "OK" in result.stdout
