"""Tests for Shell sandbox security validation (validate_command)."""

import os
import sys

# Ensure the server directory is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from infrastructure.shell_sandbox import ShellSecurityError, validate_command


class TestValidateCommand:
    """Direct unit tests for the shell sandbox security policy."""

    def test_empty_command_blocked(self):
        with pytest.raises(ShellSecurityError, match="Empty command"):
            validate_command("   ")

    def test_whitespace_only_blocked(self):
        with pytest.raises(ShellSecurityError, match="Empty command"):
            validate_command("\n\t  ")

    def test_simple_allowed(self):
        assert validate_command("echo hello world") is None

    def test_echo_allowed_with_flags(self):
        assert validate_command("echo -n hello") is None

    def test_ls_allowed(self):
        assert validate_command("ls -la") is None

    def test_git_status_allowed(self):
        assert validate_command("git status") is None

    def test_rm_blocked(self):
        with pytest.raises(ShellSecurityError, match="rm"):
            validate_command("rm -rf /tmp/x")

    def test_sudo_blocked(self):
        with pytest.raises(ShellSecurityError, match="sudo"):
            validate_command("sudo ls")

    def test_bash_blocked(self):
        with pytest.raises(ShellSecurityError, match="bash"):
            validate_command("bash -c 'ls'")

    def test_python_blocked(self):
        with pytest.raises(ShellSecurityError, match="python"):
            validate_command("python3 -c 'print(1)'")

    def test_curl_blocked(self):
        with pytest.raises(ShellSecurityError, match="curl"):
            validate_command("curl http://example.com")

    def test_root_redirect_blocked(self):
        with pytest.raises(ShellSecurityError, match="blocked pattern"):
            validate_command("echo x > /etc/passwd")

    def test_tmp_redirect_blocked(self):
        with pytest.raises(ShellSecurityError, match="blocked pattern"):
            validate_command("echo x > /tmp/x")

    def test_pipe_to_shell_blocked(self):
        with pytest.raises(ShellSecurityError, match="blocked pattern"):
            validate_command("echo x | sh")

    def test_subshell_expression_blocked(self):
        with pytest.raises(ShellSecurityError, match="blocked pattern"):
            validate_command("echo $(ls)")

    def test_backtick_expression_blocked(self):
        with pytest.raises(ShellSecurityError, match="blocked pattern"):
            validate_command("echo `ls`")

    def test_aws_metadata_blocked(self):
        with pytest.raises(ShellSecurityError, match="blocked pattern"):
            validate_command("curl http://169.254.169.254/latest/meta-data/")

    def test_gcp_metadata_blocked(self):
        with pytest.raises(ShellSecurityError, match="blocked pattern"):
            validate_command("echo metadata.google.internal")

    def test_malformed_unclosed_quote_blocked(self):
        with pytest.raises(ShellSecurityError, match="invalid syntax"):
            validate_command('echo "unclosed')

    def test_path_prefixed_command_blocked(self):
        with pytest.raises(ShellSecurityError, match="rm"):
            validate_command("/usr/bin/rm file")
