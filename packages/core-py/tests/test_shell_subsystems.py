"""
Tests for shell subsystems: commands, permissions, audit.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest


# ── ShellPermissions ──────────────────────────────────────────────


class TestShellPermissions:
    """Tests for the permissions manager."""

    def test_safe_commands_allowed(self):
        from domains.shell.permissions import ShellPermissions
        p = ShellPermissions()
        for cmd in ("ls", "cat", "echo", "help", "health", "status"):
            p.check(cmd)  # should not raise

    def test_elevated_commands_allowed(self):
        from domains.shell.permissions import ShellPermissions
        p = ShellPermissions()
        for cmd in ("alias", "set", "cd", "py"):
            p.check(cmd)

    def test_dangerous_blocked_by_default(self):
        from domains.shell.permissions import ShellPermissions
        p = ShellPermissions()
        with pytest.raises(PermissionError, match="Permission denied"):
            p.check("rm")

    def test_critical_blocked_by_default(self):
        from domains.shell.permissions import ShellPermissions
        p = ShellPermissions()
        with pytest.raises(PermissionError, match="Permission denied"):
            p.check("shutdown")

    def test_grant_allows_command(self):
        from domains.shell.permissions import ShellPermissions
        p = ShellPermissions()
        p.grant("rm")
        p.check("rm")  # should not raise

    def test_revoke_blocks_again(self):
        from domains.shell.permissions import ShellPermissions
        p = ShellPermissions()
        p.grant("rm")
        p.revoke("rm")
        with pytest.raises(PermissionError):
            p.check("rm")

    def test_classify_risk_levels(self):
        from domains.shell.permissions import ShellPermissions, Risk
        p = ShellPermissions()
        assert p.classify("ls") == Risk.SAFE
        assert p.classify("alias") == Risk.ELEVATED
        assert p.classify("rm") == Risk.DANGEROUS
        assert p.classify("shutdown") == Risk.CRITICAL

    def test_rm_rf_is_critical(self):
        from domains.shell.permissions import ShellPermissions, Risk
        p = ShellPermissions()
        assert p.classify("rm", "-rf") == Risk.CRITICAL
        assert p.classify("rm", "-fr") == Risk.CRITICAL

    def test_chmod_777_is_critical(self):
        from domains.shell.permissions import ShellPermissions, Risk
        p = ShellPermissions()
        assert p.classify("chmod", "777") == Risk.CRITICAL

    def test_unknown_command_is_elevated(self):
        from domains.shell.permissions import ShellPermissions, Risk
        p = ShellPermissions()
        assert p.classify("nonexistent_cmd") == Risk.ELEVATED

    def test_set_policy(self):
        from domains.shell.permissions import ShellPermissions, Risk
        p = ShellPermissions()
        p.set_policy(Risk.DANGEROUS, "allow")
        p.check("rm")  # should not raise now

    def test_set_policy_invalid_action(self):
        from domains.shell.permissions import ShellPermissions
        p = ShellPermissions()
        with pytest.raises(ValueError, match="must be 'allow' or 'deny'"):
            p.set_policy("dangerous", "maybe")

    def test_list_granted(self):
        from domains.shell.permissions import ShellPermissions
        p = ShellPermissions()
        p.grant("rm")
        p.grant("mv")
        granted = p.list_granted()
        assert "mv" in granted
        assert "rm" in granted

    def test_list_dangerous(self):
        from domains.shell.permissions import ShellPermissions
        p = ShellPermissions()
        dangerous = p.list_dangerous()
        assert "rm" in dangerous
        assert "shutdown" in dangerous

    def test_is_granted(self):
        from domains.shell.permissions import ShellPermissions
        p = ShellPermissions()
        assert not p.is_granted("rm")
        p.grant("rm")
        assert p.is_granted("rm")

    def test_persistence(self):
        from domains.shell.permissions import ShellPermissions
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "perms.json"
            with patch.object(ShellPermissions, "_config_path", config_path):
                p1 = ShellPermissions()
                p1.grant("rm", persist=True)

                p2 = ShellPermissions()
                assert p2.is_granted("rm")

    def test_denied_command_short_message(self):
        from domains.shell.permissions import ShellPermissions
        p = ShellPermissions()
        p._denied.add("rm")
        with pytest.raises(PermissionError, match=r"Use `permit rm` to grant\.$"):
            p.check("rm")

    def test_revoke_persist(self):
        from domains.shell.permissions import ShellPermissions
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "perms.json"
            with patch.object(ShellPermissions, "_config_path", config_path):
                p = ShellPermissions()
                p.grant("rm", persist=True)
                p.revoke("rm", persist=True)
            saved = json.loads(config_path.read_text())
            assert saved["granted"] == []

    def test_load_persistent_config(self):
        from domains.shell.permissions import ShellPermissions
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "perms.json"
            config_path.write_text(json.dumps({
                "granted": ["rm"],
                "policy": {"dangerous": "allow"},
            }))
            with patch.object(ShellPermissions, "_config_path", config_path):
                p = ShellPermissions()
            assert p.is_granted("rm")
            assert p._policy["dangerous"] == "allow"

    def test_load_corrupt_config_ignored(self):
        from domains.shell.permissions import ShellPermissions
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "perms.json"
            config_path.write_text("NOT JSON!!!")
            with patch.object(ShellPermissions, "_config_path", config_path):
                p = ShellPermissions()
            assert not p.is_granted("rm")

    def test_save_persistent_failure_ignored(self):
        from domains.shell.permissions import ShellPermissions
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "perms.json"
            with patch.object(ShellPermissions, "_config_path", config_path), \
                 patch.object(Path, "write_text", side_effect=OSError("disk full")):
                p = ShellPermissions()
                p.grant("rm", persist=True)  # should not raise
            assert p.is_granted("rm")


# ── ShellAuditLogger ──────────────────────────────────────────────


class TestShellAuditLogger:
    """Tests for the audit logger."""

    def test_command_event(self):
        from domains.shell.audit import ShellAuditLogger
        with tempfile.TemporaryDirectory() as tmp:
            logger = ShellAuditLogger(log_dir=tmp)
            logger.command("rm -rf /tmp/test", "rm", "-rf /tmp/test", 0, elapsed_ms=12.5)
            assert logger._cmd_count == 1

    def test_eval_event(self):
        from domains.shell.audit import ShellAuditLogger
        with tempfile.TemporaryDirectory() as tmp:
            logger = ShellAuditLogger(log_dir=tmp)
            logger.eval("2 + 2", "4", 0)
            assert logger._cmd_count == 1

    def test_error_event(self):
        from domains.shell.audit import ShellAuditLogger
        with tempfile.TemporaryDirectory() as tmp:
            logger = ShellAuditLogger(log_dir=tmp)
            logger.error("bad cmd", "something broke")
            # error doesn't increment cmd_count

    def test_unknown_event(self):
        from domains.shell.audit import ShellAuditLogger
        with tempfile.TemporaryDirectory() as tmp:
            logger = ShellAuditLogger(log_dir=tmp)
            logger.unknown("foobar")
            assert logger._cmd_count == 1

    def test_background_event(self):
        from domains.shell.audit import ShellAuditLogger
        with tempfile.TemporaryDirectory() as tmp:
            logger = ShellAuditLogger(log_dir=tmp)
            logger.background("sleep 10 &", 42)

    def test_startup_shutdown(self):
        from domains.shell.audit import ShellAuditLogger
        with tempfile.TemporaryDirectory() as tmp:
            logger = ShellAuditLogger(log_dir=tmp)
            logger.startup()
            logger.command("ls", "ls", "", 0)
            logger.shutdown()

    def test_log_file_written(self):
        from domains.shell.audit import ShellAuditLogger
        with tempfile.TemporaryDirectory() as tmp:
            logger = ShellAuditLogger(log_dir=tmp)
            logger.command("test", "test", "", 0)
            logger._handler.flush()
            lines = logger.log_path.read_text().strip().split("\n")
            assert len(lines) >= 1
            record = json.loads(lines[0])
            assert record["event"] == "shell.command"
            assert record["cmd"] == "test"

    def test_session_id_consistent(self):
        from domains.shell.audit import ShellAuditLogger
        with tempfile.TemporaryDirectory() as tmp:
            logger = ShellAuditLogger(log_dir=tmp)
            logger.command("a", "a", "", 0)
            logger.command("b", "b", "", 0)
            logger._handler.flush()
            lines = logger.log_path.read_text().strip().split("\n")
            s1 = json.loads(lines[0])["session"]
            s2 = json.loads(lines[1])["session"]
            assert s1 == s2

    def test_log_path(self):
        from domains.shell.audit import ShellAuditLogger
        with tempfile.TemporaryDirectory() as tmp:
            logger = ShellAuditLogger(log_dir=tmp)
            assert logger.log_path.name == "shell_audit.jsonl"
            assert logger.log_path.parent == Path(tmp)

    def test_singleton(self):
        from domains.shell.audit import get_shell_audit_logger, _audit
        import domains.shell.audit as mod
        mod._audit = None
        with tempfile.TemporaryDirectory() as tmp:
            a = get_shell_audit_logger(log_dir=tmp)
            b = get_shell_audit_logger(log_dir=tmp)
            assert a is b
        mod._audit = None


# ── ShellCommands ─────────────────────────────────────────────────


class TestShellCommands:
    """Tests for the commands module (API wrappers)."""

    def test_import(self):
        from domains.shell.commands import ShellCommands
        cmds = ShellCommands()
        assert hasattr(cmds, "health")
        assert hasattr(cmds, "models")
        assert hasattr(cmds, "load_model")

    def test_models_returns_list(self):
        from domains.shell.commands import ShellCommands
        cmds = ShellCommands()
        # Without a running server, this returns an error dict
        result = cmds.models()
        assert isinstance(result, (list, dict))

    def test_health_returns_dict(self):
        from domains.shell.commands import ShellCommands
        cmds = ShellCommands()
        result = cmds.health()
        assert isinstance(result, dict)

    def test_api_base_default(self):
        from domains.shell.commands import API_BASE
        assert API_BASE == "http://localhost:8000"
