"""Tests for shell/permissions.py — risk classification, grant/deny, persistence."""

import json
import pytest
from unittest.mock import patch, mock_open
from domains.shell.permissions import ShellPermissions, Risk


class TestRisk:
    def test_constants(self):
        assert Risk.SAFE == "safe"
        assert Risk.ELEVATED == "elevated"
        assert Risk.DANGEROUS == "dangerous"
        assert Risk.CRITICAL == "critical"


class TestShellPermissions:
    @pytest.fixture(autouse=True)
    def _no_persist(self):
        with patch.object(ShellPermissions, "_load_persistent"):
            yield

    def test_init(self):
        p = ShellPermissions()
        assert len(p._granted) == 0
        assert p._policy[Risk.SAFE] == "allow"
        assert p._policy[Risk.DANGEROUS] == "deny"

    def test_classify_safe(self):
        p = ShellPermissions()
        assert p.classify("ls") == Risk.SAFE
        assert p.classify("help") == Risk.SAFE
        assert p.classify("cat") == Risk.SAFE
        assert p.classify("grep") == Risk.SAFE

    def test_classify_elevated(self):
        p = ShellPermissions()
        assert p.classify("alias") == Risk.ELEVATED
        assert p.classify("cd") == Risk.ELEVATED
        assert p.classify("export") == Risk.ELEVATED
        assert p.classify("py") == Risk.ELEVATED

    def test_classify_dangerous(self):
        p = ShellPermissions()
        assert p.classify("rm") == Risk.DANGEROUS
        assert p.classify("chmod") == Risk.DANGEROUS
        assert p.classify("mkdir") == Risk.DANGEROUS
        assert p.classify("cp") == Risk.DANGEROUS

    def test_classify_critical(self):
        p = ShellPermissions()
        assert p.classify("shutdown") == Risk.CRITICAL
        assert p.classify("load") == Risk.CRITICAL
        assert p.classify("train") == Risk.CRITICAL
        assert p.classify("kill") == Risk.CRITICAL

    def test_classify_unknown_elevated(self):
        p = ShellPermissions()
        assert p.classify("bogus_cmd") == Risk.ELEVATED

    def test_rm_rf_escalates_to_critical(self):
        p = ShellPermissions()
        assert p.classify("rm", "-rf /") == Risk.CRITICAL
        assert p.classify("rm", "-r foo") == Risk.CRITICAL
        assert p.classify("rm", "--recursive bar") == Risk.CRITICAL

    def test_chmod_777_escalates(self):
        p = ShellPermissions()
        assert p.classify("chmod", "777 file") == Risk.CRITICAL
        assert p.classify("chmod", "000 file") == Risk.CRITICAL

    def test_check_safe_allowed(self):
        p = ShellPermissions()
        p.check("ls")  # should not raise

    def test_check_dangerous_denied(self):
        p = ShellPermissions()
        with pytest.raises(PermissionError, match="denied"):
            p.check("rm")

    def test_check_critical_denied(self):
        p = ShellPermissions()
        with pytest.raises(PermissionError, match="denied"):
            p.check("shutdown")

    def test_grant_allows(self):
        p = ShellPermissions()
        p.grant("rm")
        p.check("rm")  # should not raise
        assert p.is_granted("rm")

    def test_grant_removes_from_denied(self):
        p = ShellPermissions()
        p._denied.add("rm")
        p.grant("rm")
        assert "rm" not in p._denied

    def test_revoke(self):
        p = ShellPermissions()
        p.grant("rm")
        p.revoke("rm")
        assert not p.is_granted("rm")

    def test_set_policy_allow(self):
        p = ShellPermissions()
        p.set_policy(Risk.DANGEROUS, "allow")
        p.check("rm")  # should not raise

    def test_set_policy_invalid(self):
        p = ShellPermissions()
        with pytest.raises(ValueError, match="allow.*deny"):
            p.set_policy(Risk.SAFE, "bogus")

    def test_list_granted(self):
        p = ShellPermissions()
        p.grant("rm")
        p.grant("chmod")
        granted = p.list_granted()
        assert granted == ["chmod", "rm"]

    def test_list_dangerous(self):
        p = ShellPermissions()
        dangerous = p.list_dangerous()
        assert "rm" in dangerous
        assert "shutdown" in dangerous

    def test_check_with_args_granted(self):
        p = ShellPermissions()
        p.grant("rm /tmp/test")
        p.check("rm", "/tmp/test")  # should not raise

    def test_check_with_args_not_granted(self):
        p = ShellPermissions()
        p.grant("rm /tmp/test")
        with pytest.raises(PermissionError):
            p.check("rm", "/tmp/other")

    @patch("domains.shell.permissions.Path.write_text")
    def test_persist_save(self, mock_write):
        p = ShellPermissions()
        p.grant("rm")
        p._save_persistent()
        mock_write.assert_called_once()
        written = mock_write.call_args[0][0]
        data = json.loads(written)
        assert "rm" in data["granted"]

    def test_init_loads_persistent(self):
        with patch.object(ShellPermissions, "_load_persistent") as mock:
            p = ShellPermissions()
            mock.assert_called_once()
