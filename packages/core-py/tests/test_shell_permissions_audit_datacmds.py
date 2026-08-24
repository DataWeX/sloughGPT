"""Tests for domains/shell/permissions.py, audit.py, cmds/data_cmds.py."""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from domains.shell.permissions import (
    Risk, ShellPermissions, _RISK_MAP, _FORCE_PATTERNS,
    _SAFE, _ELEVATED, _DANGEROUS, _CRITICAL,
    set_permissions_db, reset_permissions_db,
)


@pytest.fixture(autouse=True)
def _temp_mogdb(tmp_path):
    """Point the permissions module at a temporary MogDB for every test."""
    db_path = str(tmp_path / "test_perms")
    set_permissions_db(db_path)
    yield
    reset_permissions_db()


class TestRiskConstants:
    def test_risk_values(self):
        assert Risk.SAFE == "safe"
        assert Risk.ELEVATED == "elevated"
        assert Risk.DANGEROUS == "dangerous"
        assert Risk.CRITICAL == "critical"


class TestRiskMap:
    def test_safe_commands_classified(self):
        for cmd in ["help", "exit", "ls", "cat", "pwd", "echo", "grep", "find"]:
            assert _RISK_MAP.get(cmd) == Risk.SAFE

    def test_elevated_commands_classified(self):
        for cmd in ["alias", "unalias", "set", "export", "cd", "py", "ai"]:
            assert _RISK_MAP.get(cmd) == Risk.ELEVATED

    def test_dangerous_commands_classified(self):
        for cmd in ["rm", "chmod", "mv", "cp", "mkdir", "touch"]:
            assert _RISK_MAP.get(cmd) == Risk.DANGEROUS

    def test_critical_commands_classified(self):
        for cmd in ["boot", "shutdown", "svc", "load", "unload", "train", "kill"]:
            assert _RISK_MAP.get(cmd) == Risk.CRITICAL


class TestForcePatterns:
    def test_rm_rf_is_force(self):
        assert "-rf" in _FORCE_PATTERNS["rm"]
        assert "-fr" in _FORCE_PATTERNS["rm"]

    def test_chmod_777_is_force(self):
        assert "777" in _FORCE_PATTERNS["chmod"]


class TestShellPermissions:
    def test_init_default_policy(self):
        p = ShellPermissions()
        assert p._policy[Risk.SAFE] == "allow"
        assert p._policy[Risk.ELEVATED] == "allow"
        assert p._policy[Risk.DANGEROUS] == "deny"
        assert p._policy[Risk.CRITICAL] == "deny"

    def test_check_safe_always_allowed(self):
        p = ShellPermissions()
        p.check("help")  # no exception

    def test_check_elevated_allowed(self):
        p = ShellPermissions()
        p.check("cd", "/tmp")  # no exception

    def test_check_dangerous_raises(self):
        p = ShellPermissions()
        with pytest.raises(PermissionError, match="Permission denied"):
            p.check("rm", "-rf /")

    def test_check_critical_raises(self):
        p = ShellPermissions()
        with pytest.raises(PermissionError, match="Permission denied"):
            p.check("shutdown")

    def test_check_dangerous_with_grant(self):
        p = ShellPermissions()
        p.grant("rm")
        p.check("rm", "-rf /")  # no exception

    def test_check_critical_with_grant(self):
        p = ShellPermissions()
        p.grant("shutdown")
        p.check("shutdown")  # no exception

    def test_check_denied_explicit(self):
        p = ShellPermissions()
        p._denied.add("rm")
        with pytest.raises(PermissionError, match="Permission denied"):
            p.check("rm")

    def test_grant_adds_to_granted(self):
        p = ShellPermissions()
        p.grant("rm")
        assert "rm" in p._granted

    def test_grant_removes_from_denied(self):
        p = ShellPermissions()
        p._denied.add("rm")
        p.grant("rm")
        assert "rm" not in p._denied

    def test_revoke_removes(self):
        p = ShellPermissions()
        p.grant("rm")
        p.revoke("rm")
        assert "rm" not in p._granted

    def test_set_policy_valid(self):
        p = ShellPermissions()
        p.set_policy(Risk.DANGEROUS, "allow")
        assert p._policy[Risk.DANGEROUS] == "allow"

    def test_set_policy_invalid(self):
        p = ShellPermissions()
        with pytest.raises(ValueError, match="action must be"):
            p.set_policy(Risk.DANGEROUS, "maybe")

    def test_is_granted(self):
        p = ShellPermissions()
        assert not p.is_granted("rm")
        p.grant("rm")
        assert p.is_granted("rm")

    def test_classify_safe(self):
        p = ShellPermissions()
        assert p.classify("ls") == Risk.SAFE

    def test_classify_elevated(self):
        p = ShellPermissions()
        assert p.classify("cd") == Risk.ELEVATED

    def test_classify_dangerous(self):
        p = ShellPermissions()
        assert p.classify("rm") == Risk.DANGEROUS

    def test_classify_critical(self):
        p = ShellPermissions()
        assert p.classify("shutdown") == Risk.CRITICAL

    def test_classify_unknown_is_elevated(self):
        p = ShellPermissions()
        assert p.classify("foobar") == Risk.ELEVATED

    def test_classify_force_pattern_rm_rf(self):
        p = ShellPermissions()
        assert p.classify("rm", "-rf /") == Risk.CRITICAL

    def test_classify_force_pattern_chmod_777(self):
        p = ShellPermissions()
        assert p.classify("chmod", "777 file") == Risk.CRITICAL

    def test_classify_no_force_without_args(self):
        p = ShellPermissions()
        assert p.classify("rm") == Risk.DANGEROUS

    def test_list_granted(self):
        p = ShellPermissions()
        p.grant("rm")
        p.grant("shutdown")
        granted = p.list_granted()
        assert "rm" in granted
        assert "shutdown" in granted

    def test_list_dangerous(self):
        p = ShellPermissions()
        dangerous = p.list_dangerous()
        assert "rm" in dangerous
        assert "shutdown" in dangerous

    def test_classify_leading_dashes(self):
        p = ShellPermissions()
        assert p.classify("-rf") == Risk.ELEVATED  # unknown after strip


class TestShellPermissionsPersistence:
    def test_save_and_load(self, tmp_path):
        set_permissions_db(str(tmp_path / "test_perms"))
        try:
            p = ShellPermissions()
            p.grant("rm")
            p.set_policy(Risk.DANGEROUS, "allow")
            p._save_persistent()

            p2 = ShellPermissions()
            assert "rm" in p2._granted
        finally:
            reset_permissions_db()

    def test_load_missing_file(self, tmp_path):
        set_permissions_db(str(tmp_path / "empty_perms"))
        try:
            p = ShellPermissions()
            assert p._granted == set()
        finally:
            reset_permissions_db()

    def test_load_corrupt_file(self, tmp_path):
        # With MogDB, there's no "corrupt file" scenario — just empty collection
        set_permissions_db(str(tmp_path / "clean_perms"))
        try:
            p = ShellPermissions()
            assert p._granted == set()
        finally:
            reset_permissions_db()


# ── audit.py ────────────────────────────────────────────────────────────────

from domains.shell.audit import ShellAuditLogger, get_shell_audit_logger


class TestShellAuditLogger:
    def test_init_creates_log_dir(self, tmp_path):
        log_dir = tmp_path / "audit_test"
        logger = ShellAuditLogger(log_dir=log_dir)
        assert log_dir.exists()

    def test_log_path(self, tmp_path):
        logger = ShellAuditLogger(log_dir=tmp_path)
        assert logger.log_path.name == "shell_audit.jsonl"

    def test_command_event(self, tmp_path):
        logger = ShellAuditLogger(log_dir=tmp_path)
        logger.command("ls -la", "ls", "-la", 0, elapsed_ms=1.5)
        logger._handler.flush()
        content = (tmp_path / "shell_audit.jsonl").read_text()
        assert "shell.command" in content
        assert '"cmd": "ls"' in content

    def test_command_increments_count(self, tmp_path):
        logger = ShellAuditLogger(log_dir=tmp_path)
        logger.command("ls", "ls", "", 0)
        logger.command("pwd", "pwd", "", 0)
        logger._handler.flush()
        content = (tmp_path / "shell_audit.jsonl").read_text()
        assert '"cmd_num": 1' in content
        assert '"cmd_num": 2' in content

    def test_eval_event(self, tmp_path):
        logger = ShellAuditLogger(log_dir=tmp_path)
        logger.eval("2+2", "4", 0)
        logger._handler.flush()
        content = (tmp_path / "shell_audit.jsonl").read_text()
        assert "shell.eval" in content
        assert '"expression": "2+2"' in content

    def test_eval_truncates_long_result(self, tmp_path):
        logger = ShellAuditLogger(log_dir=tmp_path)
        logger.eval("x", "a" * 300, 0)
        logger._handler.flush()
        content = (tmp_path / "shell_audit.jsonl").read_text()
        assert '"result_preview"' in content

    def test_error_event(self, tmp_path):
        logger = ShellAuditLogger(log_dir=tmp_path)
        logger.error("bad cmd", "traceback here")
        logger._handler.flush()
        content = (tmp_path / "shell_audit.jsonl").read_text()
        assert "shell.error" in content
        assert "traceback here" in content

    def test_unknown_event(self, tmp_path):
        logger = ShellAuditLogger(log_dir=tmp_path)
        logger.unknown("foobar")
        logger._handler.flush()
        content = (tmp_path / "shell_audit.jsonl").read_text()
        assert "shell.unknown" in content
        assert '"cmd": "foobar"' in content

    def test_background_event(self, tmp_path):
        logger = ShellAuditLogger(log_dir=tmp_path)
        logger.background("sleep 10 &", 1)
        logger._handler.flush()
        content = (tmp_path / "shell_audit.jsonl").read_text()
        assert "shell.background" in content
        assert '"bg_id": 1' in content

    def test_startup_event(self, tmp_path):
        logger = ShellAuditLogger(log_dir=tmp_path)
        logger.startup()
        logger._handler.flush()
        content = (tmp_path / "shell_audit.jsonl").read_text()
        assert "shell.startup" in content
        assert '"pid"' in content

    def test_shutdown_event(self, tmp_path):
        logger = ShellAuditLogger(log_dir=tmp_path)
        logger.command("ls", "ls", "", 0)
        logger.shutdown()
        logger._handler.flush()
        content = (tmp_path / "shell_audit.jsonl").read_text()
        assert "shell.shutdown" in content
        assert '"total_commands": 1' in content

    def test_session_id(self, tmp_path):
        logger = ShellAuditLogger(log_dir=tmp_path)
        assert logger._session_id is not None

    def test_command_with_background_flag(self, tmp_path):
        logger = ShellAuditLogger(log_dir=tmp_path)
        logger.command("sleep 10 &", "sleep", "10", 0, is_background=True)
        logger._handler.flush()
        content = (tmp_path / "shell_audit.jsonl").read_text()
        assert '"is_background": true' in content

    def test_command_with_pipeline_flag(self, tmp_path):
        logger = ShellAuditLogger(log_dir=tmp_path)
        logger.command("ls | grep py", "ls", "| grep py", 0, is_pipeline=True)
        logger._handler.flush()
        content = (tmp_path / "shell_audit.jsonl").read_text()
        assert '"is_pipeline": true' in content

    def test_command_with_expanded(self, tmp_path):
        logger = ShellAuditLogger(log_dir=tmp_path)
        logger.command("cat $FILE", "cat", "$FILE", 0, expanded="cat test.txt")
        logger._handler.flush()
        content = (tmp_path / "shell_audit.jsonl").read_text()
        assert '"expanded": "cat test.txt"' in content


class TestAuditSingleton:
    def test_singleton(self, tmp_path):
        import domains.shell.audit as audit_mod
        audit_mod._audit = None
        logger1 = get_shell_audit_logger(log_dir=tmp_path)
        logger2 = get_shell_audit_logger(log_dir=tmp_path)
        assert logger1 is logger2
        audit_mod._audit = None


# ── cmds/data_cmds.py ──────────────────────────────────────────────────────

from domains.shell.cmds import data_cmds
from domains.shell.console import Console
from domains.shell.io import MemoryIO


def _make_console():
    io = MemoryIO()
    return Console(io, has_readline=False)


class TestDataCmdsDatasets:
    def test_datasets_empty(self):
        console = _make_console()
        api = MagicMock()
        api.datasets.return_value = []
        rc = data_cmds.run(["datasets"], console, api, {})
        assert rc == 0

    def test_datasets_with_data(self):
        console = _make_console()
        api = MagicMock()
        api.datasets.return_value = [
            {"name": "shakespeare", "samples": 100, "size": 1048576},
            {"name": "wiki", "samples": 50, "size": 0},
        ]
        rc = data_cmds.run(["datasets"], console, api, {})
        assert rc == 0

    def test_datasets_default(self):
        console = _make_console()
        api = MagicMock()
        api.datasets.return_value = []
        rc = data_cmds.run([], console, api, {})
        assert rc == 0


class TestDataCmdsKnowledge:
    def test_knowledge_empty(self):
        console = _make_console()
        api = MagicMock()
        api.knowledge_stats.return_value = {"total_items": 0}
        rc = data_cmds.run(["knowledge"], console, api, {})
        assert rc == 0

    def test_knowledge_with_stats(self):
        console = _make_console()
        api = MagicMock()
        api.knowledge_stats.return_value = {"total_items": 5, "topics": {"ai": 3, "ml": 2}}
        rc = data_cmds.run(["knowledge"], console, api, {})
        assert rc == 0

    def test_knowledge_search_empty(self):
        console = _make_console()
        api = MagicMock()
        api.list_knowledge.return_value = []
        rc = data_cmds.run(["knowledge", "quantum"], console, api, {})
        assert rc == 0

    def test_knowledge_search_results(self):
        console = _make_console()
        api = MagicMock()
        api.list_knowledge.return_value = [{"content": "fact1"}, {"content": "fact2"}]
        rc = data_cmds.run(["knowledge", "test"], console, api, {})
        assert rc == 0


class TestDataCmdsRemember:
    def test_remember_no_args(self):
        console = _make_console()
        api = MagicMock()
        rc = data_cmds.run(["remember"], console, api, {})
        assert rc == 1

    def test_remember_success(self):
        console = _make_console()
        api = MagicMock()
        api.add_knowledge.return_value = {"status": "stored", "topic": "test"}
        rc = data_cmds.run(["remember", "some fact"], console, api, {})
        assert rc == 0

    def test_remember_failure(self):
        console = _make_console()
        api = MagicMock()
        api.add_knowledge.return_value = {"status": "error"}
        rc = data_cmds.run(["remember", "bad fact"], console, api, {})
        assert rc == 0


class TestDataCmdsRecall:
    def test_recall_no_args_empty(self):
        console = _make_console()
        api = MagicMock()
        api.knowledge_stats.return_value = {"total_items": 0}
        rc = data_cmds.run(["recall"], console, api, {})
        assert rc == 0

    def test_recall_no_args_with_facts(self):
        console = _make_console()
        api = MagicMock()
        api.knowledge_stats.return_value = {"total_items": 3, "topics": {"t": 1}}
        rc = data_cmds.run(["recall"], console, api, {})
        assert rc == 0

    def test_recall_search_no_results(self):
        console = _make_console()
        api = MagicMock()
        api.list_knowledge.return_value = []
        rc = data_cmds.run(["recall", "query"], console, api, {})
        assert rc == 0

    def test_recall_search_results(self):
        console = _make_console()
        api = MagicMock()
        api.list_knowledge.return_value = [{"topic": "t", "content": "fact"}]
        rc = data_cmds.run(["recall", "query"], console, api, {})
        assert rc == 0


class TestDataCmdsCheckpoints:
    def test_checkpoints_empty(self):
        console = _make_console()
        api = MagicMock()
        api.checkpoints.return_value = []
        rc = data_cmds.run(["checkpoints"], console, api, {})
        assert rc == 0

    def test_checkpoints_with_data(self):
        console = _make_console()
        api = MagicMock()
        api.checkpoints.return_value = [
            {"name": "cp1", "loss": 0.5, "model_type": "lstm"},
        ]
        rc = data_cmds.run(["checkpoints"], console, api, {})
        assert rc == 0


class TestDataCmdsFinetuned:
    def test_finetuned_empty(self):
        console = _make_console()
        api = MagicMock()
        api.finetuned_models.return_value = []
        rc = data_cmds.run(["finetuned"], console, api, {})
        assert rc == 0

    def test_finetuned_with_data(self):
        console = _make_console()
        api = MagicMock()
        api.finetuned_models.return_value = [
            {"model_name": "m1", "final_loss": 0.3, "epochs": 5, "size_bytes": 1048576},
        ]
        rc = data_cmds.run(["finetuned"], console, api, {})
        assert rc == 0

    def test_finetuned_load_no_name(self):
        console = _make_console()
        api = MagicMock()
        rc = data_cmds.run(["finetuned", "load"], console, api, {})
        assert rc == 1

    def test_finetuned_load_success(self):
        console = _make_console()
        api = MagicMock()
        api.load_finetuned.return_value = {"status": "loaded"}
        rc = data_cmds.run(["finetuned", "load", "mymodel"], console, api, {})
        assert rc == 0

    def test_finetuned_load_failure(self):
        console = _make_console()
        api = MagicMock()
        api.load_finetuned.return_value = {"status": "error", "error": "not found"}
        rc = data_cmds.run(["finetuned", "load", "bad"], console, api, {})
        assert rc == 1

    def test_finetuned_rm_no_name(self):
        console = _make_console()
        api = MagicMock()
        rc = data_cmds.run(["finetuned", "rm"], console, api, {})
        assert rc == 1

    def test_finetuned_rm_success(self):
        console = _make_console()
        api = MagicMock()
        api.delete_finetuned.return_value = {"status": "deleted"}
        rc = data_cmds.run(["finetuned", "rm", "mymodel"], console, api, {})
        assert rc == 0

    def test_finetuned_rm_failure(self):
        console = _make_console()
        api = MagicMock()
        api.delete_finetuned.return_value = {"status": "error"}
        rc = data_cmds.run(["finetuned", "rm", "bad"], console, api, {})
        assert rc == 1

    def test_finetuned_delete_alias(self):
        console = _make_console()
        api = MagicMock()
        api.delete_finetuned.return_value = {"status": "deleted"}
        rc = data_cmds.run(["finetuned", "delete", "m1"], console, api, {})
        assert rc == 0

    def test_finetuned_del_alias(self):
        console = _make_console()
        api = MagicMock()
        api.delete_finetuned.return_value = {"status": "deleted"}
        rc = data_cmds.run(["finetuned", "del", "m1"], console, api, {})
        assert rc == 0


class TestDataCmdsTokenizer:
    def test_tokenizer_stats(self):
        console = _make_console()
        api = MagicMock()
        api.tokenizer_stats.return_value = {"vocab_size": 50000, "merges": 49000}
        rc = data_cmds.run(["tokenizer"], console, api, {})
        assert rc == 0

    def test_tokenizer_error(self):
        console = _make_console()
        api = MagicMock()
        api.tokenizer_stats.return_value = {"error": "not loaded"}
        rc = data_cmds.run(["tokenizer"], console, api, {})
        assert rc == 0


# ── cmds/models_cmd.py ─────────────────────────────────────────────────────

from domains.shell.cmds import models_cmd


class TestModelsCmd:
    def test_models_empty(self):
        console = _make_console()
        api = MagicMock()
        api.models.return_value = []
        api._api_get.return_value = {}
        rc = models_cmd.run(["models"], console, api, {})
        assert rc == 0

    def test_models_with_data(self):
        console = _make_console()
        api = MagicMock()
        api.models.return_value = [
            {"model_id": "gpt2", "type": "text", "size_gb": 0.5},
        ]
        api._api_get.return_value = {"data": {"model_type": "gpt2"}}
        rc = models_cmd.run(["models"], console, api, {})
        assert rc == 0

    def test_models_health_exception(self):
        console = _make_console()
        api = MagicMock()
        api.models.return_value = [{"model_id": "m1"}]
        api._api_get.side_effect = Exception("connection refused")
        rc = models_cmd.run(["models"], console, api, {})
        assert rc == 0

    def test_models_default(self):
        console = _make_console()
        api = MagicMock()
        api.models.return_value = []
        api._api_get.return_value = {}
        rc = models_cmd.run([], console, api, {})
        assert rc == 0

    def test_unload(self):
        console = _make_console()
        api = MagicMock()
        api.unload_model.return_value = {"status": "unloaded"}
        rc = models_cmd.run(["unload"], console, api, {})
        assert rc == 0

    def test_precision_default(self):
        console = _make_console()
        api = MagicMock()
        api.set_precision.return_value = {"mode": "auto"}
        rc = models_cmd.run(["precision"], console, api, {})
        assert rc == 0

    def test_precision_explicit(self):
        console = _make_console()
        api = MagicMock()
        api.set_precision.return_value = {"mode": "fp16"}
        rc = models_cmd.run(["precision", "fp16"], console, api, {})
        assert rc == 0

    def test_quantize_default(self):
        console = _make_console()
        api = MagicMock()
        api.quantize_model.return_value = {"status": "ok"}
        rc = models_cmd.run(["quantize"], console, api, {})
        assert rc == 0

    def test_quantize_custom(self):
        console = _make_console()
        api = MagicMock()
        api.quantize_model.return_value = {"status": "ok"}
        rc = models_cmd.run(["quantize", "4", "asymmetric"], console, api, {})
        assert rc == 0

    def test_dequantize(self):
        console = _make_console()
        api = MagicMock()
        api.dequantize_model.return_value = {"status": "ok"}
        rc = models_cmd.run(["dequantize"], console, api, {})
        assert rc == 0


# ── cmds/souls_cmd.py ──────────────────────────────────────────────────────

from domains.shell.cmds import souls_cmd


class TestSoulsCmd:
    def test_souls_empty(self):
        console = _make_console()
        api = MagicMock()
        api.souls.return_value = []
        rc = souls_cmd.run(["souls"], console, api, {})
        assert rc == 0

    def test_souls_with_data(self):
        console = _make_console()
        api = MagicMock()
        api.souls.return_value = [
            {"name": "friendly", "description": "A friendly soul", "traits": ["warm", "curious"]},
        ]
        rc = souls_cmd.run(["souls"], console, api, {})
        assert rc == 0

    def test_souls_default(self):
        console = _make_console()
        api = MagicMock()
        api.souls.return_value = []
        rc = souls_cmd.run([], console, api, {})
        assert rc == 0

    def test_switch_no_name(self):
        console = _make_console()
        api = MagicMock()
        api.souls.return_value = []
        rc = souls_cmd.run(["switch"], console, api, {})
        assert rc == 1

    def test_switch_success(self):
        console = _make_console()
        api = MagicMock()
        api.switch_soul.return_value = {"status": "switched"}
        rc = souls_cmd.run(["switch", "friendly"], console, api, {})
        assert rc == 0

    def test_whoami(self):
        console = _make_console()
        api = MagicMock()
        api.current_soul.return_value = {"name": "friendly", "description": "A friendly soul"}
        rc = souls_cmd.run(["whoami"], console, api, {})
        assert rc == 0

    def test_whoami_no_description(self):
        console = _make_console()
        api = MagicMock()
        api.current_soul.return_value = {"name": "solo"}
        rc = souls_cmd.run(["whoami"], console, api, {})
        assert rc == 0
