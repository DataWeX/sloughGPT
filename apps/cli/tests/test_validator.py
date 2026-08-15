"""Tests for apps/cli/src/core/validator.py (Doctor / ValidationResult)."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestValidationResult:
    def test_add_pass(self):
        from core.validator import ValidationResult
        r = ValidationResult()
        r.add_pass("A", "ok")
        assert r.checks[0].passed is True
        assert r.passed is True
        assert r.failed_count == 0
        assert r.warning_count == 0

    def test_add_fail_counts(self):
        from core.validator import ValidationResult
        r = ValidationResult()
        r.add_fail("A", "broken", "fix it")
        assert r.passed is False
        assert r.failed_count == 1

    def test_add_warn_is_pass_with_prefix(self):
        from core.validator import ValidationResult
        r = ValidationResult()
        r.add_warn("A", "low disk")
        c = r.checks[0]
        assert c.passed is True
        assert c.message.startswith("Warning:")
        assert r.warning_count == 1
        assert r.passed is True


class TestDoctorChecks:
    def test_python_version_current(self, tmp_path):
        from core.validator import Doctor
        d = Doctor(root_dir=tmp_path)
        d._check_python_version()
        assert d.result.checks[0].passed is True

    def test_required_dirs_exist_pass(self, tmp_path):
        from core.validator import Doctor
        for name in ["models", "datasets", "data"]:
            (tmp_path / name).mkdir()
        d = Doctor(root_dir=tmp_path)
        d._check_required_dirs()
        assert d.result.passed is True
        assert len(d.result.checks) == 3

    def test_required_dirs_missing_fail(self, tmp_path):
        from core.validator import Doctor
        d = Doctor(root_dir=tmp_path)
        d._check_required_dirs()
        assert d.result.passed is False
        assert d.result.failed_count == 3

    def test_env_file_missing_warns(self, tmp_path):
        from core.validator import Doctor
        d = Doctor(root_dir=tmp_path)
        d._check_env_file()
        assert d.result.checks[0].passed is True
        assert d.result.warning_count == 1

    def test_env_file_present_passes(self, tmp_path):
        from core.validator import Doctor
        (tmp_path / ".env").write_text("KEY=value\n")
        d = Doctor(root_dir=tmp_path)
        d._check_env_file()
        assert d.result.checks[0].message == "Found"

    def test_api_server_unreachable_warns(self, tmp_path, monkeypatch):
        from core.validator import Doctor
        monkeypatch.setitem(sys.modules, "requests", None)
        d = Doctor(root_dir=tmp_path)
        d._check_api_server()
        assert d.result.checks[0].passed is True
        assert d.result.warning_count == 1

    def test_run_all_includes_all_checks(self, tmp_path, monkeypatch):
        from core.validator import Doctor
        monkeypatch.setitem(sys.modules, "requests", None)
        d = Doctor(root_dir=tmp_path)
        result = d.run_all()
        names = {c.name for c in result.checks}
        assert {"Python", "models", "datasets", "data", "API Server", ".env"} <= names
