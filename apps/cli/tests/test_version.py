"""Tests for apps/cli/src/core/version.py — version tracking."""
import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestVersionInfo:
    def test_str_representation(self):
        from core.version import VersionInfo
        info = VersionInfo(version="1.2.3", python_version="3.11.0", platform="Linux x86_64")
        assert str(info) == "sloughgpt v1.2.3"

    def test_to_dict(self):
        from core.version import VersionInfo
        info = VersionInfo(version="1.0.0", python_version="3.11.0", platform="Linux x86_64")
        d = info.to_dict()
        assert d["cli"] == "sloughgpt"
        assert d["version"] == "1.0.0"
        assert d["python"] == "3.11.0"
        assert d["platform"] == "Linux x86_64"

    def test_to_json(self):
        from core.version import VersionInfo
        info = VersionInfo(version="1.0.0", python_version="3.11.0", platform="Linux x86_64")
        j = info.to_json()
        parsed = json.loads(j)
        assert parsed["version"] == "1.0.0"

    def test_default_version(self):
        from core.version import VersionInfo, __version__
        info = VersionInfo()
        assert info.version == __version__

    def test_default_python_version(self):
        from core.version import VersionInfo
        info = VersionInfo()
        assert info.python_version is not None
        assert "." in info.python_version

    def test_default_platform(self):
        from core.version import VersionInfo
        info = VersionInfo()
        assert info.platform is not None


class TestGetVersion:
    def test_returns_version_info(self):
        from core.version import get_version, VersionInfo
        info = get_version()
        assert isinstance(info, VersionInfo)

    def test_has_version_string(self):
        from core.version import get_version
        info = get_version()
        assert info.version is not None


class TestFormatVersionDisplay:
    def test_contains_name_and_version(self):
        from core.version import format_version_display, CLI_NAME, __version__
        display = format_version_display()
        assert CLI_NAME in display
        assert __version__ in display

    def test_contains_platform(self):
        from core.version import format_version_display
        display = format_version_display()
        import platform
        assert platform.system() in display

    def test_contains_python_version(self):
        from core.version import format_version_display
        display = format_version_display()
        import sys
        assert str(sys.version_info.major) in display
