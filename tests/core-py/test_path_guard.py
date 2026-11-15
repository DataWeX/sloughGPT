"""
Path guard — model-supplied paths denied on BOTH tool surfaces.

The guard (domain/agents/_internal/path_guard.py) is one shared blocklist
enforced by agents.py's ToolRunner (covered in test_agent_core.py::
TestToolRunnerPathGuard) and by tools.py's ToolRegistry (covered here).
Two surfaces, ONE blocklist: a sensitive path must be denied through
either entry point, and ordinary reads must pass unchanged.

Lives in its own file (not test_tools.py) because that module is marked
``slow`` and gets deselected by default in CI and locally — a guard that
never runs is not a guard.
"""

from __future__ import annotations

import pytest

from domain.agents._internal.path_guard import path_violation
from domain.agents._internal.tools import ToolRegistry


@pytest.fixture
def registry():
    return ToolRegistry()


# ── ToolRegistry seam: execute() → guarded _run_* methods ────────────────


class TestToolRegistryPathGuard:
    @pytest.mark.asyncio
    async def test_file_read_denies_sensitive_dir(self, registry, monkeypatch, tmp_path):
        monkeypatch.setenv("HOME", str(tmp_path))
        secret = tmp_path / ".ssh" / "known_hosts"
        secret.parent.mkdir()
        secret.write_text("key material")
        result = await registry.execute("file_read", {"path": str(secret)})
        assert result.success is False
        assert "sensitive" in result.error

    @pytest.mark.asyncio
    async def test_file_read_denies_dotenv(self, registry, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("API_KEY=abc")
        result = await registry.execute("file_read", {"path": str(env_file)})
        assert result.success is False
        assert "sensitive" in result.error

    @pytest.mark.asyncio
    async def test_file_read_symlink_to_sensitive_denied(self, registry, monkeypatch, tmp_path):
        """realpath resolves the symlink BEFORE the check — a benign-looking
        link inside the workspace pointing at ~/.ssh must still be denied."""
        monkeypatch.setenv("HOME", str(tmp_path))
        secret = tmp_path / ".ssh" / "id_rsa"
        secret.parent.mkdir()
        secret.write_text("PRIVATE KEY")
        link = tmp_path / "notes.png"  # benign name, sensitive target
        link.symlink_to(secret)
        result = await registry.execute("file_read", {"path": str(link)})
        assert result.success is False
        assert "sensitive" in result.error

    @pytest.mark.asyncio
    async def test_file_read_normal_path_still_reads(self, registry, tmp_path):
        """The guard must not overblock ordinary workspace reads."""
        doc = tmp_path / "notes.txt"
        doc.write_text("hello guard")
        result = await registry.execute("file_read", {"path": str(doc)})
        assert result.success is True
        assert "hello guard" in result.output

    @pytest.mark.asyncio
    async def test_image_analysis_denies_sensitive_path(self, registry, monkeypatch, tmp_path):
        monkeypatch.setenv("HOME", str(tmp_path))
        secret = tmp_path / ".ssh" / "config"
        secret.parent.mkdir()
        secret.write_text("Host *")
        result = await registry.execute("image_analysis", {"image_path": str(secret)})
        assert result.success is False
        assert "sensitive" in result.error

    @pytest.mark.asyncio
    async def test_image_analysis_url_not_treated_as_local_path(self, registry):
        """Spec says 'Path or URL' — the guard only rejects blocklist matches;
        a URL must pass the guard and fail downstream (not_found), not
        be denied as a sensitive path."""
        result = await registry.execute(
            "image_analysis", {"image_path": "https://example.com/photo.png"}
        )
        assert result.success is False
        assert "denied" not in (result.error or "")

    @pytest.mark.asyncio
    async def test_data_analysis_denies_sensitive_path(self, registry, tmp_path):
        result = await registry.execute("data_analysis", {"data_path": str(tmp_path / ".env")})
        assert result.success is False
        assert "sensitive" in result.error

    @pytest.mark.asyncio
    async def test_data_analysis_normal_json_still_reads(self, registry, tmp_path):
        data = tmp_path / "metrics.json"
        data.write_text('{"tok_s": 19.27, "r2": 0.9999}')
        result = await registry.execute("data_analysis", {"data_path": str(data)})
        assert result.success is True
        assert "2 keys" in result.output


# ── Pure function: the shared blocklist both surfaces enforce ────────────


class TestPathViolationFunction:
    def test_allowlist_prefix_bound_on_separator(self, tmp_path):
        root = tmp_path / "workspace"
        root.mkdir()
        # sibling dir sharing the prefix must NOT match
        sibling = tmp_path / "workspace_evil"
        sibling.mkdir()
        assert path_violation(str(root / "a.txt"), [str(root)]) is None
        assert path_violation(str(sibling / "a.txt"), [str(root)]) is not None

    def test_dotenv_basename_blocked_anywhere(self, tmp_path):
        f = tmp_path / ".env.production"
        f.write_text("x=1")
        assert path_violation(str(f)) is not None

    def test_sensitive_basename_blocked_outside_home(self, tmp_path):
        f = tmp_path / "id_ed25519"
        f.write_text("key")
        assert path_violation(str(f)) is not None

    def test_ordinary_path_permitted(self, tmp_path):
        f = tmp_path / "data.csv"
        f.write_text("a,b\n1,2")
        assert path_violation(str(f)) is None
