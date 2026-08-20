"""Tests for domains.shell — get_dait_runtime singleton."""

from domains.shell import get_dait_runtime, DaitRuntime


class TestGetDaitRuntime:
    def test_returns_dait_runtime(self):
        import domains.shell as shell_mod
        shell_mod._dait_instance = None
        rt = get_dait_runtime()
        assert isinstance(rt, DaitRuntime)
        shell_mod._dait_instance = None

    def test_singleton(self):
        import domains.shell as shell_mod
        shell_mod._dait_instance = None
        a = get_dait_runtime()
        b = get_dait_runtime()
        assert a is b
        shell_mod._dait_instance = None
