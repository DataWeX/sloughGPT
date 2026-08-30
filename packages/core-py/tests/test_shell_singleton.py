"""Tests for domains.shell — get_dait_runtime singleton and Resource dataclass."""

from __future__ import annotations

import domains.shell as shell_mod
from domains.shell import get_dait_runtime, DaitRuntime
from domains.shell.runtime import Resource


# ---------------------------------------------------------------------------
# Resource dataclass
# ---------------------------------------------------------------------------

class TestResource:
    def test_basic_fields(self):
        r = Resource(name="model.bin", kind="model", path="/tmp/model.bin")
        assert r.name == "model.bin"
        assert r.kind == "model"
        assert r.path == "/tmp/model.bin"
        assert r.size_bytes == 0
        assert r.metadata == {}

    def test_custom_fields(self):
        r = Resource(name="ds.json", kind="dataset", path="/data/ds.json",
                     size_bytes=1024, metadata={"format": "json"})
        assert r.size_bytes == 1024
        assert r.metadata["format"] == "json"

    def test_size_str_bytes(self):
        r = Resource(name="f", kind="k", path="/f", size_bytes=50)
        assert r.size_str == "50B"

    def test_size_str_kilobytes(self):
        r = Resource(name="f", kind="k", path="/f", size_bytes=2048)
        assert r.size_str == "2.0K"

    def test_size_str_megabytes(self):
        r = Resource(name="f", kind="k", path="/f", size_bytes=5 * 1024 * 1024)
        assert r.size_str == "5.0M"

    def test_size_str_gigabytes(self):
        r = Resource(name="f", kind="k", path="/f", size_bytes=3 * 1024**3)
        assert r.size_str == "3.0G"

    def test_size_str_exact_kb_boundary(self):
        r = Resource(name="f", kind="k", path="/f", size_bytes=1024)
        assert r.size_str == "1.0K"

    def test_size_str_exact_mb_boundary(self):
        r = Resource(name="f", kind="k", path="/f", size_bytes=1048576)
        assert r.size_str == "1.0M"

    def test_size_str_exact_gb_boundary(self):
        r = Resource(name="f", kind="k", path="/f", size_bytes=1073741824)
        assert r.size_str == "1.0G"

    def test_size_str_zero(self):
        r = Resource(name="f", kind="k", path="/f", size_bytes=0)
        assert r.size_str == "0B"

    def test_metadata_default_factory(self):
        r1 = Resource(name="a", kind="k", path="/a")
        r2 = Resource(name="b", kind="k", path="/b")
        r1.metadata["x"] = 1
        assert r2.metadata == {}

    def test_equality(self):
        a = Resource(name="f", kind="k", path="/f")
        b = Resource(name="f", kind="k", path="/f")
        assert a == b

    def test_inequality(self):
        a = Resource(name="f", kind="k", path="/f")
        b = Resource(name="g", kind="k", path="/f")
        assert a != b

    def test_size_str_one_byte(self):
        r = Resource(name="f", kind="k", path="/f", size_bytes=1)
        assert r.size_str == "1B"

    def test_size_str_1023_bytes(self):
        r = Resource(name="f", kind="k", path="/f", size_bytes=1023)
        assert r.size_str == "1023B"

    def test_size_str_large_kb(self):
        r = Resource(name="f", kind="k", path="/f", size_bytes=1024 * 999)
        assert r.size_str.endswith("K")

    def test_size_str_large_mb(self):
        r = Resource(name="f", kind="k", path="/f", size_bytes=1048576 * 500)
        assert r.size_str.endswith("M")

    def test_size_str_terabytes_not_handled(self):
        r = Resource(name="f", kind="k", path="/f", size_bytes=1024**4)
        assert r.size_str.endswith("G")

    def test_metadata_mutable(self):
        r = Resource(name="f", kind="k", path="/f")
        r.metadata["key"] = "val"
        assert r.metadata["key"] == "val"

    def test_metadata_multiple_entries(self):
        r = Resource(name="f", kind="k", path="/f", metadata={"a": 1, "b": 2, "c": 3})
        assert len(r.metadata) == 3

    def test_name_with_spaces(self):
        r = Resource(name="my model.bin", kind="model", path="/m.bin")
        assert r.name == "my model.bin"

    def test_path_with_spaces(self):
        r = Resource(name="f", kind="k", path="/path with spaces/file.bin")
        assert r.path == "/path with spaces/file.bin"

    def test_kind_any_string(self):
        for kind in ["model", "dataset", "soul", "checkpoint", "tokenizer", "custom"]:
            r = Resource(name="f", kind=kind, path="/f")
            assert r.kind == kind

    def test_repr_contains_name(self):
        r = Resource(name="test.bin", kind="model", path="/test.bin")
        assert "test.bin" in repr(r) or "test.bin" in str(r)

    def test_hashable_not_crash(self):
        r = Resource(name="f", kind="k", path="/f")
        try:
            hash(r)
        except TypeError:
            pass

    def test_negative_size_bytes(self):
        r = Resource(name="f", kind="k", path="/f", size_bytes=-1)
        assert r.size_str == "-1B"

    def test_zero_size_exact(self):
        r = Resource(name="f", kind="k", path="/f", size_bytes=0)
        assert r.size_str == "0B"


# ---------------------------------------------------------------------------
# get_dait_runtime singleton
# ---------------------------------------------------------------------------

class TestGetDaitRuntime:
    def test_returns_dait_runtime(self):
        shell_mod._dait_instance = None
        rt = get_dait_runtime()
        assert isinstance(rt, DaitRuntime)
        shell_mod._dait_instance = None

    def test_singleton_same_instance(self):
        shell_mod._dait_instance = None
        a = get_dait_runtime()
        b = get_dait_runtime()
        assert a is b
        shell_mod._dait_instance = None

    def test_returns_same_after_reset(self):
        shell_mod._dait_instance = None
        first = get_dait_runtime()
        second = get_dait_runtime()
        assert first is second
        shell_mod._dait_instance = None

    def test_type_is_dait_runtime(self):
        shell_mod._dait_instance = None
        rt = get_dait_runtime()
        assert type(rt).__name__ == "DaitRuntime"
        shell_mod._dait_instance = None

    def test_has_expected_attributes(self):
        shell_mod._dait_instance = None
        rt = get_dait_runtime()
        assert hasattr(rt, "kernel") or hasattr(rt, "_kernel") or hasattr(rt, "__init__")
        shell_mod._dait_instance = None

    def test_concurrent_same_instance(self):
        import threading
        shell_mod._dait_instance = None
        instances = []

        def _worker():
            instances.append(get_dait_runtime())

        threads = [threading.Thread(target=_worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert all(i is instances[0] for i in instances)
        shell_mod._dait_instance = None

    def test_reset_allows_new_instance(self):
        shell_mod._dait_instance = None
        first = get_dait_runtime()
        shell_mod._dait_instance = None
        second = get_dait_runtime()
        assert first is not second
        shell_mod._dait_instance = None

    def test_module_has_dait_instance_attr(self):
        assert hasattr(shell_mod, "_dait_instance")

    def test_dait_instance_is_none_or_daitruntime(self):
        val = shell_mod._dait_instance
        assert val is None or isinstance(val, DaitRuntime)

    def test_get_dait_runtime_is_callable(self):
        assert callable(get_dait_runtime)

    def test_three_calls_same(self):
        shell_mod._dait_instance = None
        a = get_dait_runtime()
        b = get_dait_runtime()
        c = get_dait_runtime()
        assert a is b is c
        shell_mod._dait_instance = None

    def test_reset_between_pairs(self):
        shell_mod._dait_instance = None
        a = get_dait_runtime()
        shell_mod._dait_instance = None
        b = get_dait_runtime()
        shell_mod._dait_instance = None
        c = get_dait_runtime()
        assert a is not b
        assert b is not c
        assert a is not c
        shell_mod._dait_instance = None

    def test_has_api_status(self):
        shell_mod._dait_instance = None
        rt = get_dait_runtime()
        assert hasattr(rt, "api_status")
        shell_mod._dait_instance = None

    def test_has_kernel(self):
        shell_mod._dait_instance = None
        rt = get_dait_runtime()
        assert hasattr(rt, "kernel")
        shell_mod._dait_instance = None

    def test_kernel_is_not_none(self):
        shell_mod._dait_instance = None
        rt = get_dait_runtime()
        assert rt.kernel is not None
        shell_mod._dait_instance = None

    def test_dait_instance_is_none_before_first_call(self):
        shell_mod._dait_instance = None
        assert shell_mod._dait_instance is None
        get_dait_runtime()
        shell_mod._dait_instance = None

    def test_multiple_threads_see_singleton(self):
        import threading
        shell_mod._dait_instance = None
        results = [None] * 20

        def _worker(idx):
            results[idx] = get_dait_runtime()

        threads = [threading.Thread(target=_worker, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert all(r is results[0] for r in results)
        shell_mod._dait_instance = None

    def test_runtime_has_api_property(self):
        shell_mod._dait_instance = None
        rt = get_dait_runtime()
        assert hasattr(rt, "api")
        shell_mod._dait_instance = None

    def test_api_is_process_object(self):
        shell_mod._dait_instance = None
        rt = get_dait_runtime()
        from domains.shell.runtime import APIServerProcess
        assert isinstance(rt.api, APIServerProcess)
        shell_mod._dait_instance = None

    def test_has_status_summary(self):
        shell_mod._dait_instance = None
        rt = get_dait_runtime()
        assert hasattr(rt, "status_summary")
        shell_mod._dait_instance = None

    def test_status_summary_is_string(self):
        shell_mod._dait_instance = None
        rt = get_dait_runtime()
        summary = rt.status_summary
        assert isinstance(summary, str)
        shell_mod._dait_instance = None

    def test_status_summary_mentions_kernel(self):
        shell_mod._dait_instance = None
        rt = get_dait_runtime()
        summary = rt.status_summary
        assert "Kernel" in summary or "kernel" in summary
        shell_mod._dait_instance = None

    def test_has_init_system_attr(self):
        shell_mod._dait_instance = None
        rt = get_dait_runtime()
        assert hasattr(rt, "init_system")
        shell_mod._dait_instance = None

    def test_has_devices_attr(self):
        shell_mod._dait_instance = None
        rt = get_dait_runtime()
        assert hasattr(rt, "devices")
        shell_mod._dait_instance = None

    def test_has_vfs_attr(self):
        shell_mod._dait_instance = None
        rt = get_dait_runtime()
        assert hasattr(rt, "vfs")
        shell_mod._dait_instance = None
