"""Tests for downcraft.download.state — persistent download state."""

import json
import os
import tempfile
from pathlib import Path

import pytest

from downcraft.download.state import PersistentState, ModelState, FileProgress, get_state


class TestPersistentState:
    def test_create_and_get(self):
        with tempfile.TemporaryDirectory() as td:
            st = PersistentState(Path(td))
            ms = st.create("test-model", "/tmp/cache")
            assert ms.model_id == "test-model"
            assert ms.status == "queued"
            assert ms.cache_dir == "/tmp/cache"
            assert st.get("test-model") is ms

    def test_get_nonexistent(self):
        with tempfile.TemporaryDirectory() as td:
            st = PersistentState(Path(td))
            assert st.get("nonexistent") is None

    def test_set_status(self):
        with tempfile.TemporaryDirectory() as td:
            st = PersistentState(Path(td))
            st.create("m1", "/cache")
            st.set_status("m1", "complete")
            assert st.get("m1").status == "complete"
            assert st.get("m1").completed_at is not None

    def test_set_status_with_error(self):
        with tempfile.TemporaryDirectory() as td:
            st = PersistentState(Path(td))
            st.create("m1", "/cache")
            st.set_status("m1", "failed", error="connection lost")
            assert st.get("m1").error == "connection lost"

    def test_set_status_nonexistent(self):
        with tempfile.TemporaryDirectory() as td:
            st = PersistentState(Path(td))
            st.set_status("ghost", "complete")  # should not raise

    def test_update_file_progress_creates_entry(self):
        with tempfile.TemporaryDirectory() as td:
            st = PersistentState(Path(td))
            st.create("m1", "/cache")
            st.update_file_progress("m1", "weights.bin", "https://x.com/w", 100, 200)
            ms = st.get("m1")
            assert "weights.bin" in ms.files
            assert ms.files["weights.bin"].bytes_downloaded == 100
            assert ms.files["weights.bin"].total_bytes == 200
            assert ms.status == "downloading"

    def test_update_file_progress_updates_existing(self):
        with tempfile.TemporaryDirectory() as td:
            st = PersistentState(Path(td))
            st.create("m1", "/cache")
            st.update_file_progress("m1", "w.bin", "https://x.com/w", 100, 200)
            st.update_file_progress("m1", "w.bin", "https://x.com/w", 200, 200, complete=True)
            fp = st.get("m1").files["w.bin"]
            assert fp.bytes_downloaded == 200
            assert fp.complete is True

    def test_all_files_complete_marks_model_complete(self):
        with tempfile.TemporaryDirectory() as td:
            st = PersistentState(Path(td))
            st.create("m1", "/cache")
            st.update_file_progress("m1", "a.bin", "https://x.com/a", 100, 100, complete=True)
            st.update_file_progress("m1", "b.bin", "https://x.com/b", 200, 200, complete=True)
            ms = st.get("m1")
            assert ms.status == "complete"

    def test_remove(self):
        with tempfile.TemporaryDirectory() as td:
            st = PersistentState(Path(td))
            st.create("m1", "/cache")
            st.remove("m1")
            assert st.get("m1") is None

    def test_list(self):
        with tempfile.TemporaryDirectory() as td:
            st = PersistentState(Path(td))
            st.create("m1", "/cache1")
            st.create("m2", "/cache2")
            assert len(st.list()) == 2

    def test_persistence_across_reload(self):
        with tempfile.TemporaryDirectory() as td:
            st1 = PersistentState(Path(td))
            st1.create("m1", "/cache")
            st1.set_status("m1", "complete")
            st1.flush()

            st2 = PersistentState(Path(td))
            ms = st2.get("m1")
            assert ms is not None
            assert ms.status == "complete"
            assert ms.cache_dir == "/cache"

    def test_persistence_with_file_progress(self):
        with tempfile.TemporaryDirectory() as td:
            st1 = PersistentState(Path(td))
            st1.create("m1", "/cache")
            st1.update_file_progress("m1", "w.bin", "https://x.com/w", 50, 100)
            st1.flush()

            st2 = PersistentState(Path(td))
            ms = st2.get("m1")
            assert ms.files["w.bin"].bytes_downloaded == 50
            assert ms.files["w.bin"].total_bytes == 100

    def test_corrupt_state_file_does_not_crash(self):
        with tempfile.TemporaryDirectory() as td:
            state_file = Path(td) / "state.json"
            state_file.write_text("not valid json{{{")
            st = PersistentState(Path(td))  # should not raise
            assert st.list() == []

    def test_model_state_properties(self):
        ms = ModelState(
            model_id="test",
            status="downloading",
            files={
                "a": FileProgress(path="a", url="u", bytes_downloaded=30, total_bytes=100),
                "b": FileProgress(path="b", url="u", bytes_downloaded=70, total_bytes=100),
            },
        )
        assert ms.total_bytes == 200
        assert ms.bytes_downloaded == 100
        assert ms.percentage == 50.0
        assert ms.files_completed == 0
        assert ms.files_total == 2

    def test_model_state_percentage_zero(self):
        ms = ModelState(model_id="test", status="queued")
        assert ms.percentage == 0.0
        assert ms.files_completed == 0

    def test_flush_writes_to_disk(self):
        with tempfile.TemporaryDirectory() as td:
            st = PersistentState(Path(td))
            st.create("m1", "/cache")
            st.flush()
            state_file = Path(td) / "state.json"
            assert state_file.exists()
            data = json.loads(state_file.read_text())
            assert "m1" in data["models"]

    def test_thread_safety_no_crash(self):
        import threading
        with tempfile.TemporaryDirectory() as td:
            st = PersistentState(Path(td))
            errors = []

            def worker():
                try:
                    for i in range(50):
                        st.create(f"m{i}", "/cache")
                        st.set_status(f"m{i}", "complete")
                        st.get(f"m{i}")
                        st.remove(f"m{i}")
                except Exception as e:
                    errors.append(e)

            threads = [threading.Thread(target=worker) for _ in range(4)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            assert not errors, f"Thread safety errors: {errors}"
