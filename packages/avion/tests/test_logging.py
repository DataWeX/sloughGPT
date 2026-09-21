"""Logging tests — entries, context, kinds, save."""

import json
import os
import tempfile

from arken.logging import StructuredLogger


class TestEntries:
    def test_info_entry(self):
        log = StructuredLogger("t")
        log.info("hello")
        assert log.entries[0]["message"] == "hello"
        assert log.entries[0]["kind"] == "info"

    def test_context_attached(self):
        log = StructuredLogger("t")
        log.set_context(run="1")
        log.info("x")
        assert log.entries[0]["run"] == "1"

    def test_navigation_marks_failure(self):
        log = StructuredLogger("t")
        log.navigation("/chat", True)
        log.navigation("/nope", False)
        assert log.entries[0]["success"] is True
        assert log.entries[1]["success"] is False

    def test_element_task_helpers(self):
        log = StructuredLogger("t")
        log.element("clicked", "text='Start'")
        log.task_start("chat")
        log.task_end("chat", True)
        kinds = [e["kind"] for e in log.entries]
        assert kinds == ["element", "task_start", "task_end"]

    def test_of_kind_filters(self):
        log = StructuredLogger("t")
        log.info("a")
        log.navigation("/x", True)
        assert len(log.of_kind("navigation")) == 1

    def test_save_and_clear(self):
        log = StructuredLogger("t")
        log.info("a")
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "logs.json")
            log.save(p)
            assert json.load(open(p))[0]["message"] == "a"
        log.clear()
        assert log.entries == []
