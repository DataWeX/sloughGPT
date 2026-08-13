"""Tests for the CLI memory commands (apps/cli/src/commands/memory.py)."""
import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from unittest.mock import MagicMock  # noqa: E402


@pytest.fixture(autouse=True)
def fake_service(monkeypatch):
    """Substitute a fake MemoryService for all command tests."""
    svc = MagicMock()
    svc.enabled = True
    svc.stats.return_value = {"total_facts": 2, "topics": 1}
    svc.list_all.return_value = [
        {"content": "The capital of France is Paris", "topic": "geo",
         "source": "task", "score": 0.5},
    ]
    svc.retrieve.return_value = [
        {"content": "The capital of France is Paris", "topic": "geo",
         "source": "task", "score": 0.9},
    ]
    svc.store.return_value = True
    svc.remember.return_value = True
    svc.clear.return_value = 2
    import commands.memory as mem
    monkeypatch.setattr(mem, "get_memory_service", lambda: svc)
    return svc


def _ns(**kwargs):
    from types import SimpleNamespace
    return SimpleNamespace(**kwargs)


class TestStats:
    def test_prints_enabled_and_facts(self, fake_service, capsys):
        from commands.memory import cmd_memory_stats
        cmd_memory_stats(_ns())
        out = capsys.readouterr().out
        assert "Facts" in out
        assert "2" in out
        fake_service.stats.assert_called_once()


class TestEnable:
    def test_enable_turns_memory_on(self, fake_service, capsys):
        from commands.memory import cmd_memory_enable
        cmd_memory_enable(_ns(enabled=True))
        fake_service.set_enabled.assert_called_once_with(True)
        assert "enabled" in capsys.readouterr().out

    def test_disable_turns_memory_off(self, fake_service, capsys):
        from commands.memory import cmd_memory_enable
        cmd_memory_enable(_ns(enabled=False))
        fake_service.set_enabled.assert_called_once_with(False)
        assert "disabled" in capsys.readouterr().out


class TestList:
    def test_lists_items(self, fake_service, capsys):
        from commands.memory import cmd_memory_list
        cmd_memory_list(_ns(limit=10))
        out = capsys.readouterr().out
        assert "Paris" in out
        fake_service.list_all.assert_called_with(limit=10)

    def test_empty_list_prints_hint(self, fake_service, capsys):
        fake_service.list_all.return_value = []
        from commands.memory import cmd_memory_list
        cmd_memory_list(_ns(limit=10))
        out = capsys.readouterr().out
        assert "No memory stored" in out


class TestSearch:
    def test_missing_query_exits(self):
        from commands.memory import cmd_memory_search
        with pytest.raises(SystemExit) as exc:
            cmd_memory_search(_ns(query="", limit=5))
        assert exc.value.code == 2

    def test_prints_matches(self, fake_service, capsys):
        from commands.memory import cmd_memory_search
        cmd_memory_search(_ns(query="france", limit=3))
        out = capsys.readouterr().out
        assert "Paris" in out
        fake_service.retrieve.assert_called_with("france", limit=3)

    def test_no_matches_prints_hint(self, fake_service, capsys):
        fake_service.retrieve.return_value = []
        from commands.memory import cmd_memory_search
        cmd_memory_search(_ns(query="xyz", limit=3))
        assert "No memory matches" in capsys.readouterr().out


class TestStore:
    def test_missing_content_exits(self):
        from commands.memory import cmd_memory_store
        with pytest.raises(SystemExit) as exc:
            cmd_memory_store(_ns(content="", topic="t", source="cli"))
        assert exc.value.code == 2

    def test_stores_fact(self, fake_service, capsys):
        from commands.memory import cmd_memory_store
        cmd_memory_store(_ns(content="a fact", topic="t", source="cli"))
        fake_service.store.assert_called_with("a fact", "t", "cli")
        assert "Stored" in capsys.readouterr().out

    def test_failed_store_warns(self, fake_service, capsys):
        fake_service.store.return_value = False
        from commands.memory import cmd_memory_store
        cmd_memory_store(_ns(content="dup", topic="t", source="cli"))
        assert "Not stored" in capsys.readouterr().out


class TestRemember:
    def test_missing_fields_exit(self):
        from commands.memory import cmd_memory_remember
        with pytest.raises(SystemExit):
            cmd_memory_remember(_ns(user_message="", assistant_response=""))
        with pytest.raises(SystemExit):
            cmd_memory_remember(_ns(user_message="hi", assistant_response=""))

    def test_remember_stores_turn(self, fake_service, capsys):
        from commands.memory import cmd_memory_remember
        cmd_memory_remember(_ns(user_message="q", assistant_response="a"))
        fake_service.remember.assert_called_with("q", "a")
        assert "stored" in capsys.readouterr().out


class TestClear:
    def test_clear_requires_confirmation(self, fake_service, capsys, monkeypatch):
        import click
        monkeypatch.setattr(click, "confirm", lambda *a, **k: True)
        from commands.memory import cmd_memory_clear
        cmd_memory_clear(_ns(yes=False))
        fake_service.clear.assert_called_once()

    def test_clear_skipped_when_declined(self, fake_service, monkeypatch):
        import click
        monkeypatch.setattr(click, "confirm", lambda *a, **k: False)
        from commands.memory import cmd_memory_clear
        cmd_memory_clear(_ns(yes=False))
        fake_service.clear.assert_not_called()

    def test_clear_yes_flag(self, fake_service, capsys):
        from commands.memory import cmd_memory_clear
        cmd_memory_clear(_ns(yes=True))
        fake_service.clear.assert_called_once()
        assert "Cleared 2" in capsys.readouterr().out


class TestConsolidate:
    SHORT = "Machine learning learns patterns from data."
    LONG = "Machine learning learns patterns from data very effectively."
    DISTINCT = "Neural networks recognize images well."

    def test_merges_near_duplicates(self, fake_service, capsys):
        fake_service.list_all.return_value = [
            {"id": "fact_1", "content": self.SHORT, "topic": "ml"},
            {"id": "fact_2", "content": self.LONG, "topic": "ml"},
        ]
        fake_service.delete.return_value = 1
        from commands.memory import cmd_memory_consolidate
        cmd_memory_consolidate(_ns(threshold=0.80))
        assert "Consolidated 1" in capsys.readouterr().out
        fake_service.delete.assert_called_with(["fact_1"])

    def test_merges_near_duplicates_at_default_threshold(self, fake_service, capsys):
        """Default threshold (0.80) collapses near-verbatim copies (~0.845)."""
        fake_service.list_all.return_value = [
            {"id": "fact_1", "content": self.SHORT, "topic": "ml"},
            {"id": "fact_2", "content": self.LONG, "topic": "ml"},
        ]
        fake_service.delete.return_value = 1
        from commands.memory import cmd_memory_consolidate
        cmd_memory_consolidate(_ns(threshold=None))
        assert "Consolidated 1" in capsys.readouterr().out
        fake_service.delete.assert_called_with(["fact_1"])

    def test_no_duplicates_at_default_threshold(self, fake_service, capsys):
        """Default threshold keeps distinct facts about the same topic."""
        fake_service.list_all.return_value = [
            {"id": "fact_1", "content": self.SHORT, "topic": "ml"},
            {"id": "fact_2", "content": self.DISTINCT, "topic": "ml"},
        ]
        from commands.memory import cmd_memory_consolidate
        cmd_memory_consolidate(_ns(threshold=None))
        out = capsys.readouterr().out
        assert "No near-duplicates" in out
        fake_service.delete.assert_not_called()

    def test_empty_store_prints_hint(self, fake_service, capsys):
        fake_service.list_all.return_value = []
        from commands.memory import cmd_memory_consolidate
        cmd_memory_consolidate(_ns(threshold=0.80))
        assert "No memory to consolidate" in capsys.readouterr().out


class TestArchive:
    STATS = {
        "path": "/tmp/data/memory/facts.jsonl",
        "records": 2,
        "bytes": 120,
        "task_types": {"memory.store": 2},
        "oldest_ts": 100.0,
        "newest_ts": 200.0,
    }

    def _patch(self, monkeypatch, stats=None, records=None):
        from domains.memory import task_memory as tm
        monkeypatch.setattr(tm, "archive_stats", lambda: stats if stats is not None else self.STATS)
        monkeypatch.setattr(tm, "list_archive",
                            lambda limit: records if records is not None else
                            [{"ts": 200.0, "task_type": "memory.store", "task_id": "t1",
                              "content": "Redwoods can live over 2000 years"}])

    def test_archive_shows_stats_and_records(self, monkeypatch, capsys):
        self._patch(monkeypatch)
        from commands.memory import cmd_memory_archive
        cmd_memory_archive(_ns(limit=10, prune_days=None))
        out = capsys.readouterr().out
        assert "Records" in out
        assert "2" in out
        assert "Redwoods" in out

    def test_archive_empty_stats(self, monkeypatch, capsys):
        self._patch(monkeypatch, stats={"path": "/x", "records": 0, "bytes": 0,
                                        "task_types": {}, "oldest_ts": None, "newest_ts": None})
        from commands.memory import cmd_memory_archive
        cmd_memory_archive(_ns(limit=10, prune_days=None))
        assert "Records" in capsys.readouterr().out

    def test_archive_prune_confirmed(self, monkeypatch, capsys):
        from domains.memory import task_memory as tm
        calls = []
        monkeypatch.setattr(tm, "prune_archive", lambda retain_days: calls.append(retain_days) or 3)
        import click
        monkeypatch.setattr(click, "confirm", lambda *a, **k: True)
        from commands.memory import cmd_memory_archive
        cmd_memory_archive(_ns(limit=10, prune_days=30))
        assert calls == [30.0]
        assert "Pruned 3" in capsys.readouterr().out

    def test_archive_prune_declined(self, monkeypatch):
        from domains.memory import task_memory as tm
        calls = []
        monkeypatch.setattr(tm, "prune_archive", lambda retain_days: calls.append(retain_days) or 3)
        import click
        monkeypatch.setattr(click, "confirm", lambda *a, **k: False)
        from commands.memory import cmd_memory_archive
        cmd_memory_archive(_ns(limit=10, prune_days=30))
        assert calls == []
