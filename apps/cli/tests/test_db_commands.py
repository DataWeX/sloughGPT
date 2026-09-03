"""Tests for the `slo db` command group — MogDB migration, sync, and status."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import cli as _cli  # noqa: E402


class TestDbGroup:
    def test_db_group_registered(self):
        root = _cli.cli
        assert "db" in root.groups

    def test_db_subcommands_present(self):
        db_group = _cli.cli.groups["db"]
        assert set(db_group.commands.keys()) == {"migrate", "sync", "status"}

    def test_db_migrate_has_required_key_option(self):
        db_group = _cli.cli.groups["db"]
        migrate = db_group.commands["migrate"]
        dests = {opt.dest for opt in migrate.options}
        assert "key" in dests
        assert "file" in dests

    def test_db_status_has_json_flag(self):
        db_group = _cli.cli.groups["db"]
        status = db_group.commands["status"]
        dests = {opt.dest for opt in status.options}
        assert "as_json" in dests
