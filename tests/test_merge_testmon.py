"""Sharded testmon recording merge (card 20261107_112103).

The contract PR selection depends on: the shards' union is exact (no row
lost, no test landing in two shards), shared fingerprints dedup to one
row, and any violation fails loud — an under-selecting ``.testmondata``
would silently skip affected tests on PRs, which is strictly worse than
having none at all (``run_affected.sh``'s no-DB fallback runs everything).

DDL mirrors pytest-testmon 2.3.1 (``testmon/db.py`` ``init_tables``),
including the dead ``suite_execution_file_fsha`` table, so the merge is
exercised against the real schema shape without importing the package —
pytest-testmon is a CI-only dependency, never installed locally.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from scripts import merge_testmon

# Verbatim from pytest-testmon 2.3.1 testmon/db.py init_tables (the
# dead suite table included — the merge must cope with a base that has it).
_DDL = """
CREATE TABLE metadata (dataid TEXT PRIMARY KEY, data TEXT);
CREATE TABLE environment (
    id INTEGER PRIMARY KEY ASC,
    environment_name TEXT,
    system_packages TEXT,
    python_version TEXT,
    UNIQUE (environment_name, system_packages, python_version)
);
CREATE TABLE test_execution (
    id INTEGER PRIMARY KEY ASC,
    environment_id INTEGER,
    test_name TEXT,
    duration FLOAT,
    failed BIT,
    forced BIT,
    FOREIGN KEY(environment_id) REFERENCES environment(id) ON DELETE CASCADE
);
CREATE INDEX test_execution_fk_name ON test_execution (environment_id, test_name);
CREATE TABLE file_fp (
    id INTEGER PRIMARY KEY,
    filename TEXT,
    method_checksums BLOB,
    mtime FLOAT,
    fsha TEXT,
    UNIQUE (filename, fsha, method_checksums)
);
CREATE TABLE test_execution_file_fp (
    test_execution_id INTEGER,
    fingerprint_id INTEGER,
    PRIMARY KEY (test_execution_id, fingerprint_id),
    FOREIGN KEY(test_execution_id) REFERENCES test_execution(id) ON DELETE CASCADE,
    FOREIGN KEY(fingerprint_id) REFERENCES file_fp(id)
) WITHOUT ROWID;
CREATE TABLE suite_execution_file_fsha (
    suite_execution_id INTEGER,
    filename TEXT,
    fsha text,
    FOREIGN KEY(suite_execution_id) REFERENCES suite_execution(id) ON DELETE CASCADE
);
"""

ENV = ("cpython", "pip freeze", "3.12.16")


def _shard(
    path: Path,
    tests: dict[str, list[tuple[str, str, bytes]]],
    *,
    env: tuple[str, str, str] = ENV,
    version: int = 23,
) -> Path:
    """Create a shard DB: {test_name: [(filename, fsha, checksums), ...]}.

    Fingerprints dedup on (filename, fsha, checksums) within the shard
    exactly like testmon's fetch_or_create; mtime is fixed at 1.0 and
    varied only by the tests that need it.
    """
    con = sqlite3.connect(path)
    try:
        con.executescript(_DDL)
        # PRAGMA takes no bound parameters; version is our own int.
        con.execute(f"PRAGMA user_version = {int(version)}")  # noqa: S608
        env_id = con.execute(
            "INSERT INTO environment (environment_name, system_packages, "
            "python_version) VALUES (?, ?, ?)",
            env,
        ).lastrowid
        fp_ids: dict[tuple[str, str, bytes], int] = {}
        for test_name, fingerprints in tests.items():
            te_id = con.execute(
                "INSERT INTO test_execution (environment_id, test_name, "
                "duration, failed, forced) VALUES (?, ?, 0.01, 0, 0)",
                (env_id, test_name),
            ).lastrowid
            for key in fingerprints:
                if key not in fp_ids:
                    fp_ids[key] = con.execute(
                        "INSERT INTO file_fp (filename, method_checksums, mtime, "
                        "fsha) VALUES (?, ?, 1.0, ?)",
                        (key[0], key[2], key[1]),
                    ).lastrowid
                con.execute(
                    "INSERT INTO test_execution_file_fp VALUES (?, ?)",
                    (te_id, fp_ids[key]),
                )
        con.commit()
    finally:
        con.close()
    return path


def _rows(path: Path, sql: str) -> list[tuple]:
    con = sqlite3.connect(path)
    try:
        return con.execute(sql).fetchall()
    finally:
        con.close()


class TestMergeContract:
    def test_disjoint_shards_union_exactly(self, tmp_path: Path) -> None:
        a = _shard(
            tmp_path / "a.db",
            {
                "tests/test_a.py::test_x": [("domain/x.py", "sha-x", b"cx")],
                "tests/test_b.py::test_y": [
                    ("domain/x.py", "sha-x", b"cx"),
                    ("domain/y.py", "sha-y", b"cy"),
                ],
            },
        )
        b = _shard(
            tmp_path / "b.db",
            {
                "tests/test_c.py::test_z": [
                    ("domain/y.py", "sha-y", b"cy"),
                    ("domain/z.py", "sha-z", b"cz"),
                ],
                "tests/test_d.py::test_w": [("domain/z.py", "sha-z", b"cz")],
            },
        )
        out = tmp_path / ".testmondata"
        stats = merge_testmon.merge(out, [a, b])
        assert stats == {
            "tests": 4,
            "fingerprint_links": 6,
            "files": 3,
            "environments": 1,
        }
        # Each test keeps exactly its own fingerprints after the id remap.
        links = _rows(
            out,
            "SELECT te.test_name, f.filename FROM test_execution te "
            "JOIN test_execution_file_fp l ON l.test_execution_id = te.id "
            "JOIN file_fp f ON f.id = l.fingerprint_id "
            "ORDER BY te.test_name, f.filename",
        )
        assert links == [
            ("tests/test_a.py::test_x", "domain/x.py"),
            ("tests/test_b.py::test_y", "domain/x.py"),
            ("tests/test_b.py::test_y", "domain/y.py"),
            ("tests/test_c.py::test_z", "domain/y.py"),
            ("tests/test_c.py::test_z", "domain/z.py"),
            ("tests/test_d.py::test_w", "domain/z.py"),
        ]

    def test_shared_fingerprint_dedups_keeping_base_mtime(self, tmp_path: Path) -> None:
        a = _shard(tmp_path / "a.db", {"t1": [("domain/x.py", "sha-x", b"cx")]})
        b = _shard(tmp_path / "b.db", {"t2": [("domain/x.py", "sha-x", b"cx")]})
        # Second shard touched the file later — content key is identical,
        # so one row survives and the BASE shard's mtime wins (first
        # writer, the same semantics as testmon's own fetch_or_create).
        con = sqlite3.connect(b)
        con.execute("UPDATE file_fp SET mtime = 2.0")
        con.commit()
        con.close()
        out = tmp_path / ".testmondata"
        stats = merge_testmon.merge(out, [a, b])
        assert stats["files"] == 1
        assert stats["tests"] == 2
        assert _rows(out, "SELECT mtime FROM file_fp") == [(1.0,)]

    def test_same_test_name_in_different_environments_kept_apart(self, tmp_path: Path) -> None:
        a = _shard(
            tmp_path / "a.db",
            {"tests/test_x.py::test_it": [("domain/x.py", "sha-x", b"cx")]},
            env=("cpython", "pip freeze", "3.12.16"),
        )
        b = _shard(
            tmp_path / "b.db",
            {"tests/test_x.py::test_it": [("domain/x.py", "sha-x", b"cx")]},
            env=("cpython", "pip freeze", "3.11.9"),
        )
        out = tmp_path / ".testmondata"
        stats = merge_testmon.merge(out, [a, b])
        # The key is (environment, test_name): two envs, two rows, and the
        # shared fingerprint still dedups to one file row.
        assert stats["tests"] == 2
        assert stats["environments"] == 2
        assert stats["files"] == 1

    def test_dead_suite_table_is_copied_not_forgotten(self, tmp_path: Path) -> None:
        a = _shard(tmp_path / "a.db", {"t1": [("f.py", "s", b"c")]})
        b = _shard(tmp_path / "b.db", {"t2": [("f.py", "s", b"c")]})
        out = tmp_path / ".testmondata"
        merge_testmon.merge(out, [a, b])
        # Schema comes from the base copy: the dead table exists (empty —
        # nothing in testmon ever writes it) and the merge never trips on it.
        tables = {r[0] for r in _rows(out, "SELECT name FROM sqlite_master")}
        assert "suite_execution_file_fsha" in tables
        assert _rows(out, "SELECT count(*) FROM suite_execution_file_fsha") == [(0,)]


class TestMergeFailsLoud:
    def test_duplicate_test_across_shards_fails(self, tmp_path: Path) -> None:
        a = _shard(tmp_path / "a.db", {"t_dup": [("f.py", "s", b"c")]})
        b = _shard(tmp_path / "b.db", {"t_dup": [("f.py", "s", b"c")]})
        with pytest.raises(merge_testmon.MergeError, match="test_execution"):
            merge_testmon.merge(tmp_path / ".testmondata", [a, b])
        # The half-merged base copy is removed: no caller can ever save a
        # partial selection basis (CI would under-select silently).
        assert not (tmp_path / ".testmondata").exists()

    def test_schema_version_mismatch_fails(self, tmp_path: Path) -> None:
        a = _shard(tmp_path / "a.db", {"t1": [("f.py", "s", b"c")]}, version=23)
        b = _shard(tmp_path / "b.db", {"t2": [("f.py", "s", b"c")]}, version=99)
        with pytest.raises(merge_testmon.MergeError, match="version mismatch"):
            merge_testmon.merge(tmp_path / ".testmondata", [a, b])

    def test_single_shard_rejected(self, tmp_path: Path) -> None:
        a = _shard(tmp_path / "a.db", {"t1": [("f.py", "s", b"c")]})
        with pytest.raises(merge_testmon.MergeError, match="at least 2"):
            merge_testmon.merge(tmp_path / ".testmondata", [a])

    def test_missing_shard_rejected(self, tmp_path: Path) -> None:
        a = _shard(tmp_path / "a.db", {"t1": [("f.py", "s", b"c")]})
        with pytest.raises(merge_testmon.MergeError, match="not found"):
            merge_testmon.merge(tmp_path / ".testmondata", [a, tmp_path / "absent.db"])

    def test_output_must_differ_from_shards(self, tmp_path: Path) -> None:
        a = _shard(tmp_path / "a.db", {"t1": [("f.py", "s", b"c")]})
        b = _shard(tmp_path / "b.db", {"t2": [("f.py", "s", b"c")]})
        with pytest.raises(merge_testmon.MergeError, match="differ"):
            merge_testmon.merge(a, [a, b])


class TestCli:
    def test_usage_error_exits_2(self, capsys: pytest.CaptureFixture) -> None:
        assert merge_testmon.main(["merge_testmon.py"]) == 2
        assert "merge_testmon.py OUT SHARD" in capsys.readouterr().err

    def test_failed_merge_exits_1_with_reason(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        a = _shard(tmp_path / "a.db", {"t1": [("f.py", "s", b"c")]})
        rc = merge_testmon.main(["merge_testmon.py", str(tmp_path / "out"), str(a)])
        assert rc == 1
        assert "FAILED" in capsys.readouterr().err

    def test_successful_merge_exits_0_with_summary(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        a = _shard(tmp_path / "a.db", {"t1": [("f.py", "s", b"c")]})
        b = _shard(tmp_path / "b.db", {"t2": [("g.py", "s2", b"d")]})
        rc = merge_testmon.main(["merge_testmon.py", str(tmp_path / "out"), str(a), str(b)])
        assert rc == 0
        out = capsys.readouterr().out
        assert "2 shards" in out and "2 tests" in out
