#!/usr/bin/env python3
"""Merge sharded pytest-testmon recording DBs into one selection DB.

Card 20261107_112103 (ci(xdist): parallelize test-core-py): main's record
pass fans out across N runners (``SLO_TEST_SHARD=k/n``), but PRs restore
ONE ``.testmondata`` (``scripts/run_affected.sh`` + ``actions/cache``), so
the fragments must become a single DB before the single-writer cache save.

pytest-testmon 2.3.1 has no merge path (no CLI, no API), so this script
moves rows with stdlib ``sqlite3`` only. The target starts as a byte-copy
of shard 1: the schema comes from a real recording, never from a
hand-copied DDL — a testmon schema change breaks the INSERTs loudly at
runtime instead of silently drifting a re-declaration.

Rows from shard 2..N are upserted with id remapping:

=============== =============================================
environment     dedup on its UNIQUE triple, remap ids
file_fp         dedup on UNIQUE(filename, fsha, method_checksums)
test_execution  fetch-or-create on (environment, test_name) — the
                same key pytest-testmon itself records with
test_execution_file_fp  remap both ids, INSERT OR IGNORE (PK)
metadata        INSERT OR IGNORE (dataid PK)
=============== =============================================

``suite_execution_file_fsha`` is NOT copied: in 2.3.1 it is dead DDL —
its parent table ``suite_execution`` is never created and nothing in the
package reads or writes it (verified against the wheel: the only INSERT
targets are file_fp, test_execution, test_execution_file_fp,
environment and two TEMP tables).

Verification is strict because the failure mode matters: an
under-selecting DB is worse than no DB — PRs would silently skip
affected tests. Violations exit 1 and the cache save never happens;
run_affected's documented fallback (no DB -> full run, over-selecting)
is the safe direction.

Usage::

    python scripts/merge_testmon.py OUT SHARD [SHARD...]

Exit 0 prints a one-line summary; any violation exits 1 with the reason.
"""

from __future__ import annotations

import shutil
import sqlite3
import sys
from pathlib import Path

# Columns per table, pinned to pytest-testmon 2.3.1 (the CI install).
# Deliberately explicit: a column rename in a future testmon must break
# this script loudly, not silently map the wrong data.
_ENV_COLS = ("environment_name", "system_packages", "python_version")
_FP_COLS = ("filename", "method_checksums", "mtime", "fsha")
_TE_COLS = ("duration", "failed", "forced")
_META_COLS = ("dataid", "data")


class MergeError(RuntimeError):
    """A shard violates the merge contract — fail loud, save nothing."""


def _connect(path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(path)
    # Autocommit: python's implicit transaction would stay open across the
    # src reads and make DETACH fail ("database src is locked"). Atomicity
    # does not depend on it — a failure unlinks the whole output file.
    con.isolation_level = None
    con.execute("PRAGMA foreign_keys = OFF")
    return con


def _materialize(path: Path) -> None:
    """Fold any WAL sidecar (``path-wal``) into the main file.

    Artifacts upload ``.testmondata*`` so sidecars survive the runner
    teardown, but a byte-copy of the main file alone would drop their
    committed rows. Checkpoint-TRUNCATE is a no-op outside WAL mode.
    """
    con = sqlite3.connect(path)
    try:
        con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    finally:
        con.close()


def _user_version(path: Path) -> int:
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        return int(con.execute("PRAGMA user_version").fetchone()[0])
    finally:
        con.close()


def _fetch_or_create(
    dst: sqlite3.Connection,
    table: str,
    key_cols: tuple[str, ...],
    key: tuple,
    set_cols: tuple[str, ...],
    values: tuple,
) -> int:
    """Return the row id for *key* in *table*, inserting it if absent.

    Same semantics as pytest-testmon's own fetch_or_create: first writer
    wins (e.g. file_fp's mtime stays the base shard's).
    """
    where = " AND ".join(f"{c} = ?" for c in key_cols)
    row = dst.execute(
        f"SELECT id FROM {table} WHERE {where}",  # noqa: S608 - identifiers are ours
        key,
    ).fetchone()
    if row is not None:
        return int(row[0])
    cols = ", ".join(set_cols)
    marks = ", ".join("?" for _ in set_cols)
    cur = dst.execute(
        f"INSERT INTO {table} ({cols}) VALUES ({marks})",  # noqa: S608 - identifiers are ours
        values,
    )
    return int(cur.lastrowid)


def _merge_one(dst: sqlite3.Connection, shard: Path) -> None:
    """Upsert every live row of *shard* into *dst* (attached as `src`)."""
    dst.execute("ATTACH DATABASE ? AS src", (str(shard),))
    try:
        # Fresh id maps per shard: integer ids from different shard DBs
        # are unrelated, so stale entries must never leak across shards.
        for name in ("map_env", "map_fp", "map_te"):
            dst.execute(f"DROP TABLE IF EXISTS temp.{name}")  # noqa: S608
            dst.execute(
                f"CREATE TEMP TABLE {name}("  # noqa: S608
                "old_id INTEGER PRIMARY KEY, new_id INTEGER)"
            )

        # environment: dedup on the UNIQUE triple.
        for eid, *vals in dst.execute(
            f"SELECT id, {', '.join(_ENV_COLS)} FROM src.environment"  # noqa: S608
        ):
            new_id = _fetch_or_create(
                dst, "environment", _ENV_COLS, tuple(vals), _ENV_COLS, tuple(vals)
            )
            dst.execute("INSERT OR REPLACE INTO map_env VALUES (?, ?)", (eid, new_id))

        # file_fp: dedup on the UNIQUE content key (mtime excluded, as in
        # the constraint — two shards may have touched the same file).
        for fid, filename, checksums, mtime, fsha in dst.execute(
            f"SELECT id, {', '.join(_FP_COLS)} FROM src.file_fp"  # noqa: S608
        ):
            new_id = _fetch_or_create(
                dst,
                "file_fp",
                ("filename", "fsha", "method_checksums"),
                (filename, fsha, checksums),
                _FP_COLS,
                (filename, checksums, mtime, fsha),
            )
            dst.execute("INSERT OR REPLACE INTO map_fp VALUES (?, ?)", (fid, new_id))

        # test_execution: fetch-or-create on (environment, test_name).
        for tid, env_id, name, duration, failed, forced in dst.execute(
            f"SELECT id, environment_id, test_name, {', '.join(_TE_COLS)} "  # noqa: S608
            "FROM src.test_execution"
        ):
            env_row = dst.execute(
                "SELECT new_id FROM map_env WHERE old_id = ?", (env_id,)
            ).fetchone()
            if env_row is None:
                raise MergeError(
                    f"{shard}: test_execution id={tid} references "
                    f"environment id={env_id} missing from the shard"
                )
            new_id = _fetch_or_create(
                dst,
                "test_execution",
                ("environment_id", "test_name"),
                (env_row[0], name),
                ("environment_id", "test_name") + _TE_COLS,
                (env_row[0], name, duration, failed, forced),
            )
            dst.execute("INSERT OR REPLACE INTO map_te VALUES (?, ?)", (tid, new_id))

        # Links: bulk remap through the temp id maps. An inner join means
        # an unmapped id drops the row — the count verification below
        # turns any such loss into a hard failure.
        dst.execute(
            """
            INSERT OR IGNORE INTO test_execution_file_fp
                (test_execution_id, fingerprint_id)
            SELECT mte.new_id, mfp.new_id
            FROM src.test_execution_file_fp l
            JOIN map_te mte ON mte.old_id = l.test_execution_id
            JOIN map_fp mfp ON mfp.old_id = l.fingerprint_id
            """
        )

        # metadata: dataid is the PK; identical across shards in practice.
        dst.execute(
            f"INSERT OR IGNORE INTO metadata ({', '.join(_META_COLS)}) "  # noqa: S608
            f"SELECT {', '.join(_META_COLS)} FROM src.metadata"  # noqa: S608
        )
        # suite_execution_file_fsha intentionally not copied (dead table).
    finally:
        dst.execute("DETACH DATABASE src")


def _verify(dst_path: Path, shards: list[Path]) -> dict[str, int]:
    """Strict contract checks — see the module docstring for the rationale."""
    dst = _connect(dst_path)
    try:
        dst_te = dst.execute("SELECT count(*) FROM test_execution").fetchone()[0]
        dst_links = dst.execute("SELECT count(*) FROM test_execution_file_fp").fetchone()[0]
        dst_fp = dst.execute("SELECT count(*) FROM file_fp").fetchone()[0]
        dst_env = dst.execute("SELECT count(*) FROM environment").fetchone()[0]
        orphans = dst.execute("PRAGMA foreign_key_check").fetchall()
    finally:
        dst.close()

    te_total = links_total = 0
    fp_union: set[tuple] = set()
    env_union: set[tuple] = set()
    for shard in shards:
        con = _connect(shard)
        try:
            te_total += con.execute("SELECT count(*) FROM test_execution").fetchone()[0]
            links_total += con.execute("SELECT count(*) FROM test_execution_file_fp").fetchone()[0]
            fp_union.update(
                (r[0], r[1], r[2])
                for r in con.execute(
                    "SELECT filename, fsha, method_checksums FROM file_fp"  # noqa: S608
                )
            )
            env_union.update(
                tuple(r)
                for r in con.execute(
                    f"SELECT {', '.join(_ENV_COLS)} FROM environment"  # noqa: S608
                )
            )
        finally:
            con.close()

    problems = []
    # Splits are disjoint by construction (round-robin over one collection
    # order): equality proves no row was lost AND no test landed in two
    # shards (a split-contract violation, not a merge hazard to paper over).
    if dst_te != te_total:
        problems.append(
            f"test_execution: merged {dst_te} != sum of shards {te_total} "
            "(row lost, or the same test recorded in two shards)"
        )
    if dst_links != links_total:
        problems.append(
            f"test_execution_file_fp: merged {dst_links} != sum of shards "
            f"{links_total} (fingerprint links lost)"
        )
    if dst_fp != len(fp_union):
        problems.append(
            f"file_fp: merged {dst_fp} != union of shards {len(fp_union)} (fingerprint dedup wrong)"
        )
    if dst_env != len(env_union):
        problems.append(f"environment: merged {dst_env} != union of shards {len(env_union)}")
    if orphans:
        problems.append(f"foreign_key_check found {len(orphans)} orphan row(s)")
    if problems:
        raise MergeError("; ".join(problems))
    return {
        "tests": int(dst_te),
        "fingerprint_links": int(dst_links),
        "files": int(dst_fp),
        "environments": int(dst_env),
    }


def merge(out: Path, shards: list[Path]) -> dict[str, int]:
    """Merge *shards* into a fresh *out* DB. Returns verification stats."""
    if len(shards) < 2:
        raise MergeError("need at least 2 shard DBs to merge")
    if out.resolve() in {p.resolve() for p in shards}:
        raise MergeError(f"output path must differ from every shard: {out}")
    for p in shards:
        if not p.is_file():
            raise MergeError(f"shard not found: {p}")
        _materialize(p)
    versions = {p: _user_version(p) for p in shards}
    if len(set(versions.values())) != 1:
        raise MergeError(
            f"pytest-testmon schema version mismatch across shards: "
            f"{ {str(p): v for p, v in versions.items()} }"
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(shards[0], out)  # schema comes from a real recording
    try:
        dst = _connect(out)
        try:
            for shard in shards[1:]:
                _merge_one(dst, shard)
        finally:
            dst.close()
        return _verify(out, shards)
    except Exception:
        # A half-merged DB must never be mistaken for a selection basis:
        # remove it so a caller that saves unconditionally saves nothing.
        out.unlink(missing_ok=True)
        raise


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print(__doc__.split("Usage::")[1].strip().splitlines()[0], file=sys.stderr)
        return 2
    out = Path(argv[1])
    shards = [Path(a) for a in argv[2:]]
    try:
        stats = merge(out, shards)
    except MergeError as exc:
        print(f"merge_testmon: FAILED — {exc}", file=sys.stderr)
        return 1
    print(
        f"merge_testmon: {len(shards)} shards -> {out} "
        f"({stats['tests']} tests, {stats['fingerprint_links']} links, "
        f"{stats['files']} files, {stats['environments']} environments)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
