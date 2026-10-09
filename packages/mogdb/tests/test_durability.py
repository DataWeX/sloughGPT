"""Crash battery: journal durability, torn-tail recovery, compact crash-consistency (card faa1cfa7)."""

import json
import os
import random
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import pytest
from mogdb.collection import Collection

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = REPO_ROOT / "packages" / "mogdb" / "src"

# Child writer for the cross-process append test: inserts 100 docs whose
# `seq` values are unique across all 6 writers (w * 1000 + i).
WRITER_SCRIPT = (
    "import sys\n"
    "from pathlib import Path\n"
    "from mogdb.collection import Collection\n"
    "c = Collection('hammer', Path(sys.argv[1]), fsync=True)\n"
    "w = int(sys.argv[2])\n"
    "for i in range(100):\n"
    "    c.insert_one({'seq': w * 1000 + i})\n"
)

# Child for the SIGKILL hammer: writes as fast as it can, ACKing every
# insert so the parent knows exactly which seqs were acknowledged.
KILL_CHILD_SCRIPT = (
    "import sys\n"
    "from pathlib import Path\n"
    "from mogdb.collection import Collection\n"
    "c = Collection('hammer', Path(sys.argv[1]), fsync=True)\n"
    "for i in range(10000):\n"
    "    c.insert_one({'seq': i})\n"
    "    print('ACK %d' % i, flush=True)\n"
)


def journal_path(path: Path, name: str) -> Path:
    return path / f"{name}.journal.jsonl"


def snapshot_path(path: Path, name: str) -> Path:
    return path / f"{name}.mogdb"


def seq_path(path: Path, name: str) -> Path:
    return path / f"{name}.seq"


# =========================================================================
# fsync knob
# =========================================================================


class TestFsyncKnob:
    def test_constructor_accepts_fsync_kwarg(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)
            c1 = Collection("c", path, fsync=True)
            assert c1 is not None
            c2 = Collection("c", path, fsync=False)
            assert c2 is not None

    def test_fsync_true_appends_line_terminated(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)
            c = Collection("c", path, fsync=True)
            c.insert_one({"x": 1})
            c.insert_one({"x": 2})
            raw = journal_path(path, "c").read_bytes()
            assert raw.endswith(b"\n")
            lines = raw.splitlines()
            assert len(lines) == 2
            for line in lines:
                json.loads(line)


# =========================================================================
# torn / corrupt tail quarantine
# =========================================================================


class TestTornTailQuarantine:
    def test_partial_tail_line_quarantined_not_silently_dropped(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)
            c = Collection("t", path, fsync=False)
            c.insert_one({"x": 1})
            c.insert_one({"x": 2})
            # Simulate a torn write: half a JSON line with no newline.
            torn = b'{"op": "insert", "data": {"_id": "hal'
            with open(journal_path(path, "t"), "ab") as f:
                f.write(torn)

            c2 = Collection("t", path)
            assert len(c2.find()) == 2
            assert c2.open_report["quarantined"] == 1
            assert c2.open_report["replayed"] == 2
            corrupt = list(path.glob("t.corrupt-*"))
            assert corrupt, "quarantine sidecar <name>.corrupt-<ts> not created"
            assert torn in corrupt[0].read_bytes()

    def test_nul_poisoned_line_quarantined(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)
            c = Collection("t", path)
            c.insert_one({"x": 1})
            with open(journal_path(path, "t"), "ab") as f:
                f.write(b'{"op": "insert", "data": {"x": 3}}\x00\x00\n')

            c2 = Collection("t", path)
            # The poisoned entry must NOT be applied.
            assert c2.count() == 1
            assert c2.open_report["quarantined"] == 1

    def test_corrupt_middle_line_keeps_neighbors(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)
            payload = (
                json.dumps({"op": "insert", "data": {"x": 1}}).encode()
                + b"\n"
                + b"NOT JSON AT ALL\n"
                + json.dumps({"op": "insert", "data": {"x": 2}}).encode()
                + b"\n"
            )
            journal_path(path, "t").write_bytes(payload)

            c = Collection("t", path)
            assert len(c.find()) == 2
            assert c.open_report["replayed"] == 2
            assert c.open_report["quarantined"] == 1

    def test_valid_journal_reports_zero_quarantine(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)
            c = Collection("t", path)
            c.insert_one({"x": 1})
            c.insert_one({"x": 2})

            c2 = Collection("t", path)
            report = c2.open_report
            assert report["quarantined"] == 0
            assert report["replayed"] >= 1


# =========================================================================
# post-compact survival (the current data-loss bug)
# =========================================================================


class TestPostCompactSurvival:
    def test_writes_after_compact_survive_reload(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)
            c = Collection("t", path)
            c.insert_one({"a": 1})
            c.compact()
            c.insert_one({"a": 2})

            c2 = Collection("t", path)
            assert c2.count() == 2

    def test_pre_compact_journal_ops_not_double_applied(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)
            c = Collection("t", path)
            doc_id = c.insert_one({"n": 0})
            c.update_one({"_id": doc_id}, {"$inc": {"n": 1}})
            saved_journal = journal_path(path, "t").read_bytes()
            c.compact()
            # Simulate a crash between snapshot rename and journal truncate:
            # the pre-compact journal bytes are back on disk alongside the
            # snapshot that already bakes in the update.
            journal_path(path, "t").write_bytes(saved_journal)

            c2 = Collection("t", path)
            doc = c2.find_one({"_id": doc_id})
            assert doc is not None
            assert doc["n"] == 1

    def test_stale_tmp_snapshot_ignored(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)
            c = Collection("t", path)
            c.insert_one({"a": 1})
            c.compact()
            # A stale/partial temp file from an interrupted compact must be
            # ignored by load.
            (path / "t.mogdb.tmp").write_bytes(b'{"partial garbage')

            c2 = Collection("t", path)
            assert c2.count() == 1
            assert c2.find()[0]["a"] == 1


# =========================================================================
# compact crash windows
# =========================================================================


class TestCompactCrashWindows:
    def test_crash_between_snapshot_write_and_replace_leaves_journal_authoritative(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)
            c = Collection("t", path)
            for i in range(3):
                c.insert_one({"i": i})
            # Simulate crash before os.replace: a truncated .mogdb.tmp exists,
            # no final snapshot, journal intact. Build state directly — no
            # compact() on a real Collection.
            (path / "t.mogdb.tmp").write_bytes(b'{"_id": "trunc')

            c2 = Collection("t", path)
            assert c2.count() == 3

    def test_reload_uses_snapshot_plus_new_seq_ops(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)
            c = Collection("t", path)
            c.insert_one({"a": 1})
            c.compact()
            c.insert_one({"a": 2})

            c2 = Collection("t", path)
            assert c2.count() == 2
            assert sorted(d["a"] for d in c2.find()) == [1, 2]


# =========================================================================
# cross-process appends
# =========================================================================


class TestCrossProcessAppend:
    def test_concurrent_writers_lose_no_records(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)
            env = {**os.environ, "PYTHONPATH": str(SRC_DIR)}
            procs = [
                subprocess.Popen(
                    [sys.executable, "-c", WRITER_SCRIPT, str(path), str(w)],
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                for w in range(6)
            ]
            results = []
            for p in procs:
                try:
                    _, err = p.communicate(timeout=120)
                except subprocess.TimeoutExpired:
                    p.kill()
                    p.communicate()
                    pytest.fail("child writer timed out after 120s")
                results.append((p.returncode, err))
            if any(rc != 0 for rc, _ in results):
                detail = "; ".join(
                    f"w{i} rc={rc} err={tail!r}"
                    for i, (rc, err) in enumerate(results)
                    for tail in [err[-300:]]
                )
                pytest.fail(f"child writers failed: {detail}")

            c = Collection("hammer", path)
            assert c.count() == 600
            assert c.open_report["quarantined"] == 0
            present = {d["seq"] for d in c.find()}
            expected = {w * 1000 + i for w in range(6) for i in range(100)}
            assert present == expected, (
                f"lost/interleaved records: missing={sorted(expected - present)[:10]} "
                f"extra={sorted(present - expected)[:10]}"
            )


# =========================================================================
# SIGKILL hammer
# =========================================================================


class TestKillHammer:
    def test_sigkill_keeps_prefix_without_holes(self):
        rng = random.Random(0)
        for round_no in range(3):
            with tempfile.TemporaryDirectory() as td:
                path = Path(td)
                script = path / "kill_child.py"
                script.write_text(KILL_CHILD_SCRIPT)
                env = {**os.environ, "PYTHONPATH": str(SRC_DIR)}
                proc = subprocess.Popen(
                    [sys.executable, str(script), str(path)],
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,
                )
                # Hard watchdog: never let one round blow the test budget.
                watchdog = threading.Timer(25.0, proc.kill)
                watchdog.daemon = True
                watchdog.start()

                last_ack = -1
                target: int | None = None
                # Fail-safe only: the real kill trigger is `target` above.
                # Cold python+mogdb startup under load can take seconds, so
                # this budget must dwarf it (fsynced inserts run ~385/s).
                deadline = time.monotonic() + 10.0
                stderr = ""
                try:
                    assert proc.stdout is not None
                    while True:
                        line = proc.stdout.readline()
                        if not line:
                            break
                        if not line.startswith("ACK "):
                            continue
                        last_ack = int(line.split()[1])
                        if target is None and last_ack >= 30:
                            target = last_ack + rng.randint(0, 50)
                        if target is not None and last_ack >= target:
                            os.kill(proc.pid, signal.SIGKILL)
                            break
                        if time.monotonic() >= deadline:
                            os.kill(proc.pid, signal.SIGKILL)
                            break
                    _, stderr = proc.communicate(timeout=30)
                finally:
                    watchdog.cancel()
                    if proc.poll() is None:
                        proc.kill()
                        proc.wait(timeout=10)

                assert last_ack >= 30, (
                    f"round {round_no}: child never reached 30 ACKs "
                    f"(last_ack={last_ack}, rc={proc.returncode}, "
                    f"stderr={stderr[-400:]})"
                )

                # Every acknowledged write must survive the SIGKILL.
                c = Collection("hammer", path)
                present = {d["seq"] for d in c.find()}
                missing = set(range(last_ack + 1)) - present
                assert not missing, (
                    f"round {round_no}: acked writes missing after SIGKILL: "
                    f"{sorted(missing)[:10]}"
                )
                # A torn tail at most — no silently dropped whole records.
                assert c.open_report["quarantined"] <= 1, (
                    f"round {round_no}: quarantined={c.open_report['quarantined']} > 1"
                )
