from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from apps.cli.src.utils.helpers import local_soul_candidate_paths as _local_soul_candidate_paths


class TestLocalSoulCandidates(unittest.TestCase):
    def test_newest_non_default_first_when_no_canonical(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            a = d / "older.soul"
            b = d / "newer.soul"
            a.write_text("x", encoding="utf-8")
            b.write_text("y", encoding="utf-8")
            Path(f"{a}.meta.json").write_text("{}", encoding="utf-8")
            Path(f"{b}.meta.json").write_text("{}", encoding="utf-8")
            os.utime(a, (10, 10))
            os.utime(b, (99_999, 99_999))
            self.assertEqual(_local_soul_candidate_paths(d), [b, a])

    def test_canonical_first_then_newest_others(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            a = d / "older.soul"
            b = d / "newer.soul"
            default = d / "sloughgpt.soul"
            a.write_text("x", encoding="utf-8")
            b.write_text("y", encoding="utf-8")
            default.write_text("z", encoding="utf-8")
            Path(f"{a}.meta.json").write_text("{}", encoding="utf-8")
            Path(f"{b}.meta.json").write_text("{}", encoding="utf-8")
            os.utime(a, (10, 10))
            os.utime(b, (99_999, 99_999))
            os.utime(default, (50_000, 50_000))
            paths = _local_soul_candidate_paths(d)
            self.assertEqual(paths[0], default)
            self.assertEqual(paths[1], b)
            self.assertEqual(paths[2], a)

    def test_skips_junk_and_meta_less_checkpoints(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            real = d / "model_1790048650.soul"
            junk = d / "tmp_garbage.soul"
            nometa = d / "bare.soul"
            real.write_text("x", encoding="utf-8")
            junk.write_text("y", encoding="utf-8")
            nometa.write_text("z", encoding="utf-8")
            Path(f"{real}.meta.json").write_text("{}", encoding="utf-8")
            os.utime(real, (99_999, 99_999))
            os.utime(junk, (500_000, 500_000))
            os.utime(nometa, (600_000, 600_000))
            self.assertEqual(_local_soul_candidate_paths(d), [real])

    def test_recurses_immediate_subdirs(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            sub = d / "auto-training"
            sub.mkdir()
            a = d / "root.soul"
            b = sub / "trained.soul"
            a.write_text("x", encoding="utf-8")
            b.write_text("y", encoding="utf-8")
            Path(f"{a}.meta.json").write_text("{}", encoding="utf-8")
            Path(f"{b}.meta.json").write_text("{}", encoding="utf-8")
            os.utime(a, (10, 10))
            os.utime(b, (99_999, 99_999))
            self.assertEqual(_local_soul_candidate_paths(d), [b, a])

    def test_missing_dir(self) -> None:
        d = Path(tempfile.mkdtemp()) / "nope"
        self.assertEqual(_local_soul_candidate_paths(d), [])


if __name__ == "__main__":
    unittest.main()
