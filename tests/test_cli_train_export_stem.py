from __future__ import annotations

import re
import unittest

from apps.cli.src.cli import _train_export_stem_slug, _train_export_default_stem


class TestTrainExportStem(unittest.TestCase):
    def test_slug_fallback_and_whitespace(self) -> None:
        self.assertEqual(_train_export_stem_slug("", "fb"), "fb")
        self.assertEqual(_train_export_stem_slug("  hello world  ", "x"), "hello-world")
        self.assertEqual(_train_export_stem_slug("a/b\\c", "fb"), "a-b-c")

    def test_default_stem_shape(self) -> None:
        stem = _train_export_default_stem("sloughgpt", "shakespeare")
        self.assertRegex(
            stem,
            re.compile(r"^sloughgpt-shakespeare-\d{4}-\d{2}-\d{2}-\d{6}$"),
        )


if __name__ == "__main__":
    unittest.main()
