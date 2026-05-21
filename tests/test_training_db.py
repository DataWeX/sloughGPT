"""Test TrainingDB - simple conversation storage for training."""

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from importlib.util import spec_from_file_location, module_from_spec


def load_module():
    path = Path("/Users/mac/sloughGPT/packages/core-py/domains/infrastructure/training_db.py")
    spec = spec_from_file_location("training_db", path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestTrainingDB(unittest.TestCase):
    def setUp(self):
        self.module = load_module()
        self.temp_dir = tempfile.mkdtemp()
        self.db = self.module.TrainingDB(path=str(Path(self.temp_dir) / "train.db"))

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_add_pair(self):
        """Test adding a conversation pair."""
        pair = self.db.add("s1", "Hello", "Hi there", "gpt2")
        self.assertIsNotNone(pair.id)
        self.assertEqual(pair.prompt, "Hello")
        self.assertEqual(pair.response, "Hi there")

    def test_get_pairs(self):
        """Test getting pairs."""
        self.db.add("s1", "Q1", "A1", "gpt2")
        self.db.add("s1", "Q2", "A2", "gpt2")

        pairs = self.db.get_pairs()
        self.assertEqual(len(pairs), 2)

    def test_feedback(self):
        """Test setting feedback."""
        pair = self.db.add("s1", "Hello", "Hi", "gpt2")

        self.db.set_feedback(pair.id, "up")

        updated = self.db.get_pairs()[0]
        self.assertEqual(updated.quality, 1.0)
        self.assertEqual(updated.feedback, "up")

    def test_mark_used(self):
        """Test marking pairs as used."""
        p1 = self.db.add("s1", "Q1", "A1", "gpt2")
        p2 = self.db.add("s1", "Q2", "A2", "gpt2")

        self.db.mark_used([p1.id])

        pairs = self.db.get_pairs(unused_only=True)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0].id, p2.id)

    def test_stats(self):
        """Test stats."""
        self.db.add("s1", "Q1", "A1", "gpt2")
        self.db.add("s1", "Q2", "A2", "gpt2")

        stats = self.db.get_stats()
        self.assertEqual(stats["total_pairs"], 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
