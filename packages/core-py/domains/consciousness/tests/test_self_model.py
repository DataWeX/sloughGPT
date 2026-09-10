"""Tests for SelfModel."""

import tempfile
from pathlib import Path

import pytest
from domains.consciousness.self_model import SelfEpisode, SelfIdentity, SelfModel


class TestSelfIdentity:
    def test_defaults(self):
        ident = SelfIdentity()
        assert ident.name == "SloughGPT"
        assert len(ident.capabilities) > 0
        assert len(ident.limitations) > 0
        assert len(ident.values) > 0


class TestSelfModel:
    def test_init(self):
        model = SelfModel()
        assert model.identity.name == "SloughGPT"
        assert len(model.self_beliefs) > 0
        assert len(model.episodes) == 0

    def test_observe_creates_episode(self):
        model = SelfModel()
        exp = {
            "input_text": "hello",
            "response": "hi there",
            "qualia": {"valence": 0.5, "novelty": 0.3},
        }
        episode = model.observe(exp)
        assert isinstance(episode, SelfEpisode)
        assert episode.input_text == "hello"
        assert episode.response == "hi there"
        assert len(model.episodes) == 1

    def test_observe_with_feedback(self):
        model = SelfModel()
        exp = {"input_text": "q", "response": "a", "feedback_rating": 5}
        episode = model.observe(exp)
        assert episode.growth_delta > 0

    def test_observe_negative_feedback(self):
        model = SelfModel()
        exp = {"input_text": "q", "response": "a", "feedback_rating": 1}
        episode = model.observe(exp)
        assert episode.growth_delta < 0

    def test_reflect_empty(self):
        model = SelfModel()
        assert "no experiences" in model.reflect().lower()

    def test_reflect_with_episodes(self):
        model = SelfModel()
        model.observe({"input_text": "test", "response": "test"})
        reflection = model.reflect()
        assert "1 experiences" in reflection
        assert "beliefs" in reflection

    def test_get_belief(self):
        model = SelfModel()
        assert model.get_belief("competence") > 0
        assert model.get_belief("nonexistent") == 0.0

    def test_update_belief_clamps(self):
        model = SelfModel()
        model.update_belief("competence", 10.0)
        assert model.get_belief("competence") == 1.0
        model.update_belief("competence", -10.0)
        assert model.get_belief("competence") == 0.0

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = SelfModel(store_path=tmpdir)
            model.observe({"input_text": "hello", "response": "world"})
            model.save()

            loaded = SelfModel(store_path=tmpdir)
            loaded.load()
            assert len(loaded.episodes) == 1
            assert loaded.episodes[0].input_text == "hello"

    def test_load_missing_file(self):
        model = SelfModel(store_path="/tmp/nonexistent_path_12345")
        model.load()  # should not raise

    def test_beliefs_update_from_positive_episode(self):
        model = SelfModel()
        initial = model.get_belief("competence")
        model.observe({
            "input_text": "help me",
            "response": "sure, here is help",
            "feedback_rating": 5,
        })
        assert model.get_belief("competence") >= initial

    def test_growth_delta_bounded(self):
        model = SelfModel()
        exp = {"input_text": "q", "response": "a", "feedback_rating": 5}
        ep = model.observe(exp)
        assert -0.1 <= ep.growth_delta <= 0.1

    def test_insight_novelty(self):
        model = SelfModel()
        exp = {
            "input_text": "what is this new thing?",
            "response": "it is a new thing",
            "qualia": {"novelty": 0.9, "coherence": 0.5},
        }
        ep = model.observe(exp)
        # Should be one of the novelty insights
        novelty_keywords = ["novel", "unexpected", "challenged", "encountered"]
        assert any(kw in ep.self_insight.lower() for kw in novelty_keywords)
