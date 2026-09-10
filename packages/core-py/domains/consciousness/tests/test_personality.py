"""Tests for the Personality System."""

from domains.consciousness.personality import PersonalityProfile, PersonalityManager


class TestPersonalityProfile:
    def test_default_profile(self):
        profile = PersonalityProfile()
        assert len(profile.values) > 0
        assert len(profile.goals) > 0
        assert "formality" in profile.voice
        assert "openness" in profile.traits
        assert len(profile.interests) > 0
        assert len(profile.avoid) > 0

    def test_to_dict(self):
        profile = PersonalityProfile()
        d = profile.to_dict()
        assert "values" in d
        assert "goals" in d
        assert "voice" in d
        assert "style" in d
        assert "traits" in d
        assert "interests" in d
        assert "avoid" in d

    def test_from_dict(self):
        data = {
            "values": ["test"],
            "goals": ["test goal"],
            "voice": {"formality": 0.8},
            "style": {"use_examples": False},
            "traits": {"openness": 0.9},
            "interests": ["testing"],
            "avoid": ["bad things"],
        }
        profile = PersonalityProfile.from_dict(data)
        assert profile.values == ["test"]
        assert profile.goals == ["test goal"]
        assert profile.voice["formality"] == 0.8
        assert profile.style["use_examples"] is False
        assert profile.traits["openness"] == 0.9
        assert profile.interests == ["testing"]
        assert profile.avoid == ["bad things"]

    def test_from_dict_defaults(self):
        profile = PersonalityProfile.from_dict({})
        assert len(profile.values) > 0
        assert len(profile.goals) > 0


class TestPersonalityManager:
    def test_save_and_load(self, tmp_path):
        manager = PersonalityManager(store_path=str(tmp_path))
        profile = PersonalityProfile()
        profile.values = ["test_value"]
        manager.save(profile)

        loaded = manager.load()
        assert loaded.values == ["test_value"]

    def test_get_profile_caches(self, tmp_path):
        manager = PersonalityManager(store_path=str(tmp_path))
        p1 = manager.get_profile()
        p2 = manager.get_profile()
        assert p1 is p2

    def test_update_values(self, tmp_path):
        manager = PersonalityManager(store_path=str(tmp_path))
        manager.update_values(["new_value"])
        assert manager.get_profile().values == ["new_value"]

    def test_update_goals(self, tmp_path):
        manager = PersonalityManager(store_path=str(tmp_path))
        manager.update_goals(["new goal"])
        assert manager.get_profile().goals == ["new goal"]

    def test_update_voice(self, tmp_path):
        manager = PersonalityManager(store_path=str(tmp_path))
        manager.update_voice({"formality": 0.9})
        assert manager.get_profile().voice["formality"] == 0.9
        # Other defaults preserved
        assert "warmth" in manager.get_profile().voice

    def test_update_traits(self, tmp_path):
        manager = PersonalityManager(store_path=str(tmp_path))
        manager.update_traits({"openness": 0.99})
        assert manager.get_profile().traits["openness"] == 0.99

    def test_update_interests(self, tmp_path):
        manager = PersonalityManager(store_path=str(tmp_path))
        manager.update_interests(["quantum physics"])
        assert manager.get_profile().interests == ["quantum physics"]

    def test_update_avoid(self, tmp_path):
        manager = PersonalityManager(store_path=str(tmp_path))
        manager.update_avoid(["being rude"])
        assert manager.get_profile().avoid == ["being rude"]

    def test_evolve_from_episode_positive(self, tmp_path):
        manager = PersonalityManager(store_path=str(tmp_path))
        original_confidence = manager.get_profile().voice["confidence"]
        manager.evolve_from_episode({
            "growth_delta": 0.1,
            "qualia": {"novelty": 0.8, "valence": 0.6, "coherence": 0.5},
        })
        p = manager.get_profile()
        assert p.voice["confidence"] >= original_confidence
        assert p.traits["openness"] >= 0.7  # increased from novelty

    def test_evolve_from_episode_low_coherence(self, tmp_path):
        manager = PersonalityManager(store_path=str(tmp_path))
        original_verbosity = manager.get_profile().voice["verbosity"]
        manager.evolve_from_episode({
            "growth_delta": 0.0,
            "qualia": {"novelty": 0.3, "valence": 0.0, "coherence": 0.2},
        })
        assert manager.get_profile().voice["verbosity"] < original_verbosity

    def test_to_system_prompt_context(self, tmp_path):
        manager = PersonalityManager(store_path=str(tmp_path))
        ctx = manager.to_system_prompt_context()
        assert "Core values" in ctx
        assert "Goals" in ctx
        assert "Interests" in ctx

    def test_load_nonexistent_returns_default(self, tmp_path):
        manager = PersonalityManager(store_path=str(tmp_path))
        profile = manager.load()
        assert isinstance(profile, PersonalityProfile)
        assert len(profile.values) > 0

    def test_save_none_does_nothing(self, tmp_path):
        manager = PersonalityManager(store_path=str(tmp_path))
        manager._profile = None
        manager.save()  # Should not raise
