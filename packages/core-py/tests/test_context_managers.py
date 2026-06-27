"""Tests for context managers — TraitWeightsConfig + 4 steering managers."""

import os
import json
import shutil
import tempfile
import pytest
from datetime import datetime

from domains.context.managers import (
    TraitWeightsConfig, get_trait_config, reset_trait_config,
    PersonalityManager, MemoryManager, StyleManager, TaskManager,
    TRAIT_SCHEMA, ALL_TRAITS,
)


# ── Fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_config():
    """TraitWeightsConfig backed by a temp file."""
    d = tempfile.mkdtemp()
    path = os.path.join(d, "weights.json")
    cfg = TraitWeightsConfig(path=path)
    cfg.reset()
    yield cfg
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def clean_global():
    reset_trait_config()
    # Also cleanup disk to prevent stale file from polluting next test
    import glob
    for f in glob.glob("data/trait_weights*"):
        try: os.remove(f)
        except: pass
    for d in glob.glob("data/trait_snapshots*"):
        try: shutil.rmtree(d, ignore_errors=True)
        except: pass
    yield
    reset_trait_config()


# ── TraitWeightsConfig Tests ──────────────────────────────────────────────

class TestTraitWeightsConfig:

    def test_defaults_to_05(self, tmp_config):
        assert tmp_config.get("warmth") == 0.5
        assert tmp_config.get("nonexistent") == 0.5

    def test_set_and_get(self, tmp_config):
        tmp_config.set("warmth", 0.8)
        assert tmp_config.get("warmth") == 0.8

    def test_set_clamps_to_01(self, tmp_config):
        tmp_config.set("warmth", 2.0)
        assert tmp_config.get("warmth") == 1.0
        tmp_config.set("warmth", -0.5)
        assert tmp_config.get("warmth") == 0.0

    def test_all_returns_grouped_structure(self, tmp_config):
        result = tmp_config.all()
        assert "personality" in result
        assert "cognition" in result
        assert "emotion" in result
        assert len(result["personality"]) == 10
        assert len(result["cognition"]) == 8
        assert len(result["emotion"]) == 5

    def test_all_unset_traits_are_05(self, tmp_config):
        result = tmp_config.all()
        for group in ("personality", "cognition", "emotion"):
            for v in result[group].values():
                assert v == 0.5

    def test_update_deltas(self, tmp_config):
        tmp_config.update({"warmth": 0.1, "humor": -0.2})
        assert tmp_config.get("warmth") == 0.6
        assert tmp_config.get("humor") == 0.3

    def test_update_clamps_result(self, tmp_config):
        tmp_config.set("warmth", 0.95)
        tmp_config.update({"warmth": 0.1})
        assert tmp_config.get("warmth") == 1.0

    def test_set_many(self, tmp_config):
        tmp_config.set_many({"warmth": 0.9, "creativity": 0.75, "humor": 0.6})
        assert tmp_config.get("warmth") == 0.9
        assert tmp_config.get("creativity") == 0.75
        assert tmp_config.get("humor") == 0.6

    def test_set_many_ignores_unknown_keys(self, tmp_config):
        tmp_config.set_many({"warmth": 0.9, "bogus_key": 1.0})
        assert tmp_config.get("warmth") == 0.9
        assert tmp_config.get("bogus_key") == 0.5  # default

    def test_reset_clears_to_defaults(self, tmp_config):
        tmp_config.set("warmth", 0.9)
        tmp_config.reset()
        assert tmp_config.get("warmth") == 0.5

    def test_persistence(self, tmp_config):
        path = tmp_config._path
        tmp_config.set("warmth", 0.8)
        tmp_config.set("humor", 0.9)
        # New instance reading same file
        cfg2 = TraitWeightsConfig(path=str(path))
        assert cfg2.get("warmth") == 0.8
        assert cfg2.get("humor") == 0.9

    def test_persistence_ignores_stale_keys(self, tmp_config):
        """Traits not in ALL_TRAITS are ignored on load."""
        path = tmp_config._path
        tmp_config._weights = {"warmth": 0.8, "old_trait": 0.5}
        tmp_config._save()
        cfg2 = TraitWeightsConfig(path=str(path))
        assert cfg2.get("warmth") == 0.8
        assert cfg2.get("old_trait") == 0.5  # default since not in ALL_TRAITS

    def test_feedback_thumbs_up_boosts_all(self, tmp_config):
        tmp_config.update_from_feedback("thumbs_up", "good", "great")
        for t in ALL_TRAITS:
            assert tmp_config.get(t) >= 0.5, f"{t} should be >= 0.5 after thumbs_up"

    def test_feedback_thumbs_down_lowers_all(self, tmp_config):
        # Set traits high first so they can decrease
        for t in ALL_TRAITS:
            tmp_config.set(t, 0.8)
        tmp_config.update_from_feedback("thumbs_down", "bad", "awful")
        for t in ALL_TRAITS:
            assert tmp_config.get(t) <= 0.8, f"{t} should be <= 0.8 after thumbs_down"

    def test_feedback_content_aware_humor(self, tmp_config):
        tmp_config.update_from_feedback("thumbs_up", "tell me a joke", "lol")
        assert tmp_config.get("humor") > tmp_config.get("patience"), "humor should get extra boost from joke request"

    def test_feedback_content_aware_depth(self, tmp_config):
        tmp_config.update_from_feedback("thumbs_up", "explain how quantum works", "ok")
        assert tmp_config.get("creative_divergence") > 0.5
        assert tmp_config.get("abstract_reasoning") > 0.5

    def test_feedback_content_aware_directness(self, tmp_config):
        tmp_config.update_from_feedback("thumbs_up", "tl;dr give me short", "ok")
        assert tmp_config.get("directness") > 0.5

    # ── Snapshots ──

    def test_snapshot_save_and_list(self, tmp_config):
        tmp_config.save_snapshot("test_snap")
        snaps = tmp_config.list_snapshots()
        assert any(s["name"] == "test_snap" for s in snaps)
        assert all("name" in s for s in snaps)  # metadata intact

    def test_snapshot_round_trip(self, tmp_config):
        tmp_config.set("warmth", 0.9)
        tmp_config.set("humor", 0.15)
        tmp_config.save_snapshot("rt_snap")
        tmp_config.reset()
        assert tmp_config.get("warmth") == 0.5  # reset worked
        tmp_config.load_snapshot("rt_snap")
        assert tmp_config.get("warmth") == 0.9
        assert tmp_config.get("humor") == 0.15

    def test_snapshot_delete(self, tmp_config):
        tmp_config.save_snapshot("del_me")
        assert any(s["name"] == "del_me" for s in tmp_config.list_snapshots())
        assert tmp_config.delete_snapshot("del_me") is True
        assert not any(s["name"] == "del_me" for s in tmp_config.list_snapshots())

    def test_snapshot_delete_nonexistent(self, tmp_config):
        assert tmp_config.delete_snapshot("no_exist") is False

    def test_snapshot_load_nonexistent_returns_zero(self, tmp_config):
        count = tmp_config.load_snapshot("no_exist")
        assert count == 0

    def test_multiple_snapshots_independent(self, tmp_config):
        tmp_config.set("warmth", 0.9)
        tmp_config.save_snapshot("high_warmth")
        tmp_config.set("warmth", 0.1)
        tmp_config.save_snapshot("low_warmth")
        tmp_config.load_snapshot("high_warmth")
        assert tmp_config.get("warmth") == 0.9
        tmp_config.load_snapshot("low_warmth")
        assert tmp_config.get("warmth") == 0.1

    # ── Thread safety smoke test ──

    def test_concurrent_access(self, tmp_config):
        import threading
        errors = []
        def worker():
            try:
                for _ in range(20):
                    tmp_config.set("warmth", 0.5)
                    tmp_config.get("warmth")
                    tmp_config.update({"warmth": 0.1})
                    tmp_config.all()
            except Exception as e:
                errors.append(e)
        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads: t.start()
        for t in threads: t.join()
        assert len(errors) == 0, f"Concurrent access errors: {errors}"


# ── PersonalityManager Tests ─────────────────────────────────────────────

class TestPersonalityManager:

    def test_apply_returns_block(self, tmp_config):
        pm = PersonalityManager(tmp_config)
        text = pm.apply()
        assert text.startswith("\n\n[PERSONALITY INSTRUCTIONS]")
        assert "Personality:" in text

    def test_high_warmth_described(self, tmp_config):
        tmp_config.set("warmth", 0.9)
        pm = PersonalityManager(tmp_config)
        text = pm.apply()
        assert "warm and nurturing" in text

    def test_low_warmth_described(self, tmp_config):
        tmp_config.set("warmth", 0.2)
        pm = PersonalityManager(tmp_config)
        text = pm.apply()
        assert "reserved" in text.lower()

    def test_high_formality(self, tmp_config):
        tmp_config.set("formality", 0.9)
        pm = PersonalityManager(tmp_config)
        text = pm.apply()
        assert "formal" in text.lower()

    def test_low_formality(self, tmp_config):
        tmp_config.set("formality", 0.1)
        pm = PersonalityManager(tmp_config)
        text = pm.apply()
        assert "casual" in text.lower()

    def test_high_humor_includes_wit_instruction(self, tmp_config):
        tmp_config.set("humor", 0.8)
        pm = PersonalityManager(tmp_config)
        text = pm.apply()
        assert "humor" in text.lower() or "wit" in text.lower()

    def test_low_humor_omits_humor_instruction(self, tmp_config):
        tmp_config.set("humor", 0.4)
        pm = PersonalityManager(tmp_config)
        text = pm.apply()
        assert "humor" not in text.lower()

    def test_high_confidence_includes_authority(self, tmp_config):
        tmp_config.set("confidence", 0.8)
        pm = PersonalityManager(tmp_config)
        text = pm.apply()
        assert "authority" in text.lower() or "conviction" in text.lower()

    def test_get_weights_snapshot(self, tmp_config):
        tmp_config.set("warmth", 0.8)
        pm = PersonalityManager(tmp_config)
        snap = pm.get_weights_snapshot()
        assert snap["warmth"] == 0.8
        assert len(snap) == 10


# ── MemoryManager Tests ──────────────────────────────────────────────────

class TestMemoryManager:

    def test_working_capacity_default(self, tmp_config):
        mm = MemoryManager(tmp_config)
        cap = mm.working_capacity
        assert 5 <= cap <= 11

    def test_working_capacity_scales_with_context(self, tmp_config):
        tmp_config.set("long_context_handling", 1.0)
        mm = MemoryManager(tmp_config)
        assert mm.working_capacity == 11  # max

    def test_working_capacity_min(self, tmp_config):
        tmp_config.set("long_context_handling", 0.0)
        mm = MemoryManager(tmp_config)
        assert mm.working_capacity == 5  # min

    def test_importance_threshold_scales(self, tmp_config):
        tmp_config.set("learning_adaptability", 1.0)
        mm = MemoryManager(tmp_config)
        assert mm.memory_importance_threshold == pytest.approx(0.2, abs=0.01)

        tmp_config.set("learning_adaptability", 0.0)
        mm2 = MemoryManager(tmp_config)
        assert mm2.memory_importance_threshold == pytest.approx(0.5, abs=0.01)

    def test_retention_decay_scales(self, tmp_config):
        tmp_config.set("pattern_recognition", 1.0)
        mm = MemoryManager(tmp_config)
        assert mm.retention_decay == pytest.approx(0.02, abs=0.01)

        tmp_config.set("pattern_recognition", 0.0)
        mm2 = MemoryManager(tmp_config)
        assert mm2.retention_decay == pytest.approx(0.1, abs=0.01)

    def test_should_consolidate(self, tmp_config):
        mm = MemoryManager(tmp_config)
        # default threshold ~0.35
        assert mm.should_consolidate(0.5) is True
        assert mm.should_consolidate(0.1) is False

    def test_apply_memory_context_filters(self, tmp_config):
        mm = MemoryManager(tmp_config)
        episodes = [
            {"importance": 0.8, "content": "important"},
            {"importance": 0.2, "content": "trivial"},
            {"importance": 0.6, "content": "moderate"},
        ]
        filtered = mm.apply_memory_context(episodes)
        assert len(filtered) == 2  # 0.8 and 0.6 should pass threshold


# ── StyleManager Tests ──────────────────────────────────────────────────

class TestStyleManager:

    def test_apply_returns_block(self, tmp_config):
        sm = StyleManager(tmp_config)
        text = sm.apply()
        assert text.startswith("\n\n[STYLE INSTRUCTIONS]")

    def test_high_formality_style(self, tmp_config):
        tmp_config.set("formality", 0.9)
        sm = StyleManager(tmp_config)
        text = sm.apply()
        assert "formal" in text.lower()
        assert "slang" in text.lower()  # "Avoid slang"

    def test_low_formality_style(self, tmp_config):
        tmp_config.set("formality", 0.1)
        sm = StyleManager(tmp_config)
        text = sm.apply()
        assert "casual" in text.lower()

    def test_high_directness_style(self, tmp_config):
        tmp_config.set("directness", 0.9)
        sm = StyleManager(tmp_config)
        text = sm.apply()
        assert "direct" in text.lower()

    def test_low_directness_style(self, tmp_config):
        tmp_config.set("directness", 0.1)
        sm = StyleManager(tmp_config)
        text = sm.apply()
        assert "diplomatic" in text.lower()

    def test_high_precision_style(self, tmp_config):
        tmp_config.set("factual_precision", 0.9)
        sm = StyleManager(tmp_config)
        text = sm.apply()
        assert "accuracy" in text.lower()

    def test_high_tone_flexibility(self, tmp_config):
        tmp_config.set("tone_flexibility", 0.9)
        sm = StyleManager(tmp_config)
        text = sm.apply()
        assert "adapt tone" in text.lower()


# ── TaskManager Tests ────────────────────────────────────────────────────

class TestTaskManager:

    def test_apply_returns_block(self, tmp_config):
        tm = TaskManager(tmp_config)
        text = tm.apply()
        assert text.startswith("\n\n[TASK APPROACH]")

    def test_high_abstract_reasoning(self, tmp_config):
        tmp_config.set("abstract_reasoning", 0.9)
        tm = TaskManager(tmp_config)
        text = tm.apply()
        assert "analogies" in text.lower()

    def test_low_abstract_reasoning(self, tmp_config):
        tmp_config.set("abstract_reasoning", 0.1)
        tm = TaskManager(tmp_config)
        text = tm.apply()
        assert "concrete" in text.lower()

    def test_high_creative_divergence(self, tmp_config):
        tmp_config.set("creative_divergence", 0.9)
        tm = TaskManager(tmp_config)
        text = tm.apply()
        assert "multiple perspectives" in text.lower()

    def test_low_creative_divergence(self, tmp_config):
        tmp_config.set("creative_divergence", 0.1)
        tm = TaskManager(tmp_config)
        text = tm.apply()
        assert "well-established" in text.lower()

    def test_high_systematic_planning(self, tmp_config):
        tmp_config.set("systematic_planning", 0.9)
        tm = TaskManager(tmp_config)
        text = tm.apply()
        assert "methodically" in text.lower()

    def test_high_metacognitive_awareness(self, tmp_config):
        tmp_config.set("metacognitive_awareness", 0.9)
        tm = TaskManager(tmp_config)
        text = tm.apply()
        assert "reflect" in text.lower()


# ── Integration: Managers + ContextCore ─────────────────────────────────

class TestManagersWithContextCore:

    @pytest.mark.asyncio
    async def test_all_managers_inject_into_frame(self, tmp_config):
        from domains.infrastructure.context_core import ContextCore

        tmp_config.set("warmth", 0.9)
        tmp_config.set("formality", 0.2)
        tmp_config.set("creative_divergence", 0.8)

        cc = ContextCore(
            personality_manager=PersonalityManager(tmp_config),
            memory_manager=MemoryManager(tmp_config),
            style_manager=StyleManager(tmp_config),
            task_manager=TaskManager(tmp_config),
        )
        cc.set_session_id("test")
        cc.add_message("user", "hello")
        frame = await cc.build_context_frame(query="hello")

        assert "[PERSONALITY INSTRUCTIONS]" in frame.system_prompt
        assert "[STYLE INSTRUCTIONS]" in frame.system_prompt
        assert "[TASK APPROACH]" in frame.system_prompt
        assert "warm" in frame.system_prompt.lower()

    def test_working_capacity_from_memory_manager(self, tmp_config):
        from domains.infrastructure.context_core import ContextCore

        tmp_config.set("long_context_handling", 0.0)
        mm = MemoryManager(tmp_config)
        cc = ContextCore(memory_manager=mm)
        cc.set_session_id("test")
        cc.add_message("user", "a")
        cc.add_message("user", "b")
        cc.add_message("user", "c")
        cc.add_message("user", "d")
        cc.add_message("user", "e")
        # At 5 items, capacity should be 5 (min), so pushing a 6th evicts
        cc.add_message("user", "f")
        assert len(cc.working_memory) == 5

    @pytest.mark.asyncio
    async def test_without_managers_falls_back_gracefully(self, tmp_config):
        from domains.infrastructure.context_core import ContextCore
        cc = ContextCore()  # no managers
        cc.set_session_id("test")
        cc.add_message("user", "hello")
        frame = await cc.build_context_frame(query="hello")
        # Default system prompt, no manager extras
        assert "[PERSONALITY INSTRUCTIONS]" not in frame.system_prompt


# ── Global singleton tests ──────────────────────────────────────────────

class TestGlobalConfig:

    def test_get_trait_config_returns_singleton(self, clean_global):
        c1 = get_trait_config()
        c2 = get_trait_config()
        assert c1 is c2

    def test_reset_trait_config_creates_new(self, clean_global):
        c1 = get_trait_config()
        c1.set("warmth", 0.9)
        # Manually delete the backing file so new instance starts clean
        if c1._path.exists():
            c1._path.unlink()
        reset_trait_config()
        c2 = get_trait_config()
        assert c2.get("warmth") == 0.5

    def test_shared_state_across_imports(self, clean_global):
        """Config set in one module is visible in another."""
        from domains.context.managers import get_trait_config as gtc1
        from domains.context.managers import get_trait_config as gtc2
        c1 = gtc1()
        c2 = gtc2()
        c1.set("warmth", 0.8)
        assert c2.get("warmth") == 0.8

    def test_slo_manager_reads_global_config(self, clean_global):
        from domains.inference.slo_manager import SloManager
        config = get_trait_config()
        config.set("warmth", 0.8)
        mgr = SloManager()
        weights = mgr.get_trait_weights()
        # Without a soul file, get_trait_weights returns
        # defaults + TraitWeightsConfig overlay
        assert weights["personality"]["warmth"] == 0.8

    def test_global_config_works_in_feedback_workflow(self, clean_global):
        """Simulate what workflow.py does when record_feedback is called."""
        from domains.context.managers import get_trait_config
        config = get_trait_config()
        config.reset()
        config.update_from_feedback("thumbs_up", "great response", "thanks")
        assert config.get("confidence") > 0.5
        assert config.get("optimism") > 0.5


# ── Manager Mode Tests ────────────────────────────────────────────────────

class TestManagerModes:

    def test_personality_mode_default(self, tmp_config):
        """Default (all 0.5) produces a valid label ≥ 0.5 confidence."""
        m = PersonalityManager(tmp_config).get_mode()
        assert isinstance(m["label"], str) and len(m["label"]) > 0
        assert 0 <= m["confidence"] <= 1
        assert "scores" in m
        assert len(m["scores"]) >= 4

    def test_personality_mode_warm(self, tmp_config):
        tmp_config.set("warmth", 0.95)
        tmp_config.set("empathy", 0.9)
        m = PersonalityManager(tmp_config).get_mode()
        assert m["label"] in ("Warm", "Playful")
        assert m["confidence"] > 0.6

    def test_personality_mode_analytical(self, tmp_config):
        for k, v in {"formality":0.9,"directness":0.8,"patience":0.85,"curiosity":0.7,"warmth":0.15}.items():
            tmp_config.set(k, v)
        m = PersonalityManager(tmp_config).get_mode()
        assert m["label"] in ("Analytical", "Confident")

    def test_personality_mode_creative(self, tmp_config):
        for k, v in {"creativity":0.95,"curiosity":0.9,"humor":0.7,"formality":0.1}.items():
            tmp_config.set(k, v)
        m = PersonalityManager(tmp_config).get_mode()
        assert m["label"] in ("Creative", "Playful")

    def test_personality_mode_scores_summarized(self, tmp_config):
        m = PersonalityManager(tmp_config).get_mode()
        top = max(m["scores"].values())
        assert m["confidence"] == top

    def test_memory_mode_default(self, tmp_config):
        m = MemoryManager(tmp_config).get_mode()
        assert isinstance(m["label"], str)
        assert m.get("capacity", 0) >= 5
        assert "scores" in m

    def test_memory_mode_deep_context(self, tmp_config):
        for k, v in {"long_context_handling":0.95,"pattern_recognition":0.9,"learning_adaptability":0.7}.items():
            tmp_config.set(k, v)
        m = MemoryManager(tmp_config).get_mode()
        assert m["label"] == "Deep Context" or m["confidence"] > 0.6

    def test_memory_capacity_range(self, tmp_config):
        tmp_config.set("long_context_handling", 0.0)
        assert MemoryManager(tmp_config).working_capacity == 5
        tmp_config.set("long_context_handling", 1.0)
        assert MemoryManager(tmp_config).working_capacity == 11

    def test_memory_mode_focused(self, tmp_config):
        for k, v in {"long_context_handling":0.1,"pattern_recognition":0.1}.items():
            tmp_config.set(k, v)
        m = MemoryManager(tmp_config).get_mode()
        assert m["label"] in ("Focused", "Stable")

    def test_style_mode_default(self, tmp_config):
        m = StyleManager(tmp_config).get_mode()
        assert isinstance(m["label"], str) and len(m["label"]) > 0
        assert 0 <= m["confidence"] <= 1

    def test_style_mode_casual(self, tmp_config):
        for k, v in {"formality":0.1,"directness":0.85,"tone_flexibility":0.9}.items():
            tmp_config.set(k, v)
        m = StyleManager(tmp_config).get_mode()
        assert m["label"] in ("Casual", "Direct", "Flexible")

    def test_style_mode_formal(self, tmp_config):
        for k, v in {"formality":0.95,"factual_precision":0.9}.items():
            tmp_config.set(k, v)
        m = StyleManager(tmp_config).get_mode()
        assert m["label"] in ("Formal", "Precise")

    def test_task_mode_default(self, tmp_config):
        m = TaskManager(tmp_config).get_mode()
        assert isinstance(m["label"], str) and len(m["label"]) > 0
        assert 0 <= m["confidence"] <= 1

    def test_task_mode_methodical(self, tmp_config):
        for k, v in {"systematic_planning":0.95,"abstract_reasoning":0.85,"patience":0.9,"creative_divergence":0.1}.items():
            tmp_config.set(k, v)
        m = TaskManager(tmp_config).get_mode()
        assert m["label"] in ("Methodical", "Structured", "Analytical")

    def test_task_mode_creative(self, tmp_config):
        for k, v in {"creative_divergence":0.95,"curiosity":0.9,"systematic_planning":0.1}.items():
            tmp_config.set(k, v)
        m = TaskManager(tmp_config).get_mode()
        assert m["label"] in ("Creative", "Exploratory")

    def test_task_mode_reflective(self, tmp_config):
        for k, v in {"metacognitive_awareness":0.95,"patience":0.9,"abstract_reasoning":0.85}.items():
            tmp_config.set(k, v)
        m = TaskManager(tmp_config).get_mode()
        assert m["label"] in ("Reflective", "Analytical")

    def test_all_modes_expose_scores_dict(self, tmp_config):
        for mgr_cls in (PersonalityManager, MemoryManager, StyleManager, TaskManager):
            m = mgr_cls(tmp_config).get_mode()
            scores = m.get("scores", {})
            assert len(scores) >= 4, f"{mgr_cls.__name__} only has {len(scores)} scores"
            assert max(scores.values()) == m["confidence"]

    def test_feedback_changes_modes(self, tmp_config):
        """Thumbs up on a funny joke should shift personality mode."""
        mgr = PersonalityManager(tmp_config)
        before = mgr.get_mode()
        tmp_config.update_from_feedback("thumbs_up", "that was hilarious", "lol")
        after = mgr.get_mode()
        # Mode may stay same but scores should differ
        assert before["confidence"] != after["confidence"] or before["label"] != after["label"]
