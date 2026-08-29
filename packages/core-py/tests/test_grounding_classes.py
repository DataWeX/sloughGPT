"""Tests for domains.cognitive.grounding — HierarchicalContext, CurriculumLearner."""

from domains.cognitive.grounding import HierarchicalContext, CurriculumLearner


class TestHierarchicalContext:
    def test_init(self):
        hc = HierarchicalContext()
        assert hc.max_context == 4096
        assert hc.chunk_size == 512
        assert hc.hierarchy == []

    def test_build_hierarchy(self):
        hc = HierarchicalContext(chunk_size=10)
        text = "word " * 25  # 25 words
        hc.build_hierarchy(text)
        assert len(hc.hierarchy) >= 1
        assert len(hc.hierarchy[0]) == 3  # ceil(25/10) = 3

    def test_get_relevant_context_empty(self):
        hc = HierarchicalContext()
        assert hc.get_relevant_context("query") == ""

    def test_get_relevant_context(self):
        hc = HierarchicalContext(chunk_size=10)
        text = "word " * 25
        hc.build_hierarchy(text)
        ctx = hc.get_relevant_context("word")
        assert isinstance(ctx, str)
        assert len(ctx) > 0


class TestCurriculumLearner:
    def test_init(self):
        cl = CurriculumLearner()
        assert cl.current_level == 0
        assert cl.stage == "bootstrapping"

    def test_add_example(self):
        cl = CurriculumLearner()
        cl.add_example("easy", 0.1)
        cl.add_example("medium", 0.5)
        cl.add_example("hard", 0.9)
        assert len(cl.difficulty_levels) == 3

    def test_get_batch_bootstrapping(self):
        cl = CurriculumLearner()
        cl.add_example("e1", 0.05)
        cl.add_example("e2", 0.15)
        cl.add_example("e3", 0.8)
        batch = cl.get_batch(2)
        assert len(batch) == 2

    def test_update_stage_mastery(self):
        cl = CurriculumLearner()
        cl.update_stage(0.95)
        assert cl.stage == "mastery"

    def test_update_stage_progressing(self):
        cl = CurriculumLearner()
        cl.update_stage(0.8)
        assert cl.stage == "progressing"

    def test_update_stage_bootstrapping(self):
        cl = CurriculumLearner()
        cl.update_stage(0.3)
        assert cl.stage == "bootstrapping"
