"""Agent loop: model-in-the-loop iteration, run persistence, training export."""

import json

import pytest

import domain.agents._internal.run_history as rh
from domain.agents._internal.agents import Agent, AgentConfig, ToolCapability


class ScriptedInference:
    """Returns scripted plan texts, then the final answer."""

    def __init__(self, plan_texts, answer="synthesized answer"):
        self.calls = 0
        self.plan_texts = list(plan_texts)
        self.answer = answer
        self.prompts = []

    def __call__(self, prompt: str) -> dict:
        self.prompts.append(prompt)
        idx = self.calls
        self.calls += 1
        if idx < len(self.plan_texts):
            return {"text": self.plan_texts[idx]}
        return {"text": self.answer}


@pytest.fixture()
def run_store(tmp_path):
    rh.reset_agent_run_store()
    rh.set_mogdb_path(str(tmp_path / "agent_runs"))
    yield rh.get_agent_run_store()
    rh.reset_agent_run_store()
    rh.reset_mogdb()


@pytest.fixture()
def fake_runner(monkeypatch):
    async def execute(self, tool_name, args, context):
        return {"success": True, "stdout": f"obs-from-{tool_name}"}

    monkeypatch.setattr("domain.agents._internal.agents.ToolRunner.execute", execute, raising=True)
    return execute


def _agent(inference_fn, max_iterations=5, tools=None):
    config = AgentConfig(
        tools=tools or [ToolCapability.KNOWLEDGE_RETRIEVAL],
        max_iterations=max_iterations,
    )
    return Agent(config=config, inference_fn=inference_fn)


class TestLoop:
    @pytest.mark.asyncio
    async def test_replans_with_observations_until_empty_plan(self, fake_runner, run_store):
        inference = ScriptedInference(
            [
                '[{"tool": "knowledge_retrieval", "args": {"query": "x"}}]',
                "[]",
            ],
            answer="done from observations",
        )
        agent = _agent(inference)
        events = [e async for e in agent.execute_stream("research x", "s1")]
        kinds = [e["event"] for e in events]
        assert kinds == [
            "run_start",
            "tool_start",
            "tool_result",
            "final",
            "run_end",
        ]
        assert events[3]["text"] == "done from observations"
        # plan → re-plan (saw observations) → synthesize
        assert inference.calls == 3
        assert "Observations" in inference.prompts[1]
        assert "obs-from-knowledge_retrieval" in inference.prompts[1]

    @pytest.mark.asyncio
    async def test_execute_dict_consumes_same_loop(self, fake_runner, run_store):
        inference = ScriptedInference(['[{"tool": "knowledge_retrieval", "args": {}}]', "[]"])
        agent = _agent(inference)
        result = await agent.execute("research x", "s1")
        assert result["response"] == "synthesized answer"
        assert result["session_id"] == "s1"
        assert len(result["tools_used"]) == 1
        assert result["tools_used"][0]["tool"] == "knowledge_retrieval"

    @pytest.mark.asyncio
    async def test_keyword_plan_is_one_shot_then_synthesizes(self, fake_runner, run_store):
        agent = _agent(None, max_iterations=3)
        events = [e async for e in agent.execute_stream('search for "alpha"', "s2")]
        kinds = [e["event"] for e in events]
        assert kinds[0] == "run_start" and kinds[-1] == "run_end"
        # at most one tool batch — keyword planning never re-runs mid-loop
        assert kinds.count("tool_start") <= 1
        assert "final" in kinds

    @pytest.mark.asyncio
    async def test_budget_bounds_the_loop(self, fake_runner, run_store):
        inference = ScriptedInference([])  # every plan call → [] would stop early...
        # force endless planning: always return a tool until budget exhausted
        endless = ScriptedInference(['[{"tool": "knowledge_retrieval", "args": {}}]'] * 10)
        agent = _agent(endless, max_iterations=3)
        events = [e async for e in agent.execute_stream("go", "s3")]
        kinds = [e["event"] for e in events]
        assert kinds.count("tool_start") == 3
        assert inference.calls == 0
        assert kinds[-1] == "run_end"


class TestPersistenceAndExport:
    @pytest.mark.asyncio
    async def test_run_recorded_and_exportable(self, fake_runner, run_store):
        inference = ScriptedInference(['[{"tool": "knowledge_retrieval", "args": {}}]', "[]"])
        agent = _agent(inference)
        await agent.execute("research the loop", "s4")

        runs = run_store.list_runs()
        assert len(runs) == 1
        run = runs[0]
        assert run["status"] == "completed"
        assert run["goal"] == "research the loop"
        assert run["response"] == "synthesized answer"
        assert any("knowledge_retrieval" in line for line in run["logs"])

        examples = run_store.training_examples()
        assert len(examples) == 1
        ex = examples[0]
        assert ex["source"] == "agent-run"
        assert ex["messages"][0] == {"role": "user", "content": "research the loop"}
        assert ex["messages"][1] == {"role": "assistant", "content": "synthesized answer"}

    @pytest.mark.asyncio
    async def test_export_jsonl_roundtrip(self, fake_runner, run_store, tmp_path):
        inference = ScriptedInference(["[]"])
        agent = _agent(inference)
        await agent.execute("just answer", "s5")

        out = tmp_path / "agent_runs.jsonl"
        count = run_store.export_training_jsonl(str(out))
        assert count == 1
        line = out.read_text(encoding="utf-8").strip().splitlines()
        assert len(line) == 1
        parsed = json.loads(line[0])
        assert parsed["messages"][0]["role"] == "user"
        assert parsed["source"] == "agent-run"

    @pytest.mark.asyncio
    async def test_failed_runs_are_not_exported(self, fake_runner, run_store):
        inference = ScriptedInference(["[]"])
        agent = _agent(inference)
        run_id = run_store.start("abandoned goal")
        run_store.fail(run_id, "boom")
        await agent.execute("kept goal", "s6")
        examples = run_store.training_examples()
        assert len(examples) == 1
        assert examples[0]["messages"][0]["content"] == "kept goal"
