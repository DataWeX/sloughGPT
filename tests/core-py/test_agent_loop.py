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

    @pytest.mark.asyncio
    async def test_model_empty_plan_falls_back_to_keywords_on_step0(self, fake_runner, run_store):
        # The bootstrap model plans [] even for explicit tool requests —
        # step 0 must still honor a conservative keyword match.
        inference = ScriptedInference(["[]"])
        agent = _agent(inference, tools=[ToolCapability.FILE_SEARCH])
        events = [e async for e in agent.execute_stream('search for "alpha"', "s7")]
        kinds = [e["event"] for e in events]
        assert kinds.count("tool_start") == 1
        tool_events = [e for e in events if e["event"] == "tool_start"]
        assert tool_events[0]["tool"] == "file_search"

    @pytest.mark.asyncio
    async def test_no_keyword_rescue_mid_loop(self, fake_runner, run_store):
        # After observations exist, [] means synthesize — never re-keyword.
        inference = ScriptedInference(['[{"tool": "knowledge_retrieval", "args": {}}]', "[]"])
        agent = _agent(inference)
        events = [e async for e in agent.execute_stream('search for "alpha"', "s8")]
        kinds = [e["event"] for e in events]
        assert kinds.count("tool_start") == 1

    @pytest.mark.asyncio
    async def test_keyword_matches_quoted_query_anywhere_in_sentence(self, fake_runner, run_store):
        # The pre-fix pattern required the quote right after "search for" and
        # missed this exact journey phrasing.
        inference = ScriptedInference(["[]"])
        agent = _agent(inference, tools=[ToolCapability.FILE_SEARCH])
        events = [
            e async for e in agent.execute_stream('search for files matching "run_history"', "s9")
        ]
        tool_events = [e for e in events if e["event"] == "tool_start"]
        assert tool_events and tool_events[0]["tool"] == "file_search"
        assert tool_events[0]["args"]["query"] == "run_history"

    @pytest.mark.asyncio
    async def test_single_quoted_model_plan_still_parses(self, fake_runner, run_store):
        # The 0.5B often emits Python-style single-quoted dicts — the ast
        # fallback must recover them instead of logging a parse failure.
        inference = ScriptedInference(
            ["[{'tool': 'knowledge_retrieval', 'args': {'query': 'x'}}]", "[]"]
        )
        agent = _agent(inference)
        events = [e async for e in agent.execute_stream("research x", "s10")]
        tool_events = [e for e in events if e["event"] == "tool_start"]
        assert tool_events and tool_events[0]["tool"] == "knowledge_retrieval"

    @pytest.mark.asyncio
    async def test_degenerate_synthesis_falls_back_to_compose(self, fake_runner, run_store):
        loop_text = "the tool has not been executed " * 8
        inference = ScriptedInference(
            ['[{"tool": "knowledge_retrieval", "args": {}}]', "[]"], answer=loop_text
        )
        agent = _agent(inference)
        events = [e async for e in agent.execute_stream("research x", "s11")]
        finals = [e for e in events if e["event"] == "final"]
        # compose path produces an honest observation-based line, never the loop
        assert finals and finals[0]["text"].strip()
        assert "has not been executed" not in finals[0]["text"]

    @pytest.mark.asyncio
    async def test_refusal_synthesis_falls_back_to_compose(self, fake_runner, run_store):
        inference = ScriptedInference(
            ['[{"tool": "knowledge_retrieval", "args": {}}]', "[]"],
            answer="I'm sorry, but I can't assist with that request.",
        )
        agent = _agent(inference)
        events = [e async for e in agent.execute_stream("research x", "s12")]
        finals = [e for e in events if e["event"] == "final"]
        assert finals and "sorry" not in finals[0]["text"].lower()

    @pytest.mark.asyncio
    async def test_bigram_tail_loop_is_degenerate(self, fake_runner, run_store):
        # Live symptom: "…artificial intelligence, artificial intelligence, …" x3
        tail = "Models are useful. artificial intelligence artificial intelligence artificial intelligence"
        inference = ScriptedInference(
            ['[{"tool": "knowledge_retrieval", "args": {}}]', "[]"], answer=tail
        )
        agent = _agent(inference)
        events = [e async for e in agent.execute_stream("research x", "s13")]
        finals = [e for e in events if e["event"] == "final"]
        assert finals and "artificial intelligence artificial" not in finals[0]["text"]


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

    def test_refusal_responses_are_not_exported(self, run_store):
        good = run_store.start("real question")
        run_store.complete(good, response="The answer is 42.")
        bad = run_store.start("risky question")
        run_store.complete(bad, response="I'm sorry, but I'm unable to assist with that request.")
        examples = run_store.training_examples()
        assert len(examples) == 1
        assert examples[0]["run_id"] == good
        assert examples[0]["messages"][1]["content"] == "The answer is 42."
