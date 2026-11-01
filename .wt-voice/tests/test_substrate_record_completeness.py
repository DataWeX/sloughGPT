"""Substrate completeness contract — every run must leave a full record trail.

Audit instrument for card 3b6a5ca2 (user directive, handoff 5d060399):
"...make sure your episodes and memory and mind and cognition of that aspect
of the stack is structured enough to output the entire information that was
recorded during the run and the execution of those processes."

PASSING tests = experience paths that provably record full payloads today.
xfail(strict=True) = documented gaps G1..G7, each naming its gap child card
(audited on 2026-10-17, see note f32a761f): when the gap closes the xfail
turns RED (XPASS) and must be promoted to a passing test — neither the gap
nor its fix can rot silently.

Every test builds its own engine over a tmp store — the production store
(data/consciousness) is never touched.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from domain.cognition._internal.consciousness.config import ConsciousnessConfig
from domain.cognition._internal.consciousness.engine import ConsciousnessEngine

REPO = Path(__file__).resolve().parents[1]


def make_engine(tmp_path, level: int = 1) -> ConsciousnessEngine:
    """An isolated engine over a fresh store (own MindLog + SelfModel)."""
    store = tmp_path / "store"
    store.mkdir(parents=True, exist_ok=True)
    cfg = ConsciousnessConfig(level=level, store_path=str(store))
    return ConsciousnessEngine(cfg)


def episodes(engine: ConsciousnessEngine):
    return engine.mind_log.query(record_type="episode", limit=0)


def all_records(engine: ConsciousnessEngine):
    return engine.mind_log.query(limit=0)


# NOTE (main landing 2026-10-19): the audit lane's full-payload tests
# (process/dream/reflect/feedback/locus) ride with the audit lane's own
# merge — they need mind_log/locus on ConsciousnessEngine, which main
# does not have yet. G7's trail + behavioral checks below are complete
# for this tree.


# --- documented gaps: strict xfails keyed to their gap child cards --------


@pytest.mark.xfail(
    strict=True,
    reason="GAP G1 (child of 3b6a5ca2): process() records unconditionally — "
    "24 of 47 real production episodes have an empty response",
)
def test_process_never_writes_contentless_episode(tmp_path):
    eng = make_engine(tmp_path)
    eng.process("Hi", "")
    assert all(r.payload.get("response") for r in episodes(eng)), (
        "an experience with no content must not be written as an episode"
    )


# (G2 landed on main 2026-10-19 via fix/substrate-g2-nonstream-postgen —
# kept un-xfailed here as a regression guard for the non-stream post-gen call.)
def test_non_stream_chat_runs_post_gen_tasks():
    src = (REPO / "apps/api/server/routers/inference.py").read_text()
    m = re.search(r"    async def chat\((?s:.*?)\n    (?:async )?def ", src)
    assert m, "chat() body not found"
    assert "_run_post_gen_tasks" in m.group(0), (
        "the non-stream chat handler must run the same post-generation tasks "
        "(consciousness processing) the stream handler runs"
    )


@pytest.mark.xfail(
    strict=True,
    reason="GAP G3 (child of 3b6a5ca2): _mind_last is process-lifetime-only "
    "and never rehydrated from the log — 0 of 63 production records carry edges",
)
def test_dream_edge_survives_restart(tmp_path):
    eng = make_engine(tmp_path)
    eng.process("Hi", "Yo")
    eng.save()

    eng2 = make_engine(tmp_path)
    eng2.dream()
    dreams = eng2.mind_log.query(record_type="dream", limit=0)
    assert dreams and dreams[-1].edges, (
        "a dream after restart must still edge to the episode it consolidates "
        "(seed _mind_last from the log tail at boot)"
    )


@pytest.mark.xfail(
    strict=True,
    reason="GAP G4 (child of 3b6a5ca2): observe_feedback returns before any "
    "record for neutral and saturated ratings — the event itself is never logged",
)
def test_feedback_events_always_leave_a_record(tmp_path):
    eng = make_engine(tmp_path)
    before = len(all_records(eng))
    out = eng.observe_feedback("neutral")
    assert out["observed"] is True
    assert len(all_records(eng)) > before, (
        "a feedback event is experience and must leave a record even when it "
        "moves no belief (neutral or saturated)"
    )


@pytest.mark.xfail(
    strict=True,
    reason="GAP G4 (child of 3b6a5ca2): saturated confidence short-circuits "
    "with no record — 'saturated downs undercount by design'",
)
def test_saturated_feedback_still_recorded(tmp_path):
    eng = make_engine(tmp_path)
    for name in ("accuracy", "helpfulness"):
        eng.self_model.self_beliefs[name] = 1.0
    before = len(all_records(eng))
    out = eng.observe_feedback("thumbs_up")
    assert out["observed"] is True
    assert len(all_records(eng)) > before


@pytest.mark.xfail(
    strict=True,
    reason="GAP G5 (child of 3b6a5ca2): consciousness.seed appends to the "
    "self_model document directly, never through _mind_append",
)
def test_seed_endpoint_reaches_the_log():
    src = (REPO / "apps/api/server/routers/consciousness.py").read_text()
    m = re.search(r"async def seed_data\((?s:.*?)\n    (?:@endpoint|    async def )", src)
    assert m, "seed_data body not found"
    assert "_mind_append" in m.group(0) or "mind_log" in m.group(0), (
        "seeded episodes must enter through the log like every other episode"
    )


@pytest.mark.xfail(
    strict=True,
    reason="GAP G6 (child of 3b6a5ca2): 99 of 100 production episodes predate "
    "the log and were never backfilled — record-level past unrecoverable (blink)",
)
def test_document_episodes_present_in_log(tmp_path):
    store = tmp_path / "store"
    store.mkdir(parents=True, exist_ok=True)
    store.joinpath("self_model.json").write_text(
        json.dumps(
            {
                "episodes": [
                    {
                        "timestamp": 1.0,
                        "input_text": f"pre-landing {i}",
                        "response": "r",
                        "qualia": {},
                        "self_insight": "",
                        "growth_delta": 0.0,
                        "is_dream": False,
                    }
                    for i in range(3)
                ]
            }
        )
    )
    eng = make_engine(tmp_path)
    doc_eps = len(eng.self_model.episodes)
    assert doc_eps == 3
    assert len(episodes(eng)) >= doc_eps, (
        "every episode in the state document must exist in the append-only log"
    )


def test_agent_stack_records_experience(tmp_path):
    """Agent runs and journey executions must leave a record trail.

    The seam is `record_experience` (domain.cognition — the single
    substrate-recording entrypoint for non-chat runners); chat records
    through its own post-gen pipeline, which uses get_consciousness /
    mind_log directly. Any of these names in an agent/journey runner
    file counts as a trail.
    """
    hits = []
    for root in ("domain/agents", "packages/uxjourneys", "domain/journeys"):
        base = REPO / root
        if not base.exists():
            continue
        for f in base.rglob("*.py"):
            text = f.read_text(errors="ignore")
            if any(name in text for name in ("get_consciousness", "mind_log", "record_experience")):
                hits.append(str(f.relative_to(REPO)))
    assert hits, "agent/journey execution must leave at least one record trail"


def test_agent_run_lifecycle_records_experience(tmp_path, monkeypatch):
    """Behavioral half of G7: an agent run's terminal transitions each
    record exactly one substrate experience through the shared seam —
    completion and failure alike. A grep hit is not proof."""
    import domain.agents._internal.run_history as rh

    calls: list[dict] = []

    def _spy(prompt: str, response: str, *, source: str = "agent"):
        calls.append({"prompt": prompt, "response": response, "source": source})
        return None

    monkeypatch.setattr("domain.cognition.record_experience", _spy)
    db = str(tmp_path / "agent_runs")
    rh.set_mogdb_path(db)
    rh.reset_mogdb()
    try:
        store = rh.AgentRunStore(db_path=db)

        rid = store.start(goal="ship the feature")
        store.complete(rid, response="done, tests green")
        assert len(calls) == 1
        assert calls[0]["prompt"] == "ship the feature"
        assert calls[0]["response"] == "done, tests green"
        assert calls[0]["source"] == "agent.run"

        rid2 = store.start(goal="risky change")
        store.fail(rid2, "boom")
        assert len(calls) == 2
        assert calls[1]["prompt"] == "risky change"
        assert "boom" in calls[1]["response"]

        # Unknown run ids are no-ops — they must not record fake episodes.
        store.complete("run_missing", response="ghost")
        store.fail("run_missing", "ghost")
        assert len(calls) == 2
    finally:
        rh.reset_agent_run_store()
        rh.reset_mogdb()
