"""chat-sft seed fixture — card 20261101_052145 step 2 (50-100 pair seed).

Pins the committed hand-checked seed:
  * size band (50-100) — the card's step-2 contract;
  * split discipline — heldout never enters the default train array;
  * source mix — real agent-run pairs + hand-authored trajectories;
  * refusal-free — the poison class this card exists to remove;
  * tool-use mass — plan -> tool_calls -> synthesis materially present;
  * renders through the adapter for every split.
"""

from __future__ import annotations

import json
from pathlib import Path

from domain.agents._internal.run_history import looks_like_refusal
from domain.training._internal.chat_sft_adapter import build

SEED = Path(__file__).resolve().parents[1] / "fixtures" / "chat_sft_seed.jsonl"


def _records() -> list[dict]:
    assert SEED.exists(), f"seed fixture missing: {SEED}"
    records: list[dict] = []
    for line in SEED.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


class TestSeedFixture:
    def test_size_band_50_to_100(self):
        records = _records()
        assert 50 <= len(records) <= 100, f"seed has {len(records)} records"

    def test_split_discipline(self):
        records = _records()
        heldout = [r for r in records if r.get("split", "train") == "heldout"]
        train = [r for r in records if r.get("split", "train") == "train"]
        assert len(heldout) >= 10, "eval set too small to measure against"
        assert len(train) >= 50, "train slice too small to learn from"
        assert len(train) + len(heldout) == len(records)

    def test_ids_unique_and_present(self):
        records = _records()
        ids = [r.get("id") for r in records]
        assert all(ids), "every seed record carries an id"
        assert len(set(ids)) == len(ids)

    def test_roles_are_known(self):
        allowed = {"system", "user", "assistant", "tool"}
        for record in _records():
            for message in record["messages"]:
                assert message.get("role") in allowed, (
                    record.get("id"),
                    message.get("role"),
                )

    def test_source_mix(self):
        records = _records()
        sources = {r.get("source") for r in records}
        assert sources <= {"run", "seed"}, f"unexpected sources: {sources}"
        assert sum(1 for r in records if r.get("source") == "run") >= 1
        assert sum(1 for r in records if r.get("source") == "seed") >= 30

    def test_no_refusals_anywhere(self):
        for record in _records():
            for message in record["messages"]:
                if message.get("role") == "assistant" and message.get("content"):
                    assert not looks_like_refusal(message["content"]), record.get("id")

    def test_tool_use_mass_present(self):
        records = _records()
        with_calls = [r for r in records if any(m.get("tool_calls") for m in r["messages"])]
        assert len(with_calls) >= 40, "tool-use trajectories are the point of this card"

    def test_run_records_carry_provenance(self):
        runs = [r for r in _records() if r.get("source") == "run"]
        assert runs, "seed must include real agent-run pairs"
        for record in runs:
            assert str(record.get("id", "")).startswith("run_"), record.get("id")
            assert str(record.get("run_id", "")).startswith("run_"), record.get("id")

    def test_renders_every_split_through_the_adapter(self):
        records = _records()
        heldout_n = sum(1 for r in records if r.get("split") == "heldout")
        train = build(SEED)
        heldout = build(SEED, split="heldout")
        everything = build(SEED, split=None)
        assert len(train) == len(records) - heldout_n, "default train must exclude every heldout record"
        assert len(heldout) == heldout_n
        assert len(everything) == len(train) + len(heldout)
        rendered = "".join(train[i] for i in range(len(train)))
        assert "User: " in rendered
        assert "Assistant: " in rendered
        assert "Tool call: " in rendered
