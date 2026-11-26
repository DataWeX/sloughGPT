"""Tests for the OpenCode session → chat-sft trajectory exporter (card 20261123_135823).

Canned payloads mirror the documented export endpoint's shape
(``/api/experimental/session/{id}/export`` → ``{data: {info, messages}}``).
"""

import json

from domain.training._internal.chat_sft_adapter import build
from domain.training._internal.session_exporter import (
    OBSERVATION_MAX,
    assign_split,
    export_session,
    extract_trajectories,
    filter_records,
    is_repo_session,
    list_sessions,
)


def _turn_messages(observation: str = "1 failed, 2 passed") -> list[dict]:
    """One user turn: plan → tool call → observation → synthesis (+ noise types)."""
    return [
        {"type": "user", "id": "m1", "text": "Fix the failing test and run the suite"},
        {
            "type": "assistant",
            "id": "m2",
            "content": [
                {"type": "reasoning", "text": "Plan: run pytest first to see the failure."},
                {
                    "type": "tool",
                    "id": "t1",
                    "name": "shell",
                    "state": {
                        "status": "completed",
                        "input": {"command": "pytest tests/core-py/test_x.py"},
                        "content": [{"type": "text", "text": observation}],
                    },
                },
            ],
        },
        {"type": "synthetic", "id": "m2s", "text": "system noise"},
        {
            "type": "assistant",
            "id": "m3",
            "content": [
                {"type": "text", "text": "The assertion expected 2 but got 1; fixed, now green."}
            ],
        },
        {"type": "idle", "id": "m4"},
        {"type": "system", "id": "m5", "content": "agent changed"},
    ]


class TestExtract:
    def test_turn_maps_to_seed_shape(self):
        records = extract_trajectories("ses_test", _turn_messages())
        assert len(records) == 1
        record = records[0]
        assert record["source"] == "session"
        assert record["id"] == "session-ses_test-0"
        messages = record["messages"]
        assert [m["role"] for m in messages] == ["user", "assistant", "tool", "assistant"]
        assert messages[0]["content"] == "Fix the failing test and run the suite"
        assert messages[1]["content"].startswith("Plan:")
        (call,) = messages[1]["tool_calls"]
        assert call["function"]["name"] == "shell"
        assert json.loads(call["function"]["arguments"]) == {
            "command": "pytest tests/core-py/test_x.py"
        }
        assert messages[2]["content"] == "1 failed, 2 passed"
        assert messages[3]["content"].startswith("The assertion expected")

    def test_multi_step_sequence_keeps_order(self):
        messages = [
            {"type": "user", "text": "Refactor and verify"},
            {
                "type": "assistant",
                "content": [
                    {"type": "reasoning", "text": "Plan: edit then test."},
                    {
                        "type": "tool",
                        "name": "edit",
                        "state": {
                            "status": "completed",
                            "input": {"path": "a.py"},
                            "content": [{"type": "text", "text": "applied"}],
                        },
                    },
                ],
            },
            {
                "type": "assistant",
                "content": [
                    {"type": "reasoning", "text": "Plan: run the suite."},
                    {
                        "type": "tool",
                        "name": "shell",
                        "state": {
                            "status": "completed",
                            "input": {"command": "pytest"},
                            "content": [{"type": "text", "text": "all green"}],
                        },
                    },
                ],
            },
            {
                "type": "assistant",
                "content": [{"type": "text", "text": "Refactor landed, suite green."}],
            },
        ]
        (record,) = extract_trajectories("ses_x", messages)
        roles = [m["role"] for m in record["messages"]]
        assert roles == ["user", "assistant", "tool", "assistant", "tool", "assistant"]
        assert record["messages"][-1]["content"] == "Refactor landed, suite green."

    def test_prose_turn_is_candidate_not_yet_filtered(self):
        messages = [
            {"type": "user", "text": "explain your architecture"},
            {
                "type": "assistant",
                "content": [{"type": "text", "text": "I am a helpful assistant."}],
            },
        ]
        (record,) = extract_trajectories("ses_p", messages)
        assert not any(m.get("tool_calls") for m in record["messages"])

    def test_oversized_observation_truncated(self):
        records = extract_trajectories("ses_big", _turn_messages(observation="x" * 10_000))
        tool_message = records[0]["messages"][2]
        assert len(tool_message["content"]) <= OBSERVATION_MAX + 1
        assert tool_message["content"].endswith("…")

    def test_empty_prompt_turn_not_emitted(self):
        messages = [
            {"type": "user", "text": ""},
            {"type": "assistant", "content": [{"type": "text", "text": "no prompt, no record"}]},
        ]
        assert extract_trajectories("ses_e", messages) == []


class TestFilter:
    @staticmethod
    def _candidate(messages: list[dict], tag: str) -> dict:
        return {"id": f"session-{tag}", "source": "session", "messages": messages}

    def test_discipline_drops_each_poison_class(self):
        good = self._candidate(
            extract_trajectories("ses_a", _turn_messages())[0]["messages"], "good"
        )
        prose = self._candidate(
            [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "Hi! How can I help you today?"},
            ],
            "prose",
        )
        refusal = self._candidate(
            good["messages"][:-1]
            + [{"role": "assistant", "content": "I'm sorry, I cannot help with that."}],
            "refusal",
        )
        degenerate = self._candidate(
            good["messages"][:-1] + [{"role": "assistant", "content": "aaaaaaaa"}],
            "degenerate",
        )
        no_synthesis = self._candidate(good["messages"][:-1], "nosyn")
        duplicate = dict(good, id="session-dup")

        kept, stats = filter_records([good, prose, refusal, degenerate, no_synthesis, duplicate])

        assert [r["id"] for r in kept] == ["session-good"]
        assert stats == {
            "total": 6,
            "no_tool": 1,
            "no_synthesis": 1,
            "refusal": 1,
            "degenerate": 1,
            "duplicate": 1,
            "kept": 1,
        }


class TestSplit:
    def test_deterministic_with_seed_ratio(self):
        assert assign_split("session-ses_x-0") == assign_split("session-ses_x-0")
        splits = [assign_split(f"session-ses_{i}-0") for i in range(2000)]
        assert set(splits) == {"train", "heldout"}
        ratio = splits.count("heldout") / len(splits)
        assert 0.05 <= ratio <= 0.30, f"heldout ratio {ratio:.3f} off the ~16% target"


class TestApiSeam:
    def test_list_sessions_follows_cursor(self):
        calls: list[str] = []

        def fake_api(path: str) -> dict:
            calls.append(path)
            if path == "/api/session":
                return {"data": [{"id": "ses_1"}], "cursor": "page2"}
            return {"data": [{"id": "ses_2"}], "cursor": None}

        sessions = list_sessions(api=fake_api)
        assert [s["id"] for s in sessions] == ["ses_1", "ses_2"]
        assert calls == ["/api/session", "/api/session?cursor=page2"]

    def test_export_session_unwraps_messages(self):
        def fake_api(path: str) -> dict:
            assert path == "/api/experimental/session/ses_1/export"
            return {"data": {"info": {"id": "ses_1"}, "messages": [{"type": "user", "text": "hi"}]}}

        assert export_session("ses_1", api=fake_api) == [{"type": "user", "text": "hi"}]

    def test_is_repo_session_prefix_match(self):
        repo = "/home/mana/Documents/Default Project/sloughGPT"
        assert is_repo_session({"location": {"directory": repo}}, repo)
        assert is_repo_session({"location": {"directory": repo + "/.wt-chatsft"}}, repo)
        assert not is_repo_session(
            {"location": {"directory": "/home/mana/Documents/Default Project/threoiss"}}, repo
        )
        assert not is_repo_session({}, repo)

    def test_is_repo_session_accepts_checkout_family(self):
        main = "/home/mana/Documents/Default Project/sloughGPT"
        wt = "/home/mana/.cache/wt-agent-phase2"
        family = [main, wt]
        assert is_repo_session({"location": {"directory": wt}}, family)
        assert is_repo_session({"location": {"directory": main}}, family)
        assert not is_repo_session(
            {"location": {"directory": "/home/mana/Documents/Default Project/threoiss"}}, family
        )


class TestAdapterContract:
    def test_emitted_records_pass_adapter_build(self, tmp_path):
        candidates = extract_trajectories("ses_a", _turn_messages())
        kept, _ = filter_records(candidates)
        for record in kept:
            record["split"] = assign_split(record["id"])
        path = tmp_path / "session_trajectories.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in kept))

        all_records = build(path, split=None)
        trains = build(path, split="train")
        heldouts = build(path, split="heldout")
        assert len(all_records) == len(kept)
        assert len(trains) + len(heldouts) == len(all_records)
        rendered = trains[0]
        assert "User:" in rendered
        assert "Tool call: shell" in rendered
        assert "Tool: 1 failed, 2 passed" in rendered
