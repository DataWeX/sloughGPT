"""Tests for planner.engine — Kanban state machine and transition engine."""

from planner.engine import (
    CardState,
    EventType,
    KanbanEngine,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_board(columns=None, cards=None):
    """Build a minimal board dict structure."""
    if columns is None:
        columns = [
            {"name": "inbox", "wip_limit": 0, "order": 0},
            {"name": "ready", "wip_limit": 0, "order": 1},
            {"name": "doing", "wip_limit": 3, "order": 2},
            {"name": "review", "wip_limit": 2, "order": 3},
            {"name": "verify", "wip_limit": 0, "order": 4},
            {"name": "done", "wip_limit": 0, "order": 5},
            {"name": "blocked", "wip_limit": 0, "order": 6},
        ]
    if cards is None:
        cards = [
            {"id": "card-a", "column": "inbox", "title": "Card A"},
            {"id": "card-b", "column": "ready", "title": "Card B"},
            {"id": "card-c", "column": "doing", "title": "Card C"},
        ]
    return columns, cards


# ---------------------------------------------------------------------------
# State machine basics
# ---------------------------------------------------------------------------


class TestStateEnum:
    def test_all_states_exist(self):
        assert CardState.INBOX.value == "inbox"
        assert CardState.READY.value == "ready"
        assert CardState.DOING.value == "doing"
        assert CardState.REVIEW.value == "review"
        assert CardState.VERIFY.value == "verify"
        assert CardState.DONE.value == "done"
        assert CardState.BLOCKED.value == "blocked"

    def test_seven_states(self):
        assert len(CardState) == 7


class TestLegalTransitions:
    def test_inbox_to_ready(self):
        assert KanbanEngine.is_legal("inbox", "ready")

    def test_inbox_cannot_skip(self):
        assert not KanbanEngine.is_legal("inbox", "doing")
        assert not KanbanEngine.is_legal("inbox", "done")

    def test_ready_to_doing(self):
        assert KanbanEngine.is_legal("ready", "doing")

    def test_ready_to_blocked(self):
        assert KanbanEngine.is_legal("ready", "blocked")

    def test_doing_to_review(self):
        assert KanbanEngine.is_legal("doing", "review")

    def test_doing_back_to_ready(self):
        assert KanbanEngine.is_legal("doing", "ready")

    def test_review_to_verify(self):
        assert KanbanEngine.is_legal("review", "verify")

    def test_review_back_to_doing(self):
        assert KanbanEngine.is_legal("review", "doing")

    def test_verify_to_done(self):
        assert KanbanEngine.is_legal("verify", "done")

    def test_verify_back_to_review(self):
        assert KanbanEngine.is_legal("verify", "review")

    def test_done_is_terminal(self):
        assert not KanbanEngine.is_legal("done", "inbox")
        assert not KanbanEngine.is_legal("done", "ready")
        assert not KanbanEngine.is_legal("done", "doing")
        assert KanbanEngine.is_terminal("done")

    def test_blocked_to_any_prior(self):
        assert KanbanEngine.is_legal("blocked", "inbox")
        assert KanbanEngine.is_legal("blocked", "ready")
        assert KanbanEngine.is_legal("blocked", "doing")

    def test_blocked_cannot_go_to_done(self):
        assert not KanbanEngine.is_legal("blocked", "done")

    def test_illegal_transitions_rejected(self):
        # Cannot skip stages
        assert not KanbanEngine.is_legal("inbox", "review")
        assert not KanbanEngine.is_legal("inbox", "verify")
        assert not KanbanEngine.is_legal("ready", "review")
        assert not KanbanEngine.is_legal("doing", "done")

    def test_unknown_state_returns_false(self):
        assert not KanbanEngine.is_legal("bogus", "ready")
        assert not KanbanEngine.is_legal("inbox", "bogus")


class TestNextStates:
    def test_next_states_inbox(self):
        assert KanbanEngine.next_states("inbox") == ["ready"]

    def test_next_states_doing(self):
        result = KanbanEngine.next_states("doing")
        assert sorted(result) == ["blocked", "ready", "review"]

    def test_next_states_done(self):
        assert KanbanEngine.next_states("done") == []

    def test_next_states_blocked(self):
        result = KanbanEngine.next_states("blocked")
        assert sorted(result) == ["doing", "inbox", "ready"]


class TestActiveAndTerminal:
    def test_done_is_terminal(self):
        assert KanbanEngine.is_terminal("done")

    def test_inbox_is_not_terminal(self):
        assert not KanbanEngine.is_terminal("inbox")

    def test_doing_is_active(self):
        assert KanbanEngine.is_active("doing")

    def test_inbox_is_not_active(self):
        assert not KanbanEngine.is_active("inbox")

    def test_done_is_not_active(self):
        assert not KanbanEngine.is_active("done")


# ---------------------------------------------------------------------------
# Transition application
# ---------------------------------------------------------------------------


class TestTransition:
    def test_successful_move(self):
        cols, cards = make_board()
        engine = KanbanEngine()
        result = engine.transition(cols, cards, "card-a", "ready", actor="alice")
        assert result.ok
        assert result.from_state == "inbox"
        assert result.to_state == "ready"
        assert cards[0]["column"] == "ready"

    def test_idempotent_same_state(self):
        cols, cards = make_board()
        engine = KanbanEngine()
        result = engine.transition(cols, cards, "card-b", "ready")
        assert result.ok
        assert result.event.metadata.get("noop") is True

    def test_card_not_found(self):
        cols, cards = make_board()
        engine = KanbanEngine()
        result = engine.transition(cols, cards, "ghost", "ready")
        assert not result.ok
        assert "not found" in result.error

    def test_illegal_transition_rejected(self):
        cols, cards = make_board()
        engine = KanbanEngine()
        result = engine.transition(cols, cards, "card-a", "done")
        assert not result.ok
        assert "Illegal transition" in result.error
        # Card unchanged
        assert cards[0]["column"] == "inbox"

    def test_unknown_target_state(self):
        cols, cards = make_board()
        engine = KanbanEngine()
        result = engine.transition(cols, cards, "card-a", "bogus")
        assert not result.ok
        assert "Unknown state" in result.error

    def test_force_bypasses_wip(self):
        cols, cards = make_board()
        # Fill doing to limit
        cards.append({"id": "card-d", "column": "doing", "title": "D"})
        cards.append({"id": "card-e", "column": "doing", "title": "E"})
        engine = KanbanEngine()
        result = engine.transition(cols, cards, "card-b", "doing", force=True)
        assert result.ok

    def test_full_flow_inbox_to_done(self):
        cols, cards = make_board()
        engine = KanbanEngine()
        # inbox → ready → doing → review → verify → done
        path = ["ready", "doing", "review", "verify", "done"]
        for target in path:
            result = engine.transition(cols, cards, "card-a", target)
            assert result.ok, f"Failed at {target}: {result.error}"
        assert cards[0]["column"] == "done"

    def test_block_and_unblock(self):
        cols, cards = make_board()
        engine = KanbanEngine()
        result = engine.transition(cols, cards, "card-c", "blocked")
        assert result.ok
        assert result.event.type == EventType.CARD_BLOCKED

        result = engine.transition(cols, cards, "card-c", "doing")
        assert result.ok
        assert result.event.type == EventType.CARD_UNBLOCKED


# ---------------------------------------------------------------------------
# WIP enforcement
# ---------------------------------------------------------------------------


class TestWipEnforcement:
    def test_wip_limit_blocks(self):
        cols, cards = make_board()
        # Fill review (limit 2)
        cards.append({"id": "card-r1", "column": "review", "title": "R1"})
        cards.append({"id": "card-r2", "column": "review", "title": "R2"})
        engine = KanbanEngine()
        # card-c is in doing → try to move to review (full)
        result = engine.transition(cols, cards, "card-c", "review")
        assert not result.ok
        assert "WIP limit" in result.error
        assert result.event.type == EventType.WIP_EXCEEDED
        assert result.wip_limit == 2
        assert result.wip_count == 3

    def test_wip_limit_respects_existing_cards(self):
        cols, cards = make_board()
        # 1 card in doing (limit 3) — room for 2 more
        engine = KanbanEngine()
        result = engine.transition(cols, cards, "card-b", "doing")
        assert result.ok
        result = engine.transition(cols, cards, "card-a", "ready")
        assert result.ok
        # Move card-a: ready → doing (now 2 in doing)
        cards[0]["column"] = "ready"
        result = engine.transition(cols, cards, "card-a", "doing")
        assert result.ok

    def test_zero_wip_limit_means_unlimited(self):
        cols, cards = make_board()
        # Add 10 cards to "verify" (limit 0 = unlimited)
        for i in range(10):
            cards.append({"id": f"v{i}", "column": "verify", "title": f"V{i}"})
        engine = KanbanEngine()
        result = engine.transition(cols, cards, "card-c", "review")
        # review has limit 2, but we have room
        assert result.ok

    def test_wip_count_excludes_moved_card(self):
        cols, cards = make_board()
        # card-c is already in doing. Move it to review (limit 2, 0 current)
        engine = KanbanEngine()
        result = engine.transition(cols, cards, "card-c", "review")
        assert result.ok
        assert result.wip_count == 1  # only card-c, not counting itself in doing


# ---------------------------------------------------------------------------
# Event emission
# ---------------------------------------------------------------------------


class TestEvents:
    def test_event_emitted_on_move(self):
        cols, cards = make_board()
        events = []
        engine = KanbanEngine(on_event=events.append)
        engine.transition(cols, cards, "card-a", "ready", actor="alice")
        assert len(events) == 1
        assert events[0].type == EventType.CARD_MOVED
        assert events[0].card_id == "card-a"
        assert events[0].from_state == "inbox"
        assert events[0].to_state == "ready"
        assert events[0].actor == "alice"
        assert events[0].timestamp

    def test_event_emitted_on_rejection(self):
        cols, cards = make_board()
        events = []
        engine = KanbanEngine(on_event=events.append)
        engine.transition(cols, cards, "card-a", "done")
        assert len(events) == 1
        assert events[0].type == EventType.TRANSITION_REJECTED

    def test_event_log_accumulates(self):
        cols, cards = make_board()
        engine = KanbanEngine()
        engine.transition(cols, cards, "card-a", "ready")
        engine.transition(cols, cards, "card-a", "doing")
        engine.transition(cols, cards, "card-b", "doing")
        assert len(engine.event_log) == 3

    def test_event_to_dict(self):
        cols, cards = make_board()
        engine = KanbanEngine()
        result = engine.transition(cols, cards, "card-a", "ready")
        d = result.event.to_dict()
        assert d["type"] == "card_moved"
        assert d["card_id"] == "card-a"
        assert "timestamp" in d


# ---------------------------------------------------------------------------
# Board integration
# ---------------------------------------------------------------------------


class TestBoardIntegration:
    def test_transition_updates_card_timestamp(self):
        cols, cards = make_board()
        engine = KanbanEngine()
        engine.transition(cols, cards, "card-a", "ready")
        assert "updated_at" in cards[0]
        assert cards[0]["updated_at"] != ""

    def test_multiple_cards_same_column(self):
        cols, cards = make_board()
        cards.append({"id": "card-d", "column": "inbox", "title": "D"})
        engine = KanbanEngine()
        engine.transition(cols, cards, "card-a", "ready")
        engine.transition(cols, cards, "card-d", "ready")
        ready_count = sum(1 for c in cards if c["column"] == "ready")
        # card-b was already ready, plus card-a and card-d moved = 3
        assert ready_count == 3

    def test_wip_check_on_full_column(self):
        cols, cards = make_board()
        # Fill doing to 3 (limit)
        cards.append({"id": "card-d", "column": "doing", "title": "D"})
        cards.append({"id": "card-e", "column": "doing", "title": "E"})
        engine = KanbanEngine()
        # card-b: ready → doing (would be 4th, limit 3)
        result = engine.transition(cols, cards, "card-b", "doing")
        assert not result.ok
        assert result.wip_exceeded
