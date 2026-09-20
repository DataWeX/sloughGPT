"""Rules tests — priorities, max-fires, DSL, error safety."""

import asyncio

from arken.rules import Rule, RuleAction, RuleEngine, rules_from_dict


def run(coro):
    return asyncio.run(coro)


class TestEngine:
    def test_fires_matching_rule(self):
        eng = RuleEngine()
        eng.add_rule(Rule("a", lambda ctx: True, RuleAction("click")))
        assert run(eng.evaluate({})) == [RuleAction("click")]

    def test_skips_non_matching(self):
        eng = RuleEngine()
        eng.add_rule(Rule("a", lambda ctx: False, RuleAction("click")))
        assert run(eng.evaluate({})) == []

    def test_priority_order(self):
        eng = RuleEngine()
        eng.add_rule(Rule("low", lambda ctx: True, RuleAction("low"), priority=1))
        eng.add_rule(Rule("high", lambda ctx: True, RuleAction("high"), priority=10))
        assert [a.action for a in run(eng.evaluate({}))] == ["high", "low"]

    def test_max_fires(self):
        eng = RuleEngine()
        eng.add_rule(Rule("once", lambda ctx: True, RuleAction("x"), max_fires=1))
        assert len(run(eng.evaluate({}))) == 1
        assert run(eng.evaluate({})) == []
        eng.reset_counts()
        assert len(run(eng.evaluate({}))) == 1

    def test_bad_condition_does_not_crash(self):
        eng = RuleEngine()
        eng.add_rule(Rule("bad", lambda ctx: 1 / 0, RuleAction("x")))
        assert run(eng.evaluate({})) == []

    def test_remove_rule(self):
        eng = RuleEngine()
        eng.add_rule(Rule("a", lambda ctx: True, RuleAction("x")))
        assert eng.remove_rule("a") is True
        assert eng.remove_rule("a") is False


class TestDsl:
    def test_from_dict(self):
        rules = rules_from_dict(
            [
                {
                    "name": "popup",
                    "when": {"url_contains": "popup"},
                    "then": {"action": "click", "params": {"target": "Close"}},
                    "priority": 5,
                    "max_fires": 2,
                }
            ]
        )
        eng = RuleEngine()
        for r in rules:
            eng.add_rule(r)
        fired = run(eng.evaluate({"current_url": "http://x/popup/1"}))
        assert fired[0].params == {"target": "Close"}
        assert run(eng.evaluate({"current_url": "http://x/home"})) == []
