"""Rule engine — if condition holds, fire action. Priority ordered."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(frozen=True)
class RuleAction:
    """What to do when a rule fires."""

    action: str
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"action": self.action, "params": self.params}


@dataclass
class Rule:
    """A named condition → action mapping.

    condition: callable taking the context dict, returning True to fire.
    priority: higher fires first. max_fires: cap on firings (None = unlimited).
    """

    name: str
    condition: Callable[[dict[str, Any]], bool]
    action: RuleAction
    priority: int = 0
    max_fires: int | None = None
    fires: int = 0

    @property
    def exhausted(self) -> bool:
        return self.max_fires is not None and self.fires >= self.max_fires


class RuleEngine:
    """Holds rules, evaluates them against a context dict.

    Usage::

        engine = RuleEngine()
        engine.add_rule(Rule("close-popup",
                             lambda ctx: "popup" in ctx.get("url", ""),
                             RuleAction("click", {"target": "Close"})))
        fired = await engine.evaluate({"url": "...popup..."})
    """

    def __init__(self):
        self._rules: list[Rule] = []

    def add_rule(self, rule: Rule) -> None:
        self._rules.append(rule)
        self._rules.sort(key=lambda r: r.priority, reverse=True)

    def remove_rule(self, name: str) -> bool:
        before = len(self._rules)
        self._rules = [r for r in self._rules if r.name != name]
        return len(self._rules) < before

    @property
    def rules(self) -> list[Rule]:
        return list(self._rules)

    async def evaluate(self, context: dict[str, Any] | None = None) -> list[RuleAction]:
        """Fire all matching, non-exhausted rules. Returns actions fired."""
        ctx = context or {}
        fired: list[RuleAction] = []
        for rule in self._rules:
            if rule.exhausted:
                continue
            try:
                match = rule.condition(ctx)
            except Exception:
                match = False
            if match:
                rule.fires += 1
                fired.append(rule.action)
        return fired

    def reset_counts(self) -> None:
        for rule in self._rules:
            rule.fires = 0
