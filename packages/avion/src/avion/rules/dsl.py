"""Rule DSL — build rules from plain dicts (JSON/YAML friendly)."""

from __future__ import annotations

from typing import Any

from avion.rules.engine import Rule, RuleAction


def _when(spec: dict[str, Any]):
    """Turn {"url_contains": "/login", "has_failed": true} into a condition."""

    def condition(ctx: dict[str, Any]) -> bool:
        for key, want in spec.items():
            if key == "url_contains":
                if want not in str(ctx.get("current_url", "")):
                    return False
            elif key == "has_failed":
                failed = bool(ctx.get("failed", False))
                if failed != bool(want):
                    return False
            elif key == "event_count_at_least":
                if int(ctx.get("event_count", 0)) < int(want):
                    return False
            else:
                if ctx.get(key) != want:
                    return False
        return True

    return condition


def rules_from_dict(specs: list[dict[str, Any]]) -> list[Rule]:
    """Build rules from dicts.

    Example::

        rules_from_dict([{
            "name": "close-popup",
            "when": {"url_contains": "popup"},
            "then": {"action": "click", "params": {"target": "Close"}},
            "priority": 10,
            "max_fires": 1,
        }])
    """
    rules = []
    for spec in specs:
        then = spec.get("then", {})
        rules.append(
            Rule(
                name=spec["name"],
                condition=_when(spec.get("when", {})),
                action=RuleAction(
                    action=then.get("action", "noop"),
                    params=then.get("params", {}),
                ),
                priority=spec.get("priority", 0),
                max_fires=spec.get("max_fires"),
            )
        )
    return rules
