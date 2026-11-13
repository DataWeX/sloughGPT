"""Verifier — ground-truth goal checks for agent runs.

Instead of trusting the model's DONE, check the world: URL, page text,
element presence. A run only counts as success when its goal checks pass.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class GoalCheck:
    """One check against the live page.

    kind: url_contains | text_present | text_absent | element_present
    value: substring/needle, or locator describe() for element_present.
    """

    kind: str
    value: str = ""

    def to_dict(self) -> dict[str, str]:
        return {"kind": self.kind, "value": self.value}

    @classmethod
    def from_dict(cls, data: dict[str, str]) -> GoalCheck:
        return cls(kind=data["kind"], value=data.get("value", ""))


@dataclass
class VerifyResult:
    """Outcome of checking a goal."""

    passed: bool
    failed: list[str] = field(default_factory=list)
    checked: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "checked": self.checked,
            "failed": self.failed,
        }


class Verifier:
    """Runs goal checks against page getters.

    Usage::

        v = Verifier()
        result = await v.verify(
            [GoalCheck("url_contains", "/chat")],
            url="http://x/chat", text="hi",
        )
        assert result.passed
    """

    async def verify(
        self,
        checks: list[GoalCheck | dict[str, str]],
        url: str = "",
        text: str = "",
        find=None,
    ) -> VerifyResult:
        """Check a goal. find: async callable(locator)->element|None."""
        failed: list[str] = []
        done = 0
        for raw in checks:
            check = raw if isinstance(raw, GoalCheck) else GoalCheck.from_dict(raw)
            done += 1
            if check.kind == "url_contains":
                if check.value not in url:
                    failed.append(f"url {url!r} lacks {check.value!r}")
            elif check.kind == "text_present":
                if check.value not in text:
                    failed.append(f"text lacks {check.value!r}")
            elif check.kind == "text_absent":
                if check.value in text:
                    failed.append(f"text still has {check.value!r}")
            elif check.kind == "element_present":
                if find is None:
                    failed.append("no finder for element_present")
                    continue
                from avion.events.replay import parse_locator

                locator = parse_locator(check.value)
                if locator is None:
                    failed.append(f"unparseable locator {check.value!r}")
                    continue
                try:
                    el = await find(locator)
                except Exception:
                    el = None
                if el is None:
                    failed.append(f"element missing: {check.value!r}")
            else:
                failed.append(f"unknown check {check.kind!r}")
        return VerifyResult(passed=not failed, failed=failed, checked=done)
