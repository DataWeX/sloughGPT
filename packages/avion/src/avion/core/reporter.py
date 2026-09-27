"""Reporter — collect task results, render terminal/markdown/JSON."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from avion.core.task import TaskResult, TaskStatus


class Report:
    """Rendered view over a list of results."""

    def __init__(self, title: str, results: list[TaskResult]):
        self.title = title
        self.results = list(results)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.status == TaskStatus.PASSED)

    @property
    def failed(self) -> int:
        return sum(1 for r in self.results if r.status == TaskStatus.FAILED)

    def to_terminal(self) -> str:
        lines = [self.title, "=" * len(self.title)]
        for r in self.results:
            icon = "PASS" if r.status == TaskStatus.PASSED else "FAIL"
            lines.append(f"  [{icon}] {r.task_name} ({r.duration_ms:.0f}ms)")
            if r.error:
                lines.append(f"         {r.error[:120]}")
        lines.append(
            f"  Total: {len(self.results)} | Passed: {self.passed} | Failed: {self.failed}"
        )
        return "\n".join(lines)

    def to_markdown(self) -> str:
        lines = [f"# {self.title}", "", "| Task | Status | Duration |", "| --- | --- | --- |"]
        for r in self.results:
            lines.append(f"| {r.task_name} | {r.status.value} | {r.duration_ms:.0f}ms |")
        return "\n".join(lines)

    def to_json(self) -> str:
        return json.dumps(
            {
                "title": self.title,
                "passed": self.passed,
                "failed": self.failed,
                "results": [r.to_dict() for r in self.results],
            },
            indent=2,
        )


class Reporter:
    """Collects results during a session and renders reports."""

    def __init__(self, title: str = "Arken Report"):
        self.title = title
        self._results: list[TaskResult] = []

    def add(self, result: TaskResult) -> None:
        self._results.append(result)

    @property
    def results(self) -> list[TaskResult]:
        return list(self._results)

    def build(self) -> Report:
        return Report(self.title, self._results)

    def report(self, format: str = "terminal") -> str:
        built = self.build()
        if format == "markdown":
            return built.to_markdown()
        if format == "json":
            return built.to_json()
        return built.to_terminal()

    def save(self, path: str, format: str = "json") -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write(self.report(format=format))

    def summary(self) -> dict[str, Any]:
        built = self.build()
        return {
            "total": len(self._results),
            "passed": built.passed,
            "failed": built.failed,
        }

    def clear(self) -> None:
        self._results.clear()
