"""
DevTools Follower — monitors browser DevTools output during test execution.

Wraps the ComputerUseAgent to provide real-time logging, performance tracking,
and automatic error detection during user journey tests.

Usage:
    from domains.agents.devtools_follower import DevToolsFollower

    async with DevToolsFollower(base_url="http://localhost:3000") as follower:
        await follower.navigate("/training")
        await follower.click("Train")
        report = follower.report()

Or as a pytest fixture:
    @pytest.fixture
    def follower():
        return DevToolsFollower.fixture(base_url="http://localhost:3000")
"""

from __future__ import annotations

import json
import logging
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger("slo.devtools_follower")


@dataclass
class StepRecord:
    step: int
    action: str
    url: str
    timestamp: float
    duration_s: float
    console_count: int
    request_count: int
    error_count: int
    console_errors: list[str] = field(default_factory=list)
    failed_requests: list[str] = field(default_factory=list)
    success: bool = True
    detail: str = ""


@dataclass
class NetworkAnalytics:
    total_requests: int = 0
    failed_requests: int = 0
    avg_latency_ms: float = 0
    max_latency_ms: float = 0
    slowest_url: str = ""
    by_status: dict[str, int] = field(default_factory=dict)
    by_type: dict[str, int] = field(default_factory=dict)


@dataclass
class ConsoleAnalytics:
    total_messages: int = 0
    errors: int = 0
    warnings: int = 0
    info: int = 0
    error_categories: dict[str, int] = field(default_factory=dict)
    most_common_error: str = ""


@dataclass
class StepPerformance:
    action: str
    duration_s: float
    url: str
    errors: int
    requests: int
    console_errors: int


@dataclass
class FollowerReport:
    steps: list[StepRecord]
    total_steps: int
    total_console: int
    total_requests: int
    total_errors: int
    total_console_errors: int
    total_failed_requests: int
    duration_s: float
    all_console_errors: list[dict[str, Any]]
    all_failed_requests: list[dict[str, Any]]
    summary: str
    network: NetworkAnalytics | None = None
    console: ConsoleAnalytics | None = None
    step_perf: list[StepPerformance] | None = None

    def to_json(self) -> str:
        data = {
            "steps": [
                {
                    "step": s.step,
                    "action": s.action,
                    "url": s.url,
                    "duration_s": round(s.duration_s, 3),
                    "console_count": s.console_count,
                    "request_count": s.request_count,
                    "error_count": s.error_count,
                    "console_errors": s.console_errors,
                    "failed_requests": s.failed_requests,
                    "success": s.success,
                    "detail": s.detail,
                }
                for s in self.steps
            ],
            "totals": {
                "steps": self.total_steps,
                "console": self.total_console,
                "requests": self.total_requests,
                "errors": self.total_errors,
                "console_errors": self.total_console_errors,
                "failed_requests": self.total_failed_requests,
                "duration_s": round(self.duration_s, 3),
            },
            "summary": self.summary,
        }
        if self.network:
            data["network"] = {
                "total_requests": self.network.total_requests,
                "failed_requests": self.network.failed_requests,
                "avg_latency_ms": round(self.network.avg_latency_ms, 1),
                "max_latency_ms": round(self.network.max_latency_ms, 1),
                "slowest_url": self.network.slowest_url,
                "by_status": self.network.by_status,
                "by_type": self.network.by_type,
            }
        if self.console:
            data["console"] = {
                "total_messages": self.console.total_messages,
                "errors": self.console.errors,
                "warnings": self.console.warnings,
                "info": self.console.info,
                "error_categories": self.console.error_categories,
                "most_common_error": self.console.most_common_error,
            }
        if self.step_perf:
            data["step_performance"] = [
                {
                    "action": sp.action,
                    "duration_s": round(sp.duration_s, 3),
                    "url": sp.url,
                    "errors": sp.errors,
                    "requests": sp.requests,
                    "console_errors": sp.console_errors,
                }
                for sp in self.step_perf
            ]
        return json.dumps(data, indent=2)


class DevToolsFollower:
    """Follows test execution with live DevTools monitoring.

    Collects console messages, network requests, and errors at each step.
    Produces a structured report with analytics at the end.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:3000",
        headless: bool = True,
        output_path: str | None = None,
    ):
        from domains.agents.computer_use import ComputerUseAgent

        self.agent = ComputerUseAgent(base_url=base_url, headless=headless)
        self.output_path = output_path
        self._steps: list[StepRecord] = []
        self._step_counter = 0
        self._start_time: float = 0
        self._console_snapshot_idx = 0
        self._network_snapshot_idx = 0

    async def start(self) -> None:
        await self.agent.start()
        self._start_time = time.time()

    async def stop(self) -> None:
        await self.agent.stop()
        if self.output_path:
            report = self.report()
            Path(self.output_path).write_text(report.to_json())
            logger.info("Report saved to %s", self.output_path)

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, *args):
        await self.stop()

    def _take_snapshot(self) -> tuple[int, int, int]:
        console = len(self.agent._console_messages)
        network = len(self.agent._network_requests)
        errors = len(self.agent._errors)
        return console, network, errors

    def _get_new_items(self, snapshot: tuple[int, int, int]) -> dict:
        console_start = self._console_snapshot_idx
        network_start = self._network_snapshot_idx
        self._console_snapshot_idx = snapshot[0]
        self._network_snapshot_idx = snapshot[1]

        new_console = self.agent._console_messages[console_start:snapshot[0]]
        new_network = self.agent._network_requests[network_start:snapshot[1]]

        console_errors = [
            f"{m['type']}: {m['text'][:100]}"
            for m in new_console
            if m.get("type") in ("error", "warning")
        ]
        failed_requests = [
            f"{r.get('method', 'GET')} {r['url'][:80]} -> {r.get('status', '?')}"
            for r in new_network
            if r.get("status", 200) >= 400
        ]

        return {
            "console_errors": console_errors,
            "failed_requests": failed_requests,
        }

    async def step(self, action: str, detail: str = "") -> StepRecord:
        self._step_counter += 1
        snapshot_before = self._take_snapshot()
        t0 = time.time()

        record = StepRecord(
            step=self._step_counter,
            action=action,
            url=self.agent._page.url if self.agent._page else "",
            timestamp=t0,
            duration_s=0,
            console_count=snapshot_before[0],
            request_count=snapshot_before[1],
            error_count=snapshot_before[2],
        )

        try:
            await self.agent.collect_performance()
        except Exception:
            pass

        snapshot_after = self._take_snapshot()
        new_items = self._get_new_items(snapshot_after)

        record.duration_s = time.time() - t0
        record.console_count = snapshot_after[0]
        record.request_count = snapshot_after[1]
        record.error_count = snapshot_after[2]
        record.console_errors = new_items["console_errors"]
        record.failed_requests = new_items["failed_requests"]
        record.success = record.error_count == 0 and not record.console_errors

        self._steps.append(record)
        logger.info(
            "[Step %d] %s | console=%d network=%d errors=%d%s",
            record.step,
            action,
            record.console_count,
            record.request_count,
            record.error_count,
            f" | {record.detail}" if record.detail else "",
        )

        return record

    async def navigate(self, path: str) -> StepRecord:
        await self.agent.navigate(path)
        return await self.step(f"navigate to {path}")

    async def click_button(self, name: str) -> StepRecord:
        result = await self.agent.click_button(name)
        return await self.step(
            f"click button '{name}'",
            detail=f"found={result.found}" + (f" error={result.error}" if result.error else ""),
        )

    async def click_link(self, name: str) -> StepRecord:
        result = await self.agent.click_link(name)
        return await self.step(
            f"click link '{name}'",
            detail=f"found={result.found}",
        )

    async def fill_input(self, placeholder: str, value: str) -> StepRecord:
        result = await self.agent.fill_input(placeholder, value)
        return await self.step(
            f"fill '{placeholder}' = '{value}'",
            detail=f"success={result.success}",
        )

    async def wait_for_text(self, text: str, timeout: int = 10000) -> StepRecord:
        found = await self.agent.wait_for_text(text, timeout)
        return await self.step(
            f"wait for '{text}'",
            detail=f"found={found}",
        )

    async def take_screenshot(self, path: str) -> StepRecord:
        await self.agent.take_screenshot(path)
        return await self.step(f"screenshot -> {path}")

    async def get_body_text(self) -> str:
        return await self.agent.get_body_text()

    async def check_element_exists(self, role: str, name: str = "") -> bool:
        return await self.agent.element_exists(role, name)

    def _compute_network_analytics(self) -> NetworkAnalytics:
        requests = self.agent._network_requests
        total = len(requests)
        failed = sum(1 for r in requests if r.get("status", 200) >= 400)

        latencies = [r.get("latency_ms", 0) for r in requests if r.get("latency_ms")]
        avg_lat = sum(latencies) / len(latencies) if latencies else 0
        max_lat = max(latencies) if latencies else 0
        slowest = ""
        for r in requests:
            if r.get("latency_ms", 0) == max_lat and max_lat > 0:
                slowest = r.get("url", "")[:120]
                break

        by_status: dict[str, int] = Counter()
        by_type: dict[str, int] = Counter()
        for r in requests:
            status = str(r.get("status", "?"))
            by_status[status] += 1
            rtype = r.get("resourceType", "other")
            by_type[rtype] += 1

        return NetworkAnalytics(
            total_requests=total,
            failed_requests=failed,
            avg_latency_ms=avg_lat,
            max_latency_ms=max_lat,
            slowest_url=slowest,
            by_status=dict(by_status),
            by_type=dict(by_type),
        )

    def _compute_console_analytics(self) -> ConsoleAnalytics:
        messages = self.agent._console_messages
        errors = [m for m in messages if m.get("type") == "error"]
        warnings = [m for m in messages if m.get("type") == "warning"]
        infos = [m for m in messages if m.get("type") == "info"]

        categories: dict[str, int] = Counter()
        for m in errors:
            text = m.get("text", "")
            if "TypeError" in text:
                categories["TypeError"] += 1
            elif "ReferenceError" in text:
                categories["ReferenceError"] += 1
            elif "SyntaxError" in text:
                categories["SyntaxError"] += 1
            elif "network" in text.lower() or "fetch" in text.lower():
                categories["NetworkError"] += 1
            else:
                categories["Other"] += 1

        most_common = ""
        if categories:
            most_common = categories.most_common(1)[0][0]

        return ConsoleAnalytics(
            total_messages=len(messages),
            errors=len(errors),
            warnings=len(warnings),
            info=len(infos),
            error_categories=dict(categories),
            most_common_error=most_common,
        )

    def _compute_step_perf(self) -> list[StepPerformance]:
        return [
            StepPerformance(
                action=s.action,
                duration_s=s.duration_s,
                url=s.url,
                errors=s.error_count,
                requests=s.request_count,
                console_errors=len(s.console_errors),
            )
            for s in self._steps
        ]

    def report(self) -> FollowerReport:
        total_console = len(self.agent._console_messages)
        total_requests = len(self.agent._network_requests)
        total_errors = len(self.agent._errors)

        all_console_errors = [
            {"type": m["type"], "text": m["text"][:200], "timestamp": m["timestamp"]}
            for m in self.agent._console_messages
            if m.get("type") in ("error",)
        ]
        all_failed_requests = [
            {"url": r["url"][:200], "method": r.get("method", "GET"), "status": r.get("status")}
            for r in self.agent._network_requests
            if r.get("status", 200) >= 400
        ]

        total_duration = time.time() - self._start_time if self._start_time else 0

        network = self._compute_network_analytics()
        console = self._compute_console_analytics()
        step_perf = self._compute_step_perf()

        parts = [
            f"Steps: {len(self._steps)}",
            f"Console: {total_console} ({console.errors} err, {console.warnings} warn)",
            f"Network: {total_requests} ({network.failed_requests} failed, {network.avg_latency_ms:.0f}ms avg)",
            f"Errors: {total_errors}",
            f"Duration: {total_duration:.1f}s",
        ]
        if console.most_common_error:
            parts.append(f"Top error: {console.most_common_error}")

        return FollowerReport(
            steps=list(self._steps),
            total_steps=len(self._steps),
            total_console=total_console,
            total_requests=total_requests,
            total_errors=total_errors,
            total_console_errors=len(all_console_errors),
            total_failed_requests=len(all_failed_requests),
            duration_s=total_duration,
            all_console_errors=all_console_errors,
            all_failed_requests=all_failed_requests,
            summary=" | ".join(parts),
            network=network,
            console=console,
            step_perf=step_perf,
        )

    def clear(self) -> None:
        self._steps.clear()
        self._step_counter = 0
        self._console_snapshot_idx = 0
        self._network_snapshot_idx = 0
        self.agent.clear_logs()
