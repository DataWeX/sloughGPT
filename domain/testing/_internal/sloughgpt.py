"""
sloughGPT-specific UI Journey Test Configuration.

Provides pre-configured site config and pages for sloughGPT.

Usage:
    from domains.testing.sloughgpt import SLOUGHPGPT_SITE, SloughGPTJourney

    journey = SloughGPTJourney()
    result = journey.run([journey.goto("/training"), journey.check_body("train")])
"""
from __future__ import annotations

from . import Page, SiteConfig, create_site_config

# ── sloughGPT Pages ─────────────────────────────────────────────────────────

SLOUGHPGPT_PAGES = {
    "dashboard": ("", ["chat", "training", "model"]),
    "chat": ("/chat", ["chat", "message"]),
    "training": ("/training", ["train", "dataset"]),
    "datasets": ("/datasets", ["dataset", "import"]),
    "models": ("/models", ["model", "load"]),
    "agents": ("/agents", ["agent", "task"]),
    "souls": ("/souls", ["soul", "personality"]),
    "knowledge": ("/knowledge", ["knowledge", "memory"]),
    "monitoring": ("/monitoring", ["cpu", "memory"]),
    "settings": ("/settings", ["setting", "config"]),
    "planner": ("/planner", ["planner", "board"]),
    "benchmark": ("/benchmark", ["benchmark", "eval"]),
    "tokenizer": ("/tokenizer", ["token", "vocab"]),
    "errors": ("/errors", ["error", "log"]),
    "security": ("/security", ["security", "audit"]),
    "shell": ("/shell", ["shell", "terminal"]),
    "feedback": ("/feedback", ["feedback", "rating"]),
    "files": ("/files", ["file", "upload"]),
    "adapters": ("/adapters", ["adapter", "lora"]),
}

SLOUGHPGPT_TRAINING_PAGES = {
    "queue": ("/training/queue", ["queue", "job"]),
    "runs": ("/training/runs", ["run", "history"]),
    "presets": ("/training/presets", ["preset", "config"]),
    "analytics": ("/training/analytics", ["analytics", "chart"]),
    "compare": ("/training/compare", ["compare", "diff"]),
    "trends": ("/training/trends", ["trend", "chart"]),
    "insights": ("/training/insights", ["insight"]),
    "model-card": ("/training/model-card", ["model", "card"]),
}

SLOUGHPGPT_API_ENDPOINTS = {
    "health": "/health",
    "training_status": "/training/status",
    "datasets": "/datasets",
    "models": "/models",
}

# ── sloughGPT Site Config ──────────────────────────────────────────────────

SLOUGHPGPT_SITE = create_site_config(
    name="sloughGPT",
    base_url="http://localhost:3000",
    api_url="http://localhost:8000",
    pages={**SLOUGHPGPT_PAGES, **SLOUGHPGPT_TRAINING_PAGES},
)


# ── sloughGPT Journey Helper ───────────────────────────────────────────────

class SloughGPTJourney:
    """Pre-configured journey for sloughGPT."""

    def __init__(self, base_url: str = "http://localhost:3000", api_url: str = "http://localhost:8000"):
        from . import Journey

        self.config = SiteConfig(
            name="sloughGPT",
            base_url=base_url,
            api_url=api_url,
            pages=SLOUGHPGPT_SITE.pages,
        )
        self.journey = Journey(self.config)

    def goto(self, path: str):
        return self.journey.goto(path)

    def wait_for(self, text: str, timeout_s: float = 10.0):
        return self.journey.wait_for(text, timeout_s)

    def snapshot(self):
        return self.journey.snapshot()

    def check_body(self, expected: str):
        return self.journey.check_body(expected)

    def click_button(self, text: str):
        return self.journey.click_button(text)

    def check_no_console_errors(self):
        return self.journey.check_no_console_errors()

    def check_no_network_errors(self):
        return self.journey.check_no_network_errors()

    def run(self, steps, name: str = "sloughgpt_journey"):
        return self.journey.run(steps, name)

    def report(self) -> str:
        return self.journey.report()
