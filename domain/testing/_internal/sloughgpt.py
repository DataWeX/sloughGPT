"""
sloughGPT-specific UI Journey Test Configuration.

Provides pre-configured site config and pages for sloughGPT.

Usage:
    from domain.testing._internal.sloughgpt import SLOUGHPGPT_SITE, SloughGPTJourney

    journey = SloughGPTJourney()
    result = journey.run([journey.goto("/training"), journey.check_body("train")])
"""
from __future__ import annotations

from . import Page, SiteConfig, create_site_config

# ── sloughGPT Pages ─────────────────────────────────────────────────────────

SLOUGHPGPT_PAGES = {
    # Core
    "dashboard": ("", ["chat", "training", "model"]),
    "chat": ("/chat", ["chat", "message"]),
    "shell": ("/shell", ["shell", "terminal"]),
    "settings": ("/settings", ["setting", "config"]),
    "settings/workspace": ("/settings/workspace", ["workspace", "setting"]),
    "profile": ("/profile", ["profile", "user"]),
    "auth": ("/auth", ["auth", "login"]),
    "session": ("/session", ["session"]),
    "shortcuts": ("/shortcuts", ["shortcut", "key"]),
    "notifications": ("/notifications", ["notification", "alert"]),
    "developer": ("/developer", ["developer", "api"]),
    # Training
    "training": ("/training", ["train", "dataset"]),
    "training/runs": ("/training/runs", ["run", "history"]),
    "training/presets": ("/training/presets", ["preset", "config"]),
    "training/analytics": ("/training/analytics", ["analytics", "chart"]),
    "training/compare": ("/training/compare", ["compare", "diff"]),
    "training/trends": ("/training/trends", ["trend", "chart"]),
    "training/insights": ("/training/insights", ["insight"]),
    "training/model-card": ("/training/model-card", ["model", "card"]),
    "training/grid-search": ("/training/grid-search", ["grid", "search"]),
    "training/job/[id]": ("/training/job/1", ["job", "detail"]),
    # Data
    "datasets": ("/datasets", ["dataset", "import"]),
    "dataset/[id]": ("/dataset/1", ["dataset", "detail"]),
    "files": ("/files", ["file", "upload"]),
    "docstore": ("/docstore", ["document", "store"]),
    "collections": ("/collections", ["collection"]),
    "knowledge": ("/knowledge", ["knowledge", "memory"]),
    "knowledge/[id]": ("/knowledge/1", ["knowledge", "detail"]),
    "kb": ("/kb", ["knowledge", "base"]),
    "shared-data": ("/shared-data", ["shared", "data"]),
    # Models
    "models": ("/models", ["model", "load"]),
    "model/[id]": ("/model/1", ["model", "detail"]),
    "adapters": ("/adapters", ["adapter", "lora"]),
    "lora-eval": ("/lora-eval", ["lora", "eval"]),
    "meta-weights": ("/meta-weights", ["meta", "weight"]),
    "export": ("/export", ["export"]),
    "registry": ("/registry", ["registry"]),
    "plugins-cloud": ("/plugins-cloud", ["plugin", "cloud"]),
    # AI / Consciousness
    "agents": ("/agents", ["agent", "task"]),
    "agents/[id]": ("/agents/1", ["agent", "detail"]),
    "souls": ("/souls", ["soul", "personality"]),
    "personality": ("/personality", ["personality"]),
    "companion": ("/companion", ["companion"]),
    "consciousness/dashboard": ("/consciousness/dashboard", ["dashboard"]),
    "consciousness/debug": ("/consciousness/debug", ["debug"]),
    "consciousness/testing": ("/consciousness/testing", ["test"]),
    "consciousness/test-runner": ("/consciousness/test-runner", ["test", "runner"]),
    "consciousness/analytics": ("/consciousness/analytics", ["analytics"]),
    "consciousness/insights": ("/consciousness/insights", ["insight"]),
    "consciousness/monitor": ("/consciousness/monitor", ["monitor"]),
    "consciousness/playground": ("/consciousness/playground", ["playground"]),
    "consciousness/benchmark": ("/consciousness/benchmark", ["benchmark"]),
    "consciousness/versions": ("/consciousness/versions", ["version"]),
    "consciousness/history": ("/consciousness/history", ["history"]),
    "consciousness/health": ("/consciousness/health", ["health"]),
    "consciousness/alerts": ("/consciousness/alerts", ["alert"]),
    "consciousness/statistics": ("/consciousness/statistics", ["statistic"]),
    "consciousness/compare": ("/consciousness/compare", ["compare"]),
    "consciousness/training": ("/consciousness/training", ["training"]),
    "consciousness/master": ("/consciousness/master", ["master"]),
    "consciousness/settings": ("/consciousness/settings", ["setting"]),
    "consciousness/all-settings": ("/consciousness/all-settings", ["setting", "all"]),
    "consciousness/export": ("/consciousness/export", ["export"]),
    "consciousness/docs": ("/consciousness/docs", ["doc"]),
    "consciousness/help": ("/consciousness/help", ["help"]),
    "consciousness/quickstart": ("/consciousness/quickstart", ["quickstart"]),
    "consciousness/api-explorer": ("/consciousness/api-explorer", ["api", "explorer"]),
    "consciousness/personality": ("/consciousness/personality", ["personality"]),
    # Inference
    "infer": ("/infer", ["infer", "predict"]),
    "evaluate": ("/evaluate", ["evaluate", "eval"]),
    "benchmark": ("/benchmark", ["benchmark", "eval"]),
    "explain": ("/explain", ["explain"]),
    "compare": ("/compare", ["compare", "diff"]),
    "rewrite": ("/rewrite", ["rewrite"]),
    "translate": ("/translate", ["translate"]),
    "writing": ("/writing", ["writing"]),
    "magazine": ("/magazine", ["magazine"]),
    "brainstorm": ("/brainstorm", ["brainstorm"]),
    "auto-train": ("/auto-train", ["auto", "train"]),
    "self-train": ("/self-train", ["self", "train"]),
    # Monitoring / Ops
    "monitoring": ("/monitoring", ["cpu", "memory"]),
    "errors": ("/errors", ["error", "log"]),
    "security": ("/security", ["security", "audit"]),
    "audit-trail": ("/audit-trail", ["audit", "trail"]),
    "usage": ("/usage", ["usage"]),
    "rate-limit": ("/rate-limit", ["rate", "limit"]),
    "feedback": ("/feedback", ["feedback", "rating"]),
    "permissions": ("/permissions", ["permission"]),
    "users": ("/users", ["user"]),
    "members": ("/members", ["member"]),
    "admin": ("/admin", ["admin"]),
    "workspaces": ("/workspaces", ["workspace"]),
    "workspace-dashboard": ("/workspace-dashboard", ["workspace", "dashboard"]),
    "workspace-search": ("/workspace-search", ["workspace", "search"]),
    # Tools / Features
    "planner": ("/planner", ["planner", "board"]),
    "kanban": ("/kanban", ["kanban", "board"]),
    "token-tree": ("/token-tree", ["token", "tree"]),
    "tokenizer": ("/tokenizer", ["token", "vocab"]),
    "tools": ("/tools", ["tool"]),
    "workflow": ("/workflow", ["workflow"]),
    "vector": ("/vector", ["vector", "embed"]),
    "multimodal": ("/multimodal", ["multimodal", "image"]),
    "images": ("/images", ["image"]),
    "voice": ("/voice", ["voice", "audio"]),
    "phoneme": ("/phoneme", ["phoneme"]),
    "memory": ("/memory", ["memory"]),
    "learn": ("/learn", ["learn"]),
    "world": ("/world", ["world"]),
    "vm": ("/vm", ["vm", "machine"]),
    "wellness": ("/wellness", ["wellness"]),
    "decide": ("/decide", ["decide"]),
    "experiments": ("/experiments", ["experiment"]),
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
    pages=SLOUGHPGPT_PAGES,
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
