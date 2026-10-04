"""API router mount table — GENERATED FILE, DO NOT EDIT.

Regenerate with ``python scripts/gen_router_manifest.py`` after adding, removing or reordering a
router. ``python scripts/gen_router_manifest.py --check`` fails when this file is stale and runs
as part of the test gate (``tests/server/test_router_manifest.py``).

Read by ``routers.get_all_routers()``, which still imports each module
**lazily** and isolates each with try/except — this file is a table of names,
not imports, so it costs nothing at cold start (it binds no router modules).

Order here is mount order and is load-bearing: ``inference`` must precede
``chat``. See the docstring of ``scripts/gen_router_manifest.py``.
"""

from __future__ import annotations

__all__ = ["ROUTER_MODULES"]

ROUTER_MODULES: tuple[str, ...] = (
    "auth",
    "models",
    "inference",
    "feedback",
    "kb",
    "agents",
    "system",
    "souls",
    "config",
    "settings",
    "security",
    "datasets",
    "ratelimit",
    "workflow",
    "experiments",
    "benchmark",
    "user_adapters",
    "vector",
    "registry",
    "chat",
    "session",
    "meta_weights",
    "lora_eval",
    "companion",
    "multimodal",
    "tokenizer",
    "learner",
    "self_train",
    "token_tree",
    "errors",
    "mobile",
    "images",
    "files",
    "voice",
    "infer",
    "vm",
    "memory",
    "docstore",
    "shell",
    "world_render",
    "tokens",
    "profiles",
    "users",
    "tenants",
    "workspaces",
    "search",
    "openwebui",
    "cloud_training",
    "plugins",
    "tools",
    "model_stack",
    "phoneme",
    "collections",
)
