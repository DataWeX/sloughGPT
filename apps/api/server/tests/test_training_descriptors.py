"""Descriptor-projected training routes — the migration wave's contract names.

One section per migrated module (card `20261104_151036`). ``check_contract
--static`` only AST-scans ``routers/`` for descriptor names, and ``--runtime``
only assembles the ``routers`` manifest — ``training/`` mounts through
``infrastructure/startup.py`` instead. This suite therefore carries both
weights for the wave: every migrated module's SPEC must be registered in the
contract registry, its route must actually be served, and the wire must stay
byte-identical to the hand-built route it replaced (raw sites stay raw —
``envelope=False``; self-wrapped handlers keep their ``{status, data}``).
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient
from infrastructure.contract import REGISTRY


def _entry(name: str) -> dict:
    entry = next((e for e in REGISTRY.routes() if e["name"] == name), None)
    assert entry is not None, f"descriptor '{name}' not registered in the contract registry"
    return entry


def _client(router) -> TestClient:
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


# ── training/builds.py (raw wire) ─────────────────────────────────────


def test_builds_descriptor_registered():
    from training.builds import SPEC  # import registers the descriptor

    assert SPEC.name == "training.builds.list"
    entry = _entry("training.builds.list")
    assert entry["path"] == "/training/builds"
    assert entry["method"] == "GET"
    assert entry["auth_scope"] == "public"
    assert entry["idempotent"] is True


def test_builds_serves_the_raw_wire():
    """Hand-built ``{"builds": [...]}`` — no envelope may leak in."""
    from training.builds import router

    resp = _client(router).get("/training/builds")
    assert resp.status_code == 200
    body = resp.json()
    assert "builds" in body
    assert "status" not in body
