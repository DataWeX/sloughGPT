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


# ── training/control.py (raw wire, public, 200-only) ───────────────────

CONTROL_ROUTES = [
    ("training.control.status", "/training/status", "GET", True),
    ("training.control.is_running", "/training/is-running", "GET", True),
    ("training.control.start", "/training/control/start", "POST", False),
    ("training.control.pause", "/training/control/pause", "POST", False),
    ("training.control.resume", "/training/control/resume", "POST", False),
    ("training.control.stop", "/training/control/stop", "POST", False),
    ("training.control.reset", "/training/control/reset", "POST", False),
]


def test_control_descriptors_registered():
    import training.control  # noqa: F401 — import registers the seven descriptors

    for name, path, method, idempotent in CONTROL_ROUTES:
        entry = _entry(name)
        assert entry["path"] == path
        assert entry["method"] == method
        assert entry["auth_scope"] == "public"
        assert entry["idempotent"] is idempotent
        assert entry["params"] == {"type": "object", "properties": {}, "required": []}


def test_control_serves_the_raw_wire():
    """Hand-built controller dicts — no envelope may leak in."""
    from training.control import router

    client = _client(router)
    resp = client.get("/training/status")
    assert resp.status_code == 200
    body = resp.json()
    assert "running_jobs" in body and "total_tracked_jobs" in body
    assert "data" not in body  # an envelope would nest the payload under data
    resp = client.get("/training/is-running")
    assert resp.status_code == 200
    assert "is_running" in resp.json()


def test_control_openapi_declares_200_only():
    """No parameters → no validation can fail → the hand-built 200-only doc."""
    from training.control import router

    app = FastAPI()
    app.include_router(router)
    doc = app.openapi()
    for _name, path, _method, _idem in CONTROL_ROUTES:
        for method, op in doc["paths"][path].items():
            assert sorted(op["responses"]) == ["200"], (path, method)


# ── training/distill.py / visual.py / execution.py (model-style body) ──


def test_distill_descriptor_registered():
    from training.distill import SPEC_DISTILL  # import registers the descriptor

    assert SPEC_DISTILL.name == "training.distill.start"
    entry = _entry("training.distill.start")
    assert entry["path"] == "/training/distill"
    assert entry["method"] == "POST"
    assert entry["auth_scope"] == "authenticated"
    assert entry["idempotent"] is False
    # the model's fields are the documented contract (one declaration, both views)
    props = entry["params"]["properties"]
    assert props["dataset"] == {"type": "string", "description": ""}
    assert "dataset" in entry["params"]["required"]
    assert props["epochs"]["type"] == "integer"


def test_distill_validates_like_the_hand_built_route():
    from training.distill import router

    client = _client(router)
    # Missing required field → 422 from the model validator (the hand-built
    # route 422'd identically); the handler never runs.
    assert client.post("/training/distill", json={}).status_code == 422
    op = client.app.openapi()["paths"]["/training/distill"]["post"]
    assert op["tags"] == ["training-distill"]
    assert "422" in op["responses"]


def test_visual_descriptor_registered():
    from training.visual import SPEC_VISUAL  # import registers the descriptor

    assert SPEC_VISUAL.name == "training.visual.start"
    entry = _entry("training.visual.start")
    assert entry["path"] == "/training/visual-start"
    assert entry["method"] == "POST"
    assert entry["auth_scope"] == "authenticated"
    assert entry["idempotent"] is False
    props = entry["params"]["properties"]
    assert props["dataset"] == {"type": "string", "description": ""}
    assert "dataset" in entry["params"]["required"]
    assert props["batch_size"]["type"] == "integer"


def test_visual_validates_like_the_hand_built_route():
    from training.visual import router

    client = _client(router)
    assert client.post("/training/visual-start", json={}).status_code == 422
    op = client.app.openapi()["paths"]["/training/visual-start"]["post"]
    assert op["tags"] == ["training-visual"]
    assert "422" in op["responses"]


def test_execution_start_descriptor_registered():
    from training.execution import SPEC_START  # import registers the descriptor

    assert SPEC_START.name == "training.start"
    entry = _entry("training.start")
    assert entry["path"] == "/training/start"
    assert entry["method"] == "POST"
    assert entry["auth_scope"] == "authenticated"
    assert entry["idempotent"] is False
    props = entry["params"]["properties"]
    assert "name" in props and "model" in props and "dataset" in props
    assert {"name", "model"} <= set(entry["params"]["required"])


def test_execution_start_validates_like_the_hand_built_route():
    from training.execution import router

    client = _client(router)
    assert client.post("/training/start", json={}).status_code == 422
    op = client.app.openapi()["paths"]["/training/start"]["post"]
    assert op["tags"] == ["training-execution"]
    assert "422" in op["responses"]


def test_full_training_chain_keeps_the_hand_built_tag_composition():
    """Served through training/router.py, tags compose exactly as before."""
    from training.router import router

    app = FastAPI()
    app.include_router(router)
    doc = app.openapi()
    assert doc["paths"]["/training/status"]["get"]["tags"] == ["training", "training", "control"]
    assert doc["paths"]["/training/start"]["post"]["tags"] == [
        "training",
        "training-execution",
    ]
    assert doc["paths"]["/training/distill"]["post"]["tags"] == [
        "training",
        "training-execution",
        "training-distill",
    ]
    assert doc["paths"]["/training/visual-start"]["post"]["tags"] == [
        "training",
        "training-execution",
        "training-visual",
    ]
