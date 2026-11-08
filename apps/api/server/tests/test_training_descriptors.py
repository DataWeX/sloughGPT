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


# ── training/jobs_api.py (path params, query input, DELETE, FileResponse) ──


JOBS_ROUTES = [
    # name, path, method, idempotent, auth_scope
    ("training.jobs.list", "/training/jobs", "GET", True, "authenticated"),
    ("training.jobs.get", "/training/jobs/{job_id}", "GET", True, "public"),
    ("training.jobs.stop", "/training/jobs/{job_id}/stop", "POST", False, "public"),
    ("training.jobs.summary", "/training/jobs/{job_id}/summary", "GET", True, "public"),
    ("training.jobs.delete", "/training/jobs/{job_id}", "DELETE", False, "authenticated"),
    ("training.jobs.purge", "/training/jobs/purge", "POST", False, "authenticated"),
    ("training.jobs.export", "/training/export/{job_id}", "GET", True, "public"),
    ("training.jobs.export_text", "/training/export-text", "POST", False, "public"),
]


def test_jobs_descriptors_registered():
    import training.jobs_api  # noqa: F401 — import registers the eight descriptors

    for name, path, method, idem, scope in JOBS_ROUTES:
        entry = _entry(name)
        assert entry["path"] == path, name
        assert entry["method"] == method, name
        assert entry["idempotent"] is idem, name
        assert entry["auth_scope"] == scope, name
    # a path param is a required member of the contract; purge's status keeps
    # its hand-built description; export-text documents the model's fields
    assert _entry("training.jobs.get")["params"] == {
        "type": "object",
        "properties": {"job_id": {"type": "string", "description": ""}},
        "required": ["job_id"],
    }
    purge_params = _entry("training.jobs.purge")["params"]
    assert purge_params["properties"]["status"] == {
        "type": "string",
        "description": "Only purge jobs in this terminal status",
    }
    assert purge_params["required"] == []
    exported = _entry("training.jobs.export_text")["params"]["properties"]
    assert {"min_quality", "target_count"} <= set(exported)


def test_jobs_serves_the_raw_wire():
    """Hand-built payloads served unwrapped; verbs, 404s and 400s unchanged."""
    from infrastructure.exception_handlers import register_all_handlers
    from training.jobs import training_jobs
    from training.jobs_api import router

    # raise_error raises AppErrors; the app-level handlers are what turn them
    # into the 404/400 JSON responses both hand-built and projected routes serve
    app = FastAPI()
    app.include_router(router)
    register_all_handlers(app)
    client = TestClient(app)

    # list: a bare list — an envelope would wrap it in a dict
    resp = client.get("/training/jobs")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)

    # reads/actions on a missing id → the handlers' own 404s
    assert client.get("/training/jobs/no-such-job").status_code == 404
    assert client.get("/training/jobs/no-such-job/summary").status_code == 404
    assert client.post("/training/jobs/no-such-job/stop").status_code == 404

    # DELETE is the verb (a wrong verb would answer 405)
    assert client.delete("/training/jobs/no-such-job").status_code == 404

    # purge keeps its query input: a non-terminal status → the handler's 400
    assert client.post("/training/jobs/purge?status=running").status_code == 400

    # purge with no status must reach the handler's own default (None) — the
    # projection may never hand it a transport object instead. This endpoint
    # mutates the shared tracker by design, so snapshot and restore it.
    snapshot = dict(training_jobs)
    try:
        resp = client.post("/training/jobs/purge")
        assert resp.status_code == 200
        assert resp.json()["status"] == "purged"
    finally:
        training_jobs.clear()
        training_jobs.update(snapshot)

    # export-text keeps the model's ge/le constraints as 422s (handler never runs)
    assert client.post("/training/export-text", json={"min_quality": 5}).status_code == 422


def test_jobs_export_serves_the_file_response_raw(tmp_path):
    """export returns a FileResponse — envelope=False keeps the bytes raw."""
    from training.jobs import training_jobs
    from training.jobs_api import router

    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(b"weights")
    training_jobs["wave-export-test"] = {
        "id": "wave-export-test",
        "status": "completed",
        "checkpoint": str(checkpoint),
    }
    try:
        resp = _client(router).get("/training/export/wave-export-test")
        assert resp.status_code == 200
        assert resp.content == b"weights"
        assert "model.safetensors" in resp.headers.get("content-disposition", "")
    finally:
        training_jobs.pop("wave-export-test", None)


def test_jobs_openapi_matches_the_hand_built_doc():
    from training.jobs_api import router

    app = FastAPI()
    app.include_router(router)
    doc = app.openapi()

    # list publishes 200 alone; every other operation can 422 (path/query/model)
    assert sorted(doc["paths"]["/training/jobs"]["get"]["responses"]) == ["200"]
    checked = 0
    for path, methods in doc["paths"].items():
        for method, op in methods.items():
            if (path, method) == ("/training/jobs", "get"):
                continue
            assert sorted(op["responses"]) == ["200", "422"], (path, method)
            checked += 1
    assert checked == 7

    # path placeholders publish in: path; purge's status stays a query param
    get_op = doc["paths"]["/training/jobs/{job_id}"]["get"]
    assert [(p["name"], p["in"]) for p in get_op["parameters"]] == [("job_id", "path")]
    purge_op = doc["paths"]["/training/jobs/purge"]["post"]
    assert [(p["name"], p["in"]) for p in purge_op["parameters"]] == [("status", "query")]
    assert purge_op["parameters"][0]["description"] == "Only purge jobs in this terminal status"
    assert purge_op["parameters"][0]["required"] is False

    # operation titles match FastAPI's hand-built derivation byte-for-byte
    # (route.name.replace("_", " ").title()) — carried by RouteSpec.summary
    assert doc["paths"]["/training/jobs"]["get"]["summary"] == "List Training Jobs"
    assert doc["paths"]["/training/jobs/purge"]["post"]["summary"] == "Purge Training Jobs"
    assert doc["paths"]["/training/export-text"]["post"]["summary"] == "Export Feedback Pairs"
    assert doc["paths"]["/training/export/{job_id}"]["get"]["summary"] == "Export Training Job"

    # GET and DELETE share the job_id path; no stray verbs anywhere
    assert sorted(doc["paths"]["/training/jobs/{job_id}"]) == ["delete", "get"]


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
    # jobs composes through its own module tags, exactly as the hand-built
    # routes did: ['training', 'training', 'jobs']
    assert doc["paths"]["/training/jobs"]["get"]["tags"] == ["training", "training", "jobs"]
    assert doc["paths"]["/training/jobs/purge"]["post"]["tags"] == [
        "training",
        "training",
        "jobs",
    ]
    assert doc["paths"]["/training/export-text"]["post"]["tags"] == [
        "training",
        "training",
        "jobs",
    ]
