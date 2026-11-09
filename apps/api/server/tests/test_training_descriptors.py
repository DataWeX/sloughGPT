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
    # batch 3 composes the same three-level tags as the hand-built routes
    assert doc["paths"]["/training/feed"]["get"]["tags"] == ["training", "training-feed"]
    assert doc["paths"]["/training/feeds"]["get"]["tags"] == ["training", "training-feed"]
    assert doc["paths"]["/training/turbo/status"]["get"]["tags"] == ["training", "training-turbo"]
    assert doc["paths"]["/training/turbo-start"]["post"]["tags"] == ["training", "training-turbo"]
    assert doc["paths"]["/training/from-sessions-start"]["post"]["tags"] == [
        "training",
        "training-turbo",
    ]
    assert doc["paths"]["/train/resolve"]["post"]["tags"] == [
        "training",
        "training-execution",
        "training-legacy",
    ]
    assert doc["paths"]["/training/lora-finetune"]["post"]["tags"] == [
        "training",
        "training-execution",
        "training-lora",
    ]
    assert doc["paths"]["/training/load-adapter"]["post"]["tags"] == [
        "training",
        "training-execution",
        "training-lora",
    ]
    assert doc["paths"]["/training/unload-adapter"]["post"]["tags"] == [
        "training",
        "training-execution",
        "training-lora",
    ]
    assert doc["paths"]["/training/from-feedback"]["post"]["tags"] == [
        "training",
        "training-execution",
        "training-feedback",
    ]


# ── batch 3: turbo_endpoints / feeds / legacy / lora / from_feedback ────
# 10 sites projected; ``/train`` retained as the card's first listed
# exception (the projection reserves the param name ``request`` for
# raw-Request injection — see training/legacy.py's module docstring).


BATCH3_ROUTES = [
    # name, path, method, idempotent, auth_scope
    ("training.turbo.status", "/training/turbo/status", "GET", True, "public"),
    ("training.turbo.from_sessions", "/training/from-sessions-start", "POST", False, "public"),
    ("training.turbo.start", "/training/turbo-start", "POST", False, "public"),
    ("training.feeds.read", "/training/feed", "GET", True, "authenticated"),
    ("training.feeds.list", "/training/feeds", "GET", True, "authenticated"),
    ("training.legacy.resolve", "/train/resolve", "POST", False, "public"),
    ("training.lora.finetune", "/training/lora-finetune", "POST", False, "authenticated"),
    ("training.lora.load_adapter", "/training/load-adapter", "POST", False, "public"),
    ("training.lora.unload_adapter", "/training/unload-adapter", "POST", False, "public"),
    (
        "training.feedback.from_feedback",
        "/training/from-feedback",
        "POST",
        False,
        "authenticated",
    ),
]


def test_batch3_descriptors_registered():
    import training.feeds  # noqa: F401 — import registers the descriptors
    import training.from_feedback  # noqa: F401
    import training.legacy  # noqa: F401
    import training.lora  # noqa: F401
    import training.turbo_endpoints  # noqa: F401

    for name, path, method, idem, scope in BATCH3_ROUTES:
        entry = _entry(name)
        assert entry["path"] == path, name
        assert entry["method"] == method, name
        assert entry["idempotent"] is idem, name
        assert entry["auth_scope"] == scope, name
    # the model's fields document the contract (one declaration, both views)
    feed_props = _entry("training.feeds.read")["params"]["properties"]
    assert feed_props["source"] == {"type": "string", "description": ""}
    assert feed_props["offset"] == {"type": "integer", "description": ""}
    assert _entry("training.feeds.read")["params"]["required"] == []
    assert _entry("training.legacy.resolve")["params"] == {
        "type": "object",
        "properties": {"body": {"type": "string", "description": ""}},
        "required": ["body"],
    }
    # listed exception: /train carries no descriptor at all
    assert not any(e["path"] == "/train" for e in REGISTRY.routes())


def _batch3_client() -> TestClient:
    from infrastructure.exception_handlers import register_all_handlers
    from training.feeds import router as feeds_router
    from training.from_feedback import router as feedback_router
    from training.legacy import router as legacy_router
    from training.lora import router as lora_router
    from training.turbo_endpoints import router as turbo_router

    app = FastAPI()
    for r in (turbo_router, feeds_router, legacy_router, lora_router, feedback_router):
        app.include_router(r)
    # raise_error raises AppErrors; the app-level handlers are what turn them
    # into the 400/422 JSON both hand-built and projected routes serve
    register_all_handlers(app)
    return TestClient(app)


def test_batch3_serves_the_wire():
    """Envelope split + validation parity across the batch's five modules."""
    client = _batch3_client()

    # feeds read: group A — the projection's envelope wraps the payload
    # exactly as the hand-built success_response(data=...) did
    resp = client.get("/training/feed")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "success"
    assert set(body) == {"status", "data"}
    assert set(body["data"]) == {"name", "source", "offset", "next_offset", "total", "records"}
    assert body["data"]["name"] == body["data"]["source"] == "api-conversations"

    # feeds list: same wrap, data = {feeds, total}
    resp = client.get("/training/feeds")
    assert resp.status_code == 200
    body = resp.json()
    assert set(body) == {"status", "data"}
    assert set(body["data"]) == {"feeds", "total"}
    assert body["data"]["total"] == len(body["data"]["feeds"])

    # turbo status: the service's bare dict stays bare (envelope=False) —
    # no {status, data} wrap may leak in; the service boots at "idle"
    resp = client.get("/training/turbo/status")
    assert resp.status_code == 200
    body = resp.json()
    assert "data" not in body
    assert body["status"] == "idle"

    # transport validation: a non-int query member 422s before the handler
    assert client.get("/training/feed?offset=abc").status_code == 422

    # the handler's own guards keep their 400s (unknown source is checked
    # before offset/limit, so the message proves the handler ran)
    resp = client.get("/training/feed?source=nope")
    assert resp.status_code == 400
    assert "Unknown feed source" in resp.text

    # listed exception /train: a JSON-body POST has always 422'd here — the
    # hand-built route's required `request` query member cannot be supplied
    assert client.post("/train", json={"epochs": 1}).status_code == 422

    # /train/resolve keeps `body` as a required QUERY param (hand-built wire).
    # The documented divergence — a supplied ?body=x reaching an untyped
    # handler — is deliberately NOT pinned; only the missing-param 422 is
    # the shared contract.
    assert client.post("/train/resolve").status_code == 422

    # unload-adapter pre-flight: empty server state → the handler's own 400
    resp = client.post("/training/unload-adapter")
    assert resp.status_code == 400
    assert "No model loaded" in resp.text

    # lora-finetune pre-flight: unknown model file → 400 before any job exists
    resp = client.post(
        "/training/lora-finetune", json={"model_path": "/bogus/nope.slnc", "dataset": "x"}
    )
    assert resp.status_code == 400
    assert "Model not found" in resp.text

    # model ge/le constraints stay validator 422s (handler never runs)
    assert client.post("/training/from-sessions-start", json={"epochs": 0}).status_code == 422
    assert client.post("/training/from-feedback", json={"epochs": 0}).status_code == 422


def test_batch3_openapi_matches_the_hand_built_doc():
    """Byte-identical wire fields; the 4 accepted delta classes noted below."""
    client = _batch3_client()
    doc = client.app.openapi()

    # operation titles byte-identical (RouteSpec.summary, jobs precedent)
    summaries = {
        "/training/turbo/status": "Get Turbo Status",
        "/training/from-sessions-start": "Start From Sessions Unified",
        "/training/turbo-start": "Start Turbo Training Unified",
        "/training/feed": "Training Feed",
        "/training/feeds": "Training Feeds",
        "/train/resolve": "Train Resolve",
        "/training/lora-finetune": "Start Lora Finetune",
        "/training/load-adapter": "Load Adapter",
        "/training/unload-adapter": "Unload Adapter",
        "/training/from-feedback": "Train From Feedback",
    }
    for path, summary in summaries.items():
        (method,) = doc["paths"][path]  # one verb per path in this batch
        assert doc["paths"][path][method]["summary"] == summary, path

    # declared query carriers keep name + position (the real wire defect the
    # comparison caught: resolve defaults to body-input without input="query")
    feed_params = [
        (p["name"], p["in"]) for p in doc["paths"]["/training/feed"]["get"]["parameters"]
    ]
    assert feed_params == [("source", "query"), ("offset", "query"), ("limit", "query")]
    resolve_params = [
        (p["name"], p["in"]) for p in doc["paths"]["/train/resolve"]["post"]["parameters"]
    ]
    assert resolve_params == [("body", "query")]

    # no-param ops publish 200 alone — nothing can fail validation — exactly
    # the hand-built doc (turbo status, feeds list, unload adapter)
    for path in ("/training/turbo/status", "/training/feeds", "/training/unload-adapter"):
        (method,) = doc["paths"][path]
        assert sorted(doc["paths"][path][method]["responses"]) == ["200"], path
    # every other op can 422 → 200 + 422
    for path in (
        "/training/from-sessions-start",
        "/training/turbo-start",
        "/training/feed",
        "/train/resolve",
        "/training/lora-finetune",
        "/training/load-adapter",
        "/training/from-feedback",
    ):
        (method,) = doc["paths"][path]
        assert sorted(doc["paths"][path][method]["responses"]) == ["200", "422"], path

    # security rides auth_scope — HTTPBearer exactly where the hand-built
    # routes declared it (deps=require_auth_if_enabled), nowhere else
    for path in (
        "/training/feed",
        "/training/feeds",
        "/training/lora-finetune",
        "/training/from-feedback",
    ):
        (method,) = doc["paths"][path]
        assert doc["paths"][path][method].get("security") == [{"HTTPBearer": []}], path
    for path in (
        "/training/turbo/status",
        "/training/from-sessions-start",
        "/training/turbo-start",
        "/train/resolve",
        "/training/load-adapter",
        "/training/unload-adapter",
    ):
        (method,) = doc["paths"][path]
        assert "security" not in doc["paths"][path][method], path

    # Accepted doc-only deltas (4 classes, outside the brief's verb/URL/wire
    # bar — all verified byte-identical on wire fields by the batch's
    # before/after OpenAPI comparison): (1) param schema cosmetics — hand-built
    # `default`+`title` vs projection `type`+`description` on feed ×3 + resolve;
    # (2) 200-response-model schemas on feeds ×2 + turbo-start (envelope routes
    # keep the derived `-> dict`); (3) resolve's 200 schema `{}` from the popped
    # annotation; (4) 422 description/content — the class accepted since part 1.
