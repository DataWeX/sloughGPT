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


# ── sub-batch A: webhook_endpoints (9 sites) ────────────────────────────
# All nine handlers serve bare dicts and the module declares no Depends
# chain anywhere → ``envelope=False`` + ``auth_scope="public"`` on every
# route (brief edge 3: auth parity means reproducing the empty chain);
# register's flat scalars ride ``input="query"`` (the hand-built POST
# published them as query members, not a body); the DELETE verb is stated
# explicitly (derive_method can only emit GET/POST).

WEBHOOK_ROUTES = [
    # name, path, method, idempotent, auth_scope
    ("training.webhooks.list", "/training/webhooks", "GET", True, "public"),
    ("training.webhooks.register", "/training/webhooks", "POST", False, "public"),
    ("training.webhooks.stats", "/training/webhooks/stats", "GET", True, "public"),
    ("training.webhooks.retry_queue", "/training/webhooks/retry-queue", "GET", True, "public"),
    ("training.webhooks.dead_letters", "/training/webhooks/dead-letters", "GET", True, "public"),
    ("training.webhooks.test", "/training/webhooks/test", "POST", False, "public"),
    (
        "training.webhooks.unregister",
        "/training/webhooks/{webhook_id}",
        "DELETE",
        False,
        "public",
    ),
    ("training.webhooks.get", "/training/webhooks/{webhook_id}", "GET", True, "public"),
    (
        "training.webhooks.deliveries",
        "/training/webhooks/{webhook_id}/deliveries",
        "GET",
        True,
        "public",
    ),
]


def test_webhook_descriptors_registered():
    import training.webhook_endpoints  # noqa: F401 — import registers the descriptors

    for name, path, method, idem, scope in WEBHOOK_ROUTES:
        entry = _entry(name)
        assert entry["path"] == path, name
        assert entry["method"] == method, name
        assert entry["idempotent"] is idem, name
        assert entry["auth_scope"] == scope, name

    # the model's fields document the contract (one declaration, both views)
    reg_params = _entry("training.webhooks.register")["params"]
    assert list(reg_params["properties"]) == ["url", "events", "description", "secret"]
    assert reg_params["required"] == ["url", "events"]
    assert _entry("training.webhooks.dead_letters")["params"]["required"] == []
    assert _entry("training.webhooks.deliveries")["params"]["required"] == ["webhook_id"]


def test_webhook_serves_the_wire(tmp_path):
    """Raw dicts everywhere; query-POST register; 400/404/422 parity."""
    from unittest.mock import patch

    from infrastructure.exception_handlers import register_all_handlers
    from training.webhook_endpoints import router as webhook_router
    from training.webhooks import WebhookStore

    app = FastAPI()
    app.include_router(webhook_router)
    register_all_handlers(app)
    store = WebhookStore(str(tmp_path / "webhooks.db"))

    with patch("training.webhook_endpoints.get_webhook_store", return_value=store):
        client = TestClient(app)

        # list: the hand-built bare dict — no {status, data} envelope may leak
        resp = client.get("/training/webhooks")
        assert resp.status_code == 200
        body = resp.json()
        assert "data" not in body
        assert set(body) == {"webhooks", "available_events"}
        assert body["webhooks"] == []
        assert "training.completed" in body["available_events"]

        # register: flat scalars are QUERY members on POST (hand-built wire)
        resp = client.post(
            "/training/webhooks",
            params={"url": "https://example.com/hook-x", "events": '["training.completed"]'},
        )
        assert resp.status_code == 200, resp.text
        reg = resp.json()
        assert set(reg) == {"id", "url", "events", "secret", "message"}
        assert reg["message"] == "Webhook registered successfully"
        assert "training.completed" in reg["events"]

        # the registered hook round-trips raw: get by id, delivery log, then
        # unregister → gone (the projection must not wrap any of these)
        resp = client.get(f"/training/webhooks/{reg['id']}")
        assert resp.status_code == 200
        assert set(resp.json()) == {
            "id",
            "url",
            "events",
            "description",
            "is_active",
            "created_at",
        }
        resp = client.get(f"/training/webhooks/{reg['id']}/deliveries")
        assert resp.status_code == 200
        assert set(resp.json()) == {"deliveries"}
        assert resp.json()["deliveries"] == []

        # static sub-routes are never shadowed by /{webhook_id} — registration
        # order is load-bearing and the projection kept it
        for path in ("/training/webhooks/stats", "/training/webhooks/retry-queue"):
            resp = client.get(path)
            assert resp.status_code == 200, path
            assert "data" not in resp.json(), path
        resp = client.get("/training/webhooks/retry-queue")
        assert set(resp.json()) == {"retries"}
        resp = client.get("/training/webhooks/dead-letters")
        assert resp.status_code == 200
        assert set(resp.json()) == {"dead_letters"}

        # transport validation: a mistyped member 422s before the handler;
        # an ABSENT optional keeps the handler default (limit=50) — the
        # projection must pass nothing, never None
        assert client.get("/training/webhooks/dead-letters?limit=abc").status_code == 422
        assert client.get(f"/training/webhooks/{reg['id']}/deliveries?limit=abc").status_code == 422

        # required query members missing → 422 before the handler runs
        assert client.post("/training/webhooks", params={"events": "[]"}).status_code == 422
        assert client.post("/training/webhooks", params={"url": "https://x"}).status_code == 422

        # the handler's own guards keep their 400s
        resp = client.post("/training/webhooks", params={"url": "ftp://x", "events": "[]"})
        assert resp.status_code == 400
        assert "http:// or https://" in resp.text
        resp = client.post("/training/webhooks", params={"url": "https://x", "events": "not-json"})
        assert resp.status_code == 400
        assert "Invalid events format" in resp.text
        resp = client.post(
            "/training/webhooks", params={"url": "https://x", "events": '["bogus.event"]'}
        )
        assert resp.status_code == 400
        assert "Invalid events" in resp.text

        # test route: missing body member → 422 — never a live delivery
        assert client.post("/training/webhooks/test", json={}).status_code == 422

        # unknown id → 404 on the parameterized trio
        assert client.get("/training/webhooks/nope").status_code == 404
        assert client.delete("/training/webhooks/nope").status_code == 404
        assert client.get("/training/webhooks/nope/deliveries").status_code == 404

        # unregister: explicit DELETE verb, raw dict, gone afterwards
        resp = client.delete(f"/training/webhooks/{reg['id']}")
        assert resp.status_code == 200
        assert resp.json() == {"status": "deleted", "webhook_id": reg["id"]}
        assert client.get(f"/training/webhooks/{reg['id']}").status_code == 404
        assert client.get("/training/webhooks").json()["webhooks"] == []


def test_webhook_openapi_matches_the_hand_built_doc():
    """Byte-identical wire fields; the accepted doc-only deltas noted below."""
    from training.router import router as training_router

    app = FastAPI()
    app.include_router(training_router)
    doc = app.openapi()

    summaries = {
        "/training/webhooks": {"get": "List Webhooks", "post": "Register Webhook"},
        "/training/webhooks/stats": {"get": "Get Webhook Stats"},
        "/training/webhooks/retry-queue": {"get": "Get Webhook Retry Queue"},
        "/training/webhooks/dead-letters": {"get": "Get Webhook Dead Letters"},
        "/training/webhooks/test": {"post": "Test Webhook"},
        "/training/webhooks/{webhook_id}": {"delete": "Unregister Webhook", "get": "Get Webhook"},
        "/training/webhooks/{webhook_id}/deliveries": {"get": "Get Webhook Deliveries"},
    }
    for path, ops in summaries.items():
        assert set(doc["paths"][path]) == set(ops), path
        for method, summary in ops.items():
            op = doc["paths"][path][method]
            assert op["summary"] == summary, (path, method)
            assert op["tags"] == ["training", "training-webhooks"], (path, method)
            assert "security" not in op, (path, method)  # the chain was empty

    # register: flat scalars stay QUERY members on POST — no request body
    reg = doc["paths"]["/training/webhooks"]["post"]
    assert [(p["name"], p["in"], p.get("required")) for p in reg["parameters"]] == [
        ("url", "query", True),
        ("events", "query", True),
        ("description", "query", False),
        ("secret", "query", False),
    ]
    assert "requestBody" not in reg

    # limit rides the query (optional); webhook_id is contract-required path
    dl = doc["paths"]["/training/webhooks/dead-letters"]["get"]
    assert [(p["name"], p["in"], p.get("required")) for p in dl["parameters"]] == [
        ("limit", "query", False)
    ]
    delv = doc["paths"]["/training/webhooks/{webhook_id}/deliveries"]["get"]
    assert [(p["name"], p["in"], p.get("required")) for p in delv["parameters"]] == [
        ("webhook_id", "path", True),
        ("limit", "query", False),
    ]

    # responses: no-param ops publish 200 alone — nothing can fail validation
    for path, method in (
        ("/training/webhooks", "get"),
        ("/training/webhooks/stats", "get"),
        ("/training/webhooks/retry-queue", "get"),
    ):
        assert sorted(doc["paths"][path][method]["responses"]) == ["200"], (path, method)
    # every op with a parameter can 422 → 200 + 422
    for path, method in (
        ("/training/webhooks", "post"),
        ("/training/webhooks/dead-letters", "get"),
        ("/training/webhooks/test", "post"),
        ("/training/webhooks/{webhook_id}", "delete"),
        ("/training/webhooks/{webhook_id}", "get"),
        ("/training/webhooks/{webhook_id}/deliveries", "get"),
    ):
        assert sorted(doc["paths"][path][method]["responses"]) == ["200", "422"], (path, method)

    # the test route keeps its model-style body — the same TestWebhookRequest
    # class, so the $ref is byte-identical to the hand-built doc
    body = doc["paths"]["/training/webhooks/test"]["post"]["requestBody"]
    assert body["required"] is True
    assert (
        body["content"]["application/json"]["schema"]["$ref"]
        == "#/components/schemas/TestWebhookRequest"
    )

    # Accepted doc-only deltas (4 classes, verified by the before/after
    # OpenAPI diff — 50 field diffs, zero on wire fields): (1) operationId —
    # the registry's stable `training.webhooks.<name>.<verb>` scheme vs
    # FastAPI's function-name derivation; (2) the additive `x-contract`
    # extension (name/version/auth_scope/idempotent — no hand-built key is
    # overwritten); (3) param schema cosmetics — hand-built `title`/`default`/
    # `anyOf[null]` vs projection `type`+`description`; (4) 422
    # description/content — the class accepted since part 1.


# ── sub-batch B: training/router.py (27 sites) ───────────────────────────
# 19 GET (idempotent=True) + 5 POST + 3 explicit DELETE (``derive_method``
# can only emit GET/POST); 2 authenticated — the only routes whose
# hand-built signature carried ``Depends(require_auth_if_enabled)``; 13
# group-A (inner ``success_response`` stripped — the projection's wrap
# emits the identical ``{status, data}`` bytes) vs 14 raw (bare dicts, the
# ``message="loaded"`` self-wrap, ``FileResponse``/raw ``Response``
# passthroughs, and the 2 SSE streams' self-wrapped idle errors).

ROUTER_ROUTES = [
    # name, path, method, idempotent, auth_scope
    ("training.finetuned.list", "/training/finetuned-models", "GET", True, "public"),
    (
        "training.finetuned.load",
        "/training/finetuned-models/{name}/load",
        "POST",
        False,
        "public",
    ),
    (
        "training.finetuned.delete",
        "/training/finetuned-models/{name}",
        "DELETE",
        False,
        "authenticated",
    ),
    ("training.recovery.check", "/recovery/check", "GET", True, "public"),
    ("training.recovery.recoverable", "/recovery/recoverable", "GET", True, "public"),
    ("training.recovery.events", "/recovery/events", "GET", True, "public"),
    ("training.recovery.recover", "/recovery/recover/{job_id}", "POST", False, "public"),
    (
        "training.recovery.abandon",
        "/recovery/abandon/{job_id}",
        "DELETE",
        False,
        "authenticated",
    ),
    ("training.recovery.stats", "/recovery/stats", "GET", True, "public"),
    ("training.log.get", "/training/log", "GET", True, "public"),
    ("training.stop.all", "/training/stop", "POST", False, "public"),
    ("training.checkpoints.list", "/training/checkpoints", "GET", True, "public"),
    ("training.checkpoints.delete", "/training/checkpoints/{name}", "DELETE", False, "public"),
    ("training.checkpoints.load", "/training/checkpoints/{name}/load", "POST", False, "public"),
    (
        "training.checkpoints.download",
        "/training/checkpoints/{name}/download",
        "GET",
        True,
        "public",
    ),
    ("training.checkpoints.info", "/training/checkpoints/{name}/info", "GET", True, "public"),
    ("training.metrics.export", "/training/metrics/export", "GET", True, "public"),
    (
        "training.from_sessions.cancel",
        "/training/from-sessions/cancel",
        "GET",
        True,
        "public",
    ),
    ("training.stream.live", "/training/stream", "GET", True, "public"),
    ("training.stream.from_sessions", "/training/from-sessions-stream", "GET", True, "public"),
    ("training.recommend.get", "/training/recommend", "GET", True, "public"),
    ("training.trends.get", "/training/trends", "GET", True, "public"),
    ("training.monitor.status", "/monitor", "GET", True, "public"),
    ("training.monitor.alerts", "/monitor/alerts", "GET", True, "public"),
    ("training.monitor.metrics", "/monitor/metrics", "GET", True, "public"),
    ("training.monitor.reset", "/monitor/reset", "POST", False, "public"),
    ("training.monitor.resources", "/monitor/resources", "GET", True, "public"),
]

# Captured hand-built summaries (FastAPI derives them from ``route.name`` —
# on a projected route that name would be the generated endpoint's, so every
# value is pinned explicitly in its ``RouteSpec``).
ROUTER_SUMMARIES = {
    ("GET", "/training/finetuned-models"): "List Finetuned Models",
    ("POST", "/training/finetuned-models/{name}/load"): "Load Finetuned Model",
    ("DELETE", "/training/finetuned-models/{name}"): "Delete Finetuned Model",
    ("GET", "/recovery/check"): "Check Crashed Jobs",
    ("GET", "/recovery/recoverable"): "Get Recoverable Jobs",
    ("GET", "/recovery/events"): "Get Recovery Events",
    ("POST", "/recovery/recover/{job_id}"): "Recover Job",
    ("DELETE", "/recovery/abandon/{job_id}"): "Abandon Recovery",
    ("GET", "/recovery/stats"): "Get Recovery Stats",
    ("GET", "/training/log"): "Training Log",
    ("POST", "/training/stop"): "Training Stop",
    ("GET", "/training/checkpoints"): "Training List Checkpoints",
    ("DELETE", "/training/checkpoints/{name}"): "Training Delete Checkpoint",
    ("POST", "/training/checkpoints/{name}/load"): "Training Load Checkpoint",
    ("GET", "/training/checkpoints/{name}/download"): "Training Download Checkpoint",
    ("GET", "/training/checkpoints/{name}/info"): "Training Checkpoint Info",
    ("GET", "/training/metrics/export"): "Training Metrics Export",
    ("GET", "/training/from-sessions/cancel"): "Training Cancel From Sessions",
    ("GET", "/training/stream"): "Training Stream",
    ("GET", "/training/from-sessions-stream"): "Training From Sessions Stream",
    ("GET", "/training/recommend"): "Get Training Recommendation",
    ("GET", "/training/trends"): "Get Training Trends",
    ("GET", "/monitor"): "Training Monitor Status",
    ("GET", "/monitor/alerts"): "Training Monitor Alerts",
    ("GET", "/monitor/metrics"): "Training Monitor Metrics",
    ("POST", "/monitor/reset"): "Training Monitor Reset",
    ("GET", "/monitor/resources"): "Training Monitor Resources",
}

# The 14 no-parameter operations publish 200 alone (nothing can fail
# validation — FastAPI derives the same way on the hand-built routes); the
# other 13 carry parameters → 200 + 422.
ROUTER_NO_422 = {
    ("GET", "/monitor"),
    ("POST", "/monitor/reset"),
    ("GET", "/monitor/resources"),
    ("GET", "/recovery/events"),
    ("GET", "/recovery/recoverable"),
    ("GET", "/recovery/stats"),
    ("GET", "/training/checkpoints"),
    ("GET", "/training/finetuned-models"),
    ("GET", "/training/from-sessions-stream"),
    ("GET", "/training/from-sessions/cancel"),
    ("GET", "/training/log"),
    ("GET", "/training/metrics/export"),
    ("POST", "/training/stop"),
    ("GET", "/training/stream"),
}

# Exactly the two routes whose hand-built signature carried
# ``Depends(require_auth_if_enabled)`` publish a security requirement.
ROUTER_SECURED = {
    ("DELETE", "/training/finetuned-models/{name}"),
    ("DELETE", "/recovery/abandon/{job_id}"),
}


def test_router_descriptors_registered():
    import training.router  # noqa: F401 — import registers the descriptors

    for name, path, method, idem, scope in ROUTER_ROUTES:
        entry = _entry(name)
        assert entry["path"] == path, name
        assert entry["method"] == method, name
        assert entry["idempotent"] is idem, name
        assert entry["auth_scope"] == scope, name

    # edge 4: declared parameters keep the hand-built 422s alive — path
    # params contract-required, query optionals declared-but-not-required
    assert _entry("training.finetuned.load")["params"]["required"] == ["name"]
    assert _entry("training.recovery.check")["params"]["required"] == []
    trends = _entry("training.trends.get")["params"]
    assert list(trends["properties"]) == ["model", "dataset", "limit"]
    assert trends["required"] == []
    assert _entry("training.monitor.alerts")["params"]["required"] == []


def test_router_serves_the_wire(tmp_path):
    """Group A reproduces ``{status, data}``; raw sites stay unwrapped."""
    from unittest.mock import AsyncMock, MagicMock, patch

    from infrastructure.exception_handlers import register_all_handlers
    from training.router import router as training_router

    app = FastAPI()
    app.include_router(training_router)
    register_all_handlers(app)
    client = TestClient(app)

    engine = MagicMock()
    monitor = engine.get_training_monitor.return_value
    monitor.check_resources.return_value = {"cpu": 50}
    monitor.get_alerts.return_value = []
    engine.get_log = AsyncMock(return_value=["line-1"])
    engine.load_outcomes.return_value = []
    engine.get_state.return_value.config = {}
    engine.get_all_checkpoint_data = AsyncMock(return_value=[])
    engine.load_stored_checkpoint = AsyncMock(return_value={"name": "w"})
    engine.download_checkpoint_path = AsyncMock(return_value=None)

    store = MagicMock()
    store.get_stats.return_value = {"total": 1}
    store.detect_crashed_jobs.return_value = []
    store.get_recoverable_jobs.return_value = []
    store.get.return_value = None

    with (
        patch("domain.training.engine.get_training_engine", return_value=engine),
        patch("training.sse_stream.stop_all_training", return_value={"stopped": []}),
        patch("training.router.get_job_store", return_value=store),
        patch("training.process_events.boot_snapshot", return_value=[]),
    ):
        # group A: the inner success_response was stripped — the projection's
        # wrap must emit the identical {status, data} bytes the hand-built
        # route served
        body = client.get("/training/log").json()
        assert body == {"status": "success", "data": {"lines": ["line-1"], "total": 1}}

        body = client.post("/training/stop").json()
        assert body == {"status": "success", "data": {"stopped": []}}

        body = client.get("/training/trends").json()
        assert body["status"] == "success"
        assert body["data"] == {
            "runs": [],
            "models": [],
            "summary": {
                "total_runs": 0,
                "avg_quality": 0,
                "best_quality": 0,
                "avg_loss": 0,
                "trend": "stable",
            },
        }

        body = client.get("/training/recommend").json()
        assert body["status"] == "success"
        assert isinstance(body["data"], dict)

        body = client.get("/monitor/resources").json()
        assert body == {"status": "success", "data": {"cpu": 50}}

        # absent optional query keys are omitted → the handler default (50)
        # survives; the projection must pass nothing, never None
        body = client.get("/monitor/alerts").json()
        assert body == {"status": "success", "data": {"alerts": [], "total": 0}}
        monitor.get_alerts.assert_called_with(severity=None, limit=50)

        # transport validation: a mistyped member 422s before the handler
        # (these three never reach the engine/store mocks above)
        assert client.get("/monitor/alerts?limit=abc").status_code == 422
        assert client.get("/training/trends?limit=abc").status_code == 422
        assert client.get("/recovery/check?timeout_seconds=abc").status_code == 422

        # raw wire: bare dicts — no envelope may leak
        body = client.get("/training/finetuned-models").json()
        assert "status" not in body and "models" in body

        body = client.get("/recovery/stats").json()
        assert "status" not in body
        assert body["crashed_jobs"] == 0 and body["recoverable_jobs"] == 0

        assert client.get("/recovery/recoverable").json() == {"count": 0, "jobs": []}
        assert client.get("/recovery/check").json()["detected_crashes"] == 0
        events = client.get("/recovery/events").json()
        assert "status" not in events and events["count"] == 0

        # raw Response passthrough: attachment headers survive, no wrap
        resp = client.get("/training/metrics/export")
        assert resp.status_code == 200
        assert "training-metrics.json" in resp.headers.get("content-disposition", "")
        assert set(resp.json()) == {"exported_at", "total_checkpoints", "checkpoints"}

        # message="loaded" self-wrap is hand-built wire: exactly one wrap —
        # envelope=False must not re-wrap it
        body = client.post("/training/checkpoints/w/load").json()
        assert body == {
            "status": "success",
            "data": {"name": "w"},
            "message": "loaded",
        }

        # FileResponse passthrough: bytes, not a JSON envelope — then the
        # missing-checkpoint guard keeps its raw 404
        ckpt = tmp_path / "w.ckpt"
        ckpt.write_bytes(b"CKPT")
        engine.download_checkpoint_path = AsyncMock(return_value=str(ckpt))
        resp = client.get("/training/checkpoints/w/download")
        assert resp.status_code == 200
        assert resp.content == b"CKPT"
        engine.download_checkpoint_path = AsyncMock(return_value=None)
        assert client.get("/training/checkpoints/nope/download").status_code == 404

        # both SSE streams: idle is a self-wrapped 200 — raw wire, one wrap,
        # never the projection's envelope (which would nest the error dict)
        for path in ("/training/stream", "/training/from-sessions-stream"):
            resp = client.get(path)
            assert resp.status_code == 200, path
            body = resp.json()
            assert body["status"] == "success", path
            assert body["data"]["status"] == "error", path
            assert body["data"]["code"] == "E_STATE_IDLE", path

        # handler guards keep their raw error shapes on the authenticated
        # routes too (SLO_AUTH_REQUIRED defaults off — the dependency is
        # still in the chain, which is what the OpenAPI security pins)
        resp = client.delete("/training/finetuned-models/nope")
        assert resp.status_code == 404
        assert "Fine-tuned model not found" in resp.text
        resp = client.delete("/recovery/abandon/nope")
        assert resp.status_code == 404
        assert "Job not found" in resp.text


def test_router_openapi_matches_the_hand_built_doc():
    """Byte-identical wire fields; the accepted doc-only deltas noted below."""
    from training.router import router as training_router

    app = FastAPI()
    app.include_router(training_router)
    doc = app.openapi()

    assert len(ROUTER_SUMMARIES) == 27
    for (method, path), summary in ROUTER_SUMMARIES.items():
        op = doc["paths"][path][method.lower()]
        assert op["summary"] == summary, (path, method)
        assert op["tags"] == ["training"], (path, method)  # chain tag, no inner tags
        assert "requestBody" not in op, (path, method)  # every op is body-free
        responses = sorted(op["responses"])
        expected = ["200"] if (method, path) in ROUTER_NO_422 else ["200", "422"]
        assert responses == expected, (path, method)
        assert bool(op.get("security")) is ((method, path) in ROUTER_SECURED), (path, method)

    # parameter parity: path params contract-required, query optionals stay
    # optional and in the hand-built order
    assert [
        (p["name"], p["in"], p.get("required"))
        for p in doc["paths"]["/training/recommend"]["get"]["parameters"]
    ] == [("dataset_path", "query", False), ("method", "query", False)]
    assert [
        (p["name"], p["in"], p.get("required"))
        for p in doc["paths"]["/training/trends"]["get"]["parameters"]
    ] == [("model", "query", False), ("dataset", "query", False), ("limit", "query", False)]
    assert [
        (p["name"], p["in"], p.get("required"))
        for p in doc["paths"]["/monitor/alerts"]["get"]["parameters"]
    ] == [("severity", "query", False), ("limit", "query", False)]
    assert [
        (p["name"], p["in"], p.get("required"))
        for p in doc["paths"]["/recovery/check"]["get"]["parameters"]
    ] == [("timeout_seconds", "query", False)]
    assert [
        (p["name"], p["in"], p.get("required"))
        for p in doc["paths"]["/training/checkpoints/{name}/download"]["get"]["parameters"]
    ] == [("name", "path", True)]
    assert [
        (p["name"], p["in"], p.get("required"))
        for p in doc["paths"]["/recovery/abandon/{job_id}"]["delete"]["parameters"]
    ] == [("job_id", "path", True)]
    for path, method in (
        ("/training/log", "get"),
        ("/training/stream", "get"),
        ("/training/stop", "post"),
        ("/monitor", "get"),
    ):
        assert "parameters" not in doc["paths"][path][method], (path, method)

    # Accepted doc-only deltas (verified by the before/after OpenAPI diff —
    # 155 field diffs, zero on wire fields): (1) operationId — the
    # registry's stable `training.<name>.<verb>` scheme vs FastAPI's
    # function-name derivation; (2) the additive `x-contract` extension;
    # (3) param schema cosmetics — hand-built `title`/`default`/`anyOf[null]`
    # vs projection `type`+`description`; (4) the standing 422
    # description/content class; (5) 200-response-model schemas on the 13
    # group-A envelope routes (envelope routes keep the derived `-> dict`).


# ── sub-batch C: training/live_stream.py (1 site) ────────────────────────
# The brief's edge-2 exception clause was NOT needed: the SSE route is
# projectable exactly like router.py's two streams — raw-Request injection
# serves ``request``, name-based auth detection serves ``auth_user``, and
# ``envelope=False`` passes the StreamingResponse through untouched.


def test_live_stream_descriptor_registered():
    import training.live_stream  # noqa: F401 — import registers the descriptor

    entry = _entry("training.live_stream")
    assert entry["path"] == "/training/live-stream"
    assert entry["method"] == "GET"
    assert entry["idempotent"] is True
    assert entry["auth_scope"] == "authenticated"  # the hand-built Depends chain
    assert entry["params"]["properties"] == {}
    assert entry["params"]["required"] == []


def test_live_stream_serves_sse():
    """First frame over HTTP: standard SSE envelope, headers byte-identical."""
    import json as _json
    from unittest.mock import patch

    from training.router import router as training_router

    app = FastAPI()
    app.include_router(training_router)
    client = TestClient(app)

    # The route is an infinite SSE loop and TestClient runs the app to
    # completion, so the test drives the generator's own disconnect check
    # (the same fake-request trick TestLiveSnapshot uses directly) — cycle 1
    # emits the first frame, cycle 2 breaks the loop. The bare app has no
    # middleware, so nothing else touches the check.
    calls = {"n": 0}

    async def _disconnect_after_first_frame(_self):
        calls["n"] += 1
        return calls["n"] > 1

    with (
        patch("starlette.requests.Request.is_disconnected", _disconnect_after_first_frame),
        patch("training.live_stream.LIVE_STREAM_INTERVAL", 0.05),
    ):
        resp = client.get("/training/live-stream")

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    assert resp.headers["cache-control"] == "no-cache"
    frames = [
        _json.loads(line[len("data: ") :])
        for line in resp.text.splitlines()
        if line.startswith("data: ")
    ]
    assert frames, "expected at least one SSE frame"
    frame = frames[0]
    assert frame["stream"] == "training-live"
    assert frame["phase"] == "LIVE"
    assert frame["status"] == "working"
    # the first frame is the baseline: changed=True by contract
    assert frame["data"]["changed"] is True
    assert set(frame["data"]) >= {"sources", "totals", "jobs", "at", "changed"}
    assert isinstance(frame["data"]["jobs"], dict)


def test_live_stream_openapi_matches_the_hand_built_doc():
    """URL/verb/tags/summary/description/security/response exact."""
    from training.router import router as training_router

    app = FastAPI()
    app.include_router(training_router)
    op = app.openapi()["paths"]["/training/live-stream"]["get"]

    assert op["summary"] == "Training Live Stream"
    assert op["tags"] == ["training", "training-feed"]  # parent + module chain
    assert sorted(op["responses"]) == ["200"]  # no params → nothing can 422
    assert op["security"] == [{"HTTPBearer": []}]  # edge 3: the Depends chain
    assert "parameters" not in op
    assert "requestBody" not in op
    desc = op.get("description") or ""
    assert desc.splitlines()[0] == "SSE stream of feed arrivals and active training jobs."
    assert len(desc.splitlines()) == 11  # the handler docstring, verbatim

    # Accepted doc-only deltas (before/after diff — 2 field diffs, zero on
    # wire fields): (1) operationId — registry scheme vs FastAPI's
    # function-name derivation; (2) the additive `x-contract` extension.
