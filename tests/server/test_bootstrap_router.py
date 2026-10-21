"""
Tests for the bootstrap router — the /boot one-shot read-model (card 342d729f).

GET /bootstrap composes the landing journey's first-paint domain reads from
the member endpoints' own handlers via the route table: members present are
served (envelope-unwrapped), members absent are null + "not_ready", and a
failing member degrades to an error entry — the aggregate answers 200 even
during the model-load phase, never 503.
"""

from fastapi import APIRouter, Depends, FastAPI, Query, Request
from fastapi.testclient import TestClient
from infrastructure.auth import require_auth_if_enabled
from infrastructure.exception_handlers import register_all_handlers
from schemas.common import endpoint

from apps.api.server.routers.bootstrap import router

MEMBER_KEYS = {
    "health",
    "startup_progress",
    "models",
    "consciousness_status",
    "souls",
    "sessions",
    "training_jobs",
    "turbo_status",
    "knowledge_stats",
    "feedback_stats",
    "meta_weights_stats",
    "datasets",
}


def _make_app(*extra_routers: APIRouter) -> FastAPI:
    _app = FastAPI()
    register_all_handlers(_app)
    _app.include_router(router)
    for extra in extra_routers:
        _app.include_router(extra)
    return _app


def _client(*extra_routers: APIRouter) -> TestClient:
    return TestClient(_make_app(*extra_routers), raise_server_exceptions=False)


def _envelope_router() -> APIRouter:
    """Stub members at real member paths: envelope-shaped, bare-shaped,
    sync, and request-param members."""
    r = APIRouter()

    @r.get("/health")
    async def health() -> dict:
        return {"status": "success", "data": {"model_loaded": True, "device": "cpu"}}

    @r.get("/souls")
    async def souls() -> list:
        return [{"name": "ada"}]

    @r.get("/knowledge/stats")
    def knowledge_stats() -> dict:  # sync member — must be thread-offloaded
        return {"status": "success", "data": {"total": 3}}

    @r.get("/meta-weights/stats")
    async def meta_weights(req: Request) -> dict:
        return {"status": "success", "data": {"path": req.url.path}}

    @r.get("/datasets")
    async def datasets(q: str | None = Query(None), type: str | None = Query(None)) -> dict:
        # Real datasets shape: Query params + a bare (non-envelope) body —
        # doubles as the Query-default resolution and passthrough test.
        return {"q": q, "type": type}

    @r.get("/training/jobs")
    @endpoint("tests.training_jobs")
    async def training_jobs(auth_user: dict = Depends(require_auth_if_enabled)) -> dict:
        # The real member shape: @endpoint's sync-looking wrapper around an
        # async handler, with route-level auth injected by FastAPI.
        return {"status": "success", "data": {"auth_user": auth_user}}

    return r


class TestBootstrapPartialPayload:
    def test_answers_200_with_all_members_not_ready_when_only_bootstrap_mounted(self):
        resp = _client().get("/bootstrap")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "success"
        members = body["data"]["members"]
        errors = body["data"]["errors"]
        assert set(members) == MEMBER_KEYS
        assert all(v is None for v in members.values())
        assert set(errors) == set(MEMBER_KEYS)
        assert all(err == "not_ready" for err in errors.values())

    def test_member_registered_later_becomes_available(self):
        client = _client(_envelope_router())
        resp = client.get("/bootstrap")
        members = resp.json()["data"]["members"]
        errors = resp.json()["data"]["errors"]
        # The five stubbed members resolve; the rest stay not_ready.
        assert members["health"] == {"model_loaded": True, "device": "cpu"}
        assert "health" not in errors
        assert errors["sessions"] == "not_ready"


class TestBootstrapComposition:
    def test_unwraps_envelopes_and_passes_bare_payloads_through(self):
        members = _client(_envelope_router()).get("/bootstrap").json()["data"]["members"]
        assert members["health"] == {"model_loaded": True, "device": "cpu"}
        assert members["knowledge_stats"] == {"total": 3}
        assert members["souls"] == [{"name": "ada"}]
        assert members["datasets"] == {"q": None, "type": None}

    def test_endpoint_wrapped_async_member_is_awaited(self):
        # @endpoint's wrapper looks sync to iscoroutinefunction; the
        # aggregate must unwrap-detect and await it (live regression:
        # wrapped async members used to surface as "coroutine is not
        # iterable" jsonable_encoder failures).
        members = _client(_envelope_router()).get("/bootstrap").json()["data"]["members"]
        assert members["training_jobs"] == {"auth_user": None}

    def test_query_defaults_resolve_to_their_defaults_not_query_objects(self):
        members = _client(_envelope_router()).get("/bootstrap").json()["data"]["members"]
        # auth disabled in tests: require_auth_if_enabled returns None, and
        # the member body's `if auth_user and auth_user.get(...)` skips.
        assert members["datasets"] == {"q": None, "type": None}

    def test_sync_member_is_called(self):
        members = _client(_envelope_router()).get("/bootstrap").json()["data"]["members"]
        assert members["knowledge_stats"] == {"total": 3}

    def test_member_request_param_receives_the_bootstrap_request(self):
        members = _client(_envelope_router()).get("/bootstrap").json()["data"]["members"]
        # By design the injected request is the aggregate's own request.
        assert members["meta_weights_stats"] == {"path": "/bootstrap"}

    def test_no_errors_when_stubbed_members_succeed(self):
        errors = _client(_envelope_router()).get("/bootstrap").json()["data"]["errors"]
        assert set(errors) == MEMBER_KEYS - {
            "health",
            "souls",
            "knowledge_stats",
            "meta_weights_stats",
            "datasets",
            "training_jobs",
        }


class TestBootstrapDegradation:
    def test_failing_member_degrades_to_null_and_error_not_503(self):
        r = APIRouter()

        @r.get("/souls")
        async def souls() -> dict:
            raise RuntimeError("store offline")

        client = _client(r)
        resp = client.get("/bootstrap")
        assert resp.status_code == 200
        body = resp.json()
        assert body["data"]["members"]["souls"] is None
        assert "RuntimeError: store offline" in body["data"]["errors"]["souls"]
        # Other members unaffected: unstubbed ones are not_ready, not errors.
        assert body["data"]["errors"]["health"] == "not_ready"
        assert body["data"]["members"]["health"] is None

    def test_member_with_uninferable_required_param_degrades_loudly(self):
        r = APIRouter()

        @r.get("/souls")
        async def souls(weird: int) -> dict:  # required, no default — uncallable
            return {}

        resp = _client(r).get("/bootstrap")
        assert resp.status_code == 200
        errors = resp.json()["data"]["errors"]
        assert "required parameter 'weird'" in errors["souls"]
