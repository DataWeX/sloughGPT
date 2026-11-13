"""Conversation AppScope — the per-instance config seam (Phase 2).

Contracts proven here (server layer):
  - AppScopeDelta (wire twin) ↔ AppScope (domain) field parity
  - ``chat.sessions.update`` registry row honesty (params match the
    UpsertSessionRequest body; result matches what upsert_session returns)
  - PUT /chat/sessions/{id} persists the scope delta, and the list row
    carries it back (the mini-app's config rides the index)
  - ``_resolve_turn_scope``: delta over global applied onto a bare
    TurnRequest, wire-explicit fields win, corrupt deltas fail safe
  - the ``max_new_tokens`` → ``max_tokens`` alias carries wire precedence
    through ``ChatRequest.to_turn()`` (same fact, same layer)
"""

import uuid

from infrastructure.contract import REGISTRY
from routers.inference import AppScopeDelta, SessionModel, UpsertSessionRequest, _instance

from apps.api.server.tests.test_support import get_test_client
from domain.inference.app_scope import AppScope
from domain.inference.engine import TurnRequest

client = get_test_client()


def _session_with_scope(sid: str, scope: dict) -> None:
    """Seed the router's session cache with a delta (no disk I/O)."""
    _instance._session_memory_cache[sid] = {
        "id": sid,
        "messages": [],
        "created_at": "2026-11-10T00:00:00",
        "updated_at": "2026-11-10T00:00:00",
        "scope": scope,
    }


class TestAppScopeParity:
    def test_delta_and_domain_declare_the_same_fields(self):
        delta_fields = set(AppScopeDelta.model_fields)
        domain_fields = {f.name for f in AppScope.__dataclass_fields__.values()}
        assert delta_fields == domain_fields, delta_fields ^ domain_fields

    def test_delta_bounds_mirror_chat_request(self):
        """A delta can never hold a value the wire would reject."""
        from schemas.chat import ChatRequest

        for name in ("temperature", "top_p", "top_k", "repetition_penalty", "max_tokens"):
            delta_field = AppScopeDelta.model_fields[name]
            wire_field = ChatRequest.model_fields[name]
            assert delta_field.metadata == wire_field.metadata, name

    def test_delta_rejects_unknown_fields(self):
        import pytest
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            AppScopeDelta.model_validate({"temperature": 0.5, "nope": 1})

    def test_session_model_carries_scope(self):
        schema = SessionModel.model_json_schema()
        assert "scope" in schema["properties"]
        assert "AppScopeDelta" in schema.get("$defs", {})


class TestChatSessionsUpdateDescriptor:
    def test_chat_sessions_update_registered(self):
        routes = {(r["method"], r["path"]): r for r in REGISTRY.routes()}
        entry = routes[("PUT", "/chat/sessions/{session_id}")]
        assert entry["name"] == "chat.sessions.update"
        assert entry["auth_scope"] == "authenticated"
        assert entry["idempotent"] is True  # PUT is idempotent by HTTP semantics
        assert entry["module"] == "routers.inference"

    def test_params_match_upsert_request_body(self):
        """Honesty: declared params == the body fields the route validates."""
        routes = {(r["method"], r["path"]): r for r in REGISTRY.routes()}
        entry = routes[("PUT", "/chat/sessions/{session_id}")]
        assert set(entry["params"]["properties"]) == set(UpsertSessionRequest.model_fields)

    def test_result_matches_handler_payload(self):
        routes = {(r["method"], r["path"]): r for r in REGISTRY.routes()}
        entry = routes[("PUT", "/chat/sessions/{session_id}")]
        assert set(entry["result"]["properties"]) == {"session_id"}
        assert entry["result"]["required"] == ["session_id"]

    def test_served_route_matches_descriptor(self):
        sid = f"desc_{uuid.uuid4().hex[:8]}"
        client.post("/chat/sessions", json={"session_id": sid, "name": "d"})
        resp = client.put(f"/chat/sessions/{sid}", json={"name": "renamed"})
        assert resp.status_code == 200
        assert resp.json()["data"]["session_id"] == sid
        client.delete(f"/chat/sessions/{sid}")


class TestScopePersistence:
    def test_put_persists_scope_and_list_carries_it(self):
        sid = f"scope_{uuid.uuid4().hex[:8]}"
        client.post("/chat/sessions", json={"session_id": sid, "name": "mini-app"})
        resp = client.put(
            f"/chat/sessions/{sid}",
            json={"scope": {"temperature": 0.1, "max_tokens": 64, "use_rag": False}},
        )
        assert resp.status_code == 200

        rows = client.get("/chat/sessions").json()["data"]
        row = next(r for r in rows if r["id"] == sid)
        assert row["scope"] == {"temperature": 0.1, "max_tokens": 64, "use_rag": False}
        client.delete(f"/chat/sessions/{sid}")

    def test_scopeless_session_row_has_null_scope(self):
        sid = f"noscope_{uuid.uuid4().hex[:8]}"
        client.post("/chat/sessions", json={"session_id": sid, "name": "plain"})
        rows = client.get("/chat/sessions").json()["data"]
        row = next(r for r in rows if r["id"] == sid)
        assert row["scope"] is None
        client.delete(f"/chat/sessions/{sid}")

    def test_metadata_update_without_scope_leaves_delta_alone(self):
        sid = f"keep_{uuid.uuid4().hex[:8]}"
        client.post("/chat/sessions", json={"session_id": sid, "name": "orig"})
        client.put(f"/chat/sessions/{sid}", json={"scope": {"temperature": 0.3}})
        client.put(f"/chat/sessions/{sid}", json={"name": "renamed"})  # no scope key
        rows = client.get("/chat/sessions").json()["data"]
        row = next(r for r in rows if r["id"] == sid)
        assert row["name"] == "renamed"
        assert row["scope"] == {"temperature": 0.3}
        client.delete(f"/chat/sessions/{sid}")

    def test_put_rejects_out_of_bounds_delta(self):
        sid = f"bounds_{uuid.uuid4().hex[:8]}"
        client.post("/chat/sessions", json={"session_id": sid, "name": "b"})
        resp = client.put(f"/chat/sessions/{sid}", json={"scope": {"temperature": 99}})
        assert resp.status_code == 422
        client.delete(f"/chat/sessions/{sid}")


class TestResolveTurnScope:
    def test_delta_applies_onto_bare_turnrequest(self):
        sid = f"seam_{uuid.uuid4().hex[:8]}"
        _session_with_scope(
            sid,
            {
                "max_tokens": 64,
                "system_prompt": "SCOPE PROMPT",
                "knowledge": ["fact-a"],
                "use_rag": False,
            },
        )
        req = TurnRequest(messages=[], model_fields_set=set())
        won = _instance._resolve_turn_scope(sid, req)
        assert req.max_tokens == 64
        assert req.use_rag is False
        assert "SCOPE PROMPT" in req.system_prompt  # additive compose
        assert req.knowledge == ["fact-a"]
        assert won == frozenset({"max_tokens", "use_rag"})
        _instance._session_memory_cache.pop(sid, None)

    def test_wire_explicit_field_wins_over_delta(self):
        sid = f"wire_{uuid.uuid4().hex[:8]}"
        _session_with_scope(sid, {"temperature": 0.1})
        req = TurnRequest(messages=[], model_fields_set={"temperature"})
        won = _instance._resolve_turn_scope(sid, req)
        assert req.temperature == 0.7  # TurnRequest default — wire won
        assert "temperature" not in won
        _instance._session_memory_cache.pop(sid, None)

    def test_session_without_scope_falls_back_to_global(self):
        sid = f"glob_{uuid.uuid4().hex[:8]}"
        _session_with_scope(sid, {})  # record exists, no delta
        req = TurnRequest(messages=[], model_fields_set=set())
        _instance._resolve_turn_scope(sid, req)
        from domain.settings import get_settings

        assert req.max_tokens == get_settings().settings.generation.max_new_tokens
        _instance._session_memory_cache.pop(sid, None)

    def test_corrupt_delta_fails_safe(self):
        from domain.settings import get_settings

        sid = f"bad_{uuid.uuid4().hex[:8]}"
        # one corrupt field alongside a valid one: validation must drop the
        # WHOLE delta (not salvage field-by-field) and fall through to global
        _session_with_scope(sid, {"temperature": "hot", "max_tokens": 64})
        req = TurnRequest(messages=[], model_fields_set=set())
        won = _instance._resolve_turn_scope(sid, req)  # must not raise
        gen = get_settings().settings.generation
        assert req.max_tokens == gen.max_new_tokens  # 64 NOT salvaged
        assert req.temperature == gen.temperature  # global filled, delta gone
        assert won == frozenset()
        _instance._session_memory_cache.pop(sid, None)

    def test_conversation_won_fields_reach_meta_weights_as_explicit(self):
        """The seam's return value unions into ``explicit=`` — regression pin."""
        sid = f"exp_{uuid.uuid4().hex[:8]}"
        _session_with_scope(sid, {"top_p": 0.99})
        req = TurnRequest(messages=[], model_fields_set=set())
        won = _instance._resolve_turn_scope(sid, req)
        assert (req.model_fields_set | won) == frozenset({"top_p"})
        _instance._session_memory_cache.pop(sid, None)

    def test_max_new_tokens_alias_carries_wire_precedence(self):
        """``max_new_tokens`` is the same fact as ``max_tokens`` — it must outrank
        the lower layers like any other wire-explicit field (the alias resolves
        in model_post_init, which does NOT touch model_fields_set)."""
        from schemas.chat import ChatRequest

        sid = f"alias_{uuid.uuid4().hex[:8]}"
        _session_with_scope(sid, {"max_tokens": 64})
        wire = ChatRequest(
            messages=[{"role": "user", "content": "hi"}],
            max_new_tokens=200,
        )
        req = wire.to_turn()
        assert req.max_tokens == 200  # alias resolved into the field
        won = _instance._resolve_turn_scope(sid, req)
        assert req.max_tokens == 200  # ...and the delta did NOT swallow it
        assert "max_tokens" not in won
        _instance._session_memory_cache.pop(sid, None)
