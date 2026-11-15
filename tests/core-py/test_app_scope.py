"""AppScope — the per-conversation instance config (Phase 2: conversation = instance boundary).

Contracts proven here (domain layer, no I/O — the resolver is pure):
  - precedence: conversation delta > global GenerationSettings, per field
  - presence-based, never value-based: ``False``/``0`` in a delta are SET
  - layers never write back: resolution flows one way (global → conversation)
  - ``max_tokens`` ⇄ ``GenerationSettings.max_new_tokens`` rename bridging
  - apply: wire-explicit (``model_fields_set``) outranks both lower layers
  - apply returns the conversation-won set (the seam's ``explicit=`` input
    for ``_apply_meta_weights`` — a heuristic may not move a scoped value)
  - additive composition: ``system_prompt``/``knowledge`` are never
    transported by apply_scope (the seam composes them)
  - the frozen value object never holds a mutable field (knowledge → tuple)
"""

from dataclasses import fields

from domain.inference.app_scope import (
    SCOPE_TRANSPORT_FIELDS,
    AppScope,
    apply_scope,
    resolve_scope,
)
from domain.inference.engine import TurnRequest
from domain.settings import GenerationSettings


def _global(**overrides) -> GenerationSettings:
    return GenerationSettings(**overrides)


class TestResolveScope:
    def test_conversation_wins_over_global_per_field(self):
        scope = resolve_scope(
            conversation={"temperature": 0.2},
            global_settings=_global(temperature=0.9, top_p=0.5),
        )
        assert scope.temperature == 0.2  # conversation won
        assert scope.top_p == 0.5  # global filled the unset field

    def test_global_supplies_generation_params_only(self):
        scope = resolve_scope(conversation={}, global_settings=_global())
        # generation params come from the global layer...
        assert scope.temperature is not None
        assert scope.max_tokens is not None
        # ...but context toggles / instructions are conversation-owned only
        assert scope.use_rag is None
        assert scope.use_context_core is None
        assert scope.system_prompt is None
        assert scope.knowledge is None

    def test_presence_based_false_and_zero_are_set_values(self):
        scope = resolve_scope(
            conversation={"use_rag": False, "top_k": 0},
            global_settings=_global(top_k=40),
        )
        assert scope.use_rag is False  # False is set, not absent
        assert scope.top_k == 0  # 0 is set — would be lost by truthiness

    def test_absent_conversation_and_no_global_falls_to_unset(self):
        scope = resolve_scope(conversation=None, global_settings=None)
        assert scope.temperature is None  # TurnRequest default untouched
        assert scope.set_fields == frozenset()

    def test_max_tokens_reads_generation_max_new_tokens(self):
        scope = resolve_scope(conversation={}, global_settings=_global(max_new_tokens=512))
        assert scope.max_tokens == 512  # rename bridged, value intact

    def test_conversation_max_tokens_beats_global_max_new_tokens(self):
        scope = resolve_scope(
            conversation={"max_tokens": 64},
            global_settings=_global(max_new_tokens=512),
        )
        assert scope.max_tokens == 64

    def test_knowledge_coerced_to_tuple(self):
        scope = resolve_scope(conversation={"knowledge": ["a", "b"]})
        assert scope.knowledge == ("a", "b")
        assert isinstance(scope.knowledge, tuple)  # frozen object, no mutation

    def test_resolution_writes_back_nothing(self):
        delta = {"temperature": 0.1}
        global_settings = _global(temperature=0.9)
        resolve_scope(conversation=delta, global_settings=global_settings)
        assert delta == {"temperature": 0.1}  # input dict untouched
        assert global_settings.temperature == 0.9  # global untouched

    def test_scope_is_frozen(self):
        scope = resolve_scope(conversation={"temperature": 0.5})
        try:
            scope.temperature = 1.0  # noqa: B010 — deliberately invalid
        except AttributeError:
            pass
        else:
            raise AssertionError("AppScope must be frozen")

    def test_all_fields_declared_match_transport_plus_composed(self):
        names = {f.name for f in fields(AppScope)}
        # every field is either transported, composed at the seam, or both
        assert names == set(SCOPE_TRANSPORT_FIELDS) | {"system_prompt", "knowledge"}


class TestApplyScope:
    def _req(self, fields_set=frozenset()) -> TurnRequest:
        return TurnRequest(messages=[], model_fields_set=set(fields_set))

    def test_applies_scope_where_wire_left_unset(self):
        req = self._req()
        scope = resolve_scope(
            conversation={"max_tokens": 64, "use_rag": False},
            global_settings=_global(),
        )
        won = apply_scope(req, scope, conversation={"max_tokens": 64, "use_rag": False})
        assert req.max_tokens == 64
        assert req.use_rag is False
        # ...and both are reported as conversation-won (explicit at the seam)
        assert won == frozenset({"max_tokens", "use_rag"})

    def test_wire_explicit_outranks_conversation_and_global(self):
        req = self._req(fields_set={"temperature"})
        scope = resolve_scope(
            conversation={"temperature": 0.1},
            global_settings=_global(temperature=0.9),
        )
        won = apply_scope(req, scope, conversation={"temperature": 0.1})
        assert req.temperature == 0.7  # TurnRequest default — wire won both layers
        assert "temperature" not in won

    def test_returns_conversation_won_fields_only(self):
        req = self._req()
        delta = {"max_tokens": 64, "use_rag": False}
        # temperature comes from the GLOBAL layer, max_tokens/use_rag from
        # the conversation — only the latter are "conversation-won".
        scope = resolve_scope(conversation=delta, global_settings=_global())
        won = apply_scope(req, scope, conversation=delta)
        assert won == frozenset({"max_tokens", "use_rag"})
        assert req.temperature is not None  # global still applied

    def test_unset_scope_leaves_turnrequest_defaults(self):
        req = self._req()
        won = apply_scope(req, AppScope(), conversation=None)
        assert req.temperature == 0.7
        assert req.max_tokens == 128
        assert won == frozenset()

    def test_does_not_transport_additive_fields(self):
        req = self._req()
        scope = resolve_scope(
            conversation={"system_prompt": "hello", "knowledge": ["k1"]},
            global_settings=_global(),
        )
        won = apply_scope(req, scope, conversation={"system_prompt": "hello", "knowledge": ["k1"]})
        assert req.system_prompt is None  # composed by the seam, not transported
        assert req.knowledge is None
        assert "system_prompt" not in won and "knowledge" not in won

    def test_does_not_mutate_model_fields_set(self):
        req = self._req()
        scope = resolve_scope(conversation={"max_tokens": 64})
        apply_scope(req, scope, conversation={"max_tokens": 64})
        assert req.model_fields_set == set()  # wire-explicit tracking stays wire's

    def test_conversation_zero_temperature_is_applied(self):
        req = self._req()
        scope = resolve_scope(conversation={"temperature": 0.0})
        won = apply_scope(req, scope, conversation={"temperature": 0.0})
        assert req.temperature == 0.0  # 0.0 is a set value
        assert won == frozenset({"temperature"})
