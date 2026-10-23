"""CONSCIOUSNESS_PROC pipeline step (card 243016c3) — reflection injection path.

The chat pipeline pulls the consciousness engine's propose-only reflection
narrative (reflect(apply=False) — never mutates beliefs) and prepends it as a
system message. Reflection is ADVISORY: a disabled engine, an empty narrative,
or any failure skips injection and the chat proceeds with unchanged messages
(try/except skip pattern, unlike KNOWLEDGE_PROC which errors the stream).
"""

from __future__ import annotations

from types import SimpleNamespace

from routers.inference import _apply_consciousness_reflection


class _FakeEngine:
    def __init__(self, narrative: str = "I am curious about compilers.", enabled: bool = True):
        self._narrative = narrative
        self.config = SimpleNamespace(is_enabled=lambda: enabled)

    def reflect(self, apply: bool = True):
        assert apply is False, "prompt injection must be propose-only"
        return SimpleNamespace(narrative=self._narrative)


_MSGS = [{"role": "user", "content": "hi"}]


def _patch_engine(monkeypatch, engine) -> None:
    import domain.cognition as cognition

    monkeypatch.setattr(cognition, "get_consciousness", lambda *a, **k: engine)


class TestConsciousnessInjection:
    async def test_injects_reflection_narrative(self, monkeypatch):
        _patch_engine(monkeypatch, _FakeEngine())
        out = await _apply_consciousness_reflection(list(_MSGS), "corr")
        assert out[0]["role"] == "system"
        assert "curious about compilers" in out[0]["content"]
        assert out[1:] == _MSGS

    async def test_disabled_engine_is_noop(self, monkeypatch):
        _patch_engine(monkeypatch, _FakeEngine(enabled=False))
        assert await _apply_consciousness_reflection(list(_MSGS), "corr") == _MSGS

    async def test_empty_narrative_is_noop(self, monkeypatch):
        _patch_engine(monkeypatch, _FakeEngine(""))
        assert await _apply_consciousness_reflection(list(_MSGS), "corr") == _MSGS

    async def test_engine_failure_skips_injection(self, monkeypatch):
        def boom(*a, **k):
            raise RuntimeError("consciousness unavailable")

        import domain.cognition as cognition

        monkeypatch.setattr(cognition, "get_consciousness", boom)
        assert await _apply_consciousness_reflection(list(_MSGS), "corr") == _MSGS

    async def test_reflect_error_skips_injection(self, monkeypatch):
        class _BoomEngine:
            config = SimpleNamespace(is_enabled=lambda: True)

            def reflect(self, apply: bool = True):
                raise ValueError("self model empty")

        _patch_engine(monkeypatch, _BoomEngine())
        assert await _apply_consciousness_reflection(list(_MSGS), "corr") == _MSGS
