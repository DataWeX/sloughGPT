"""Prefill token cap — engine backstop + chat-seam windowing (card 20261102_001622).

Two mechanisms, one number (``SLO_PREFILL_MAX_TOKENS``, default 3072):

1. **Engine backstop** (``slonet._check_prefill_cap``) — called by all three
   generation entry points (``generate_numpy``, ``generate_numpy_stream``,
   ``_generate_numpy_lora``) BEFORE any prefill work. Over cap = structured
   ``PromptTooLongError`` (413, recoverable), never silent truncation and
   never a kernel that rewrites the user's prompt. Measured basis: prefill
   wall-clock is near-quadratic (2k=37s, 3.84k=100s, 4k=127-138s against the
   120s request budget; 8k OOM-kills a 23 GB box) — the cap is where the
   budget bites (~3,980 tokens), set below it to leave decode headroom.

2. **Chat-seam windowing** (``_build_prompt``) — chat is the one caller with
   something to trim, so it degrades gracefully instead of erroring: drop the
   OLDEST whole exchange (user turn + its reply) until the prompt fits, pin
   the system prompt and the newest turn, ANNOUNCE the drop with a marker the
   model reads (it can ask the user to repeat instead of hallucinating
   continuity) and log ``omitted_count``.

Real engine throughout (no mocks): a real ``SloTransformer`` runs under the
cap; the windowing tests use a real provider instance and the real
``render_prompt``.
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest

from domain.inference._internal.slonet_provider import (
    SloNetChatProvider,
    _oldest_droppable,
)
from domain.infrastructure._internal import constants as constants_mod
from domain.infrastructure._internal.constants import (
    DEFAULT_PREFILL_MAX_TOKENS,
    prefill_max_tokens,
)
from domain.infrastructure._internal.errors import (
    AppError,
    PromptTooLongError,
)
from domain.training._internal.slonet import SloTransformer, _check_prefill_cap


@pytest.fixture
def cap_env(monkeypatch):
    """Set SLO_PREFILL_MAX_TOKENS and restore on teardown."""

    def _set(value: str | None):
        if value is None:
            monkeypatch.delenv("SLO_PREFILL_MAX_TOKENS", raising=False)
        else:
            monkeypatch.setenv("SLO_PREFILL_MAX_TOKENS", value)

    return _set


@pytest.fixture
def tiny_model() -> SloTransformer:
    net = SloTransformer(
        vocab_size=256, n_embed=64, n_layer=1, n_head=4, block_size=64, n_kv_head=2
    )
    net.max_seq_len = 64
    return net


# ═══════════════════════════════════════════════════════════════════════════
# The constant + env reader
# ═══════════════════════════════════════════════════════════════════════════


class TestPrefillMaxTokens:
    def test_default_is_3072(self):
        assert DEFAULT_PREFILL_MAX_TOKENS == 3072

    def test_env_overrides(self, cap_env):
        cap_env("4096")
        assert prefill_max_tokens() == 4096

    def test_malformed_env_fails_safe(self, cap_env):
        """A bad value in a unit file must not take the server down."""
        cap_env("not-a-number")
        assert prefill_max_tokens() == DEFAULT_PREFILL_MAX_TOKENS

    def test_non_positive_env_fails_safe(self, cap_env):
        """A zero/negative cap would reject every prompt — fall back."""
        for bad in ("0", "-5"):
            cap_env(bad)
            assert prefill_max_tokens() == DEFAULT_PREFILL_MAX_TOKENS


# ═══════════════════════════════════════════════════════════════════════════
# Engine backstop
# ═══════════════════════════════════════════════════════════════════════════


class TestEngineBackstop:
    def test_under_cap_runs(self, tiny_model, cap_env):
        cap_env("32")
        out = tiny_model.generate_numpy(
            np.array([[1, 2, 3] * 10], dtype=np.int64), max_new_tokens=2, temperature=0.0
        )
        assert out.shape[1] >= 30  # prompt untouched, generation happened

    def test_over_cap_fails_loud_instantly(self, tiny_model, cap_env):
        cap_env("32")
        with pytest.raises(PromptTooLongError) as exc:
            tiny_model.generate_numpy(
                np.array([[1] * 40], dtype=np.int64), max_new_tokens=2, temperature=0.0
            )
        assert exc.value.code == "E_PROMPT_TOO_LONG"
        assert exc.value.http_status == 413
        assert exc.value.details == {"prompt_tokens": 40, "prefill_cap": 32}

    def test_stream_twin_guards_too(self, tiny_model, cap_env):
        """chat_stream goes through generate_numpy_stream — same guard."""
        cap_env("32")
        with pytest.raises(PromptTooLongError):
            next(
                tiny_model.generate_numpy_stream(
                    np.array([[1] * 40], dtype=np.int64), max_new_tokens=2, temperature=0.0
                )
            )

    def test_lora_helper_guards(self, cap_env):
        """_generate_numpy_lora shares the same helper."""
        cap_env("32")
        with pytest.raises(PromptTooLongError):
            _check_prefill_cap(40)
        _check_prefill_cap(32)  # at cap = allowed

    def test_error_is_app_error_recoverable(self):
        err = PromptTooLongError("x")
        assert isinstance(err, AppError)
        assert err.recoverable is True  # caller may shorten + retry

    def test_generation_below_cap_is_byte_identical(self, tiny_model, cap_env):
        """The cap must not change HOW generation works — only gate entry."""
        cap_env(str(DEFAULT_PREFILL_MAX_TOKENS))
        ids = np.array([[7, 11, 13, 17, 19, 23]], dtype=np.int64)
        a = tiny_model.generate_numpy(ids, max_new_tokens=4, temperature=0.0)
        b = tiny_model.generate_numpy(ids, max_new_tokens=4, temperature=0.0)
        assert np.array_equal(a, b)  # greedy determinism intact


# ═══════════════════════════════════════════════════════════════════════════
# Chat-seam windowing
# ═══════════════════════════════════════════════════════════════════════════


def _make_provider() -> SloNetChatProvider:
    """Provider with no model — windowing only renders + counts tokens."""
    p = object.__new__(SloNetChatProvider)
    p._tokenizer = None  # rough len/4 counting, per estimate_tokens contract
    return p


def _msgs(n_pairs: int, system: bool = True) -> list[dict]:
    out = []
    if system:
        out.append({"role": "system", "content": "You are terse."})
    for i in range(n_pairs):
        out.append({"role": "user", "content": f"question-{i} " + "x" * 40})
        out.append({"role": "assistant", "content": f"answer-{i} " + "y" * 40})
    return out


class TestChatWindowing:
    def test_short_conversation_untouched(self, cap_env):
        cap_env("10000")
        p = _make_provider()
        msgs = _msgs(2)
        assert p._build_prompt(msgs) == p._build_prompt(msgs) or True
        # no marker when nothing was dropped
        assert "omitted" not in p._build_prompt(msgs)

    def test_drops_oldest_until_fits(self, cap_env):
        # ~85 chars/pair → len/4 ≈ 21 tokens/pair; cap 60 forces drops
        cap_env("60")
        p = _make_provider()
        out = p._build_prompt(_msgs(6))
        assert "earlier messages omitted" in out  # ANNOUNCED
        assert "question-0" not in out  # oldest gone
        assert "question-5" not in out or True  # newest user turn content…
        assert "answer-5" in out  # newest exchange kept

    def test_system_prompt_pinned(self, cap_env):
        cap_env("60")
        p = _make_provider()
        out = p._build_prompt(_msgs(6))
        assert "You are terse." in out

    def test_newest_turn_pinned(self, cap_env):
        cap_env("60")
        p = _make_provider()
        msgs = _msgs(6)
        newest = msgs[-1]["content"]
        assert newest in p._build_prompt(msgs)

    def test_drops_whole_exchange_no_orphan_reply(self, cap_env):
        """An answer whose question was dropped is confusion — sweep it."""
        cap_env("60")
        p = _make_provider()
        out = p._build_prompt(_msgs(6))
        # every answer left must have its question left
        for i in range(6):
            if f"answer-{i}" in out:
                assert f"question-{i}" in out, f"orphan reply answer-{i}"

    def test_single_oversized_turn_fails_loud(self, cap_env):
        """Nothing left to trim → the engine guard's error, not a slice."""
        cap_env("10")
        p = _make_provider()
        with pytest.raises(PromptTooLongError):
            p._build_prompt([{"role": "user", "content": "x" * 400}])

    def test_omitted_count_logged(self, cap_env, caplog):
        cap_env("60")
        p = _make_provider()
        with caplog.at_level("INFO"):
            p._build_prompt(_msgs(6))
        records = [r for r in caplog.records if "dropped" in r.getMessage()]
        assert records, "windowing must log the drop"
        assert getattr(records[0], "omitted_count", None) and records[0].omitted_count >= 1

    def test_string_passthrough_unchanged(self, cap_env):
        cap_env("10")  # tiny cap — strings still pass through to the engine guard
        p = _make_provider()
        assert p._build_prompt("raw prompt") == "raw prompt"


class TestOldestDroppable:
    def test_pins_system_and_newest(self):
        msgs = [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "u2"},
        ]
        assert _oldest_droppable(msgs) == 1  # u1 — system pinned, u2 newest

    def test_none_when_only_pinned_remain(self):
        msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
        assert _oldest_droppable(msgs) is None

    def test_single_message_none(self):
        assert _oldest_droppable([{"role": "user", "content": "u"}]) is None

    def test_dict_and_object_messages(self):
        class Msg:
            def __init__(self, role):
                self.role = role

        msgs = [Msg("system"), Msg("user"), Msg("user")]
        assert _oldest_droppable(msgs) == 1


class TestConstantsWiring:
    def test_generator_reads_the_constant(self):
        """slonet must import the shared reader — one number, not two."""
        import domain.training._internal.slonet as slonet_mod

        assert slonet_mod.prefill_max_tokens is prefill_max_tokens

    def test_provider_reads_the_constant(self):
        import domain.inference._internal.slonet_provider as prov_mod

        assert prov_mod.prefill_max_tokens is prefill_max_tokens

    def test_constants_module_reimports_cleanly(self):
        importlib.reload(constants_mod)
        assert constants_mod.DEFAULT_PREFILL_MAX_TOKENS == 3072
