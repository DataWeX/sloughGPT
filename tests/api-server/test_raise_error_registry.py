"""``raise_error()``: registry resolution, loud degradation, two-channel messages.

The static gate (``tests/core-py/test_error_registry_drift.py``) asserts that
every code in use is declared. These tests pin the *behaviour* of the resolver
itself, which used to swallow ``(ValueError, KeyError)`` for an unregistered
code and hand back a generic ``AppError``/400 with no log line anywhere:

  * a registered code resolves to its registry class and status
  * an unregistered code still raises (behaviour preserved) but says so loudly
  * ``user_message`` keeps internal detail on the log channel only
"""

from __future__ import annotations

import logging

import pytest
from schemas.common import classify_and_raise, raise_error


class TestRegistryResolution:
    """A registered code must classify the way the registry says it does."""

    def test_code_resolves_its_registered_class_and_status(self):
        from domain.infrastructure._internal.errors import AppError, ConflictError

        with pytest.raises(AppError) as excinfo:
            raise_error("resource is busy", "E_BUSY")

        assert isinstance(excinfo.value, ConflictError)
        assert excinfo.value.http_status == 409

    @pytest.mark.parametrize(
        ("code", "expected_status"),
        [
            ("E_MISSING_TEXT", 400),  # raised with no explicit status_code
            ("E_VALIDATION", 422),
            ("E_SERIALIZATION", 500),
            ("E_NO_MODEL", 503),
        ],
    )
    def test_status_matches_what_the_call_sites_send(self, code, expected_status):
        with pytest.raises(Exception) as excinfo:
            raise_error("test", code)

        assert excinfo.value.http_status == expected_status

    def test_explicit_status_code_still_wins_over_the_registry(self):
        with pytest.raises(Exception) as excinfo:
            raise_error("test", "E_MISSING_TEXT", status_code=418)

        assert excinfo.value.http_status == 418

    def test_legacy_alias_resolves_by_member_name(self):
        """``E_CREATE_FAILED`` is declared as an alias of ``E_BAD_REQUEST``.

        A value-only lookup raises ValueError for it, which is exactly how the
        alias used to degrade to a generic AppError.
        """
        from domain.infrastructure._internal.errors import ValidationError

        with pytest.raises(Exception) as excinfo:
            raise_error("creation failed", "E_CREATE_FAILED")

        assert isinstance(excinfo.value, ValidationError)
        assert excinfo.value.http_status == 400
        assert excinfo.value.code == "E_CREATE_FAILED"


class TestUnknownCodeIsLoud:
    """Silent downgrade is the defect; behaviour must stay, the silence must go."""

    @staticmethod
    def test_unregistered_code_still_raises_and_logs_a_warning(caplog):
        with caplog.at_level(logging.WARNING, logger="slo.errors"):
            with pytest.raises(Exception) as excinfo:
                raise_error("test", "E_DEFINITELY_NOT_REGISTERED")

        # Behaviour preserved: generic AppError, status 400.
        assert excinfo.value.http_status == 400

        warnings = [r for r in caplog.records if r.name == "slo.errors"]
        assert warnings, "an unregistered code must be logged, not swallowed"
        assert "E_DEFINITELY_NOT_REGISTERED" in warnings[0].getMessage()
        assert "unregistered code" in warnings[0].getMessage()


class TestHandlerWiring:
    """The handler must send ``user_message`` to the client, ``message`` to the log.

    The unit tests above check the exception's attributes; this checks the one
    place the two channels could be swapped again — the AppError handler's
    JSON body.
    """

    @pytest.fixture(autouse=True)
    def _register_route(self):
        from fastapi import APIRouter
        from main import app

        router = APIRouter(prefix="/_test_tax", tags=["_test_tax"])

        @router.get("/internal-failure")
        async def _internal_failure():
            from domain.infrastructure._internal.errors import AppError

            raise AppError(
                message="raw: /models/secret.soul is corrupt (Errno 2)",
                code="E_DOMAIN",
                user_message="An unexpected internal error occurred.",
                http_status=500,
            )

        app.include_router(router)

    @staticmethod
    def test_response_carries_the_client_text_and_hides_the_internal_one():
        from test_support import get_test_client

        resp = get_test_client().get("/_test_tax/internal-failure")

        assert resp.status_code == 500
        body = resp.json()
        assert body["code"] == "E_DOMAIN"
        assert body["error"] == "An unexpected internal error occurred."
        assert "secret.soul" not in resp.text
        assert "Errno 2" not in resp.text


class TestTwoChannels:
    """``message`` is developer detail; ``user_message`` is what the client sees."""

    @staticmethod
    def test_user_message_override_keeps_internal_detail_off_the_client():
        internal = "raw: /models/secret.soul is corrupt (Errno 2)"

        with pytest.raises(Exception) as excinfo:
            raise_error(
                internal,
                "E_DOMAIN",
                status_code=500,
                user_message="An unexpected internal error occurred.",
            )

        assert excinfo.value.message == internal  # still on the log channel
        assert excinfo.value.user_message == "An unexpected internal error occurred."
        assert "secret.soul" not in excinfo.value.user_message

    @staticmethod
    def test_message_is_used_for_both_channels_by_default():
        """Callers that write a purpose-built message stay wire-identical."""
        with pytest.raises(Exception) as excinfo:
            raise_error("Checkpoint not found: demo", "E_NOT_FOUND", status_code=404)

        assert excinfo.value.message == "Checkpoint not found: demo"
        assert excinfo.value.user_message == "Checkpoint not found: demo"

    @staticmethod
    def test_classify_and_raise_fallback_does_not_ship_the_exception_text(monkeypatch):
        """The catch-all used to do ``raise_error(str(e), "E_DOMAIN", 500)``."""
        import domain.infrastructure._internal.errors as err_mod

        def _broken_classifier(_exc):
            raise TypeError("classifier exploded")

        monkeypatch.setattr(err_mod, "classify_exception", _broken_classifier)

        with pytest.raises(Exception) as excinfo:
            classify_and_raise(ValueError("raw: /models/secret.soul"), source="test.fallback")

        assert excinfo.value.http_status == 500
        assert excinfo.value.message == "raw: /models/secret.soul"
        assert excinfo.value.user_message == "An unexpected internal error occurred."
        assert "secret.soul" not in excinfo.value.user_message
