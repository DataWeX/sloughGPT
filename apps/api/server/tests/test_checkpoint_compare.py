"""Checkpoint compare: same prompt, two checkpoints, served model untouched."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from infrastructure.exception_handlers import register_app_error_handler
from training.router import router

app = FastAPI()
register_app_error_handler(app)
app.include_router(router)
client = TestClient(app)

from domain.training._internal import checkpoints as ckpt_mod  # noqa: E402
from domain.training._internal.service import (  # noqa: E402
    build_checkpoint_provider,
    compare_checkpoints,
    load_checkpoint,
)

# ── HTTP contract ──────────────────────────────────────────────────────────────


class TestCompareEndpoint:
    def test_returns_both_answers(self):
        with patch(
            "domain.training._internal.service.compare_checkpoints",
            new_callable=AsyncMock,
            return_value={
                "a": {"name": "cp-1.soul", "text": "hello from a"},
                "b": {"name": "cp-2.soul", "text": "hello from b"},
            },
        ):
            resp = client.post(
                "/training/checkpoints/compare",
                json={"a": "cp-1.soul", "b": "cp-2.soul", "prompt": "Say hi"},
            )

        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["a"]["text"] == "hello from a"
        assert data["b"]["text"] == "hello from b"
        assert data["a"]["name"] == "cp-1.soul"

    def test_forwards_prompt_and_token_budget(self):
        with patch(
            "domain.training._internal.service.compare_checkpoints",
            new_callable=AsyncMock,
            return_value={"a": {"name": "x", "text": ""}, "b": {"name": "y", "text": ""}},
        ) as mock_cmp:
            resp = client.post(
                "/training/checkpoints/compare",
                json={"a": "x.soul", "b": "y.soul", "prompt": "Hi", "max_new_tokens": 64},
            )

        assert resp.status_code == 200
        mock_cmp.assert_awaited_once_with("x.soul", "y.soul", "Hi", 64)

    def test_rejects_empty_prompt(self):
        resp = client.post(
            "/training/checkpoints/compare",
            json={"a": "cp-1.soul", "b": "cp-2.soul", "prompt": ""},
        )
        assert resp.status_code == 422

    def test_rejects_missing_prompt(self):
        resp = client.post(
            "/training/checkpoints/compare",
            json={"a": "cp-1.soul", "b": "cp-2.soul"},
        )
        assert resp.status_code == 422

    def test_unknown_checkpoint_is_reported_as_not_found(self):
        with patch(
            "domain.training._internal.service.compare_checkpoints",
            new_callable=AsyncMock,
            side_effect=FileNotFoundError("Checkpoint not found: ghost.soul"),
        ):
            resp = client.post(
                "/training/checkpoints/compare",
                json={"a": "ghost.soul", "b": "cp-2.soul", "prompt": "Hi"},
            )

        assert resp.status_code >= 400
        body = resp.json()
        assert "not found" in str(body).lower() or "ghost" in str(body).lower()


# ── compare must not touch the served model ───────────────────────────────────


def _fake_provider(answer: str) -> MagicMock:
    provider = MagicMock()
    provider.chat = AsyncMock(return_value=answer)
    return provider


class TestCompareNeverServesTheCheckpoints:
    @pytest.mark.asyncio
    async def test_never_registers_a_provider(self):
        provider_a = _fake_provider("answer a")
        provider_b = _fake_provider("answer b")
        providers = [(provider_a, {"name": "cp-1.soul"}), (provider_b, {"name": "cp-2.soul"})]

        with (
            patch.object(ckpt_mod, "build_checkpoint_provider", AsyncMock(side_effect=providers)),
            patch("domain.models._internal.provider.register_provider") as mock_register,
        ):
            result = await compare_checkpoints("cp-1.soul", "cp-2.soul", "Prompt", 32)

        assert mock_register.call_count == 0
        assert result["a"] == {"name": "cp-1.soul", "text": "answer a"}
        assert result["b"] == {"name": "cp-2.soul", "text": "answer b"}

    @pytest.mark.asyncio
    async def test_both_sides_use_identical_sampling(self):
        provider_a = _fake_provider("a")
        provider_b = _fake_provider("b")

        with patch.object(
            ckpt_mod,
            "build_checkpoint_provider",
            AsyncMock(side_effect=[(provider_a, {"name": "1"}), (provider_b, {"name": "2"})]),
        ):
            await compare_checkpoints("1", "2", "Prompt", 48)

        call_a = provider_a.chat.await_args
        call_b = provider_b.chat.await_args
        assert call_a.kwargs == call_b.kwargs
        assert call_a.kwargs["max_tokens"] == 48
        assert call_a.args == call_b.args == ([{"role": "user", "content": "Prompt"}],)

    @pytest.mark.asyncio
    async def test_stops_before_generating_when_a_checkpoint_is_missing(self):
        with patch.object(
            ckpt_mod,
            "build_checkpoint_provider",
            AsyncMock(side_effect=FileNotFoundError("Checkpoint not found: ghost")),
        ):
            with pytest.raises(FileNotFoundError):
                await compare_checkpoints("ghost", "cp-2.soul", "Prompt")


# ── the load path still swaps the served model ────────────────────────────────


class TestLoadStillRegisters:
    @pytest.mark.asyncio
    async def test_load_registers_slonet_and_default(self):
        provider = _fake_provider("unused")
        with (
            patch.object(
                ckpt_mod,
                "build_checkpoint_provider",
                AsyncMock(return_value=(provider, {"name": "cp-1.soul"})),
            ),
            patch("domain.models._internal.provider.register_provider") as mock_register,
        ):
            info = await load_checkpoint("cp-1.soul")

        assert info["name"] == "cp-1.soul"
        registered = [call.args[0] for call in mock_register.call_args_list]
        assert registered == ["slonet", "default"]
        assert all(call.args[1] is provider for call in mock_register.call_args_list)

    def test_build_is_exported_from_the_service_facade(self):
        from domain.training._internal import service

        assert service.build_checkpoint_provider is build_checkpoint_provider
        assert service.compare_checkpoints is compare_checkpoints
