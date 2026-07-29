"""Integration tests for SloNetServer -> SloNetChatProvider -> setup_providers.

Tests:
1. ``SloNetChatProvider.set_server()`` attaches a ``SloNetServer``
2. ``chat()`` and ``chat_stream()`` delegate to the server when attached
3. ``setup_providers(slonet_server=...)`` wires the server correctly
"""

import asyncio
import threading
from unittest.mock import MagicMock, patch

import pytest
import numpy as np

from domains.infrastructure.slonet_server import SloNetServer
from domains.inference.slonet_provider import SloNetChatProvider
from domains.models.provider import (
    setup_providers,
    get_provider,
    list_providers,
    ProviderRouter,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clean_registries():
    import domains.models.provider as mod
    mod._providers.clear()
    mod._processors.clear()
    yield
    mod._providers.clear()
    mod._processors.clear()


@pytest.fixture
def mock_model():
    m = MagicMock()
    m.generate_numpy.return_value = np.array([[101, 102, 103]], dtype=np.int64)
    m.generate_numpy_stream.return_value = iter([np.int64(101), np.int64(102)])
    m.layers = [MagicMock()]
    m.layers[0].weight.shape = (256, 64)
    m.max_seq_len = 2048
    m.parameters.return_value = []
    m._config = {"n_embd": 64, "n_head": 4}
    return m


@pytest.fixture
def mock_tokenizer():
    t = MagicMock()
    t.encode.return_value = [10, 20, 30]
    t.decode.return_value = "hello world"
    t.eos_token_id = 0
    return t


@pytest.fixture
def server(mock_model, mock_tokenizer):
    return SloNetServer(
        model=mock_model,
        tokenizer=mock_tokenizer,
        model_id="test-slonet",
        enable_warmup=False,
    )


def _make_provider(mock_model, mock_tokenizer):
    inst = SloNetChatProvider.__new__(SloNetChatProvider)
    inst._model = mock_model
    inst._tokenizer = mock_tokenizer
    inst._hf_model_id = "test-model"
    inst._model_id = "test-model"
    inst._device = "cpu"
    inst._parser = None
    inst._quant_engine = None
    return inst


# ---------------------------------------------------------------------------
# Server -> Provider attachment
# ---------------------------------------------------------------------------

class TestServerAttachment:
    def test_set_server_stores_reference(self, server):
        provider = _make_provider(MagicMock(), MagicMock())
        provider.set_server(server)
        assert provider.get_server() is server

    def test_get_server_returns_none_when_not_set(self):
        provider = _make_provider(MagicMock(), MagicMock())
        assert provider.get_server() is None

    def test_set_server_overwrites_previous(self, server):
        other = MagicMock()
        provider = _make_provider(MagicMock(), MagicMock())
        provider.set_server(other)
        provider.set_server(server)
        assert provider.get_server() is server


# ---------------------------------------------------------------------------
# chat() delegation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestChatDelegation:
    async def test_chat_delegates_to_server(self, server):
        provider = _make_provider(MagicMock(), MagicMock())
        provider.set_server(server)
        result = await provider.chat(
            [{"role": "user", "content": "hello"}],
            max_tokens=50, temperature=0.8,
        )
        assert result == "hello world"

    async def test_chat_without_server_uses_to_thread(self, mock_model, mock_tokenizer):
        provider = _make_provider(mock_model, mock_tokenizer)
        result = await provider.chat(
            [{"role": "user", "content": "hello"}],
            max_tokens=50,
        )
        assert result == "hello world"
        assert mock_model.generate_numpy.called

    async def test_chat_handles_server_error(self, server):
        server._model.generate_numpy.side_effect = RuntimeError("gen fail")
        provider = _make_provider(MagicMock(), MagicMock())
        provider.set_server(server)
        with pytest.raises(RuntimeError, match="gen fail"):
            await provider.chat(
                [{"role": "user", "content": "hello"}],
            )


# ---------------------------------------------------------------------------
# chat_stream() delegation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestChatStreamDelegation:
    async def test_chat_stream_delegates_to_server(self, server):
        provider = _make_provider(MagicMock(), MagicMock())
        provider.set_server(server)
        tokens = []
        async for token in provider.chat_stream(
            [{"role": "user", "content": "hello"}],
            max_tokens=50, temperature=0.8,
        ):
            tokens.append(token)
        assert len(tokens) == 2

    async def test_chat_stream_with_cancel(self, server):
        provider = _make_provider(MagicMock(), MagicMock())
        provider.set_server(server)
        cancel = threading.Event()
        cancel.set()
        tokens = []
        async for token in provider.chat_stream(
            [{"role": "user", "content": "hello"}],
            cancel_event=cancel,
        ):
            tokens.append(token)
        assert len(tokens) == 0

    async def test_chat_stream_without_server(self, mock_model, mock_tokenizer):
        provider = _make_provider(mock_model, mock_tokenizer)
        tokens = []
        async for token in provider.chat_stream(
            [{"role": "user", "content": "hello"}],
        ):
            tokens.append(token)
        assert len(tokens) > 0

    async def test_chat_stream_handles_server_error(self, server):
        server._model.generate_numpy_stream.side_effect = RuntimeError("stream fail")
        provider = _make_provider(MagicMock(), MagicMock())
        provider.set_server(server)
        with pytest.raises(RuntimeError, match="stream fail"):
            async for _ in provider.chat_stream(
                [{"role": "user", "content": "hello"}],
            ):
                pass


# ---------------------------------------------------------------------------
# setup_providers() wiring
# ---------------------------------------------------------------------------

class TestSetupProvidersWiring:
    def test_attaches_server_to_provider(self, server, mock_model, mock_tokenizer):
        provider = _make_provider(mock_model, mock_tokenizer)
        with patch.object(provider, "set_server") as mock_set:
            setup_providers(
                slonet_provider=provider,
                slonet_server=server,
            )
            mock_set.assert_called_once_with(server)

    def test_registers_slonet_native_provider(self, server, mock_model, mock_tokenizer):
        provider = _make_provider(mock_model, mock_tokenizer)
        setup_providers(
            slonet_provider=provider,
            slonet_server=server,
        )
        assert "slonet-native" in list_providers()

    def test_default_router_uses_slonet(self, server, mock_model, mock_tokenizer):
        provider = _make_provider(mock_model, mock_tokenizer)
        setup_providers(
            slonet_provider=provider,
            slonet_server=server,
        )
        router = get_provider("default")
        assert isinstance(router, ProviderRouter)
        assert router._text_name == "slonet-native"

    def test_no_server_no_attachment(self, mock_model, mock_tokenizer):
        provider = _make_provider(mock_model, mock_tokenizer)
        with patch.object(provider, "set_server") as mock_set:
            setup_providers(slonet_provider=provider)
            mock_set.assert_not_called()

    def test_full_chain_generate(self, server, mock_model, mock_tokenizer):
        provider = _make_provider(mock_model, mock_tokenizer)
        setup_providers(
            slonet_provider=provider,
            slonet_server=server,
        )
        router = get_provider("default")
        assert router is not None
        assert router._text_name == "slonet-native"
        assert server.warmup_completed is False

    def test_auto_detect_cached_slnc(self, tmp_path):
        import sys

        slnc_path = tmp_path / "model.slnc"
        slnc_path.write_text("not-real")
        model_id = "Auto/Detect"

        mock_cfg = MagicMock()
        mock_cfg.autoload_model = model_id

        fake_mod = MagicMock()
        fake_mod.get_config = MagicMock(return_value=mock_cfg)
        sys.modules["domains.infrastructure.config"] = fake_mod

        mock_provider = MagicMock()
        mock_provider.model_id = model_id
        mock_provider.set_server = MagicMock()

        try:
            with patch("domains.infrastructure.safetensors_loader._get_model_dir", return_value=tmp_path):
                with patch("domains.inference.slonet_provider.SloNetChatProvider") as mock_cls:
                    mock_cls.from_slnc.return_value = mock_provider
                    setup_providers()
                    mock_cls.from_slnc.assert_called_once()

            assert "slonet-native" in list_providers()
            router = get_provider("default")
            assert isinstance(router, ProviderRouter)
            assert router._text_name == "slonet-native"
        finally:
            if "domains.infrastructure.config" in sys.modules:
                del sys.modules["domains.infrastructure.config"]


# ---------------------------------------------------------------------------
# Pool server integration
# ---------------------------------------------------------------------------

class TestPoolServerIntegration:
    @pytest.fixture
    def pool_server(self):
        factory = MagicMock()
        model = MagicMock()
        model.generate_numpy.return_value = np.array([[101, 102, 103]], dtype=np.int64)
        model.generate_numpy_stream.return_value = iter([np.int64(42)])
        factory.return_value = model
        tokenizer = MagicMock()
        tokenizer.encode.return_value = [10, 20, 30]
        tokenizer.decode.return_value = "pool result"
        tokenizer.eos_token_id = 0
        return SloNetServer(
            model_factory=factory,
            tokenizer=tokenizer,
            max_workers=3,
            enable_warmup=False,
        )

    @pytest.mark.asyncio
    async def test_pool_chat_delegates_to_server(self, pool_server):
        provider = _make_provider(MagicMock(), MagicMock())
        provider.set_server(pool_server)
        result = await provider.chat(
            [{"role": "user", "content": "hello"}],
            max_tokens=50, temperature=0.8,
        )
        assert result == "pool result"
        stats = pool_server.pool_stats()
        assert stats["mode"] == "pool"
        assert stats["created"] == 1

    @pytest.mark.asyncio
    async def test_pool_chat_stream_delegates_to_server(self, pool_server):
        provider = _make_provider(MagicMock(), MagicMock())
        provider.set_server(pool_server)
        tokens = []
        async for token in provider.chat_stream(
            [{"role": "user", "content": "hello"}],
        ):
            tokens.append(token)
        assert len(tokens) >= 1

    @pytest.mark.asyncio
    async def test_pool_chat_concurrent_requests(self, pool_server):
        provider = _make_provider(MagicMock(), MagicMock())
        provider.set_server(pool_server)
        results = await asyncio.gather(*[
            provider.chat([{"role": "user", "content": "hi"}])
            for _ in range(3)
        ])
        assert all(r == "pool result" for r in results)
        stats = pool_server.pool_stats()
        assert stats["created"] <= 3

    @pytest.mark.asyncio
    async def test_pool_chat_stream_cancellation(self, pool_server):
        provider = _make_provider(MagicMock(), MagicMock())
        provider.set_server(pool_server)
        cancel = threading.Event()
        cancel.set()
        tokens = []
        async for token in provider.chat_stream(
            [{"role": "user", "content": "hello"}],
            cancel_event=cancel,
        ):
            tokens.append(token)
        assert len(tokens) == 0

    @pytest.mark.asyncio
    async def test_pool_chat_metrics_after_use(self, pool_server):
        provider = _make_provider(MagicMock(), MagicMock())
        provider.set_server(pool_server)
        await provider.chat([{"role": "user", "content": "hello"}])
        m = pool_server.get_metrics()
        assert m["requests_total"] == 1
        assert m["requests_completed"] == 1

    def test_pool_setup_providers_wire(self, pool_server):
        provider = _make_provider(MagicMock(), MagicMock())
        with patch.object(provider, "set_server") as mock_set:
            setup_providers(
                slonet_provider=provider,
                slonet_server=pool_server,
            )
            mock_set.assert_called_once_with(pool_server)

    @pytest.mark.asyncio
    async def test_pool_chain_generate(self, pool_server):
        provider = _make_provider(MagicMock(), MagicMock())
        setup_providers(
            slonet_provider=provider,
            slonet_server=pool_server,
        )
        router = get_provider("default")
        assert router._text_name == "slonet-native"
        result = await provider.chat([{"role": "user", "content": "hello"}])
        assert result == "pool result"
