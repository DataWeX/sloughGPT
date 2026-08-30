"""Tests for meta_weights router — GetMetaWeightsRequest, MetaWeightsRouter construction."""

import sys
import pytest
from pathlib import Path
from types import SimpleNamespace

pytest.importorskip("fastapi")

_server_dir = str(Path(__file__).resolve().parents[3] / "apps" / "api" / "server")
if _server_dir not in sys.path:
    sys.path.insert(0, _server_dir)

from apps.api.server.routers.meta_weights import MetaWeightsRouter, GetMetaWeightsRequest


class TestGetMetaWeightsRequest:
    def test_fields(self):
        req = GetMetaWeightsRequest(user_message="hello", k=5, user_id="u1")
        assert req.user_message == "hello"
        assert req.k == 5
        assert req.user_id == "u1"

    def test_defaults(self):
        req = GetMetaWeightsRequest(user_message="hi")
        assert req.k == 10
        assert req.user_id == "default"

    def test_custom_k(self):
        req = GetMetaWeightsRequest(user_message="test", k=20)
        assert req.k == 20

    def test_empty_message(self):
        req = GetMetaWeightsRequest(user_message="")
        assert req.user_message == ""

    def test_user_id_preserved(self):
        req = GetMetaWeightsRequest(user_message="m", user_id="custom_user")
        assert req.user_id == "custom_user"

    def test_negative_k(self):
        req = GetMetaWeightsRequest(user_message="m", k=-1)
        assert req.k == -1

    def test_zero_k(self):
        req = GetMetaWeightsRequest(user_message="m", k=0)
        assert req.k == 0

    def test_large_k(self):
        req = GetMetaWeightsRequest(user_message="m", k=1000)
        assert req.k == 1000

    def test_special_chars_message(self):
        req = GetMetaWeightsRequest(user_message="hello <world> & \"test\"")
        assert req.user_message == 'hello <world> & "test"'


class TestMetaWeightsRouter:
    def test_router_has_prefix(self):
        router = MetaWeightsRouter()
        assert router.router.prefix == "/meta-weights"

    def test_router_has_routes(self):
        router = MetaWeightsRouter()
        routes = [r.path for r in router.router.routes]
        assert "/ping" in routes
        assert "/get" in routes
        assert "/stats" in routes

    def test_ping_route_exists(self):
        router = MetaWeightsRouter()
        methods = []
        for route in router.router.routes:
            if hasattr(route, "path") and route.path == "/ping":
                methods.extend(route.methods)
        assert "GET" in methods

    def test_get_route_exists(self):
        router = MetaWeightsRouter()
        methods = []
        for route in router.router.routes:
            if hasattr(route, "path") and route.path == "/get":
                methods.extend(route.methods)
        assert "POST" in methods

    def test_stats_route_exists(self):
        router = MetaWeightsRouter()
        methods = []
        for route in router.router.routes:
            if hasattr(route, "path") and route.path == "/stats":
                methods.extend(route.methods)
        assert "GET" in methods

    def test_router_is_instance(self):
        router = MetaWeightsRouter()
        assert hasattr(router, "router")

    def test_multiple_routers_independent(self):
        r1 = MetaWeightsRouter()
        r2 = MetaWeightsRouter()
        assert r1.router is not r2.router
