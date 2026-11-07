"""Tests for the /registry router (model registry proxy)."""

from unittest.mock import MagicMock, patch

from test_support import get_test_client


def _data(resp):
    body = resp.json()
    return body.get("data", body)


MOCK_MODELS = [
    {"model_id": "m1", "type": "local", "status": "loaded"},
    {"model_id": "m2", "type": "remote", "status": "ready"},
]

MOCK_HEALTH = {
    "models_loaded": 1,
    "models_registered": 2,
    "healthy": True,
    "degraded": False,
    "has_errors": False,
    "default_model": "m1",
    "models": MOCK_MODELS,
}


def _make_registry():
    reg = MagicMock()
    reg.list_models.return_value = MOCK_MODELS
    reg.health_summary.return_value = MOCK_HEALTH
    return reg


class TestRegistryListModels:
    def test_list_models(self):
        with patch(
            "domain.infrastructure.model_registry.get_model_registry",
            return_value=_make_registry(),
        ):
            client = get_test_client()
            resp = client.get("/registry/models")
            assert resp.status_code == 200
            data = _data(resp)
            assert "models" in data
            assert data["count"] == 2

    def test_list_models_empty(self):
        reg = _make_registry()
        reg.list_models.return_value = []
        with patch("domain.infrastructure.model_registry.get_model_registry", return_value=reg):
            client = get_test_client()
            resp = client.get("/registry/models")
            assert resp.status_code == 200
            data = _data(resp)
            assert data["count"] == 0


class TestRegistryGetModel:
    def test_get_model_found(self):
        with patch(
            "domain.infrastructure.model_registry.get_model_registry",
            return_value=_make_registry(),
        ):
            client = get_test_client()
            resp = client.get("/registry/models/m1")
            assert resp.status_code == 200
            data = _data(resp)
            assert data["model_id"] == "m1"

    def test_get_model_not_found(self):
        with patch(
            "domain.infrastructure.model_registry.get_model_registry",
            return_value=_make_registry(),
        ):
            client = get_test_client()
            resp = client.get("/registry/models/nonexistent")
            assert resp.status_code == 404


class TestRegistryBest:
    def test_best_model(self):
        with patch(
            "domain.infrastructure.model_registry.get_model_registry",
            return_value=_make_registry(),
        ):
            client = get_test_client()
            resp = client.get("/registry/best")
            assert resp.status_code == 200
            data = _data(resp)
            assert "models_loaded" in data
            assert data["models_registered"] == 2


class TestRegistryStats:
    def test_stats(self):
        with patch(
            "domain.infrastructure.model_registry.get_model_registry",
            return_value=_make_registry(),
        ):
            client = get_test_client()
            resp = client.get("/registry/stats")
            assert resp.status_code == 200
            data = _data(resp)
            assert "models_loaded" in data
            assert data["healthy"] is True


class TestRegistryArtifactsLineage:
    """GET /registry/artifacts?lineage=true projects provenance parents."""

    def _patch_list(self, **expected):
        return patch(
            "domain.infrastructure.artifact_registry.list_artifacts",
            side_effect=lambda kind=None, q=None, **kw: (
                [{"id": "a", "name": "a", "parents": ["dataset:ds1"]}]
                if kw.get("with_provenance")
                else [{"id": "a", "name": "a"}]
            ),
        )

    def test_lineage_param_off_by_default(self):
        with self._patch_list() as mock_list:
            client = get_test_client()
            resp = client.get("/registry/artifacts")
            assert resp.status_code == 200
            mock_list.assert_called_once_with(None, None, with_provenance=False)
            [artifact] = _data(resp)["artifacts"]
            assert "parents" not in artifact

    def test_lineage_true_attaches_parents(self):
        with self._patch_list():
            client = get_test_client()
            resp = client.get("/registry/artifacts", params={"lineage": "true"})
            assert resp.status_code == 200
            [artifact] = _data(resp)["artifacts"]
            assert artifact["parents"] == ["dataset:ds1"]

    def test_unknown_kind_still_400(self):
        client = get_test_client()
        resp = client.get("/registry/artifacts", params={"kind": "nope"})
        assert resp.status_code == 400
