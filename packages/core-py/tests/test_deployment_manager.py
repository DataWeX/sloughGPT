"""Tests for domains/infrastructure/deployment/__init__.py."""

import asyncio

import pytest

from domains.infrastructure.deployment import (
    ComponentException,
    Deployment,
    DeploymentEnvironment,
    DeploymentManager,
    DeploymentStatus,
)


@pytest.fixture
def fast_sleep(monkeypatch):
    real_sleep = asyncio.sleep

    async def fake_sleep(delay):
        await real_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)


@pytest.fixture
def manager():
    return DeploymentManager()


def _mk_deployment(did="d1", env=DeploymentEnvironment.DEVELOPMENT):
    return Deployment(
        deployment_id=did,
        environment=env,
        status=DeploymentStatus.PENDING,
        config={},
        created_at=1.0,
        started_at=None,
        completed_at=None,
        error_message=None,
    )


class TestDeploymentEnums:
    def test_environments(self):
        assert DeploymentEnvironment("production") is DeploymentEnvironment.PRODUCTION

    def test_statuses(self):
        assert DeploymentStatus("rolled_back") is DeploymentStatus.ROLLED_BACK


class TestDeploymentManager:
    async def test_initialize(self, manager):
        await manager.initialize()
        assert manager.is_initialized is True

    async def test_initialize_failure_raises(self, manager, monkeypatch):
        def boom(msg, **kwargs):
            raise RuntimeError("init failed")

        monkeypatch.setattr(manager.logger, "info", boom)
        with pytest.raises(ComponentException):
            await manager.initialize()

    async def test_shutdown_no_active(self, manager):
        await manager.initialize()
        await manager.shutdown()
        assert manager.is_initialized is False

    async def test_shutdown_cancels_active(self, manager, fast_sleep):
        manager.deployments["d1"] = _mk_deployment()
        manager.active_deployments["d1"] = asyncio.create_task(
            _never_completes()
        )
        await manager.shutdown()
        assert manager.active_deployments["d1"].done()

    async def test_shutdown_failure_raises(self, manager, monkeypatch):
        def boom(msg, **kwargs):
            raise RuntimeError("shutdown failed")

        monkeypatch.setattr(manager.logger, "info", boom)
        with pytest.raises(ComponentException):
            await manager.shutdown()

    async def test_scale(self, manager):
        assert await manager.scale("svc1", 3) is True

    async def test_deploy_development_completes(self, manager, fast_sleep):
        did = await manager.deploy({"build": "v1"}, "development")
        assert did.startswith("deploy_")
        task = manager.active_deployments[did]
        await asyncio.gather(task, return_exceptions=True)
        d = manager.deployments[did]
        assert d.status is DeploymentStatus.COMPLETED
        assert d.completed_at is not None
        assert manager.stats["total_deployments"] == 1
        assert manager.stats["successful_deployments"] == 1
        assert did not in manager.active_deployments

    async def test_deploy_staging_requires_approval(self, manager, fast_sleep):
        did = await manager.deploy({}, "staging")
        task = manager.active_deployments[did]
        await asyncio.gather(task, return_exceptions=True)
        assert manager.deployments[did].status is DeploymentStatus.COMPLETED

    async def test_deploy_invalid_environment_raises(self, manager):
        with pytest.raises(ComponentException, match="Deployment start failed"):
            await manager.deploy({}, "mars")

    async def test_deploy_failure_marks_failed(self, manager, fast_sleep, monkeypatch):
        async def boom(deployment):
            raise RuntimeError("deploy step boom")

        monkeypatch.setattr(manager, "_deploy_application", boom)
        did = await manager.deploy({}, "development")
        task = manager.active_deployments[did]
        await asyncio.gather(task, return_exceptions=True)
        d = manager.deployments[did]
        assert d.status is DeploymentStatus.FAILED
        assert "boom" in d.error_message
        assert manager.stats["failed_deployments"] == 1

    async def test_get_deployment_status(self, manager, fast_sleep):
        did = await manager.deploy({}, "development")
        status = await manager.get_deployment_status(did)
        assert status["deployment_id"] == did
        assert status["environment"] == "development"

    async def test_get_deployment_status_missing_raises(self, manager):
        with pytest.raises(ComponentException, match="not found"):
            await manager.get_deployment_status("nope")

    async def test_rollback_not_found_raises(self, manager):
        with pytest.raises(ComponentException, match="not found"):
            await manager.rollback("nope")

    async def test_rollback_inactive(self, manager):
        manager.deployments["d1"] = _mk_deployment()
        assert await manager.rollback("d1") is True
        assert manager.deployments["d1"].status is DeploymentStatus.ROLLED_BACK
        assert manager.stats["rolled_back_deployments"] == 1

    async def test_rollback_cancels_active(self, manager, fast_sleep):
        manager.deployments["d1"] = _mk_deployment()
        manager.active_deployments["d1"] = asyncio.create_task(_never_completes())
        assert await manager.rollback("d1") is True
        assert manager.deployments["d1"].status is DeploymentStatus.ROLLED_BACK
        assert "d1" not in manager.active_deployments

    async def test_get_deployment_history(self, manager, fast_sleep):
        did1 = await manager.deploy({}, "development")
        await asyncio.gather(
            manager.active_deployments[did1], return_exceptions=True
        )
        did2 = await manager.deploy({}, "staging")
        await asyncio.gather(
            manager.active_deployments[did2], return_exceptions=True
        )
        history = await manager.get_deployment_history()
        assert len(history) == 2
        assert history[0]["created_at"] >= history[1]["created_at"]

    async def test_get_deployment_history_filtered(self, manager, fast_sleep):
        did1 = await manager.deploy({}, "development")
        await asyncio.gather(
            manager.active_deployments[did1], return_exceptions=True
        )
        history = await manager.get_deployment_history(environment="staging")
        assert history == []

    async def test_get_deployment_history_limit(self, manager, fast_sleep):
        for _ in range(3):
            did = await manager.deploy({}, "development")
            await asyncio.gather(
                manager.active_deployments[did], return_exceptions=True
            )
        history = await manager.get_deployment_history(limit=2)
        assert len(history) == 2

    async def test_deploy_health_check_runs(self, manager, fast_sleep):
        did = await manager.deploy({}, "development")
        await asyncio.gather(
            manager.active_deployments[did], return_exceptions=True
        )
        assert manager.deployments[did].status is DeploymentStatus.COMPLETED


async def _never_completes():
    await asyncio.Event().wait()
