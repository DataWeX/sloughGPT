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
    """Make asyncio.sleep non-blocking but leave asyncio.Event.wait alone."""
    real_sleep = asyncio.sleep

    async def fake_sleep(delay):
        await real_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)


@pytest.fixture
def manager():
    return DeploymentManager()


def _mk_deployment(did="d1", env=DeploymentEnvironment.DEVELOPMENT, config=None):
    if config is None:
        config = {"version": "v1", "image": "app:latest"}
    return Deployment(
        deployment_id=did,
        environment=env,
        status=DeploymentStatus.PENDING,
        config=config,
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
        assert manager.get_service_replicas("svc1") == 3

    async def test_scale_negative_raises(self, manager):
        with pytest.raises(ComponentException, match="cannot be negative"):
            await manager.scale("svc1", -1)

    async def test_scale_tracks_changes(self, manager):
        await manager.scale("svc1", 3)
        await manager.scale("svc1", 5)
        assert manager.get_service_replicas("svc1") == 5

    async def test_get_service_replicas_default(self, manager):
        assert manager.get_service_replicas("unknown") == 1

    async def test_deploy_development_completes(self, manager, fast_sleep):
        did = await manager.deploy({"version": "v1", "image": "app:v1"}, "development")
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
        did = await manager.deploy({"version": "v1", "image": "app:v1"}, "staging")
        # Let the task reach the approval wait point
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        manager.approve_deployment(did)
        task = manager.active_deployments[did]
        await asyncio.gather(task, return_exceptions=True)
        assert manager.deployments[did].status is DeploymentStatus.COMPLETED

    async def test_deploy_staging_denied(self, manager, fast_sleep):
        did = await manager.deploy({"version": "v1", "image": "app:v1"}, "staging")
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        manager.deny_deployment(did)
        task = manager.active_deployments[did]
        await asyncio.gather(task, return_exceptions=True)
        assert manager.deployments[did].status is DeploymentStatus.FAILED
        assert "denied" in manager.deployments[did].error_message.lower()

    async def test_deploy_invalid_environment_raises(self, manager):
        with pytest.raises(ComponentException, match="Deployment start failed"):
            await manager.deploy({"version": "v1", "image": "app:v1"}, "mars")

    async def test_deploy_missing_config_keys_fails(self, manager, fast_sleep):
        did = await manager.deploy({}, "development")
        task = manager.active_deployments[did]
        await asyncio.gather(task, return_exceptions=True)
        d = manager.deployments[did]
        assert d.status is DeploymentStatus.FAILED
        assert "Missing required config keys" in d.error_message

    async def test_deploy_failure_marks_failed(self, manager, fast_sleep, monkeypatch):
        async def boom(deployment):
            raise RuntimeError("deploy step boom")

        monkeypatch.setattr(manager, "_deploy_application", boom)
        did = await manager.deploy({"version": "v1", "image": "app:v1"}, "development")
        task = manager.active_deployments[did]
        await asyncio.gather(task, return_exceptions=True)
        d = manager.deployments[did]
        assert d.status is DeploymentStatus.FAILED
        assert "boom" in d.error_message
        assert manager.stats["failed_deployments"] == 1

    async def test_get_deployment_status(self, manager, fast_sleep):
        did = await manager.deploy({"version": "v1", "image": "app:v1"}, "development")
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
        did1 = await manager.deploy({"version": "v1", "image": "app:v1"}, "development")
        await asyncio.gather(
            manager.active_deployments[did1], return_exceptions=True
        )
        did2 = await manager.deploy({"version": "v2", "image": "app:v2"}, "development")
        await asyncio.gather(
            manager.active_deployments[did2], return_exceptions=True
        )
        history = await manager.get_deployment_history()
        assert len(history) == 2
        assert history[0]["created_at"] >= history[1]["created_at"]

    async def test_get_deployment_history_filtered(self, manager, fast_sleep):
        did1 = await manager.deploy({"version": "v1", "image": "app:v1"}, "development")
        await asyncio.gather(
            manager.active_deployments[did1], return_exceptions=True
        )
        history = await manager.get_deployment_history(environment="staging")
        assert history == []

    async def test_get_deployment_history_limit(self, manager, fast_sleep):
        for i in range(3):
            did = await manager.deploy({"version": f"v{i}", "image": "app:v1"}, "development")
            await asyncio.gather(
                manager.active_deployments[did], return_exceptions=True
            )
        history = await manager.get_deployment_history(limit=2)
        assert len(history) == 2

    async def test_deploy_health_check_runs(self, manager, fast_sleep):
        did = await manager.deploy({"version": "v1", "image": "app:v1"}, "development")
        await asyncio.gather(
            manager.active_deployments[did], return_exceptions=True
        )
        assert manager.deployments[did].status is DeploymentStatus.COMPLETED


class TestApprovalFlow:
    async def test_approve_deployment(self, manager, fast_sleep):
        did = await manager.deploy({"version": "v1", "image": "app:v1"}, "staging")
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        manager.approve_deployment(did)
        task = manager.active_deployments[did]
        await asyncio.gather(task, return_exceptions=True)
        assert manager.deployments[did].status is DeploymentStatus.COMPLETED

    async def test_approve_unknown_raises(self, manager):
        with pytest.raises(ComponentException, match="No pending approval"):
            manager.approve_deployment("nonexistent")

    async def test_deny_deployment(self, manager, fast_sleep):
        did = await manager.deploy({"version": "v1", "image": "app:v1"}, "production")
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        manager.deny_deployment(did)
        task = manager.active_deployments[did]
        await asyncio.gather(task, return_exceptions=True)
        assert manager.deployments[did].status is DeploymentStatus.FAILED
        assert "denied" in manager.deployments[did].error_message.lower()


class TestDeployArtifacts:
    async def test_creates_deploy_directory(self, manager, fast_sleep):
        did = await manager.deploy({"version": "v1", "image": "app:v1"}, "development")
        task = manager.active_deployments[did]
        await asyncio.gather(task, return_exceptions=True)
        from pathlib import Path
        deploy_dir = Path("/tmp") / "deployments" / did
        assert deploy_dir.exists()
        assert (deploy_dir / "VERSION").exists()
        assert (deploy_dir / "VERSION").read_text().strip() == "v1"
        assert (deploy_dir / "runtime.json").exists()
        assert (deploy_dir / "manifest.json").exists()

    async def test_health_check_failure(self, manager, fast_sleep, monkeypatch):
        async def bad_check(deployment):
            raise ComponentException("health check failed")

        monkeypatch.setattr(manager, "_run_health_checks", bad_check)
        did = await manager.deploy({"version": "v1", "image": "app:v1"}, "development")
        task = manager.active_deployments[did]
        await asyncio.gather(task, return_exceptions=True)
        assert manager.deployments[did].status is DeploymentStatus.FAILED


async def _never_completes():
    await asyncio.Event().wait()
