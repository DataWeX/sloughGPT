"""Tests for domains.infrastructure.deployment — enums, dataclass, manager init."""

import asyncio
import pytest
from domains.infrastructure.deployment import (
    DeploymentEnvironment, DeploymentStatus, Deployment, DeploymentManager,
)


class TestDeploymentEnvironment:
    def test_values(self):
        assert DeploymentEnvironment.DEVELOPMENT.value == "development"
        assert DeploymentEnvironment.STAGING.value == "staging"
        assert DeploymentEnvironment.PRODUCTION.value == "production"
        assert DeploymentEnvironment.TESTING.value == "testing"

    def test_all_members(self):
        assert len(DeploymentEnvironment) == 4

    def test_member_names(self):
        names = {e.name for e in DeploymentEnvironment}
        assert names == {"DEVELOPMENT", "STAGING", "PRODUCTION", "TESTING"}

    def test_from_value(self):
        assert DeploymentEnvironment("development") == DeploymentEnvironment.DEVELOPMENT
        assert DeploymentEnvironment("staging") == DeploymentEnvironment.STAGING
        assert DeploymentEnvironment("production") == DeploymentEnvironment.PRODUCTION
        assert DeploymentEnvironment("testing") == DeploymentEnvironment.TESTING

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError):
            DeploymentEnvironment("invalid")

    def test_identity(self):
        assert DeploymentEnvironment.DEVELOPMENT is DeploymentEnvironment.DEVELOPMENT

    def test_not_equal_to_string(self):
        assert DeploymentEnvironment.DEVELOPMENT != "development"

    def test_iteration(self):
        vals = [e.value for e in DeploymentEnvironment]
        assert "development" in vals
        assert "production" in vals

    def test_hash(self):
        s = {DeploymentEnvironment.DEVELOPMENT, DeploymentEnvironment.PRODUCTION}
        assert len(s) == 2

    def test_repr(self):
        r = repr(DeploymentEnvironment.DEVELOPMENT)
        assert "DEVELOPMENT" in r


class TestDeploymentStatus:
    def test_values(self):
        assert DeploymentStatus.PENDING.value == "pending"
        assert DeploymentStatus.IN_PROGRESS.value == "in_progress"
        assert DeploymentStatus.COMPLETED.value == "completed"
        assert DeploymentStatus.FAILED.value == "failed"
        assert DeploymentStatus.ROLLED_BACK.value == "rolled_back"

    def test_all_members(self):
        assert len(DeploymentStatus) == 5

    def test_member_names(self):
        names = {s.name for s in DeploymentStatus}
        assert names == {"PENDING", "IN_PROGRESS", "COMPLETED", "FAILED", "ROLLED_BACK"}

    def test_from_value(self):
        assert DeploymentStatus("pending") == DeploymentStatus.PENDING
        assert DeploymentStatus("in_progress") == DeploymentStatus.IN_PROGRESS
        assert DeploymentStatus("completed") == DeploymentStatus.COMPLETED

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError):
            DeploymentStatus("nonexistent")

    def test_identity(self):
        assert DeploymentStatus.PENDING is DeploymentStatus.PENDING

    def test_not_equal_to_string(self):
        assert DeploymentStatus.PENDING != "pending"

    def test_iteration(self):
        vals = [s.value for s in DeploymentStatus]
        assert "pending" in vals
        assert "failed" in vals

    def test_hash(self):
        s = {DeploymentStatus.PENDING, DeploymentStatus.FAILED}
        assert len(s) == 2

    def test_repr(self):
        r = repr(DeploymentStatus.COMPLETED)
        assert "COMPLETED" in r

    def test_status_ordering(self):
        statuses = [DeploymentStatus.PENDING, DeploymentStatus.IN_PROGRESS,
                    DeploymentStatus.COMPLETED]
        assert len(statuses) == 3


class TestDeployment:
    def test_fields(self):
        d = Deployment(
            deployment_id="dep-1",
            environment=DeploymentEnvironment.PRODUCTION,
            status=DeploymentStatus.PENDING,
            config={"image": "v1"},
            created_at=1000.0,
            started_at=None,
            completed_at=None,
            error_message=None,
        )
        assert d.deployment_id == "dep-1"
        assert d.environment == DeploymentEnvironment.PRODUCTION
        assert d.config["image"] == "v1"

    def test_completed_deployment(self):
        d = Deployment(
            deployment_id="dep-2",
            environment=DeploymentEnvironment.STAGING,
            status=DeploymentStatus.COMPLETED,
            config={},
            created_at=1000.0,
            started_at=1001.0,
            completed_at=1010.0,
            error_message=None,
        )
        assert d.started_at is not None
        assert d.completed_at is not None

    def test_failed_deployment(self):
        d = Deployment(
            deployment_id="dep-3",
            environment=DeploymentEnvironment.DEVELOPMENT,
            status=DeploymentStatus.FAILED,
            config={"version": "1.0"},
            created_at=1000.0,
            started_at=1001.0,
            completed_at=1005.0,
            error_message="Build failed",
        )
        assert d.error_message == "Build failed"
        assert d.status == DeploymentStatus.FAILED

    def test_rolled_back_deployment(self):
        d = Deployment(
            deployment_id="dep-4",
            environment=DeploymentEnvironment.PRODUCTION,
            status=DeploymentStatus.ROLLED_BACK,
            config={},
            created_at=1000.0,
            started_at=1001.0,
            completed_at=1020.0,
            error_message=None,
        )
        assert d.status == DeploymentStatus.ROLLED_BACK

    def test_config_is_mutable(self):
        d = Deployment(
            deployment_id="dep-5",
            environment=DeploymentEnvironment.DEVELOPMENT,
            status=DeploymentStatus.PENDING,
            config={"image": "v1"},
            created_at=1000.0,
            started_at=None,
            completed_at=None,
            error_message=None,
        )
        d.config["version"] = "2.0"
        assert d.config["version"] == "2.0"

    def test_none_timestamps(self):
        d = Deployment(
            deployment_id="dep-6",
            environment=DeploymentEnvironment.DEVELOPMENT,
            status=DeploymentStatus.PENDING,
            config={},
            created_at=1000.0,
            started_at=None,
            completed_at=None,
            error_message=None,
        )
        assert d.started_at is None
        assert d.completed_at is None

    def test_in_progress_timestamps(self):
        d = Deployment(
            deployment_id="dep-7",
            environment=DeploymentEnvironment.STAGING,
            status=DeploymentStatus.IN_PROGRESS,
            config={},
            created_at=1000.0,
            started_at=1001.0,
            completed_at=None,
            error_message=None,
        )
        assert d.started_at is not None
        assert d.completed_at is None


class TestDeploymentManager:
    def setup_method(self):
        self.mgr = DeploymentManager()

    def test_defaults(self):
        assert self.mgr.deployments == {}
        assert self.mgr.is_initialized is False

    def test_environment_configs(self):
        cfgs = self.mgr.environment_configs
        assert DeploymentEnvironment.DEVELOPMENT in cfgs
        assert DeploymentEnvironment.PRODUCTION in cfgs
        assert cfgs[DeploymentEnvironment.PRODUCTION]["require_approval"] is True

    def test_stats_init(self):
        assert self.mgr.stats["total_deployments"] == 0
        assert self.mgr.stats["successful_deployments"] == 0

    def test_component_name(self):
        assert self.mgr.component_name == "deployment_manager"

    def test_dev_auto_deploy(self):
        cfg = self.mgr.environment_configs[DeploymentEnvironment.DEVELOPMENT]
        assert cfg["auto_deploy"] is True

    def test_prod_no_auto_deploy(self):
        cfg = self.mgr.environment_configs[DeploymentEnvironment.PRODUCTION]
        assert cfg["auto_deploy"] is False

    def test_staging_requires_approval(self):
        cfg = self.mgr.environment_configs[DeploymentEnvironment.STAGING]
        assert cfg["require_approval"] is True

    def test_dev_no_approval(self):
        cfg = self.mgr.environment_configs[DeploymentEnvironment.DEVELOPMENT]
        assert cfg["require_approval"] is False

    def test_health_check_timeout_dev(self):
        cfg = self.mgr.environment_configs[DeploymentEnvironment.DEVELOPMENT]
        assert cfg["health_check_timeout"] == 30

    def test_health_check_timeout_prod(self):
        cfg = self.mgr.environment_configs[DeploymentEnvironment.PRODUCTION]
        assert cfg["health_check_timeout"] == 120

    def test_active_deployments_init(self):
        assert self.mgr.active_deployments == {}

    def test_service_replicas_init(self):
        assert self.mgr._service_replicas == {}

    def test_approval_events_init(self):
        assert self.mgr._approval_events == {}

    def test_denied_deployments_init(self):
        assert self.mgr._denied_deployments == set()

    def test_stats_keys(self):
        assert "total_deployments" in self.mgr.stats
        assert "successful_deployments" in self.mgr.stats
        assert "failed_deployments" in self.mgr.stats
        assert "rolled_back_deployments" in self.mgr.stats

    def test_env_configs_count(self):
        assert len(self.mgr.environment_configs) == 3

    def test_approval_timeout_dev(self):
        cfg = self.mgr.environment_configs[DeploymentEnvironment.DEVELOPMENT]
        assert "approval_timeout" in cfg

    def test_approval_timeout_prod(self):
        cfg = self.mgr.environment_configs[DeploymentEnvironment.PRODUCTION]
        assert cfg["approval_timeout"] == 60


class TestDeploymentManagerAsync:
    @pytest.fixture
    def mgr(self):
        return DeploymentManager()

    @pytest.mark.asyncio
    async def test_initialize(self, mgr):
        await mgr.initialize()
        assert mgr.is_initialized is True

    @pytest.mark.asyncio
    async def test_shutdown(self, mgr):
        await mgr.initialize()
        await mgr.shutdown()
        assert mgr.is_initialized is False

    @pytest.mark.asyncio
    async def test_deploy(self, mgr):
        await mgr.initialize()
        deployment_id = await mgr.deploy(
            config={"image": "v1", "version": "1.0"},
            environment="development",
        )
        assert deployment_id.startswith("deploy_")
        assert deployment_id in mgr.deployments

    @pytest.mark.asyncio
    async def test_deploy_increments_total(self, mgr):
        await mgr.initialize()
        await mgr.deploy(
            config={"image": "v1", "version": "1.0"},
            environment="development",
        )
        assert mgr.stats["total_deployments"] == 1

    @pytest.mark.asyncio
    async def test_get_deployment_status(self, mgr):
        await mgr.initialize()
        deployment_id = await mgr.deploy(
            config={"image": "v1", "version": "1.0"},
            environment="development",
        )
        status = await mgr.get_deployment_status(deployment_id)
        assert status["deployment_id"] == deployment_id
        assert "environment" in status
        assert "status" in status

    @pytest.mark.asyncio
    async def test_get_deployment_status_nonexistent(self, mgr):
        await mgr.initialize()
        with pytest.raises(Exception):
            await mgr.get_deployment_status("nonexistent-id")

    @pytest.mark.asyncio
    async def test_scale(self, mgr):
        await mgr.initialize()
        result = await mgr.scale("my-service", 5)
        assert result is True
        assert mgr.get_service_replicas("my-service") == 5

    @pytest.mark.asyncio
    async def test_scale_default_replicas(self, mgr):
        await mgr.initialize()
        assert mgr.get_service_replicas("unknown-service") == 1

    @pytest.mark.asyncio
    async def test_get_deployment_history(self, mgr):
        await mgr.initialize()
        await mgr.deploy(
            config={"image": "v1", "version": "1.0"},
            environment="development",
        )
        history = await mgr.get_deployment_history()
        assert len(history) >= 1

    @pytest.mark.asyncio
    async def test_get_deployment_history_filter(self, mgr):
        await mgr.initialize()
        await mgr.deploy(
            config={"image": "v1", "version": "1.0"},
            environment="development",
        )
        history = await mgr.get_deployment_history(environment="development")
        assert len(history) >= 1

    @pytest.mark.asyncio
    async def test_approve_deployment(self, mgr):
        await mgr.initialize()
        # Set up a manual approval event
        event = asyncio.Event()
        mgr._approval_events["test-deploy"] = event
        mgr.approve_deployment("test-deploy")
        assert event.is_set()

    @pytest.mark.asyncio
    async def test_deny_deployment(self, mgr):
        await mgr.initialize()
        event = asyncio.Event()
        mgr._approval_events["test-deploy"] = event
        mgr.deny_deployment("test-deploy")
        assert "test-deploy" in mgr._denied_deployments
        assert event.is_set()

    @pytest.mark.asyncio
    async def test_approve_nonexistent_raises(self, mgr):
        await mgr.initialize()
        with pytest.raises(Exception):
            mgr.approve_deployment("nonexistent")

    @pytest.mark.asyncio
    async def test_rollback(self, mgr):
        await mgr.initialize()
        deployment_id = await mgr.deploy(
            config={"image": "v1", "version": "1.0"},
            environment="development",
        )
        # Wait for deployment to be in-flight
        await asyncio.sleep(0.01)
        result = await mgr.rollback(deployment_id)
        assert result is True
        assert mgr.deployments[deployment_id].status == DeploymentStatus.ROLLED_BACK

    @pytest.mark.asyncio
    async def test_rollback_nonexistent_raises(self, mgr):
        await mgr.initialize()
        with pytest.raises(Exception):
            await mgr.rollback("nonexistent-id")

    @pytest.mark.asyncio
    async def test_scale_negative_raises(self, mgr):
        await mgr.initialize()
        with pytest.raises(Exception):
            await mgr.scale("svc", -1)
