"""Tests for domains.infrastructure.deployment — enums, dataclass, manager init."""

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


class TestDeploymentStatus:
    def test_values(self):
        assert DeploymentStatus.PENDING.value == "pending"
        assert DeploymentStatus.IN_PROGRESS.value == "in_progress"
        assert DeploymentStatus.COMPLETED.value == "completed"
        assert DeploymentStatus.FAILED.value == "failed"
        assert DeploymentStatus.ROLLED_BACK.value == "rolled_back"

    def test_all_members(self):
        assert len(DeploymentStatus) == 5


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
