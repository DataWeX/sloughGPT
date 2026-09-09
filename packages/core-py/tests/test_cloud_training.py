"""Tests for cloud training providers."""

import pytest
from domains.training.cloud import (
    CloudTrainingConfig,
    CloudTrainingStatus,
    LocalTrainingProvider,
    AWSSageMakerProvider,
    GCPVertexProvider,
    get_provider,
)


class TestCloudTrainingConfig:
    def test_defaults(self):
        config = CloudTrainingConfig()
        assert config.provider == "local"
        assert config.region == "us-east-1"
        assert config.instance_type == "ml.g4dn.xlarge"

    def test_custom(self):
        config = CloudTrainingConfig(provider="aws", region="eu-west-1")
        assert config.provider == "aws"
        assert config.region == "eu-west-1"


class TestCloudTrainingStatus:
    def test_defaults(self):
        status = CloudTrainingStatus(job_id="j1", provider="local", status="completed")
        assert status.progress == 0.0
        assert status.error == ""


class TestLocalTrainingProvider:
    def test_submit_job(self):
        provider = LocalTrainingProvider()
        config = CloudTrainingConfig()
        job_id = provider.submit_job(config, "dataset.txt", "train.py", {})
        assert job_id == "local_001"

    def test_get_status(self):
        provider = LocalTrainingProvider()
        status = provider.get_status("j1")
        assert status.status == "completed"

    def test_cancel_job(self):
        provider = LocalTrainingProvider()
        assert provider.cancel_job("j1") is True

    def test_list_jobs(self):
        provider = LocalTrainingProvider()
        assert provider.list_jobs() == []


class TestGetProvider:
    def test_local(self):
        provider = get_provider("local")
        assert isinstance(provider, LocalTrainingProvider)

    def test_unknown(self):
        with pytest.raises(ValueError):
            get_provider("unknown")

    def test_aws(self):
        provider = get_provider("aws")
        assert isinstance(provider, AWSSageMakerProvider)

    def test_gcp(self):
        provider = get_provider("gcp")
        assert isinstance(provider, GCPVertexProvider)
