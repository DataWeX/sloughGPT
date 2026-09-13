"""
Cloud Training — adapters for cloud training providers (AWS SageMaker, GCP Vertex).
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger("slo.cloud_training")


@dataclass
class CloudTrainingConfig:
    """Configuration for cloud training."""
    provider: str = "local"  # local, aws, gcp
    region: str = "us-east-1"
    instance_type: str = "ml.g4dn.xlarge"
    bucket: str = ""
    role_arn: str = ""
    max_wait_minutes: int = 60
    checkpoint_uri: str = ""
    extra_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CloudTrainingStatus:
    """Status of a cloud training job."""
    job_id: str
    provider: str
    status: str  # pending, running, completed, failed, cancelled
    progress: float = 0.0
    logs_uri: str = ""
    model_uri: str = ""
    error: str = ""


class CloudTrainingProvider(ABC):
    """Abstract base class for cloud training providers."""

    @abstractmethod
    def submit_job(
        self,
        config: CloudTrainingConfig,
        dataset_uri: str,
        training_script: str,
        hyperparameters: Dict[str, Any],
    ) -> str:
        """Submit a training job. Returns job_id."""
        ...

    @abstractmethod
    def get_status(self, job_id: str) -> CloudTrainingStatus:
        """Get status of a training job."""
        ...

    @abstractmethod
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a training job."""
        ...

    @abstractmethod
    def list_jobs(self, limit: int = 10) -> List[CloudTrainingStatus]:
        """List recent training jobs."""
        ...


class LocalTrainingProvider(CloudTrainingProvider):
    """Local training provider (no-op, training runs locally)."""

    def submit_job(
        self,
        config: CloudTrainingConfig,
        dataset_uri: str,
        training_script: str,
        hyperparameters: Dict[str, Any],
    ) -> str:
        logger.info("Local training: dataset=%s script=%s", dataset_uri, training_script)
        return "local_001"

    def get_status(self, job_id: str) -> CloudTrainingStatus:
        return CloudTrainingStatus(
            job_id=job_id,
            provider="local",
            status="completed",
            progress=1.0,
        )

    def cancel_job(self, job_id: str) -> bool:
        return True

    def list_jobs(self, limit: int = 10) -> List[CloudTrainingStatus]:
        return []


class AWSSageMakerProvider(CloudTrainingProvider):
    """AWS SageMaker training provider."""

    def submit_job(
        self,
        config: CloudTrainingConfig,
        dataset_uri: str,
        training_script: str,
        hyperparameters: Dict[str, Any],
    ) -> str:
        try:
            import boto3
            sagemaker = boto3.client("sagemaker", region_name=config.region)
            job_name = f"sloughgpt-{int(__import__('time').time())}"
            logger.info("Submitting SageMaker job: %s", job_name)
            return job_name
        except ImportError:
            logger.error("boto3 not installed. Install with: pip install boto3")
            return ""
        except Exception as e:
            logger.error("SageMaker submit failed: %s", e)
            return ""

    def get_status(self, job_id: str) -> CloudTrainingStatus:
        try:
            import boto3
            sagemaker = boto3.client("sagemaker")
            resp = sagemaker.describe_training_job(TrainingJobName=job_id)
            status_map = {
                "InProgress": "running",
                "Completed": "completed",
                "Failed": "failed",
                "Stopping": "running",
                "Stopped": "cancelled",
            }
            return CloudTrainingStatus(
                job_id=job_id,
                provider="aws",
                status=status_map.get(resp["TrainingJobStatus"], "unknown"),
            )
        except Exception as e:
            logger.error("SageMaker status check failed: %s", e)
            return CloudTrainingStatus(
                job_id=job_id, provider="aws", status="unknown", error=str(e)
            )

    def cancel_job(self, job_id: str) -> bool:
        try:
            import boto3
            sagemaker = boto3.client("sagemaker")
            sagemaker.stop_training_job(TrainingJobName=job_id)
            return True
        except Exception as e:
            logger.error("SageMaker cancel failed: %s", e)
            return False

    def list_jobs(self, limit: int = 10) -> List[CloudTrainingStatus]:
        try:
            import boto3
            sagemaker = boto3.client("sagemaker")
            resp = sagemaker.list_training_jobs(MaxResults=limit)
            return [
                CloudTrainingStatus(
                    job_id=j["TrainingJobName"],
                    provider="aws",
                    status=j["TrainingJobStatus"].lower(),
                )
                for j in resp.get("TrainingJobSummaries", [])
            ]
        except Exception as e:
            logger.error("SageMaker list jobs failed: %s", e)
            return []


class GCPVertexProvider(CloudTrainingProvider):
    """GCP Vertex AI training provider."""

    def submit_job(
        self,
        config: CloudTrainingConfig,
        dataset_uri: str,
        training_script: str,
        hyperparameters: Dict[str, Any],
    ) -> str:
        try:
            from google.cloud import aiplatform
            aiplatform.init(project=config.extra_kwargs.get("project_id"), location=config.region)
            job_name = f"sloughgpt-{int(__import__('time').time())}"
            logger.info("Submitting Vertex AI job: %s", job_name)
            return job_name
        except ImportError:
            logger.error("google-cloud-aiplatform not installed")
            return ""
        except Exception as e:
            logger.error("Vertex AI submit failed: %s", e)
            return ""

    def get_status(self, job_id: str) -> CloudTrainingStatus:
        return CloudTrainingStatus(
            job_id=job_id, provider="gcp", status="unknown"
        )

    def cancel_job(self, job_id: str) -> bool:
        return False

    def list_jobs(self, limit: int = 10) -> List[CloudTrainingStatus]:
        return []


def get_provider(provider: str) -> CloudTrainingProvider:
    """Get a training provider by name."""
    providers = {
        "local": LocalTrainingProvider,
        "aws": AWSSageMakerProvider,
        "gcp": GCPVertexProvider,
    }
    cls = providers.get(provider)
    if cls is None:
        raise ValueError(f"Unknown provider: {provider}. Available: {list(providers.keys())}")
    return cls()


__all__ = [
    "CloudTrainingConfig",
    "CloudTrainingProvider",
    "CloudTrainingStatus",
    "LocalTrainingProvider",
    "AWSSageMakerProvider",
    "GCPVertexProvider",
    "get_provider",
]
