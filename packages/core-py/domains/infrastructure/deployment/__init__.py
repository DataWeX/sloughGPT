"""
Deployment Manager Implementation

This module provides deployment management capabilities for
different environments and deployment strategies.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

from ...__init__ import BaseComponent, ComponentException

try:
    from ...__init__ import IDeploymentManager
except ImportError:
    class IDeploymentManager:
        pass


class DeploymentEnvironment(Enum):
    """Deployment environments"""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


class DeploymentStatus(Enum):
    """Deployment status"""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


@dataclass
class Deployment:
    """Deployment information"""

    deployment_id: str
    environment: DeploymentEnvironment
    status: DeploymentStatus
    config: Dict[str, Any]
    created_at: float
    started_at: Optional[float]
    completed_at: Optional[float]
    error_message: Optional[str]


class DeploymentManager(BaseComponent, IDeploymentManager):
    """Advanced deployment management system"""

    def __init__(self) -> None:
        super().__init__("deployment_manager")
        self.logger = logging.getLogger(f"slo.{self.component_name}")

        # Deployment tracking
        self.deployments: Dict[str, Deployment] = {}
        self.active_deployments: Dict[str, asyncio.Task] = {}
        self._approval_events: Dict[str, asyncio.Event] = {}
        self._denied_deployments: set = set()

        # Environment configurations
        self.environment_configs = {
            DeploymentEnvironment.DEVELOPMENT: {
                "auto_deploy": True,
                "require_approval": False,
                "health_check_timeout": 30,
                "approval_timeout": 10,
            },
            DeploymentEnvironment.STAGING: {
                "auto_deploy": True,
                "require_approval": True,
                "health_check_timeout": 60,
                "approval_timeout": 30,
            },
            DeploymentEnvironment.PRODUCTION: {
                "auto_deploy": False,
                "require_approval": True,
                "health_check_timeout": 120,
                "approval_timeout": 60,
            },
        }

        # Statistics
        self.stats = {
            "total_deployments": 0,
            "successful_deployments": 0,
            "failed_deployments": 0,
            "rolled_back_deployments": 0,
        }

        # Service tracking
        self._service_replicas: Dict[str, int] = {}

        self.is_initialized = False

    async def initialize(self) -> None:
        """Initialize deployment manager"""
        try:
            self.logger.info("Initializing Deployment Manager...",
                extra={"tag": "INFRA"})
            self.is_initialized = True
            self.logger.info("Deployment Manager initialized successfully",
                extra={"tag": "INFRA"})

        except Exception as e:
            self.logger.error("Failed to initialize Deployment Manager: %s", e,
                extra={"tag": "INFRA"})
            raise ComponentException(f"Deployment Manager initialization failed: {e}")

    async def shutdown(self) -> None:
        """Shutdown deployment manager"""
        try:
            self.logger.info("Shutting down Deployment Manager...",
                extra={"tag": "INFRA"})

            # Cancel active deployments
            for deployment_id, task in self.active_deployments.items():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

            self.is_initialized = False
            self.logger.info("Deployment Manager shutdown successfully",
                extra={"tag": "INFRA"})

        except Exception as e:
            self.logger.error("Failed to shutdown Deployment Manager: %s", e,
                extra={"tag": "INFRA"})
            raise ComponentException(f"Deployment Manager shutdown failed: {e}")

    async def scale(self, service_id: str, replicas: int) -> bool:
        """Scale a service to the given number of replicas."""
        if replicas < 0:
            raise ComponentException("Replica count cannot be negative")
        old = self._service_replicas.get(service_id, 1)
        self._service_replicas[service_id] = replicas
        self.logger.info(
            "Scaled service %s: %d -> %d replicas",
            service_id, old, replicas, extra={"tag": "INFRA"},
        )
        return True

    def get_service_replicas(self, service_id: str) -> int:
        """Get current replica count for a service."""
        return self._service_replicas.get(service_id, 1)

    async def deploy(self, config: Dict[str, Any], environment: str) -> str:
        """Deploy to environment"""
        try:
            deployment_id = f"deploy_{int(time.time())}_{len(self.deployments)}"

            # Create deployment record
            deployment = Deployment(
                deployment_id=deployment_id,
                environment=DeploymentEnvironment(environment),
                status=DeploymentStatus.PENDING,
                config=config,
                created_at=time.time(),
                started_at=None,
                completed_at=None,
                error_message=None,
            )

            self.deployments[deployment_id] = deployment
            self.stats["total_deployments"] += 1

            # Start deployment task
            task = asyncio.create_task(self._execute_deployment(deployment))
            self.active_deployments[deployment_id] = task

            self.logger.info("Started deployment %s to %s", deployment_id, environment,
                extra={"tag": "INFRA"})
            return deployment_id

        except Exception as e:
            self.logger.error("Failed to start deployment: %s", e,
                extra={"tag": "INFRA"})
            raise ComponentException(f"Deployment start failed: {e}")

    async def get_deployment_status(self, deployment_id: str) -> Dict[str, Any]:
        """Get deployment status"""
        if deployment_id not in self.deployments:
            raise ComponentException(f"Deployment {deployment_id} not found")

        deployment = self.deployments[deployment_id]

        return {
            "deployment_id": deployment.deployment_id,
            "environment": deployment.environment.value,
            "status": deployment.status.value,
            "created_at": deployment.created_at,
            "started_at": deployment.started_at,
            "completed_at": deployment.completed_at,
            "error_message": deployment.error_message,
            "is_active": deployment_id in self.active_deployments,
        }

    async def rollback(self, deployment_id: str) -> bool:
        """Rollback deployment"""
        try:
            if deployment_id not in self.deployments:
                raise ComponentException(f"Deployment {deployment_id} not found")

            deployment = self.deployments[deployment_id]

            # Cancel active deployment
            if deployment_id in self.active_deployments:
                task = self.active_deployments[deployment_id]
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                del self.active_deployments[deployment_id]

            # Update status
            deployment.status = DeploymentStatus.ROLLED_BACK
            deployment.completed_at = time.time()

            self.stats["rolled_back_deployments"] += 1

            self.logger.info("Rolled back deployment %s", deployment_id,
                extra={"tag": "INFRA"})
            return True

        except Exception as e:
            self.logger.error("Failed to rollback deployment %s: %s", deployment_id, e,
                extra={"tag": "INFRA"})
            raise ComponentException(f"Rollback failed: {e}")

    async def get_deployment_history(
        self, environment: Optional[str] = None, limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get deployment history"""
        deployments = list(self.deployments.values())

        # Filter by environment
        if environment:
            deployments = [d for d in deployments if d.environment.value == environment]

        # Sort by creation time (newest first)
        deployments.sort(key=lambda d: d.created_at, reverse=True)

        # Limit results
        deployments = deployments[:limit]

        return [
            {
                "deployment_id": d.deployment_id,
                "environment": d.environment.value,
                "status": d.status.value,
                "created_at": d.created_at,
                "duration": (d.completed_at or time.time()) - d.created_at,
            }
            for d in deployments
        ]

    # Private helper methods

    async def _execute_deployment(self, deployment: Deployment) -> None:
        """Execute deployment process"""
        try:
            # Update status
            deployment.status = DeploymentStatus.IN_PROGRESS
            deployment.started_at = time.time()

            env_config = self.environment_configs[deployment.environment]

            # Check approval requirement
            if env_config["require_approval"]:
                await self._wait_for_approval(deployment)

            # Execute deployment steps
            await self._prepare_deployment(deployment)
            await self._deploy_application(deployment)
            await self._run_health_checks(deployment)

            # Mark as completed
            deployment.status = DeploymentStatus.COMPLETED
            deployment.completed_at = time.time()
            self.stats["successful_deployments"] += 1

            self.logger.info("Deployment %s completed successfully", deployment.deployment_id,
                extra={"tag": "INFRA"})

        except Exception as e:
            # Mark as failed
            deployment.status = DeploymentStatus.FAILED
            deployment.completed_at = time.time()
            deployment.error_message = str(e)
            self.stats["failed_deployments"] += 1

            self.logger.error("Deployment %s failed: %s", deployment.deployment_id, e,
                extra={"tag": "INFRA"})

        finally:
            # Remove from active deployments
            if deployment.deployment_id in self.active_deployments:
                del self.active_deployments[deployment.deployment_id]

    async def _wait_for_approval(self, deployment: Deployment) -> None:
        """Wait for deployment approval via event signal.

        Approval can be granted externally by calling ``approve_deployment()``
        or denied via ``deny_deployment()``.  Times out after the environment's
        ``approval_timeout`` (falls back to ``health_check_timeout``) and raises.
        """
        event = asyncio.Event()
        self._approval_events[deployment.deployment_id] = event
        env_cfg = self.environment_configs[deployment.environment]
        timeout = env_cfg.get("approval_timeout", env_cfg["health_check_timeout"])

        self.logger.info(
            "Deployment %s awaiting approval (timeout=%ds)",
            deployment.deployment_id, timeout,
            extra={"tag": "INFRA"},
        )
        try:
            await asyncio.wait_for(event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            raise ComponentException(
                f"Deployment {deployment.deployment_id} approval timed out after {timeout}s"
            )
        finally:
            self._approval_events.pop(deployment.deployment_id, None)

        # Check if denied
        if deployment.deployment_id in self._denied_deployments:
            self._denied_deployments.discard(deployment.deployment_id)
            raise ComponentException(f"Deployment {deployment.deployment_id} was denied")

    def approve_deployment(self, deployment_id: str) -> None:
        """Approve a pending deployment."""
        event = self._approval_events.get(deployment_id)
        if event is None:
            raise ComponentException(f"No pending approval for {deployment_id}")
        event.set()

    def deny_deployment(self, deployment_id: str) -> None:
        """Deny a pending deployment."""
        self._denied_deployments.add(deployment_id)
        event = self._approval_events.get(deployment_id)
        if event is not None:
            event.set()

    async def _prepare_deployment(self, deployment: Deployment) -> None:
        """Prepare deployment environment.

        Validates config keys, creates a backup manifest of the current state,
        and records the preparation timestamp.
        """
        import json as _json
        from pathlib import Path

        self.logger.info("Preparing deployment %s", deployment.deployment_id,
            extra={"tag": "INFRA"})

        # Validate required config keys
        required = {"version", "image"}
        missing = required - set(deployment.config.keys())
        if missing:
            raise ComponentException(
                f"Missing required config keys: {', '.join(sorted(missing))}"
            )

        # Create deployment directory
        deploy_dir = Path("/tmp") / "deployments" / deployment.deployment_id
        deploy_dir.mkdir(parents=True, exist_ok=True)

        # Write backup manifest
        manifest = {
            "deployment_id": deployment.deployment_id,
            "environment": deployment.environment.value,
            "config": deployment.config,
            "prepared_at": time.time(),
        }
        (deploy_dir / "manifest.json").write_text(_json.dumps(manifest, indent=2))

        # Store deploy dir on deployment for later phases
        deployment.config["_deploy_dir"] = str(deploy_dir)
        deployment.config["_prepare_time"] = time.time()

    async def _deploy_application(self, deployment: Deployment) -> None:
        """Deploy the application.

        Writes a version marker file and simulates applying configuration
        by writing a runtime config file into the deployment directory.
        """
        import json as _json
        from pathlib import Path

        self.logger.info("Deploying application for %s", deployment.deployment_id,
            extra={"tag": "INFRA"})

        deploy_dir = Path(deployment.config.get("_deploy_dir", "/tmp"))
        deploy_dir.mkdir(parents=True, exist_ok=True)

        # Write version marker
        version = deployment.config.get("version", "unknown")
        (deploy_dir / "VERSION").write_text(version)

        # Write runtime config
        runtime = {
            "version": version,
            "image": deployment.config.get("image", ""),
            "deployed_at": time.time(),
            "environment": deployment.environment.value,
            "environment_config": {
                k: v for k, v in self.environment_configs[deployment.environment].items()
            },
        }
        (deploy_dir / "runtime.json").write_text(_json.dumps(runtime, indent=2))

        deployment.config["_deploy_time"] = time.time()

    async def _run_health_checks(self, deployment: Deployment) -> None:
        """Run post-deployment health checks.

        Verifies the deployment directory exists and the version marker was
        written correctly.  Repeats checks up to ``health_check_timeout / 2``
        times with 1s intervals.
        """
        import json as _json
        from pathlib import Path

        self.logger.info("Running health checks for %s", deployment.deployment_id,
            extra={"tag": "INFRA"})

        deploy_dir = Path(deployment.config.get("_deploy_dir", "/tmp"))
        version = deployment.config.get("version", "unknown")
        timeout = self.environment_configs[deployment.environment]["health_check_timeout"]
        max_attempts = max(1, timeout // 2)

        for attempt in range(1, max_attempts + 1):
            # Check version marker
            version_file = deploy_dir / "VERSION"
            if version_file.exists() and version_file.read_text().strip() == version:
                # Check runtime config
                runtime_file = deploy_dir / "runtime.json"
                if runtime_file.exists():
                    runtime = _json.loads(runtime_file.read_text())
                    if runtime.get("version") == version:
                        self.logger.info(
                            "Health check passed for %s (attempt %d/%d)",
                            deployment.deployment_id, attempt, max_attempts,
                            extra={"tag": "INFRA"},
                        )
                        return
            if attempt < max_attempts:
                await asyncio.sleep(1)

        raise ComponentException(
            f"Health checks failed for {deployment.deployment_id} after {max_attempts} attempts"
        )


__all__ = ["DeploymentEnvironment", "DeploymentStatus", "Deployment", "DeploymentManager"]
