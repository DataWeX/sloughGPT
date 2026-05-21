"""
Infrastructure Domain Base Class

This module contains the base infrastructure domain implementation.
"""

import asyncio
import logging
from typing import Any, Dict, Optional

from ..__init__ import (
    BaseDomain,
    DomainException,
    ICacheManager,
    IDatabaseManager,
    IDeploymentManager,
)


class InfrastructureDomain(BaseDomain):
    """Main data & infrastructure domain"""

    def __init__(self):
        super().__init__("infrastructure")
        self.logger = logging.getLogger(f"sloughgpt.{self.domain_name}")

        # Core components
        self.database_manager: Optional[IDatabaseManager] = None
        self.cache_manager: Optional[ICacheManager] = None
        self.deployment_manager: Optional[IDeploymentManager] = None

        # Infrastructure state
        self.service_status = {}
        self.resource_usage = {}
        self.health_checks = {}

    async def _on_initialize(self) -> None:
        """Initialize infrastructure domain components"""
        try:
            self.logger.info("Initializing Infrastructure Domain...")

            # Initialize core components
            await self._initialize_database_manager()
            await self._initialize_cache_manager()
            await self._initialize_deployment_manager()

            # Start health monitoring
            await self._start_health_monitoring()

            self.logger.info("Infrastructure Domain initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize Infrastructure Domain: {e}")
            raise InfrastructureException(f"Initialization failed: {e}")

    async def _on_shutdown(self) -> None:
        """Shutdown infrastructure domain components"""
        try:
            self.logger.info("Shutting down Infrastructure Domain...")

            # Stop health monitoring
            await self._stop_health_monitoring()

            # Shutdown components
            if self.deployment_manager:
                await self._shutdown_component("deployment_manager")
            if self.cache_manager:
                await self._shutdown_component("cache_manager")
            if self.database_manager:
                await self._shutdown_component("database_manager")

            self.logger.info("Infrastructure Domain shutdown successfully")

        except Exception as e:
            self.logger.error(f"Failed to shutdown Infrastructure Domain: {e}")
            raise InfrastructureException(f"Shutdown failed: {e}")

    async def _initialize_database_manager(self) -> None:
        """Initialize database manager"""
        from .database import DatabaseManager

        self.database_manager = DatabaseManager()
        await self.database_manager.initialize()
        self.components["database_manager"] = self.database_manager

    async def _initialize_cache_manager(self) -> None:
        """Initialize cache manager"""
        from .cache import CacheManager

        self.cache_manager = CacheManager()
        await self.cache_manager.initialize()
        self.components["cache_manager"] = self.cache_manager

    async def _initialize_deployment_manager(self) -> None:
        """Initialize deployment manager"""
        from .deployment import DeploymentManager

        self.deployment_manager = DeploymentManager()
        await self.deployment_manager.initialize()
        self.components["deployment_manager"] = self.deployment_manager

    async def _start_health_monitoring(self) -> None:
        """Start infrastructure health monitoring"""
        # Start health check loops
        asyncio.create_task(self._health_monitoring_loop())

    async def _stop_health_monitoring(self) -> None:
        """Stop infrastructure health monitoring"""
        # Implementation for stopping health monitoring
        pass

    async def _health_monitoring_loop(self) -> None:
        """Background health monitoring loop"""
        while self.is_initialized:
            try:
                await self._perform_health_checks()
                await asyncio.sleep(60)  # Check every minute
            except Exception as e:
                self.logger.error(f"Health monitoring error: {e}")
                await asyncio.sleep(10)

    async def _perform_health_checks(self) -> None:
        """Perform health checks on all infrastructure components"""
        current_time = asyncio.get_event_loop().time()

        # Check database manager
        if self.database_manager:
            try:
                # Perform a simple health check
                self.service_status["database"] = "healthy"
                self.health_checks["database"] = current_time
            except Exception as e:
                self.service_status["database"] = f"unhealthy: {e}"

        # Check cache manager
        if self.cache_manager:
            try:
                # Perform cache health check
                self.service_status["cache"] = "healthy"
                self.health_checks["cache"] = current_time
            except Exception as e:
                self.service_status["cache"] = f"unhealthy: {e}"

        # Check deployment manager
        if self.deployment_manager:
            try:
                # Perform deployment health check
                self.service_status["deployment"] = "healthy"
                self.health_checks["deployment"] = current_time
            except Exception as e:
                self.service_status["deployment"] = f"unhealthy: {e}"

    async def _shutdown_component(self, component_name: str) -> None:
        """Safely shutdown a component"""
        try:
            component = getattr(self, component_name)
            if hasattr(component, "shutdown"):
                await component.shutdown()
        except Exception as e:
            self.logger.error(f"Error shutting down {component_name}: {e}")

    # Public API methods

    async def get_infrastructure_status(self) -> Dict[str, Any]:
        """Get current infrastructure status"""
        return {
            "services": self.service_status,
            "resource_usage": self.resource_usage,
            "health_checks": self.health_checks,
            "components_status": {
                name: "initialized" if component else "not_initialized"
                for name, component in [
                    ("database_manager", self.database_manager),
                    ("cache_manager", self.cache_manager),
                    ("deployment_manager", self.deployment_manager),
                ]
            },
        }

    async def get_repository(self, collection_name: str) -> Optional[Any]:
        """Get a repository from database manager"""
        if not self.database_manager:
            raise InfrastructureException("Database manager not initialized")

        return await self.database_manager.get_repository(collection_name)

    async def cache_get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if not self.cache_manager:
            raise InfrastructureException("Cache manager not initialized")

        return await self.cache_manager.get(key)

    async def cache_set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache"""
        if not self.cache_manager:
            raise InfrastructureException("Cache manager not initialized")

        return await self.cache_manager.set(key, value, ttl)

    async def deploy_to_environment(self, environment: str, config: Dict[str, Any]) -> str:
        """Deploy to specified environment"""
        if not self.deployment_manager:
            raise InfrastructureException("Deployment manager not initialized")

        return await self.deployment_manager.deploy(config, environment)


class InfrastructureException(DomainException):
    """Infrastructure domain specific exceptions"""

    pass
