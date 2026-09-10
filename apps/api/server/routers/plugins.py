"""
Plugins Router — endpoints for plugin management.
"""

import logging
from pathlib import Path

from domains.shared import find_repo_root
from fastapi import APIRouter, Depends
from infrastructure.auth import require_auth_if_enabled
from schemas.common import classify_and_raise, endpoint, safe_audit_log, success_response

logger = logging.getLogger("slo.routers.plugins")


class PluginsRouter:
    """Router for plugin management."""

    def __init__(self):
        self.router = APIRouter(prefix="/plugins", tags=["plugins"])
        self._register_routes()

    def _register_routes(self):
        self.router.add_api_route(
            "", self.list_plugins, methods=["GET"]
        )
        self.router.add_api_route(
            "/{plugin_name}/enable", self.enable_plugin, methods=["POST"]
        )
        self.router.add_api_route(
            "/{plugin_name}/disable", self.disable_plugin, methods=["POST"]
        )
        self.router.add_api_route(
            "/reload", self.reload_plugins, methods=["POST"]
        )

    @endpoint("plugins.list")
    async def list_plugins(self) -> dict:
        """List all loaded plugins."""
        try:
            from domains.plugins import get_plugin_manager
            pm = get_plugin_manager()
            plugins = pm.list_plugins()
            return success_response(data={"plugins": plugins})
        except Exception as e:
            classify_and_raise(e, source="plugins.list")

    @endpoint("plugins.enable")
    async def enable_plugin(
        self,
        plugin_name: str,
        auth_user: dict = Depends(require_auth_if_enabled),
    ) -> dict:
        """Enable a plugin."""
        try:
            from domains.plugins import get_plugin_manager
            pm = get_plugin_manager()
            meta = pm._metadata.get(plugin_name)
            if meta:
                meta.enabled = True
            safe_audit_log("plugins.enable", resource=plugin_name)
            return success_response(data={"plugin": plugin_name, "enabled": True})
        except Exception as e:
            classify_and_raise(e, source="plugins.enable")

    @endpoint("plugins.disable")
    async def disable_plugin(
        self,
        plugin_name: str,
        auth_user: dict = Depends(require_auth_if_enabled),
    ) -> dict:
        """Disable a plugin."""
        try:
            from domains.plugins import get_plugin_manager
            pm = get_plugin_manager()
            meta = pm._metadata.get(plugin_name)
            if meta:
                meta.enabled = False
            safe_audit_log("plugins.disable", resource=plugin_name)
            return success_response(data={"plugin": plugin_name, "enabled": False})
        except Exception as e:
            classify_and_raise(e, source="plugins.disable")

    @endpoint("plugins.reload")
    async def reload_plugins(
        self,
        auth_user: dict = Depends(require_auth_if_enabled),
    ) -> dict:
        """Reload plugins from plugin directories."""
        try:
            from domains.plugins import get_plugin_manager
            pm = get_plugin_manager()
            repo_root = find_repo_root(Path(__file__).resolve())
            plugin_dir = repo_root / "plugins"
            loaded = pm.load_from_directory(plugin_dir) if plugin_dir.exists() else 0
            safe_audit_log("plugins.reload")
            return success_response(data={"loaded": loaded})
        except Exception as e:
            classify_and_raise(e, source="plugins.reload")


router = PluginsRouter().router
