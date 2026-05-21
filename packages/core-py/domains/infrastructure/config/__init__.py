"""
Configuration Manager Implementation

This module provides configuration management capabilities
for the entire system.
"""

import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List

from ...__init__ import BaseComponent, ComponentException


@dataclass
class ConfigurationItem:
    """Configuration item with metadata"""

    key: str
    value: Any
    description: str
    config_type: str
    is_sensitive: bool
    created_at: float
    updated_at: float


class ConfigurationManager(BaseComponent):
    """Advanced configuration management system"""

    def __init__(self) -> None:
        super().__init__("configuration_manager")
        self.logger = logging.getLogger(f"sloughgpt.{self.component_name}")

        # Configuration storage
        self.configurations: Dict[str, ConfigurationItem] = {}
        self.config_files: List[str] = []

        # Default configuration
        self.default_config = {
            "system.name": "SloughGPT",
            "system.version": "2.0.0",
            "system.environment": "development",
            "database.type": "sqlite",
            "database.connection_string": "sloughgpt.db",
            "cache.max_size": 10000,
            "cache.default_ttl": 3600,
            "log.level": "INFO",
            "api.port": 8000,
            "api.host": "localhost",
        }

        # Configuration schema
        self.config_schema = {
            "system.name": {"type": "string", "required": True},
            "system.version": {"type": "string", "required": True},
            "system.environment": {
                "type": "enum",
                "values": ["development", "staging", "production"],
                "required": True,
            },
            "database.type": {
                "type": "enum",
                "values": ["sqlite", "postgresql", "mysql"],
                "required": True,
            },
            "database.connection_string": {"type": "string", "required": True},
            "cache.max_size": {"type": "integer", "min": 100, "max": 100000, "required": True},
            "cache.default_ttl": {"type": "integer", "min": 60, "max": 86400, "required": True},
            "log.level": {
                "type": "enum",
                "values": ["DEBUG", "INFO", "WARNING", "ERROR"],
                "required": True,
            },
            "api.port": {"type": "integer", "min": 1024, "max": 65535, "required": True},
            "api.host": {"type": "string", "required": True},
        }

        self.is_initialized = False

    async def initialize(self) -> None:
        """Initialize configuration manager"""
        try:
            self.logger.info("Initializing Configuration Manager...")

            # Load default configuration
            await self._load_default_configuration()

            # Load configuration from files
            await self._load_configuration_files()

            # Load environment variables
            await self._load_environment_variables()

            # Validate configuration
            await self._validate_configuration()

            self.is_initialized = True
            self.logger.info("Configuration Manager initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize Configuration Manager: {e}")
            raise ComponentException(f"Configuration Manager initialization failed: {e}")

    async def shutdown(self) -> None:
        """Shutdown configuration manager"""
        try:
            self.logger.info("Shutting down Configuration Manager...")

            # Save configuration to files
            await self._save_configuration()

            self.is_initialized = False
            self.logger.info("Configuration Manager shutdown successfully")

        except Exception as e:
            self.logger.error(f"Failed to shutdown Configuration Manager: {e}")
            raise ComponentException(f"Configuration Manager shutdown failed: {e}")

    async def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        try:
            if key in self.configurations:
                return self.configurations[key].value
            return default

        except Exception as e:
            self.logger.error(f"Failed to get configuration key {key}: {e}")
            return default

    async def set(
        self,
        key: str,
        value: Any,
        description: str = "",
        config_type: str = "string",
        is_sensitive: bool = False,
    ) -> bool:
        """Set configuration value"""
        try:
            # Validate against schema if exists
            if key in self.config_schema:
                if not await self._validate_value(key, value):
                    raise ComponentException(f"Invalid value for {key}: {value}")

            # Create or update configuration item
            current_time = time.time()

            if key in self.configurations:
                # Update existing
                self.configurations[key].value = value
                self.configurations[key].updated_at = current_time
                if description:
                    self.configurations[key].description = description
            else:
                # Create new
                self.configurations[key] = ConfigurationItem(
                    key=key,
                    value=value,
                    description=description,
                    config_type=config_type,
                    is_sensitive=is_sensitive,
                    created_at=current_time,
                    updated_at=current_time,
                )

            self.logger.debug(f"Set configuration {key} = {value if not is_sensitive else '***'}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to set configuration key {key}: {e}")
            return False

    async def delete(self, key: str) -> bool:
        """Delete configuration key"""
        try:
            if key in self.configurations:
                del self.configurations[key]
                self.logger.debug(f"Deleted configuration key: {key}")
                return True
            return False

        except Exception as e:
            self.logger.error(f"Failed to delete configuration key {key}: {e}")
            return False

    async def get_all(self, include_sensitive: bool = False) -> Dict[str, Any]:
        """Get all configuration"""
        try:
            all_config = {}

            for key, item in self.configurations.items():
                if include_sensitive or not item.is_sensitive:
                    all_config[key] = {
                        "value": item.value,
                        "description": item.description,
                        "type": item.config_type,
                        "created_at": item.created_at,
                        "updated_at": item.updated_at,
                    }

            return dict(all_config)

        except Exception as e:
            self.logger.error(f"Failed to get all configuration: {e}")
            return {}

    async def reload(self) -> bool:
        """Reload configuration from sources"""
        try:
            self.logger.info("Reloading configuration...")

            # Clear current configuration
            self.configurations.clear()

            # Reload all sources
            await self._load_default_configuration()
            await self._load_configuration_files()
            await self._load_environment_variables()
            await self._validate_configuration()

            self.logger.info("Configuration reloaded successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to reload configuration: {e}")
            return False

    async def export_config(self, file_path: str, include_sensitive: bool = False) -> bool:
        """Export configuration to file"""
        try:
            all_config = await self.get_all(include_sensitive=include_sensitive)

            # Prepare export data
            export_data: Dict[str, Any] = {"exported_at": time.time(), "configuration": {}}

            for key, data in all_config.items():
                export_data["configuration"][key] = data["value"]

            # Write to file
            with open(file_path, "w") as f:
                json.dump(export_data, f, indent=2)

            self.logger.info(f"Configuration exported to {file_path}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to export configuration: {e}")
            return False

    # Private helper methods

    async def _load_default_configuration(self) -> None:
        """Load default configuration"""
        for key, value in self.default_config.items():
            if key not in self.configurations:
                await self.set(key, value, f"Default configuration for {key}")

    async def _load_configuration_files(self) -> None:
        """Load configuration from files"""
        config_files = [
            "config.json",
            "config.local.json",
            os.path.expanduser("~/.sloughgpt/config.json"),
        ]

        for file_path in config_files:
            if os.path.exists(file_path):
                try:
                    with open(file_path, "r") as f:
                        file_config = json.load(f)

                    for key, value in file_config.items():
                        await self.set(key, value, f"Loaded from {file_path}")

                    self.config_files.append(file_path)
                    self.logger.info(f"Loaded configuration from {file_path}")

                except Exception as e:
                    self.logger.warning(f"Failed to load configuration from {file_path}: {e}")

    async def _load_environment_variables(self) -> None:
        """Load configuration from environment variables"""
        env_prefix = "SLOUGHGPT_"

        for key, value in os.environ.items():
            if key.startswith(env_prefix):
                config_key = key[len(env_prefix) :].lower().replace("_", ".")

                # Convert string value to appropriate type
                converted_value = await self._convert_env_value(value)

                await self.set(
                    config_key, converted_value, f"Loaded from environment variable {key}"
                )

    async def _convert_env_value(self, value: str) -> Any:
        """Convert environment variable value to appropriate type"""
        # Try boolean
        if value.lower() in ["true", "false"]:
            return value.lower() == "true"

        # Try integer
        try:
            return int(value)
        except ValueError:
            pass

        # Try float
        try:
            return float(value)
        except ValueError:
            pass

        # Return as string
        return value

    async def _validate_configuration(self) -> None:
        """Validate all configuration against schema"""
        errors = []

        # Check required fields
        for key, schema in self.config_schema.items():
            if schema.get("required", False) and key not in self.configurations:
                errors.append(f"Required configuration key missing: {key}")

        # Validate values
        for key, schema in self.config_schema.items():
            if key in self.configurations:
                value = self.configurations[key].value
                if not await self._validate_value(key, value):
                    errors.append(f"Invalid value for {key}: {value}")

        if errors:
            error_msg = "Configuration validation failed:\n" + "\n".join(errors)
            raise ComponentException(error_msg)

    async def _validate_value(self, key: str, value: Any) -> bool:
        """Validate a configuration value against schema"""
        if key not in self.config_schema:
            return True  # No schema, assume valid

        schema = self.config_schema[key]
        value_type = schema["type"]

        # Type validation
        if value_type == "string":
            if not isinstance(value, str):
                return False
        elif value_type == "integer":
            if not isinstance(value, int):
                return False
        elif value_type == "boolean":
            if not isinstance(value, bool):
                return False
        elif value_type == "enum":
            if value not in schema["values"]:
                return False

        # Range validation
        if value_type == "integer":
            if "min" in schema and int(value) < int(schema.get("min", 0)):
                return False
            if "max" in schema and int(value) > int(schema.get("max", float("inf"))):
                return False

        return True

    async def _save_configuration(self) -> None:
        """Save configuration to file"""
        config_file = "config.runtime.json"

        try:
            all_config = await self.get_all(include_sensitive=False)

            # Prepare save data
            save_data = {}
            for key, data in all_config.items():
                save_data[key] = data["value"]

            # Write to file
            with open(config_file, "w") as f:
                json.dump(save_data, f, indent=2)

            self.logger.info(f"Configuration saved to {config_file}")

        except Exception as e:
            self.logger.error(f"Failed to save configuration: {e}")


__all__ = ["ConfigurationItem", "ConfigurationManager"]

