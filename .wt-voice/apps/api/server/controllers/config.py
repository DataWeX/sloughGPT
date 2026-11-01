"""
Config Controller - Business logic for configuration
"""

from typing import Any


class ConfigController:
    """Controller for configuration management"""

    def __init__(self):
        self._config = {
            "temperature": 0.7,
            "top_p": 0.85,
            "top_k": 40,
            "repetition_penalty": 1.15,
            "max_new_tokens": 200,
            "max_context_length": 1024,
        }

    def get_generation_config(self) -> dict[str, Any]:
        """Get current generation config"""
        return self._config.copy()

    def update_generation_config(self, **kwargs) -> dict[str, Any]:
        """Update generation config"""
        for key, value in kwargs.items():
            if value is not None and key in self._config:
                self._config[key] = value
        return self._config.copy()


_config_controller: ConfigController | None = None


def get_config_controller() -> ConfigController:
    global _config_controller
    if _config_controller is None:
        _config_controller = ConfigController()
    return _config_controller
