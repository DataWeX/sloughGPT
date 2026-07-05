"""
Config Controller - Business logic for configuration
"""
from typing import Dict, Any, Optional


class ConfigController:
    """Controller for configuration management"""

    def __init__(self):
        self._config = {
            "temperature": 0.8,
            "top_p": 0.9,
            "top_k": 50,
            "repetition_penalty": 1.2,
            "max_new_tokens": 200,
            "max_context_length": 1024,
        }

    def get_generation_config(self) -> Dict[str, Any]:
        """Get current generation config"""
        return self._config.copy()

    def update_generation_config(self, **kwargs) -> Dict[str, Any]:
        """Update generation config"""
        for key, value in kwargs.items():
            if value is not None and key in self._config:
                self._config[key] = value
        return self._config.copy()


_config_controller: Optional[ConfigController] = None


def get_config_controller() -> ConfigController:
    global _config_controller
    if _config_controller is None:
        _config_controller = ConfigController()
    return _config_controller
