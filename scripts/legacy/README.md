# Legacy / snapshot modules

`utils_json_device_helpers.py` was a repo-root **`utils.py`** snapshot (JSON helpers + `get_device` helpers). Nothing in the installable package imports it; prefer **`domains.shared.utils`** and **`config_loader.get_device`** for new code.
