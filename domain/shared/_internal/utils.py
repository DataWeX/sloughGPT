"""SloughGPT shared utilities — back-compat facade.

The implementation now lives in focused modules; every name below is still
importable from ``domain.shared._internal.utils`` (and most from
``domain.shared``) so existing imports keep working:

    ids           generate_id
    hashing       hash_string
    formatting    format_size, format_time
    jsonio        load_json, save_json
    ops           merge_dicts, clamp, validate_config
    concurrency   retry, Timer, Cache, RateLimiter
    net           find_available_port
    paths         find_repo_root, find_server_python
    timestamps    get_timestamp, utc_now_iso, to_iso, parse_iso,
                  normalize_iso, is_valid_iso

New code should import from the specific module (or from ``domain.shared``)
rather than from this facade.
"""

from __future__ import annotations

from domain.shared._internal.concurrency import Cache, RateLimiter, Timer, retry
from domain.shared._internal.formatting import format_size, format_time
from domain.shared._internal.hashing import hash_string
from domain.shared._internal.ids import generate_id
from domain.shared._internal.jsonio import load_json, save_json
from domain.shared._internal.net import find_available_port
from domain.shared._internal.ops import clamp, merge_dicts, validate_config
from domain.shared._internal.paths import find_repo_root, find_server_python
from domain.shared._internal.timestamps import (
    get_timestamp,
    is_valid_iso,
    normalize_iso,
    parse_iso,
    to_iso,
    utc_now_iso,
)

__all__ = [
    "Cache",
    "RateLimiter",
    "Timer",
    "clamp",
    "find_available_port",
    "find_repo_root",
    "find_server_python",
    "format_size",
    "format_time",
    "generate_id",
    "get_timestamp",
    "hash_string",
    "is_valid_iso",
    "load_json",
    "merge_dicts",
    "normalize_iso",
    "parse_iso",
    "retry",
    "save_json",
    "to_iso",
    "utc_now_iso",
    "validate_config",
]
