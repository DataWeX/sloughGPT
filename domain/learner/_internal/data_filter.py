"""Data filtering for knowledge memory.

Byte-identical with the canonical implementation; re-exported from
:mod:`domain.knowledge._internal.data_filter` to keep a single source of
truth.
"""

from domain.knowledge._internal.data_filter import (
    DEFAULT_CONFIG,
    DataFilter,
    _get_collection,
    _hashed,
    _load_config,
    _matches_blacklist,
    _matches_whitelist,
    _save_config,
    _score_quality,
    _score_relevance,
    get_data_filter,
    reset_data_filter_db,
    set_data_filter_db,
)

__all__ = [
    "DEFAULT_CONFIG",
    "DataFilter",
    "_get_collection",
    "_hashed",
    "_load_config",
    "_matches_blacklist",
    "_matches_whitelist",
    "_save_config",
    "_score_quality",
    "_score_relevance",
    "get_data_filter",
    "reset_data_filter_db",
    "set_data_filter_db",
]
