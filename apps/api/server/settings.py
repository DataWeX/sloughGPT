"""
Centralized server configuration from environment.

Canonical prefixes are ``MAN_*`` (matching the project name SloughGPT).
The misspelled ``SLAUGHGPT_*`` vars are still read as fallbacks for existing deployments.
"""

from __future__ import annotations

import os
import secrets
from dataclasses import dataclass
from functools import lru_cache


def _parse_api_key_list(raw: str) -> frozenset[str]:
    return frozenset(x.strip() for x in raw.split(",") if x.strip())


from typing import Optional

def _getenv_prefer_canonical(canonical: str, legacy_typo: str) -> Optional[str]:
    """Prefer ``MAN_*``; if unset or blank, use legacy ``SLAUGHGPT_*`` (historical typo)."""
    v = os.getenv(canonical)
    if v is not None and str(v).strip() != "":
        return str(v)
    leg = os.getenv(legacy_typo)
    if leg is not None and str(leg).strip() != "":
        return str(leg)
    return None


def _getenv_api_keys_raw() -> str:
    c = os.getenv("MAN_API_KEYS")
    if c is not None and str(c).strip() != "":
        return str(c)
    return os.getenv("SLAUGHGPT_API_KEYS", "")


@dataclass(frozen=True)
class SecuritySettings:
    """Auth material loaded once at process start."""

    primary_api_key: str
    jwt_secret: str
    jwt_algorithm: str
    jwt_expiration_hours: int
    valid_api_keys: frozenset[str]


@lru_cache(maxsize=1)
def get_security_settings() -> SecuritySettings:
    """Return cached security settings (singleton for the process)."""
    primary = _getenv_prefer_canonical("MAN_API_KEY", "SLAUGHGPT_API_KEY") or secrets.token_urlsafe(32)
    jwt_secret = (
        _getenv_prefer_canonical("MAN_JWT_SECRET", "SLAUGHGPT_JWT_SECRET") or secrets.token_urlsafe(64)
    )
    keys = set(_parse_api_key_list(_getenv_api_keys_raw()))
    if primary:
        keys.add(primary)
    return SecuritySettings(
        primary_api_key=primary,
        jwt_secret=jwt_secret,
        jwt_algorithm="HS256",
        jwt_expiration_hours=24,
        valid_api_keys=frozenset(keys),
    )
