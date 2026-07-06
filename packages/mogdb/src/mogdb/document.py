"""Document model and ObjectId for MogDB."""

import hashlib
import secrets
import time
from typing import Any, Dict


def ObjectId() -> str:
    """Generate a unique document ID (24-char hex).

    Format matches the general shape of MongoDB ObjectIds for familiarity
    but is purely Python-generated. The first 8 hex characters are the
    lower 32 bits of the ms timestamp; the remaining 16 are random.
    """
    t = int(time.time() * 1000) & 0xFFFFFFFF
    r = secrets.token_hex(8)
    return f"{t:08x}{r}"


class Document(Dict[str, Any]):
    """A dict subclass representing a single MogDB document.

    Every document has a reserved ``_id`` field (auto-generated if missing)
    and an internal ``_created`` timestamp.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        if "_id" not in self:
            self["_id"] = ObjectId()
        if "_created" not in self:
            self["_created"] = time.time()
        self["_updated"] = time.time()

    @property
    def id(self) -> str:
        return str(self["_id"])

    def content_hash(self) -> str:
        """SHA-256 of the JSON-serialised content (excluding metadata fields)."""
        import json

        clean = {k: v for k, v in self.items() if not k.startswith("_")}
        raw = json.dumps(clean, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def copy_data(self) -> Dict[str, Any]:
        """Return a plain dict copy with all fields."""
        return dict(self)
