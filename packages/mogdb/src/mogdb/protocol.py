"""Wire protocol for the MogDB network server.

Uses a simple JSON-framing protocol over TCP. Each message is a single
JSON object terminated by ``\\n``. The protocol is synchronous
(request → response) with an integer ``id`` field for correlation.

Request schema::

    {
        "id": 1,                    # int — client-generated correlation id
        "cmd": "insert_one",        # str — command name
        "collection": "users",      # str — collection name
        "args": { ... }             # dict — command-specific arguments
    }

Response schema::

    {
        "id": 1,                    # int — echoes the request id
        "ok": true,                 # bool — success indicator
        "result": ...               # any — command-specific result
    }

Error schema::

    {
        "id": 1,
        "ok": false,
        "error": "message"          # str — human-readable error
    }
"""

import json
import socket
from typing import Any, Dict, List, Optional


_CMD_ARGS: Dict[str, List[str]] = {
    "ping": [],
    "create_collection": ["name"],
    "drop_collection": ["name"],
    "list_collections": [],
    "compact": [],
    "insert_one": ["collection", "doc"],
    "insert_many": ["collection", "docs"],
    "find": ["collection"],
    "find_one": ["collection"],
    "count": ["collection"],
    "update_one": ["collection", "query", "update"],
    "update_many": ["collection", "query", "update"],
    "delete_one": ["collection"],
    "delete_many": ["collection"],
    "drop": ["collection"],
    "auth": ["password"],
}


def validate_request(req: Dict[str, Any]) -> Optional[str]:
    """Validate a decoded request dict. Returns an error message or ``None``."""
    if not isinstance(req.get("id"), int):
        return "missing or invalid 'id' (must be int)"
    cmd = req.get("cmd")
    if not isinstance(cmd, str) or cmd not in _CMD_ARGS:
        return f"unknown or missing cmd: {cmd!r}"
    expected = _CMD_ARGS[cmd]
    for arg in expected:
        if arg not in req:
            return f"cmd {cmd!r} missing required arg: {arg!r}"
    if "collection" in expected:
        coll = req.get("collection", "")
        if not isinstance(coll, str) or not coll.strip():
            return "collection name must be a non-empty string"
    return None


def encode_response(req_id: int, result: Any = None) -> bytes:
    """Encode a success response."""
    msg = {"id": req_id, "ok": True, "result": result}
    return (json.dumps(msg, default=str) + "\n").encode()


def encode_error(req_id: int, error: str) -> bytes:
    """Encode an error response."""
    msg = {"id": req_id, "ok": False, "error": error}
    return (json.dumps(msg, default=str) + "\n").encode()


def read_message(sock: socket.socket, buffer: bytes) -> tuple[Optional[Dict], bytes]:
    """Read one JSON message from *sock* with an internal *buffer*.

    Returns ``(message_dict, remaining_buffer)``.

    *message_dict* is ``None`` if a complete message isn't available yet.
    The caller should maintain the buffer across connection lifetime.
    """
    while b"\n" not in buffer:
        chunk = sock.recv(65536)
        if not chunk:
            return None, buffer
        buffer += chunk

    line, buffer = buffer.split(b"\n", 1)
    if not line.strip():
        return None, buffer
    try:
        msg = json.loads(line)
    except json.JSONDecodeError as exc:
        return None, buffer
    return msg, buffer
