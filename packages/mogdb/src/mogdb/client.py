"""Remote MogDB client — connects to a MogDB server over TCP.

Provides the same collection-style API as the local ``MogDB`` class but
operates over the wire protocol::

    from mogdb import MogDBClient

    client = MogDBClient("localhost", 27017)
    client.connect()
    client.auth("password")       # if server requires auth

    users = client.collection("users")
    users.insert_one({"name": "Alice", "age": 30})
    result = users.find({"age": {"$gt": 25}})

    client.close()
"""

import json
import logging
import socket
import threading
from typing import Any, Dict, List, Optional

from .protocol import encode_error, encode_response, read_message

logger = logging.getLogger("mogdb.client")


class MogDBError(Exception):
    """Raised when the server returns an error response."""


class RemoteCollection:
    """Proxy for a server-side collection.

    Created by ``MogDBClient.collection()``. All methods send a TCP
    message to the server and wait for the response.
    """

    def __init__(self, client: "MogDBClient", name: str):
        self._client = client
        self.name = name

    def _call(self, cmd: str, **kwargs: Any) -> Any:
        return self._client._call(cmd, self.name, **kwargs)

    def insert_one(self, doc: Dict[str, Any]) -> str:
        result = self._call("insert_one", doc=doc)
        return result["_id"]

    def insert_many(self, docs: List[Dict[str, Any]]) -> List[str]:
        result = self._call("insert_many", docs=docs)
        return result["_ids"]

    def find(
        self,
        query: Optional[Dict[str, Any]] = None,
        sort: Optional[List[tuple]] = None,
        limit: Optional[int] = None,
        skip: int = 0,
    ) -> List[Dict[str, Any]]:
        return self._call("find", query=query, sort=sort, limit=limit, skip=skip)

    def find_one(self, query: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        return self._call("find_one", query=query)

    def count(self, query: Optional[Dict[str, Any]] = None) -> int:
        return self._call("count", query=query)

    def update_one(self, query: Dict[str, Any], update: Dict[str, Any]) -> int:
        result = self._call("update_one", query=query, update=update)
        return result["modified"]

    def update_many(self, query: Dict[str, Any], update: Dict[str, Any]) -> int:
        result = self._call("update_many", query=query, update=update)
        return result["modified"]

    def delete_one(self, query: Dict[str, Any]) -> int:
        result = self._call("delete_one", query=query)
        return result["deleted"]

    def delete_many(self, query: Dict[str, Any]) -> int:
        result = self._call("delete_many", query=query)
        return result["deleted"]

    def drop(self) -> None:
        self._call("drop")


class MogDBClient:
    """Client connection to a remote MogDB server.

    Parameters
    ----------
    host:
        Server hostname or IP.
    port:
        Server port.
    timeout:
        Socket timeout in seconds (default 10).
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 27017, timeout: float = 10):
        self.host = host
        self.port = port
        self.timeout = timeout
        self._sock: Optional[socket.socket] = None
        self._buffer = b""
        self._next_id = 1
        self._lock = threading.Lock()

    def connect(self) -> None:
        """Connect to the server."""
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.settimeout(self.timeout)
        self._sock.connect((self.host, self.port))
        logger.info("connected to mogdb://%s:%d", self.host, self.port)

    def auth(self, password: str) -> bool:
        """Authenticate with the server. Returns ``True`` on success."""
        result = self._call("auth", password=password)
        return result.get("authenticated", False)

    def close(self) -> None:
        """Close the connection."""
        if self._sock:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None

    def collection(self, name: str) -> RemoteCollection:
        """Get a proxy for a server-side collection."""
        return RemoteCollection(self, name)

    def ping(self) -> str:
        """Health check — returns ``"pong"``."""
        return self._call("ping")

    def create_collection(self, name: str) -> None:
        """Explicitly create a collection on the server."""
        self._call("create_collection", name=name)

    def drop_collection(self, name: str) -> None:
        """Drop a collection on the server."""
        self._call("drop_collection", name=name)

    def list_collections(self) -> List[str]:
        """List all collections on the server."""
        return self._call("list_collections")

    def compact(self) -> int:
        """Compact all collections on the server."""
        result = self._call("compact")
        return result["compacted"]

    # ------------------------------------------------------------------
    # internal
    # ------------------------------------------------------------------

    def _call(self, cmd: str, collection: str = "", **kwargs: Any) -> Any:
        if not self._sock:
            raise MogDBError("not connected")

        with self._lock:
            req_id = self._next_id
            self._next_id += 1
            msg: Dict[str, Any] = {"id": req_id, "cmd": cmd}
            if collection:
                msg["collection"] = collection
            # Flatten kwargs into msg (or nest under args)
            for k, v in kwargs.items():
                if v is not None:
                    msg[k] = v

            payload = (json.dumps(msg, default=str) + "\n").encode()
            try:
                self._sock.sendall(payload)
            except OSError as exc:
                raise MogDBError(f"send failed: {exc}") from exc

            # Read response
            while True:
                resp, self._buffer = read_message(self._sock, self._buffer)
                if resp is not None:
                    break

            if resp.get("id") != req_id:
                raise MogDBError(f"response id mismatch: got {resp.get('id')}, expected {req_id}")

            if not resp.get("ok", False):
                raise MogDBError(resp.get("error", "unknown error"))

            return resp.get("result")


# Backward-compatible alias
MogDBClientRemote = MogDBClient
