"""Standalone TCP server for MogDB.

Listens on a configurable host/port, accepts JSON-framed messages via
the MogDB wire protocol, dispatches to collection CRUD, and returns
JSON responses.

Usage::

    python -m mogdb.server --port 27017 --dbpath data/mogdb

Or via the CLI::

    mogdb-server --port 27017
"""

import argparse
import json
import logging
import os
import socket
import threading
from typing import Any, Dict, Optional

from .database import MogDB
from .protocol import (
    encode_error,
    encode_response,
    read_message,
    validate_request,
)

logger = logging.getLogger("mogdb.server")

# Default server password for simple auth
_DEFAULT_PASSWORD = "mogdb"

# Token bucket rate-limiter state
_rate_limiters: Dict[str, "TokenBucket"] = {}


class TokenBucket:
    """Simple token-bucket rate limiter per connection."""

    def __init__(self, rate: float = 1000, burst: int = 100):
        self.rate = rate
        self.burst = burst
        self.tokens = burst
        self.last = __import__("time").time()
        self._lock = threading.Lock()

    def consume(self, tokens: int = 1) -> bool:
        with self._lock:
            now = __import__("time").time()
            elapsed = now - self.last
            self.tokens = min(self.burst, self.tokens + elapsed * self.rate)
            self.last = now
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False


class MogDBServer:
    """TCP server serving MogDB over the wire protocol.

    Parameters
    ----------
    host:
        Bind address (default ``127.0.0.1``).
    port:
        Bind port (default ``27017``).
    dbpath:
        Filesystem path for the MogDB database.
    password:
        If set, clients must send ``{"cmd": "auth", "password": "..."}``
        before any data commands.
    max_connections:
        Maximum simultaneous client connections.
    rate_limit:
        Maximum requests per second per connection.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 27017,
        dbpath: str = "data/mogdb",
        password: Optional[str] = None,
        max_connections: int = 100,
        rate_limit: float = 1000,
    ):
        self.host = host
        self.port = port
        self.password = password
        self.max_connections = max_connections
        self.rate_limit = rate_limit
        self._db = MogDB(dbpath)
        self._server: Optional[socket.socket] = None
        self._running = False
        self._active_connections = 0
        self._lock = threading.Lock()
        self._auth_clients: set[int] = set()

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the server (blocking — runs until ``stop()``)."""
        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server.bind((self.host, self.port))
        self._server.listen(self.max_connections)
        self._running = True
        logger.info(
            "MogDB server listening on %s:%s (db: %s)",
            self.host, self.port, self._db._root,
        )

        try:
            while self._running:
                client, addr = self._server.accept()
                with self._lock:
                    if self._active_connections >= self.max_connections:
                        client.close()
                        logger.warning("rejected connection from %s (max=%d)", addr, self.max_connections)
                        continue
                    self._active_connections += 1
                thread = threading.Thread(
                    target=self._handle_client,
                    args=(client, addr),
                    daemon=True,
                )
                thread.start()
        except KeyboardInterrupt:
            logger.info("server stopped by KeyboardInterrupt")
        finally:
            self.stop()

    def stop(self) -> None:
        """Stop the server and close the socket."""
        self._running = False
        if self._server:
            try:
                self._server.close()
            except OSError:
                pass
            self._server = None
        self._db.close()
        logger.info("MogDB server stopped")

    # ------------------------------------------------------------------
    # client handler
    # ------------------------------------------------------------------

    def _handle_client(self, client: socket.socket, addr: tuple) -> None:
        thread_id = threading.get_ident()
        buffer = b""
        limiter = TokenBucket(rate=self.rate_limit)
        authenticated = not self.password  # no auth required if no password set

        logger.debug("client connected: %s (thread=%d)", addr, thread_id)
        try:
            while self._running:
                if not limiter.consume():
                    client.sendall(encode_error(0, "rate limit exceeded"))
                    continue

                msg, buffer = read_message(client, buffer)
                if msg is None:
                    continue

                req_id = msg.get("id", 0)
                cmd = msg.get("cmd", "")

                # auth is always allowed
                if cmd == "auth":
                    pw = msg.get("password", "")
                    ok = pw == self.password
                    authenticated = ok
                    if ok:
                        with self._lock:
                            self._auth_clients.add(thread_id)
                        client.sendall(encode_response(req_id, {"authenticated": True}))
                    else:
                        client.sendall(encode_response(req_id, {"authenticated": False}))
                    continue

                if not authenticated:
                    client.sendall(encode_error(req_id, "not authenticated"))
                    continue

                error = validate_request(msg)
                if error:
                    client.sendall(encode_error(req_id, error))
                    continue

                response = self._dispatch(cmd, msg)
                client.sendall(response)
        except ConnectionResetError:
            logger.debug("client disconnected: %s", addr)
        except Exception as exc:
            logger.error("error handling client %s: %s", addr, exc)
            try:
                client.sendall(encode_error(0, f"internal error: {exc}"))
            except OSError:
                pass
        finally:
            with self._lock:
                self._active_connections -= 1
                self._auth_clients.discard(thread_id)
            try:
                client.close()
            except OSError:
                pass

    # ------------------------------------------------------------------
    # command dispatch
    # ------------------------------------------------------------------

    def _dispatch(self, cmd: str, msg: Dict[str, Any]) -> bytes:
        req_id = msg.get("id", 0)
        coll_name = msg.get("collection", "")
        args: Dict[str, Any] = msg.get("args", msg)

        try:
            if cmd == "ping":
                return encode_response(req_id, "pong")

            if cmd == "create_collection":
                name = args.get("name", coll_name)
                self._db.collection(name)
                return encode_response(req_id, {"collection": name})

            if cmd == "drop_collection":
                name = args.get("name", coll_name)
                self._db.drop_collection(name)
                return encode_response(req_id, {"dropped": name})

            if cmd == "list_collections":
                return encode_response(req_id, self._db.list_collections())

            if cmd == "compact":
                total = self._db.compact_all()
                return encode_response(req_id, {"compacted": total})

            if cmd == "insert_one":
                doc = args.get("doc", {})
                oid = self._db.collection(coll_name).insert_one(doc)
                return encode_response(req_id, {"_id": oid})

            if cmd == "insert_many":
                docs = args.get("docs", [])
                ids = self._db.collection(coll_name).insert_many(docs)
                return encode_response(req_id, {"_ids": ids})

            if cmd == "find":
                query = args.get("query")
                sort = args.get("sort")
                limit = args.get("limit")
                skip = args.get("skip", 0)
                docs = self._db.collection(coll_name).find(query, sort=sort, limit=limit, skip=skip)
                return encode_response(req_id, docs)

            if cmd == "find_one":
                query = args.get("query")
                doc = self._db.collection(coll_name).find_one(query)
                return encode_response(req_id, doc)

            if cmd == "count":
                query = args.get("query")
                n = self._db.collection(coll_name).count(query)
                return encode_response(req_id, n)

            if cmd == "update_one":
                query = args.get("query", {})
                update = args.get("update", {})
                modified = self._db.collection(coll_name).update_one(query, update)
                return encode_response(req_id, {"modified": modified})

            if cmd == "update_many":
                query = args.get("query", {})
                update = args.get("update", {})
                modified = self._db.collection(coll_name).update_many(query, update)
                return encode_response(req_id, {"modified": modified})

            if cmd == "delete_one":
                query = args.get("query", {})
                deleted = self._db.collection(coll_name).delete_one(query)
                return encode_response(req_id, {"deleted": deleted})

            if cmd == "delete_many":
                query = args.get("query", {})
                deleted = self._db.collection(coll_name).delete_many(query)
                return encode_response(req_id, {"deleted": deleted})

            if cmd == "drop":
                self._db.collection(coll_name).drop()
                return encode_response(req_id, {"dropped": coll_name})

            return encode_error(req_id, f"unknown cmd: {cmd!r}")

        except Exception as exc:
            logger.exception("error dispatching %s", cmd)
            return encode_error(req_id, str(exc))


def main() -> None:
    """CLI entry point for the standalone server."""
    parser = argparse.ArgumentParser(description="MogDB — embedded document database server")
    parser.add_argument("--host", default="127.0.0.1", help="bind address")
    parser.add_argument("--port", type=int, default=27017, help="bind port")
    parser.add_argument("--dbpath", default="data/mogdb", help="database directory")
    parser.add_argument("--password", default=None, help="require authentication")
    parser.add_argument("--max-connections", type=int, default=100, help="max concurrent clients")
    parser.add_argument("--rate-limit", type=float, default=1000, help="max req/s per connection")
    parser.add_argument("--verbose", "-v", action="store_true", help="debug logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    server = MogDBServer(
        host=args.host,
        port=args.port,
        dbpath=args.dbpath,
        password=args.password,
        max_connections=args.max_connections,
        rate_limit=args.rate_limit,
    )
    try:
        server.start()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
