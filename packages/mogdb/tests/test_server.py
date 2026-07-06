"""Tests for MogDB server, client, and wire protocol."""

import json
import os
import socket
import tempfile
import threading
import time

import pytest

from mogdb import MogDBServer, MogDBClient, MogDBError
from mogdb.protocol import validate_request, encode_response, encode_error


# =========================================================================
# Protocol unit tests
# =========================================================================

class TestProtocol:
    def test_validate_ping(self):
        assert validate_request({"id": 1, "cmd": "ping"}) is None

    def test_validate_missing_id(self):
        err = validate_request({"cmd": "ping"})
        assert err is not None
        assert "id" in err

    def test_validate_unknown_cmd(self):
        err = validate_request({"id": 1, "cmd": "nope"})
        assert err is not None
        assert "unknown" in err

    def test_validate_insert_missing_doc(self):
        err = validate_request({"id": 1, "cmd": "insert_one", "collection": "x"})
        assert err is not None
        assert "doc" in err

    def test_validate_insert_ok(self):
        assert validate_request({"id": 1, "cmd": "insert_one", "collection": "x", "doc": {"a": 1}}) is None

    def test_validate_empty_collection(self):
        err = validate_request({"id": 1, "cmd": "insert_one", "collection": "", "doc": {}})
        assert err is not None
        assert "collection" in err

    def test_encode_response(self):
        data = encode_response(42, {"ok": True})
        msg = json.loads(data.decode().strip())
        assert msg["id"] == 42
        assert msg["ok"] is True
        assert msg["result"] == {"ok": True}

    def test_encode_error(self):
        data = encode_error(7, "something broke")
        msg = json.loads(data.decode().strip())
        assert msg["id"] == 7
        assert msg["ok"] is False
        assert msg["error"] == "something broke"


# =========================================================================
# Integration: server + client over TCP
# =========================================================================


@pytest.fixture
def server_port():
    """Start a MogDBServer on a random port, yield the port, stop."""
    import random
    port = random.randint(20000, 30000)
    with tempfile.TemporaryDirectory() as tmp:
        dbpath = os.path.join(tmp, "mogdb")
        server = MogDBServer(host="127.0.0.1", port=port, dbpath=dbpath, password=None)
        t = threading.Thread(target=server.start, daemon=True)
        t.start()
        time.sleep(0.2)  # wait for server to start
        yield port
        server.stop()


@pytest.fixture
def client(server_port):
    c = MogDBClient("127.0.0.1", server_port)
    c.connect()
    yield c
    c.close()


class TestServerClient:
    def test_ping(self, client):
        assert client.ping() == "pong"

    def test_insert_and_find(self, client):
        coll = client.collection("test_coll")
        oid = coll.insert_one({"name": "Alice", "age": 30})
        assert len(oid) == 24

        doc = coll.find_one({"name": "Alice"})
        assert doc is not None
        assert doc["name"] == "Alice"
        assert doc["age"] == 30

    def test_insert_many_and_count(self, client):
        coll = client.collection("many")
        ids = coll.insert_many([{"x": 1}, {"x": 2}, {"x": 3}])
        assert len(ids) == 3
        assert coll.count() == 3

    def test_find_with_query(self, client):
        coll = client.collection("query_test")
        coll.insert_many([{"v": i} for i in range(10)])
        results = coll.find({"v": {"$gt": 5}})
        assert len(results) == 4
        assert all(r["v"] > 5 for r in results)

    def test_find_limit_skip_sort(self, client):
        coll = client.collection("ordered")
        coll.insert_many([{"n": i} for i in range(10)])
        results = coll.find(sort=[("n", -1)], limit=3, skip=2)
        assert len(results) == 3
        assert results[0]["n"] == 7
        assert results[1]["n"] == 6
        assert results[2]["n"] == 5

    def test_update_one(self, client):
        coll = client.collection("up_test")
        oid = coll.insert_one({"x": 1})
        modified = coll.update_one({"_id": oid}, {"$set": {"x": 99}})
        assert modified == 1
        doc = coll.find_one({"_id": oid})
        assert doc["x"] == 99

    def test_update_many(self, client):
        coll = client.collection("up_many")
        coll.insert_many([{"tag": "a"}, {"tag": "a"}, {"tag": "b"}])
        modified = coll.update_many({"tag": "a"}, {"$set": {"tag": "z"}})
        assert modified == 2
        assert coll.count({"tag": "z"}) == 2

    def test_delete_one(self, client):
        coll = client.collection("del_one")
        coll.insert_many([{"v": 1}, {"v": 2}])
        deleted = coll.delete_one({"v": 1})
        assert deleted == 1
        assert coll.count() == 1

    def test_delete_many(self, client):
        coll = client.collection("del_many")
        coll.insert_many([{"v": 1}, {"v": 1}, {"v": 2}])
        deleted = coll.delete_many({"v": 1})
        assert deleted == 2
        assert coll.count() == 1

    def test_drop(self, client):
        coll = client.collection("drop_test")
        coll.insert_one({"x": 1})
        assert coll.count() == 1
        coll.drop()
        assert coll.count() == 0

    def test_list_collections(self, client):
        client.collection("alpha").insert_one({"x": 1})
        client.collection("beta").insert_one({"x": 2})
        names = client.list_collections()
        assert "alpha" in names
        assert "beta" in names

    def test_create_and_drop_collection(self, client):
        client.create_collection("brand_new")
        names = client.list_collections()
        assert "brand_new" in names
        client.drop_collection("brand_new")
        names = client.list_collections()
        assert "brand_new" not in names

    def test_compact(self, client):
        coll = client.collection("compact_test")
        coll.insert_many([{"i": i} for i in range(50)])
        coll.delete_many({"i": {"$gte": 25}})
        count = client.compact()
        assert count == 25

    def test_insert_preserves_id(self, client):
        coll = client.collection("custom_id")
        oid = coll.insert_one({"_id": "my-key", "val": 1})
        assert oid == "my-key"
        doc = coll.find_one({"_id": "my-key"})
        assert doc["val"] == 1


class TestServerAuth:
    def test_auth_required(self):
        import random
        port = random.randint(20000, 30000)
        with tempfile.TemporaryDirectory() as tmp:
            server = MogDBServer(host="127.0.0.1", port=port, dbpath=os.path.join(tmp, "auth"), password="secret")
            t = threading.Thread(target=server.start, daemon=True)
            t.start()
            time.sleep(0.2)

            c = MogDBClient("127.0.0.1", port)
            c.connect()

            # Without auth, operations should fail
            with pytest.raises(MogDBError, match="not authenticated"):
                c.collection("x").insert_one({"a": 1})

            # Wrong password
            assert c.auth("wrong") is False

            # Correct password
            assert c.auth("secret") is True

            # Now operations work
            oid = c.collection("x").insert_one({"a": 1})
            assert len(oid) == 24

            c.close()
            server.stop()

    def test_ping_without_auth(self):
        """Ping does not work without auth either."""
        import random
        port = random.randint(20000, 30000)
        with tempfile.TemporaryDirectory() as tmp:
            server = MogDBServer(host="127.0.0.1", port=port, dbpath=os.path.join(tmp, "ping"), password="pw")
            t = threading.Thread(target=server.start, daemon=True)
            t.start()
            time.sleep(0.2)

            c = MogDBClient("127.0.0.1", port)
            c.connect()
            with pytest.raises(MogDBError, match="not authenticated"):
                c.ping()
            c.close()
            server.stop()


class TestServerErrors:
    def test_unknown_command(self, client):
        with pytest.raises(MogDBError, match="unknown or missing cmd"):
            client._call("nonexistent_cmd")

    def test_missing_collection(self, client):
        with pytest.raises(MogDBError):
            client._call("insert_one", doc={})

    def test_connection_refused(self):
        c = MogDBClient("127.0.0.1", 1)
        with pytest.raises((ConnectionRefusedError, OSError)):
            c.connect()


class TestServerConcurrency:
    def test_concurrent_clients(self, server_port):
        """Multiple clients can connect and operate simultaneously."""
        results = {}
        errors = []

        def worker(uid):
            try:
                c = MogDBClient("127.0.0.1", server_port)
                c.connect()
                coll = c.collection(f"concurrent_{uid}")
                for i in range(10):
                    coll.insert_one({"worker": uid, "i": i})
                results[uid] = coll.count()
                c.close()
            except Exception as e:
                errors.append((uid, str(e)))

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors: {errors}"
        # Each worker's collection should have 10 docs
        for uid in range(5):
            c = MogDBClient("127.0.0.1", server_port)
            c.connect()
            assert c.collection(f"concurrent_{uid}").count() == 10
            c.close()
