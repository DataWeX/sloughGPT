"""CLI client for mogdb-server — interactive or one-shot."""

import argparse
import cmd
import logging
import shlex
from typing import Optional

from .client import MogDBClient, MogDBError

logger = logging.getLogger("mogdb.cli")


class MogDBShell(cmd.Cmd):
    """Interactive REPL for a remote MogDB server."""

    intro = "MogDB shell. Type help or ? to list commands.\n"
    prompt = "mogdb> "

    def __init__(self, client: MogDBClient):
        super().__init__()
        self._client = client
        self._collection: Optional[str] = None

    def _require_collection(self) -> bool:
        if not self._collection:
            print("No collection selected. Use `use <name>` first.")
            return False
        return True

    def do_use(self, arg: str) -> None:
        """use <name> — Switch to a collection."""
        self._collection = arg.strip()
        if self._collection:
            print(f"Switched to collection '{self._collection}'")

    def do_insert(self, arg: str) -> None:
        """insert <json> — Insert a document into the current collection."""
        if not self._require_collection():
            return
        import json
        try:
            doc = json.loads(arg)
        except json.JSONDecodeError as e:
            print(f"Invalid JSON: {e}")
            return
        try:
            oid = self._client.collection(self._collection).insert_one(doc)
            print(f"Inserted: {oid}")
        except MogDBError as e:
            print(f"Error: {e}")

    def do_find(self, arg: str) -> None:
        """find [<json_query>] — Find documents in the current collection."""
        if not self._require_collection():
            return
        query = None
        if arg.strip():
            import json
            try:
                query = json.loads(arg)
            except json.JSONDecodeError as e:
                print(f"Invalid JSON: {e}")
                return
        try:
            docs = self._client.collection(self._collection).find(query)
            import json
            for d in docs:
                print(json.dumps(d, default=str))
            print(f"({len(docs)} results)")
        except MogDBError as e:
            print(f"Error: {e}")

    def do_count(self, arg: str) -> None:
        """count [<json_query>] — Count documents."""
        if not self._require_collection():
            return
        query = None
        if arg.strip():
            import json
            try:
                query = json.loads(arg)
            except json.JSONDecodeError:
                query = None
        try:
            n = self._client.collection(self._collection).count(query)
            print(n)
        except MogDBError as e:
            print(f"Error: {e}")

    def do_delete(self, arg: str) -> None:
        """delete <json_query> — Delete documents matching query."""
        if not self._require_collection():
            return
        import json
        try:
            query = json.loads(arg)
        except json.JSONDecodeError as e:
            print(f"Invalid JSON: {e}")
            return
        try:
            n = self._client.collection(self._collection).delete_many(query)
            print(f"Deleted {n} document(s)")
        except MogDBError as e:
            print(f"Error: {e}")

    def do_collections(self, arg: str) -> None:
        """collections — List all collections."""
        try:
            cols = self._client.list_collections()
            if cols:
                print("\n".join(cols))
            else:
                print("(empty)")
        except MogDBError as e:
            print(f"Error: {e}")

    def do_ping(self, arg: str) -> None:
        """ping — Health check."""
        try:
            print(self._client.ping())
        except MogDBError as e:
            print(f"Error: {e}")

    def do_exit(self, arg: str) -> bool:
        """exit — Exit the shell."""
        print("Bye.")
        return True

    def do_EOF(self, arg: str) -> bool:
        return self.do_exit(arg)


def main() -> None:
    parser = argparse.ArgumentParser(description="MogDB CLI client")
    parser.add_argument("--host", default="127.0.0.1", help="server host")
    parser.add_argument("--port", type=int, default=27017, help="server port")
    parser.add_argument("--password", default=None, help="server password")
    parser.add_argument("-c", "--command", default=None, help="one-shot command")
    args = parser.parse_args()

    client = MogDBClient(host=args.host, port=args.port)
    try:
        client.connect()
    except ConnectionRefusedError:
        print(f"Could not connect to mogdb://{args.host}:{args.port}")
        return 1

    if args.password:
        if not client.auth(args.password):
            print("Authentication failed")
            return 1

    if args.command:
        MogDBShell(client).onecmd(args.command)
    else:
        MogDBShell(client).cmdloop()

    client.close()
    return 0


if __name__ == "__main__":
    main()
