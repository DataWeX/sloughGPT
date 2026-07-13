"""Migrate training.db JSON → MogDB training data collection.

Run once: python3 -m scripts.migrate_training_data
"""

import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def migrate():
    training_db = REPO_ROOT / "data" / "training.db"
    if not training_db.exists():
        print(f"no training.db at {training_db}")
        return

    with open(training_db) as f:
        pairs = json.load(f)

    print(f"found {len(pairs)} pairs in training.db")

    # Add mogdb src to path
    sys.path.insert(0, str(REPO_ROOT / "packages" / "mogdb" / "src"))

    from domains.training.mobile_training_store import MobileTrainingStore

    store_path = str(REPO_ROOT / "packages" / "data" / "mobile_training")
    store = MobileTrainingStore(store_path)

    migrated = 0
    skipped = 0
    for p in pairs:
        user_msg = p.get("prompt", "")
        assistant_msg = p.get("response", "")
        if not user_msg or not assistant_msg:
            skipped += 1
            continue

        # Parse timestamp
        ts_str = p.get("timestamp", "")
        try:
            # "2026-04-16T17:03:53.175478" → epoch
            from datetime import datetime, timezone
            dt = datetime.fromisoformat(ts_str)
            timestamp = dt.timestamp()
        except Exception:
            timestamp = time.time()

        store.add_pair(
            user_msg=user_msg,
            assistant_msg=assistant_msg,
            session_id=p.get("session_id", ""),
            quality=p.get("quality", 0.0),
            model=p.get("model", ""),
        )
        migrated += 1

    # Mark already-used pairs as used and synced
    used_ids = []
    for p in store._col.find({"synced": False}):
        used_ids.append(p["_id"])
    if used_ids:
        store.mark_used(used_ids)
        store.mark_synced(used_ids)

    print(f"migrated: {migrated}, skipped: {skipped}, total in store: {store.count()}")
    stats = store.stats()
    print(f"stats: {json.dumps(stats, indent=2)}")
    store.close()


if __name__ == "__main__":
    migrate()
