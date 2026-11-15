# DataStep chips: kind filter + name humanization

**Status:** done
**Sprint:** —
**gh:** —

## Task
DataStep's "Your Datasets" chip cloud renders every cache entry, including
system stores (Auth-Mogdb, Backups, Errors Json…). Backend already
classifies `kind` (d31ba0b4) and DatasetSelector already filters
(25b7c8e2) — chips never consumed it.

## Body
1. Filter chip list to `kind == null || kind === 'dataset'` (same rule
   as DatasetSelector).
2. Humanize display names: strip storage suffixes (` Json`, ` Mogdb`),
   replace `_`/`-` with spaces, title-case.
3. Keep full raw id in tooltip / aria for power users.

## Constraints
- Match DatasetSelector's kind filter exactly (untagged entries show).
- No backend changes — brain already exists in cache_tags.py.
