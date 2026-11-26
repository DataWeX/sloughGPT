"""Distill-corpus adapter — build the formalized distillation corpus from its v1 manifest.

Card 20261125_203428 (user ruling: standardize the corpus BEFORE training scale —
a corpus nobody wrote down trains nobody knows what).

    sources (manifest) → readers (journal replay, jsonl, paired logs)
                       → filters (manifest transformations: contamination, length, model allowlist)
                       → dedup (highest tier wins) → split (held-out, deterministic hash)
                       → build() writes input.txt + heldout.jsonl + build_stats.json

Composition ONLY — no trainer imports (known-debt rule): the built corpus is
consumed by the EXISTING distillation path — ``apps/api/server/training/distill.py``
resolves ``data/<dataset>/input.txt`` (candidate #1), so ``dataset="distill_corpus"``
hands off with zero new wiring. The manifest is the SloughGPT Standard v1 contract
(``dataset_manifest.load_manifest``); count/hash drift raises ``ManifestError`` —
the gate is the build itself, never a silent pass.

Source stability (the generation-counter question for append-only data):
  - ``frozen``   — exact record_count AND sha256 prefix must match (static artifacts)
  - ``growing``  — record_count is a MINIMUM (append-only logs/journals)
  - ``empty_ok`` — must hold the declared count of files (0 today; bump manifest when populated)
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any

from domain.training._internal.dataset_manifest import ManifestError, load_manifest

#: Template markers that mark a record as corrupted self-play output.
_DEFAULT_MARKERS = ("<|im_start|>", "<|im_end|>")
#: Tier order for dedup: the highest-quality copy of a duplicate wins.
_TIER_ORDER = ("A", "B", "C")


def _norm_text(messages: list[dict[str, Any]]) -> str:
    """Normalized turn text — the dedup key (whitespace-strip only, exact)."""
    return "\n".join(
        f"{str(m.get('role', ''))}:{str(m.get('content', '')).strip()}" for m in messages
    )


def _render(messages: list[dict[str, Any]]) -> str:
    """Render one conversation to plain text (the format distill.py tokenizes)."""
    lines = [
        f"{str(m.get('role', 'unknown')).capitalize()}: {m.get('content', '')}" for m in messages
    ]
    return "\n".join(lines) + "\n\n"


def _content_len(messages: list[dict[str, Any]]) -> int:
    return max((len(str(m.get("content", ""))) for m in messages), default=0)


def _resolve_paths(base: Path, uri: str) -> list[Path]:
    """Source uri → existing paths (glob-aware, relative to the manifest's directory)."""
    pure = PurePosixPath(uri)
    if any(ch in uri for ch in "*?["):
        base_dir = (base / str(pure.parent)).resolve()
        return (
            sorted(p for p in base_dir.glob(pure.name) if p.exists()) if base_dir.is_dir() else []
        )
    p = base / uri
    return [p] if p.exists() else []


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and isinstance(obj.get("messages"), list):
            records.append(obj)
    return records


def _read_paired_logs(paths: list[Path]) -> list[dict[str, Any]]:
    """response_logs raw: EVERY parseable line → a record (stability counts are raw).

    Junk filtering (model allowlist, empty pairs) happens in the filter stage so
    the manifest's record_count — the source inventory — matches what is read here.
    """
    records: list[dict[str, Any]] = []
    for path in paths:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue
            records.append(
                {
                    "messages": [
                        {"role": "user", "content": obj.get("user_message") or ""},
                        {"role": "assistant", "content": obj.get("assistant_response") or ""},
                    ],
                    "_model": str(obj.get("model", "")),
                }
            )
    return records


def _read_sessions_journal(path: Path) -> list[dict[str, Any]]:
    """Replay the docstore sessions journal to live state; live sessions → records."""
    state: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        op, data = obj.get("op"), obj.get("data")
        if not isinstance(data, dict):
            continue
        sid = data.get("id")
        if op == "insert":
            state[sid] = data
        elif op == "update" and sid in state:
            state[sid] = {**state[sid], **data}
        elif op == "delete":
            state.pop(sid, None)
    return [
        {"messages": s["messages"]}
        for s in state.values()
        if isinstance(s.get("messages"), list) and s["messages"]
    ]


def _read_agent_runs(dir_path: Path) -> list[dict[str, Any]]:
    """Agent runs (future primary source): jsonl with a ``messages`` list per line.

    Absent/empty today (declared ``empty_ok`` in the manifest) — when runs start
    landing this picks them up without a schema change (count version-bump only).
    """
    if not dir_path.is_dir():
        return []
    out: list[dict[str, Any]] = []
    for path in sorted(dir_path.rglob("*.jsonl")):
        out.extend(_read_jsonl(path))
    return out


def _check_count(actual: int, src: dict[str, Any]) -> None:
    """Stability check — frozen = exact, growing = minimum, empty_ok = file count."""
    stability = src.get("stability", "frozen")
    uri = src.get("uri", "")
    declared = int(src.get("record_count", -1))
    if stability == "growing":
        if actual < declared:
            raise ManifestError(
                f"Growing source {uri} collapsed: read {actual} records, "
                f"manifest minimum {declared} — journal/log regression, investigate"
            )
    elif actual != declared:
        raise ManifestError(
            f"Source {uri} drifted: read {actual} records, manifest says {declared} — "
            "source changed under the spec; recon and version-bump the manifest"
        )


def _check_hash(path: Path, src: dict[str, Any]) -> None:
    """sha256 prefix must match for frozen sources (growing = as-of note, never checked)."""
    if src.get("stability") != "frozen" or not src.get("sha256_prefix"):
        return
    if not path.is_file():
        return
    digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    if digest != src["sha256_prefix"]:
        raise ManifestError(
            f"Frozen source {src.get('uri')} hash drift: {digest} != {src['sha256_prefix']}"
        )


def build(
    manifest_path: str | Path,
    out_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Build the corpus from its manifest. Returns build stats (also written to disk).

    Outputs (in ``out_dir`` or the manifest's directory):
      - ``input.txt``        train split, plain text (distill.py candidate #1)
      - ``heldout.jsonl``    held-out split, structured records (never trained on)
      - ``build_stats.json`` per-source read/kept counts + content hashes (auditable)
    """
    mp = Path(manifest_path).expanduser()
    manifest = load_manifest(mp)  # SloughGPT Standard v1 contract
    base = mp.parent
    out = Path(out_dir) if out_dir else base
    out.mkdir(parents=True, exist_ok=True)

    cfg = {t["name"]: t.get("config", {}) for t in manifest.get("transformations", [])}
    deg = cfg.get("filter_degenerate", {})
    min_len = int(deg.get("min_content_len", 20))
    markers = tuple(cfg.get("drop_template_contamination", {}).get("markers", _DEFAULT_MARKERS))
    allowlist = [str(a) for a in deg.get("model_allowlist", [])]
    tier_order = [str(t) for t in cfg.get("dedup_exact", {}).get("tier_order", _TIER_ORDER)]
    split_cfg = cfg.get("split_heldout", {})
    heldout_frac = float(split_cfg.get("heldout_fraction", 0.1))
    seed = int(split_cfg.get("seed", 42))

    by_tier: dict[str, list[dict[str, Any]]] = {t: [] for t in tier_order}
    observed: dict[str, int] = {}
    for src in manifest["sources"]:
        uri = str(src["uri"])
        tier = str(src.get("quality_tier", "C"))
        if tier.endswith("_target"):
            tier = "A"
        if tier not in by_tier:
            by_tier[tier] = []
        filters = src.get("filters", [])
        paths = _resolve_paths(base, uri)

        # derived provenance-only sources: hash-check if frozen, never read FOR RECORDS.
        # Line count is still observed (provenance honesty) but never enters the record pool.
        if "derived_source_excluded_from_build" in filters:
            if not paths:
                raise ManifestError(f"Declared source missing: {uri}")
            _check_hash(paths[0], src)
            with paths[0].open(encoding="utf-8", errors="replace") as fh:
                observed[uri] = sum(1 for _ in fh)
            continue

        # empty_ok = file-count expectation (agent_runs: 0 files until populated)
        if src.get("stability") == "empty_ok":
            n_files = sum(1 for p in Path(base / uri).rglob("*") if p.is_file()) if paths else 0
            if n_files != int(src.get("record_count", 0)):
                raise ManifestError(
                    f"{uri} holds {n_files} files but manifest declares "
                    f"{src.get('record_count')} — recon the new source, version-bump, rebuild"
                )
            observed[uri] = n_files
            continue

        if not paths:
            raise ManifestError(f"Declared source missing: {uri}")

        if "journal_replay_to_live_state" in filters:
            recs = _read_sessions_journal(paths[0])
        elif "model_allowlist" in filters:
            recs = _read_paired_logs(paths)
        elif "tolerate_empty_until_populated" in filters:
            recs = _read_agent_runs(paths[0])
        elif paths[0].is_dir():
            recs = [r for p in sorted(paths[0].rglob("*.jsonl")) for r in _read_jsonl(p)]
        else:
            recs = _read_jsonl(paths[0])

        # one record_count check for single-file sources; multi-file (glob) = growing only
        if len(paths) == 1:
            _check_count(len(recs), src)
            _check_hash(paths[0], src)
        elif src.get("stability") != "frozen":
            _check_count(len(recs), src)
        observed[uri] = len(recs)

        # hand-checked content (respect_existing_split) bypasses content filters only
        hand_checked = "respect_existing_split" in filters
        for r in recs:
            msgs = r.get("messages", [])
            if "model_allowlist" in filters:
                model = str(r.get("_model", "")).lower()
                if not any(a in model for a in allowlist):
                    continue  # model_allowlist — only teacher-served models survive
                if not all(str(m.get("content", "")).strip() for m in msgs):
                    continue  # drop_empty_pairs
            text = _norm_text(msgs)
            if any(m in text for m in markers):
                continue  # drop_template_contamination
            if not hand_checked and _content_len(msgs) < min_len:
                continue  # filter_degenerate
            rec: dict[str, Any] = {"messages": msgs, "source": uri, "tier": tier}
            if hand_checked:
                # fixture's split is declared both ways: heldout marked, rest = its train
                rec["forced_split"] = "heldout" if str(r.get("split", "")) == "heldout" else "train"
            by_tier[tier].append(rec)

    # dedup — highest tier first, so the best copy of a duplicate survives
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    dropped_dupes = 0
    for tier in tier_order:
        for rec in by_tier.get(tier, []):
            key = _norm_text(rec["messages"])
            if key in seen:
                dropped_dupes += 1
                continue
            seen.add(key)
            unique.append(rec)

    # split — deterministic hash (order-independent), forced where declared
    def _bucket(key: str) -> float:
        h = hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()
        return int(h[:8], 16) / 0xFFFFFFFF

    train_recs: list[dict[str, Any]] = []
    heldout: list[dict[str, Any]] = []
    for rec in unique:
        key = _norm_text(rec["messages"])
        forced = rec.pop("forced_split", None)
        is_held = (
            forced == "heldout" if forced in ("train", "heldout") else _bucket(key) < heldout_frac
        )
        (heldout if is_held else train_recs).append(rec)

    train_text = "".join(_render(r["messages"]) for r in train_recs)
    (out / "input.txt").write_text(train_text, encoding="utf-8")
    with (out / "heldout.jsonl").open("w", encoding="utf-8") as fh:
        for r in heldout:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    stats: dict[str, Any] = {
        "dataset_id": manifest["dataset_id"],
        "manifest_version": manifest["version"],
        "observed_per_source": observed,
        "read_total": sum(observed.values()),
        "kept_unique": len(unique),
        "dropped_duplicates": dropped_dupes,
        "train_records": len(train_recs),
        "heldout_records": len(heldout),
        "train_chars": len(train_text),
        "train_sha256": hashlib.sha256(train_text.encode()).hexdigest()[:16],
        "heldout_sha256": hashlib.sha256(
            "".join(_norm_text(r["messages"]) for r in heldout).encode()
        ).hexdigest()[:16],
    }
    (out / "build_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats
