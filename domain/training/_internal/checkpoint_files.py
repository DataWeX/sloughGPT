"""Checkpoint file ownership — the one predicate for "which files are this job's".

Three operations must agree about a checkpoint's identity:

- **WRITE** ``SloughGPTTrainer.save_checkpoint`` names the file,
- **READ**  ``CheckpointManager`` scan and crash-recovery find the file,
- **DELETE** ``_prune_stale_checkpoints`` / ``_enforce_disk_limit`` evict it.

When they disagree the failure is silent and lands on the reader. The
directory is shared — ``models/auto-training`` is the default
``checkpoint_dir`` for every job — so a run that pruned directory-wide
deleted checkpoints belonging to *other* jobs, and those jobs' Resume button
then failed with "checkpoint missing" on a file that had been perfectly valid
when it was recorded. Ownership enforced while loading and ignored while
unlinking is precisely that bug: the read side was job-scoped
(``stem_prefix``), the delete side was not.

A second, smaller divergence lived between the two write paths: the trainer
named files after a **raw** ``Path(data_path).parent.name`` while the API
sanitised its stem. A dataset parent containing a space or a dot produced a
file the sanitising reader could never own, so the resume scan came back empty
with nothing to report. Both paths now go through :func:`stem_for`.

The ``.soul`` *filename grammar* (suffix, legacy spellings, sidecar names) is
**not** owned here — that is ``soul_path`` / ``soul_read_candidates`` /
``soul_meta_path`` in ``domain.inference``. This module only decides ownership
and hands naming to those functions via the callers.
"""

from __future__ import annotations

from pathlib import Path

from domain.infrastructure._internal.atomic_write import is_temp_name

# Cap on a checkpoint stem. One number for every writer: the copies this
# replaces disagreed (80 for feed stems, 120 everywhere else), and a reader
# re-sanitising with a different cap than the writer used can no longer match
# the file it is looking for. 120 keeps ``<stem>_<10-digit-ts>.soul`` well
# inside the 255-byte ``NAME_MAX`` for a single-component filename.
STEM_MAX_LEN = 120


def sanitize_stem(raw: object, cap: int = STEM_MAX_LEN) -> str:
    """Return ``raw`` as a filename-safe checkpoint stem.

    Every character outside ``[A-Za-z0-9_-]`` becomes ``_``. Note
    ``str.isalnum()`` is Unicode-aware, so a non-ASCII dataset name survives
    as itself rather than degrading to underscores — as it always has here.

    Args:
        raw: Value to clean; ``None`` and other falsy values yield ``""``.
        cap: Maximum length, defaults to :data:`STEM_MAX_LEN`.

    Returns:
        The cleaned stem, truncated to ``cap`` characters.
    """
    cleaned = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(raw or ""))
    return cleaned[:cap]


def stem_for(data_path: str, soul_name: str = "assistant") -> str:
    """The stem this run's checkpoints are written under and pruned by.

    Derives from the dataset path's parent directory — ``.../corpus.jsonl``
    under ``.../python_flask/`` is ``python_flask`` — then sanitises it, so the
    name the writer uses and the name the pruner matches cannot drift apart.

    Args:
        data_path: Dataset file or directory the trainer was pointed at. An
            empty string falls back to ``soul_name`` (and a bare filename with
            no parent component falls back too, where ``parent.name`` is ``""``).
        soul_name: Name recorded on the profile, used when ``data_path`` names
            nothing useful.

    Returns:
        A non-empty stem, capped at :data:`STEM_MAX_LEN`. Never ``""``: an
        empty stem would make :func:`is_owned` match nothing and quietly turn
        retention off, so the last-resort default is ``"checkpoint"``.
    """
    candidate = Path(data_path).parent.name if data_path else soul_name
    return sanitize_stem(candidate) or sanitize_stem(soul_name) or "checkpoint"


def is_owned(path: str | Path, stem: str | None) -> bool:
    """Whether ``path`` is one of the files written under ``stem``.

    The rule is ``<stem>`` itself or ``<stem>_<anything>`` — a checkpoint
    directory is shared, so a prefix test must not let ``api_conversations``
    claim ``api_conversations_other_1790.soul`` while ``apiconversations`` stays
    a different job entirely. The underscore is what separates them.

    Args:
        path: Candidate file; only its stem is inspected.
        stem: Ownership stem, or ``None``/``""`` to own nothing.

    Returns:
        True when ``path`` belongs to ``stem``.
    """
    if not stem:
        return False
    name = Path(path).stem
    return name == stem or name.startswith(stem + "_")


def is_temp_artifact(path: str | Path) -> bool:
    """Whether ``path`` is a half-written temp, not a finished checkpoint.

    The ``_`` boundary above would otherwise admit ``run_1790.tmp.npz`` as an
    owned file: its stem is ``run_1790.tmp``, which starts with ``run_``. A
    scan would then try to resume from it and a prune would count it toward the
    retention window.
    """
    return is_temp_name(Path(path).name)


def list_owned(
    directory: str | Path,
    stem: str | None = None,
    suffixes: tuple[str, ...] = (".soul", ".npz"),
) -> list[Path]:
    """Checkpoint files under ``directory``, newest first.

    The single glob both the scanner and the pruner go through, so read and
    delete can no longer drift apart: whatever this returns is what gets
    loaded, and only what this returns gets unlinked.

    Args:
        directory: Checkpoint directory; a missing directory yields ``[]``
            (the caller decides whether that is an error).
        stem: Restrict to one job's files. ``None`` means no ownership filter —
            for callers that pinned a path themselves, never for retention.
        suffixes: Which extensions to consider. The scan wants both ``.soul``
            and ``.npz`` (either can resume); retention deliberately keeps its
            historical ``(".soul",)`` scope — the ``.npz`` fallback is not a
            file this run wrote, and sweeping it up would change what survives
            a prune.

    Returns:
        Matching files sorted by mtime descending, ties broken by name so the
        order is deterministic.

    Raises:
        OSError: Propagated from ``stat`` — callers wrap it when a failed scan
            must not abort the operation.
    """
    directory = Path(directory)
    if not directory.is_dir():
        return []
    candidates: list[Path] = []
    for suffix in suffixes:
        candidates.extend(directory.glob(f"*{suffix}"))
    files = [
        p for p in candidates if not is_temp_artifact(p) and (stem is None or is_owned(p, stem))
    ]
    return sorted(files, key=lambda p: (p.stat().st_mtime, p.name), reverse=True)
