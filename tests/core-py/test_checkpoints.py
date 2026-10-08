"""Tests for training.checkpoints — find_checkpoint, load_soul, list_checkpoints, etc."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from domain.training._internal.checkpoints import (
    TRAINED_DIR,
    _load_soul_from_path,
    checkpoint_info,
    ckpt_roots,
    compare_checkpoints,
    delete_checkpoint,
    download_checkpoint_path,
    export_all_metrics,
    find_checkpoint,
    get_all_checkpoint_data,
    is_trained_checkpoint,
    list_checkpoints,
    load_lora_soul,
    load_soul,
)
from domain.training._internal.state import CHECKPOINTS_DIR, LORA_DIR, TURBO_DIR


def _reset_dirs():
    for d in (CHECKPOINTS_DIR, TURBO_DIR, LORA_DIR):
        d.mkdir(parents=True, exist_ok=True)


def _make_soul_file(path: Path, content: str = "x" * 5000):
    path.write_text(content)


def _make_soul_with_meta(path: Path, meta: dict):
    path.write_text("x" * 5000)
    meta_path = path.with_suffix(path.suffix + ".meta.json")
    meta_path.write_text(json.dumps(meta))


@pytest.fixture(autouse=True)
def _isolated_ckpt_dirs(tmp_path, monkeypatch):
    """Send every checkpoint dir to tmp_path instead of the real repo tree.

    These helpers used to write 5KB fixtures straight into
    ``models/auto-training/`` — a directory that also holds real checkpoints
    (multi-hundred-KB ``*.soul`` plus their ``.points.json``) — and
    ``_reset_dirs()`` only ever called ``mkdir``, so nothing was ever
    removed. Debris accumulated across runs and leaked into production
    listings. ``find_checkpoint`` returns files from whatever these constants
    say, so redirecting them is enough to isolate the whole module.
    """
    import sys

    import domain.training._internal.checkpoints as ckpt_mod

    test_mod = sys.modules[__name__]
    mapping = {
        "CHECKPOINTS_DIR": tmp_path / "checkpoints",
        "TURBO_DIR": tmp_path / "turbo",
        "LORA_DIR": tmp_path / "lora",
        "TRAINED_DIR": tmp_path / "models",
    }
    for name, path in mapping.items():
        path.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(ckpt_mod, name, path)
        # This module imported three of these by name; patch them too so the
        # assertions below write and read the same tmp dirs the code does.
        if name in vars(test_mod):
            monkeypatch.setattr(test_mod, name, path)
    return mapping


# ── find_checkpoint ─────────────────────────────────────────────────────────


class TestFindCheckpoint:
    def setup_method(self):
        _reset_dirs()

    def test_find_soul_file(self):
        _make_soul_file(CHECKPOINTS_DIR / "test_ckpt.soul")
        result = find_checkpoint("test_ckpt.soul")
        assert result is not None
        assert result.name == "test_ckpt.soul"

    def test_find_soul_by_name(self):
        _make_soul_file(CHECKPOINTS_DIR / "model.soul")
        result = find_checkpoint("model")
        assert result is not None

    def test_find_slo_file(self):
        _make_soul_file(CHECKPOINTS_DIR / "model.slo")
        result = find_checkpoint("model.slo")
        assert result is not None

    def test_find_in_turbo_dir(self):
        _make_soul_file(TURBO_DIR / "turbo_ckpt.soul")
        result = find_checkpoint("turbo_ckpt.soul")
        assert result is not None

    def test_not_found(self):
        result = find_checkpoint("nonexistent_model_xyz")
        assert result is None

    def test_no_path_traversal(self):
        # The lookup is SANITIZED, not rejected: the input collapses to its
        # basename, so a traversal request resolves inside the checkpoint
        # dirs or not at all.
        #
        # Asserting `is None` here (the previous version) was backwards: this
        # test creates evil.soul first, so sanitization FINDS it and the old
        # assertion failed — while a build with the basename stripping removed
        # would resolve ../../../etc/evil.soul, miss, and pass. The old check
        # was satisfied exactly when the protection was gone.
        _make_soul_file(CHECKPOINTS_DIR / "evil.soul")

        result = find_checkpoint("../../../etc/evil.soul")

        assert result is not None
        assert result.resolve() == (CHECKPOINTS_DIR / "evil.soul").resolve()

    def test_traversal_never_escapes_the_checkpoint_dirs(self):
        # A file reachable only by stepping OUT of the checkpoint dirs must
        # never come back, whatever the input spelling.
        outside = CHECKPOINTS_DIR.parent / "escapee.soul"
        outside.write_text("x" * 100)
        try:
            for spelling in ("../escapee.soul", "../../models/escapee.soul"):
                result = find_checkpoint(spelling)
                if result is not None:
                    assert result.resolve() != outside.resolve()
                    assert str(result.resolve()).startswith(str(CHECKPOINTS_DIR.resolve()))
        finally:
            outside.unlink(missing_ok=True)

    def test_finds_legacy_sibling_from_canonical_name(self):
        # A file that exists ONLY at the double-appended spelling. Asking for
        # the canonical name used to come back not-found here, while
        # domain.inference.load_soul read it without complaint — two owners,
        # two answers to "where is this checkpoint?".
        _make_soul_file(CHECKPOINTS_DIR / "orphan.soul.soul")
        result = find_checkpoint("orphan.soul")
        assert result is not None
        assert result.name == "orphan.soul.soul"

    def test_named_spelling_wins_over_canonical(self):
        # Both spellings can hold DIFFERENT checkpoints. The name the caller
        # gave is the file it asked for — canonical-first would silently swap
        # in the neighbour, and nothing downstream would catch it.
        _make_soul_file(CHECKPOINTS_DIR / "shadow.soul")
        _make_soul_file(CHECKPOINTS_DIR / "shadow.soul.soul")

        assert find_checkpoint("shadow.soul.soul").name == "shadow.soul.soul"
        assert find_checkpoint("shadow.soul").name == "shadow.soul"


# ── load_soul ───────────────────────────────────────────────────────────────


class TestLoadSoul:
    def setup_method(self):
        _reset_dirs()

    def test_load_with_meta_json(self):
        meta = {"soul_name": "my-custom-soul", "final_train_loss": 0.5}
        _make_soul_with_meta(CHECKPOINTS_DIR / "test.soul", meta)
        result = load_soul("test")
        assert result is not None
        # soul_name "my-custom-soul" → after replace "-soul" → "my-custom" != "test"
        assert result["soul"] == "my-custom"
        assert result["loss"] == 0.5

    def test_load_small_soul_skipped(self):
        _make_soul_file(CHECKPOINTS_DIR / "tiny.soul", "x" * 100)
        result = load_soul("tiny")
        assert result is None

    def test_load_slo_file(self):
        _make_soul_file(CHECKPOINTS_DIR / "model.slo")
        with patch("domain.training._internal.checkpoints.read_slo_json_header", return_value=None):
            result = load_soul("model")
        assert result is not None

    def test_load_not_found(self):
        result = load_soul("nonexistent_xyz")
        assert result is None

    def test_load_from_turbo_dir(self):
        meta = {"soul_name": "turbo-soul"}
        _make_soul_with_meta(TURBO_DIR / "turbo_test.soul", meta)
        result = load_soul("turbo_test")
        assert result is not None

    def test_load_with_traits(self):
        meta = {
            "soul_name": "trait-soul",
            "personality_traits": {"curious": 0.8, "creative": 0.6},
            "lineage": "slonet",
        }
        _make_soul_with_meta(CHECKPOINTS_DIR / "traits.soul", meta)
        result = load_soul("traits")
        assert result["traits"]["curious"] == 0.8

    def test_load_with_soul_name_same_as_stem(self):
        meta = {"soul_name": "test"}
        _make_soul_with_meta(CHECKPOINTS_DIR / "test.soul", meta)
        result = load_soul("test")
        assert result["soul"] == "unknown"

    def test_load_finds_legacy_sibling_from_canonical_name(self):
        # The live failure this fixed: checkpoint_info() serves this path, so a
        # canonical name for a file parked at x.soul.soul returned 404
        # "Checkpoint not found" for a checkpoint sitting right there.
        _make_soul_with_meta(CHECKPOINTS_DIR / "orphan.soul.soul", {"soul_name": "orphan-soul"})
        result = load_soul("orphan.soul")
        assert result is not None
        assert result["name"] == "orphan.soul.soul"

    def test_load_prefers_the_named_spelling(self):
        # Two different checkpoints, one stem. Reading either spelling must
        # yield its own row, not the neighbour's.
        _make_soul_with_meta(CHECKPOINTS_DIR / "shadow.soul", {"soul_name": "first-soul"})
        _make_soul_with_meta(CHECKPOINTS_DIR / "shadow.soul.soul", {"soul_name": "second-soul"})

        assert load_soul("shadow.soul")["name"] == "shadow.soul"
        assert load_soul("shadow.soul.soul")["name"] == "shadow.soul.soul"

    def test_row_carries_the_declared_identity_axes(self):
        meta = {
            "soul_name": "axis-soul",
            "provenance": "training",
            "tier": "canonical",
        }
        _make_soul_with_meta(CHECKPOINTS_DIR / "axis.soul", meta)

        row = load_soul("axis")
        assert row["provenance"] == "training"
        assert row["tier"] == "canonical"

    def test_row_falls_back_for_tier_but_never_invents_provenance(self):
        # A checkpoint written before either field existed: tier is answerable
        # from the container's spelling, provenance is not answerable at all —
        # so one back-fills and the other stays empty rather than guessed.
        _make_soul_with_meta(CHECKPOINTS_DIR / "old.soul", {"soul_name": "old-soul"})

        row = load_soul("old")
        assert row["tier"] == "canonical"
        assert row["provenance"] == ""


# ── load_lora_soul ──────────────────────────────────────────────────────────


class TestLoadLoraSoul:
    def setup_method(self):
        LORA_DIR.mkdir(parents=True, exist_ok=True)

    def test_load_lora(self):
        meta = {"soul_name": "lora-soul"}
        _make_soul_with_meta(LORA_DIR / "lora_test.soul", meta)
        result = load_lora_soul("lora_test")
        assert result is not None

    def test_load_lora_not_found(self):
        result = load_lora_soul("nonexistent_lora")
        assert result is None


# ── _load_soul_from_path ───────────────────────────────────────────────────


class TestLoadSoulFromPath:
    def setup_method(self):
        _reset_dirs()

    def test_load_no_meta(self):
        path = CHECKPOINTS_DIR / "nometa.soul"
        _make_soul_file(path)
        with patch("domain.training._internal.checkpoints.read_slo_json_header", return_value=None):
            result = _load_soul_from_path(path)
        assert result is not None
        assert result["soul"] == "unknown"

    def test_load_with_created_at(self):
        meta = {"soul_name": "dated", "created_at": "2024-01-01"}
        _make_soul_with_meta(CHECKPOINTS_DIR / "dated.soul", meta)
        result = _load_soul_from_path(CHECKPOINTS_DIR / "dated.soul")
        assert result["created_at"] == "2024-01-01"


# ── async functions ─────────────────────────────────────────────────────────


class TestAsyncFunctions:
    def setup_method(self):
        _reset_dirs()

    @pytest.mark.asyncio
    async def test_list_checkpoints(self):
        meta = {"soul_name": "list-test"}
        _make_soul_with_meta(CHECKPOINTS_DIR / "list_test.soul", meta)
        result = await list_checkpoints()
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_delete_checkpoint_not_found(self):
        result = await delete_checkpoint("nonexistent")
        assert result == []

    @pytest.mark.asyncio
    async def test_delete_checkpoint_invalid_name(self):
        with pytest.raises(ValueError, match="Invalid"):
            await delete_checkpoint("../evil")

    @pytest.mark.asyncio
    async def test_delete_checkpoint(self):
        _make_soul_file(CHECKPOINTS_DIR / "del_test.soul")
        result = await delete_checkpoint("del_test.soul")
        assert "del_test.soul" in result
        assert not (CHECKPOINTS_DIR / "del_test.soul").exists()

    @pytest.mark.asyncio
    async def test_download_checkpoint_path_not_found(self):
        result = await download_checkpoint_path("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_download_checkpoint_path_invalid(self):
        with pytest.raises(ValueError, match="Invalid"):
            await download_checkpoint_path("../evil")

    @pytest.mark.asyncio
    async def test_download_checkpoint_path(self):
        _make_soul_file(CHECKPOINTS_DIR / "dl_test.soul")
        result = await download_checkpoint_path("dl_test.soul")
        assert result is not None

    @pytest.mark.asyncio
    async def test_checkpoint_info_not_found(self):
        with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
            await checkpoint_info("nonexistent")

    @pytest.mark.asyncio
    async def test_get_all_checkpoint_data(self):
        result = await get_all_checkpoint_data()
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_export_all_metrics(self):
        result = await export_all_metrics()
        assert "exported_at" in result
        assert "total_checkpoints" in result
        assert "checkpoints" in result


class TestListAndDetailAgree:
    """A checkpoint the list reports must always be openable.

    The search roots used to be hand-written per call site: the scan walked
    ``LORA_DIR`` while ``load_soul``/``find_checkpoint``/``delete_checkpoint``
    did not. A checkpoint in ``data/user_adapters`` was therefore listed —
    and even downloadable — yet 404'd the instant it was opened, and Delete
    silently removed nothing. ``ckpt_roots()`` is the single declaration.
    """

    @pytest.mark.asyncio
    async def test_every_root_yields_an_openable_checkpoint(self):
        # One file per root; models/ only ever lists final saves, so match
        # that spelling there.
        _make_soul_file(CHECKPOINTS_DIR / "probe_auto.soul")
        _make_soul_file(TURBO_DIR / "probe_turbo.soul")
        _make_soul_file(LORA_DIR / "probe_lora.soul")
        _make_soul_file(TRAINED_DIR / "probe_final_trained.soul")

        assert set(ckpt_roots()) == {CHECKPOINTS_DIR, TURBO_DIR, LORA_DIR, TRAINED_DIR}

        listed = {row["name"] for row in await list_checkpoints()}
        for name in (
            "probe_auto.soul",
            "probe_turbo.soul",
            "probe_lora.soul",
            "probe_final_trained.soul",
        ):
            assert name in listed, f"{name} must appear in the list"
            # Listed, then all three consumers must agree it exists — this is
            # exactly the trio that disagreed before ckpt_roots().
            assert find_checkpoint(name) is not None, name
            assert load_soul(name) is not None, name
            info = await checkpoint_info(name)
            assert info.get("name") == name

    @pytest.mark.asyncio
    async def test_sidecarless_file_is_unnamed_not_absent(self):
        # No sidecar means nothing NAMES the file — not that it is gone.
        # Reporting FileNotFoundError here made the list and the detail
        # endpoint contradict each other about the same bytes.
        _make_soul_file(CHECKPOINTS_DIR / "probe_naked.soul")

        info = await checkpoint_info("probe_naked.soul")
        assert info["soul"] == "unknown"
        # Byte-derived identity still describes it for the caller.
        assert info.get("model_path")

    @pytest.mark.asyncio
    async def test_detail_states_format_even_with_a_stub_sidecar(self):
        # checkpoint_info gated this enrichment on soul == "unknown", so a
        # sidecar that merely NAMED the file suppressed format entirely. The
        # dialog then dropped its whole Identity block — "not recorded"
        # rendered as "nothing to show", and the file's real container was
        # never stated anywhere.
        _make_soul_with_meta(CHECKPOINTS_DIR / "probe_stub.soul", {"soul_name": "stub"})

        info = await checkpoint_info("probe_stub.soul")
        assert info["soul"] == "stub"
        assert info.get("format"), "detail must say what container this is"

    @pytest.mark.asyncio
    async def test_enrichment_never_overwrites_declared_identity(self):
        # Derived values fill gaps only — a stored hash/tier stays exactly as
        # the writer wrote it, or read-time derivation could drift identity.
        _make_soul_with_meta(
            CHECKPOINTS_DIR / "probe_declared.soul",
            {
                "soul_name": "declared",
                "integrity_hash": "storedhash0000",
                "tier": "interchange",
                "provenance": "export",
            },
        )

        info = await checkpoint_info("probe_declared.soul")
        assert info["integrity_hash"] == "storedhash0000"
        assert info["tier"] == "interchange"
        assert info["provenance"] == "export"

    @pytest.mark.asyncio
    async def test_lookup_and_listing_never_classify_each_row(self, monkeypatch):
        # The mirror of detail's enrichment. find_checkpoint answers "where
        # is it" and list_checkpoints answers "what rows exist" — neither
        # probes bytes. classify_soul per row is the 81.74s cmd_models
        # defect (read_sidecar's size guard fixed half of it; not calling
        # it at all fixes the rest), and a timing assertion would flake
        # under the load this box runs at, so spy instead: the call either
        # happens or it does not.
        _make_soul_file(CHECKPOINTS_DIR / "probe_spy.soul")
        calls: list[str] = []

        def _spy(path):
            calls.append(str(path))
            raise AssertionError(f"classify_soul leaked into the scan path: {path}")

        monkeypatch.setattr("domain.inference.classify_soul", _spy)

        assert find_checkpoint("probe_spy.soul") is not None
        rows = await list_checkpoints()
        assert rows, "fixture must produce at least one row to be meaningful"
        assert calls == [], f"classify_soul ran over the listing: {calls[:3]}"


class TestTrainedCheckpointSpelling:
    """The legacy double-append spelling is a first-class final job save.

    ``is_trained_checkpoint`` replaced three hand-written
    ``endswith("_trained.soul")`` checks in the scan, in ``delete_checkpoint``
    and in ``download_checkpoint_path``. All three rejected
    ``<stem>_trained.soul.soul``, so four real checkpoints (~100MB) were
    invisible, undeletable and undownloadable — while ``load_soul`` read them
    happily. The disagreement between "can read it" and "can see it" was the
    bug.
    """

    def test_both_spellings_are_final_saves(self):
        assert is_trained_checkpoint("journey_select_trained.soul")
        assert is_trained_checkpoint("journey_select_trained.soul.soul")

    @pytest.mark.parametrize(
        "name",
        [
            "bench_shakespeare.soul",  # real file in models/, but not a job save
            "model.soul",
            "tmp_1791172994.soul",
            "journey_select.soul",  # stem lacks the marker
        ],
    )
    def test_non_saves_are_rejected(self, name):
        # models/ holds benchmarks and ad-hoc exports too; widening this
        # predicate carelessly would make every stray file listable and
        # therefore deletable.
        assert not is_trained_checkpoint(name)

    @pytest.mark.asyncio
    async def test_legacy_spelling_appears_in_the_scan(self):
        _make_soul_file(TRAINED_DIR / "legacy_trained.soul.soul")

        rows = await list_checkpoints()

        assert "legacy_trained.soul.soul" in {r["name"] for r in rows}

    @pytest.mark.asyncio
    async def test_models_root_still_excludes_non_saves(self):
        _make_soul_file(TRAINED_DIR / "bench_shakespeare.soul")

        rows = await list_checkpoints()

        assert "bench_shakespeare.soul" not in {r["name"] for r in rows}

    @pytest.mark.asyncio
    async def test_legacy_spelling_can_be_downloaded(self):
        _make_soul_file(TRAINED_DIR / "legacy_trained.soul.soul")

        found = await download_checkpoint_path("legacy_trained.soul.soul")

        assert found is not None
        assert found.endswith("legacy_trained.soul.soul")

    @pytest.mark.asyncio
    async def test_legacy_spelling_can_be_deleted(self):
        _make_soul_file(TRAINED_DIR / "legacy_trained.soul.soul")

        deleted = await delete_checkpoint("legacy_trained.soul.soul")

        assert "legacy_trained.soul.soul" in deleted
        assert not (TRAINED_DIR / "legacy_trained.soul.soul").exists()


class TestRowAddressedActions:
    """Actions carry the row's address, so they touch the file shown.

    A bare name sweeps every root for a first match — with same-named twins
    in two roots, Load/Compare could act on the other file. The address is
    optional (CLI/job records still pass names alone) but mandatory-proof:
    given a bad address, refuse rather than fall back to the sweep.
    """

    @pytest.mark.asyncio
    async def test_compare_addresses_each_side_by_path(self, monkeypatch):
        built: list[tuple[str, str | None]] = []

        class _FakeProvider:
            async def chat(self, *_args, **_kwargs):
                return "ok"

        async def _fake_build(name, path=None):
            built.append((name, path))
            return _FakeProvider(), {"name": name}

        monkeypatch.setattr(
            "domain.training._internal.checkpoints.build_checkpoint_provider", _fake_build
        )

        result = await compare_checkpoints(
            "twin",
            "twin",
            "hi",
            path_a="models/checkpoints/a/twin.soul",
            path_b="models/lora/b/twin.soul",
        )

        assert built == [
            ("twin", "models/checkpoints/a/twin.soul"),
            ("twin", "models/lora/b/twin.soul"),
        ]
        assert result["a"]["text"] == "ok"
        assert result["b"]["text"] == "ok"

    @pytest.mark.asyncio
    async def test_compare_without_paths_stays_name_based(self, monkeypatch):
        built: list[tuple[str, str | None]] = []

        class _FakeProvider:
            async def chat(self, *_args, **_kwargs):
                return "ok"

        async def _fake_build(name, path=None):
            built.append((name, path))
            return _FakeProvider(), {"name": name}

        monkeypatch.setattr(
            "domain.training._internal.checkpoints.build_checkpoint_provider", _fake_build
        )

        await compare_checkpoints("left", "right", "hi")

        # Legacy callers (CLI, job records) get exactly the old contract.
        assert built == [("left", None), ("right", None)]

    @pytest.mark.asyncio
    async def test_delete_by_path_refuses_a_directory(self):
        # A path param can address a directory squatting in a root;
        # unlink() on it would raise. Refuse — files only.
        squat = CHECKPOINTS_DIR / "squat.soul"
        squat.mkdir()

        deleted = await delete_checkpoint("squat.soul", path=str(squat))

        assert deleted == []
        assert squat.is_dir()
