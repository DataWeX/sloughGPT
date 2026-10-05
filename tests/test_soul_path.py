"""Structure-aware .soul filename grammar — one owner, canonical write, tolerant read."""

from __future__ import annotations

import json
import os
import struct
from pathlib import Path

import pytest

from domain.inference import (
    SloProfile,
    classify_soul,
    load_soul,
    save_soul,
    soul_path,
    soul_read_candidates,
)
from domain.inference._internal.slo_format import (
    SOUL_SUFFIXES,
    SOUL_TIER_POLICY,
    read_sidecar,
)


class TestSoulPathWriteGrammar:
    def test_bare_stem_gets_extension(self):
        assert soul_path("models/x") == "models/x.soul"

    def test_canonical_form_is_idempotent(self):
        assert soul_path("models/x.soul") == "models/x.soul"

    def test_legacy_double_append_collapses_to_single(self):
        assert soul_path("models/x.soul.soul") == "models/x.soul"
        assert soul_path("models/x.soul.soul.soul") == "models/x.soul"

    def test_dot_in_stem_is_not_treated_as_extension(self):
        assert soul_path("models/gpt2.5-model") == "models/gpt2.5-model.soul"

    def test_directory_components_are_never_inspected(self):
        assert soul_path("models/v2.3/x") == "models/v2.3/x.soul"
        assert soul_path("models/v2.3/x.soul.soul") == "models/v2.3/x.soul"

    @pytest.mark.parametrize("ext", [".pt", ".npz", ".safetensors", ".slo"])
    def test_foreign_extensions_raise_instead_of_guessing(self, ext):
        with pytest.raises(ValueError, match="not a .soul checkpoint path"):
            soul_path(f"models/x{ext}")

    def test_sou_is_our_alias_not_a_foreign_format(self):
        # .sou is absent from the foreign list deliberately: we never WRITE a
        # .sou at all, so collapsing the initiator alias cannot mislabel bytes
        # on disk — whereas .pt/.slo would, and still raise above.
        assert soul_path("models/x.sou") == "models/x.soul"


class TestSoulReadCandidates:
    def test_bare_name_probes_canonical_then_legacy_slo(self):
        assert soul_read_candidates("m/x") == ["m/x.soul", "m/x.slo"]

    def test_canonical_name_probes_legacy_double_append(self):
        assert soul_read_candidates("m/x.soul") == ["m/x.soul", "m/x.soul.soul"]

    def test_legacy_double_append_probes_itself_first(self):
        # As given FIRST, like .sou and .slo. Both spellings can be genuinely
        # different checkpoints, so canonical-first silently returned the file
        # the caller did not name.
        assert soul_read_candidates("m/x.soul.soul") == ["m/x.soul.soul", "m/x.soul"]

    def test_slo_read_form_falls_back_to_soul(self):
        assert soul_read_candidates("m/x.slo") == ["m/x.slo", "m/x.soul"]

    def test_sou_read_form_falls_back_to_soul(self):
        # As given FIRST, like .slo: .sou names a different model, so it is
        # not a spelling of .soul to be normalized away.
        assert soul_read_candidates("m/x.sou") == ["m/x.sou", "m/x.soul"]


class TestSaveLoadRoundtrip:
    def test_save_canonicalizes_and_pairs_meta(self, tmp_path):
        out = save_soul(None, str(tmp_path / "demo.soul.soul"), weights_only=True)
        assert out == str(tmp_path / "demo.soul")
        assert (tmp_path / "demo.soul").is_file()
        assert (tmp_path / "demo.soul.meta.json").is_file()
        assert not (tmp_path / "demo.soul.soul").exists()
        assert not (tmp_path / "demo.soul.soul.meta.json").exists()

    def test_save_accepts_bare_stem(self, tmp_path):
        out = save_soul(None, str(tmp_path / "demo"), weights_only=True)
        assert out == str(tmp_path / "demo.soul")
        assert (tmp_path / "demo.soul").is_file()

    def test_load_reads_legacy_double_append_via_canonical_name(self, tmp_path):
        save_soul(None, str(tmp_path / "demo.soul"), weights_only=True)
        os.rename(tmp_path / "demo.soul", tmp_path / "demo.soul.soul")
        os.rename(
            tmp_path / "demo.soul.meta.json",
            tmp_path / "demo.soul.soul.meta.json",
        )

        loaded = load_soul(str(tmp_path / "demo.soul"))
        assert loaded is not None

    def test_load_opens_legacy_name_when_given_directly(self, tmp_path):
        save_soul(None, str(tmp_path / "demo.soul"), weights_only=True)
        os.rename(tmp_path / "demo.soul", tmp_path / "demo.soul.soul")

        loaded = load_soul(str(tmp_path / "demo.soul.soul"))
        assert loaded is not None


class TestReadProbeAdvances:
    """The probe must advance past a candidate, not die on the first miss.

    soul_read_candidates() promises an ordered list; without these, a
    plain-text .slo sitting beside the real checkpoint blocks the load.
    """

    def test_text_slo_advances_to_soul_sibling(self, tmp_path):
        # A .slo profile opens with the bytes "SOUL" — it clears the magic
        # check, so the advance has to happen on the parse failure, not the
        # magic check.
        (tmp_path / "a.slo").write_text("SOUL myprofile\nLINEAGE slo\n")
        save_soul(None, str(tmp_path / "a.soul"), weights_only=True)

        assert load_soul(str(tmp_path / "a.slo")) is not None

    def test_lone_text_slo_raises_valueerror_not_struct_error(self, tmp_path):
        # No .soul sibling to fall through to: the parse failure is the final
        # answer. struct.error must never escape — the contract is ValueError.
        (tmp_path / "b.slo").write_text("SOUL lone\n")

        with pytest.raises(ValueError, match="Invalid .soul file"):
            load_soul(str(tmp_path / "b.slo"))

    def test_missing_file_lists_every_probed_candidate(self, tmp_path):
        with pytest.raises(FileNotFoundError) as exc:
            load_soul(str(tmp_path / "nothing.soul"))

        msg = str(exc.value)
        assert "nothing.soul" in msg
        assert "nothing.soul.soul" in msg


class TestSouIsADifferentModel:
    """``.sou`` names a different model, not a spelling of ``.soul``.

    Both files are valid SOUL containers, so the magic check, the parse and
    the finder globs all pass on whichever one the probe opens first. Probe
    order is the ONLY defence against handing back our model when the caller
    asked for theirs — which is exactly what the pre-fix order (canonical
    first) got wrong.
    """

    @staticmethod
    def _write_pair(tmp_path):
        """Put ``m.sou`` (theirs) beside ``m.soul`` (ours), same stem.

        save_soul canonicalizes every write to ``.soul``, so the ``.sou`` half
        is produced by renaming — the same way a foreign file would arrive.
        """
        save_soul(
            None,
            str(tmp_path / "theirs_src"),
            soul_profile=SloProfile(name="theirs"),
            weights_only=True,
        )
        save_soul(
            None,
            str(tmp_path / "m"),
            soul_profile=SloProfile(name="ours"),
            weights_only=True,
        )
        os.rename(tmp_path / "theirs_src.soul", tmp_path / "m.sou")
        assert (tmp_path / "m.sou").is_file()
        assert (tmp_path / "m.soul").is_file()

    def test_sou_request_returns_the_sou_model_not_the_soul_one(self, tmp_path):
        self._write_pair(tmp_path)

        soul, _state = load_soul(str(tmp_path / "m.sou"))
        assert soul.name == "theirs"

    def test_soul_request_still_returns_the_soul_model(self, tmp_path):
        # The sibling .sou must never hijack a canonical read either.
        self._write_pair(tmp_path)

        soul, _state = load_soul(str(tmp_path / "m.soul"))
        assert soul.name == "ours"

    def test_sou_absent_falls_through_to_soul(self, tmp_path):
        # Degrade, don't fail: a .sou that was never there still finds our
        # model sitting beside it.
        save_soul(
            None,
            str(tmp_path / "m"),
            soul_profile=SloProfile(name="ours"),
            weights_only=True,
        )
        assert not (tmp_path / "m.sou").exists()

        soul, _state = load_soul(str(tmp_path / "m.sou"))
        assert soul.name == "ours"


def _write_checkpoint(tmp_path, stem: str, soul_name: str) -> Path:
    """Write one checkpoint and rename it onto *stem* — canonical spelling."""
    save_soul(
        None,
        str(tmp_path / f"_{stem}_src"),
        soul_profile=SloProfile(name=soul_name),
        weights_only=True,
    )
    target = tmp_path / f"{stem}.soul"
    os.rename(tmp_path / f"_{stem}_src.soul", target)
    os.rename(
        tmp_path / f"_{stem}_src.soul.meta.json",
        tmp_path / f"{stem}.soul.meta.json",
    )
    return target


def _read_header(path: Path) -> dict:
    """Parse the JSON embedded in a .soul body — no sidecar involved.

    Layout is ``SOUL | <I version> | <I json_len> | json_bytes``. Going
    through the bytes rather than a loader keeps this test honest: it proves
    the declaration is IN the header a sidecar-less reader parses, not merely
    reachable through some other path that happened to read the sidecar.
    """
    with open(path, "rb") as fh:
        magic = fh.read(4)
        assert magic == b"SOUL", f"not a soul container: {magic!r}"
        fh.read(4)  # version
        (length,) = struct.unpack("<I", fh.read(4))
        return json.loads(fh.read(length))


class TestClassifySoul:
    """classify_soul — what a model file IS, derived from its own bytes.

    The motivating defect: ``journey_select_trained.soul`` and
    ``journey_select_trained.soul.soul`` are two different training runs, yet
    both sidecars declare ``name=sloughgpt`` / ``version=1.0.0`` /
    ``base_model=sloughgpt``. Nothing in the declared identity separates them,
    and canonical-first probing made the second one unreachable entirely.
    """

    @staticmethod
    def _make_shadowed_pair(tmp_path) -> tuple[Path, Path]:
        """Two DISTINCT checkpoints occupying one stem — the real-world case."""
        canonical = _write_checkpoint(tmp_path, "ck", "alpha")

        save_soul(
            None,
            str(tmp_path / "_second"),
            soul_profile=SloProfile(name="beta"),
            weights_only=True,
        )
        legacy = tmp_path / "ck.soul.soul"
        os.rename(tmp_path / "_second.soul", legacy)
        os.rename(tmp_path / "_second.soul.meta.json", tmp_path / "ck.soul.soul.meta.json")
        return canonical, legacy

    def test_distinct_checkpoints_under_one_name_are_flagged(self, tmp_path):
        canonical, _legacy = self._make_shadowed_pair(tmp_path)

        ident = classify_soul(str(canonical))

        assert ident.shadowed is True
        assert len(ident.variants) == 2
        # The uniqueness key: two hashes, so the name no longer pretends to
        # identify which checkpoint you mean.
        assert len({v.integrity_hash for v in ident.variants}) == 2

    def test_canonical_wins_the_probe_while_the_sibling_survives(self, tmp_path):
        canonical, legacy = self._make_shadowed_pair(tmp_path)

        ident = classify_soul(str(canonical))

        assert ident.resolved == str(canonical)
        assert str(legacy) in ident.siblings

    def test_naming_the_legacy_spelling_resolves_to_itself(self, tmp_path):
        # Naming the legacy spelling must return THAT file. This assertion
        # previously read `resolved == canonical`, which pinned the defect in
        # place: canonical-first swapped in the neighbour, so the checkpoint
        # the list displays was unaddressable. The probe order is now unified
        # with .sou/.slo — as given first — so the shadow is still flagged but
        # the caller gets what they asked for.
        canonical, legacy = self._make_shadowed_pair(tmp_path)

        ident = classify_soul(str(legacy))

        assert ident.spelling == "legacy-double"
        assert ident.resolved == str(legacy)
        assert ident.shadowed is True
        assert ident.siblings == (str(canonical),)

    def test_legacy_only_file_is_reachable_and_flagged(self, tmp_path):
        # Only the double-appended spelling exists. The canonical name alone
        # would raise FileNotFoundError, so the fallback is what makes it load.
        save_soul(
            None,
            str(tmp_path / "_lonely"),
            soul_profile=SloProfile(name="lonely"),
            weights_only=True,
        )
        legacy = tmp_path / "lonely.soul.soul"
        os.rename(tmp_path / "_lonely.soul", legacy)
        os.rename(
            tmp_path / "_lonely.soul.meta.json",
            tmp_path / "lonely.soul.soul.meta.json",
        )

        ident = classify_soul(str(tmp_path / "lonely.soul"))

        assert ident.exists is True
        assert ident.orphan_legacy is True
        assert ident.resolved == str(legacy)
        # Read from ITS OWN sidecar, not the canonical name's.
        assert ident.integrity_hash

    def test_identical_copies_are_not_a_shadow(self, tmp_path):
        # Same bytes under both spellings is a harmless duplicate: nothing is
        # unreachable because either one loads the same checkpoint.
        canonical = _write_checkpoint(tmp_path, "dup", "same")
        legacy = tmp_path / "dup.soul.soul"
        import shutil

        shutil.copyfile(canonical, legacy)
        shutil.copyfile(tmp_path / "dup.soul.meta.json", tmp_path / "dup.soul.soul.meta.json")

        ident = classify_soul(str(canonical))

        assert len(ident.variants) == 2
        assert ident.shadowed is False

    def test_missing_path_reports_absence_without_raising(self, tmp_path):
        ident = classify_soul(str(tmp_path / "nope.soul"))

        assert ident.exists is False
        assert ident.resolved is None
        assert ident.format == ""
        assert ident.shadowed is False and ident.orphan_legacy is False
        assert ident.variants == ()

    @pytest.mark.parametrize("raw", ["", ".", "..", "models", "/dev/null/nope"])
    def test_garbage_input_never_raises(self, raw):
        # Classifying is what you reach for when you don't know what you have;
        # raising on an odd input would defeat the purpose.
        ident = classify_soul(raw)
        assert isinstance(ident.spelling, str)

    def test_foreign_extension_is_classified_not_rejected(self, tmp_path):
        # soul_path refuses .safetensors, and rightly so — but classify must
        # still answer, else it dies exactly when it is most needed.
        foreign = tmp_path / "weights.safetensors"
        foreign.write_bytes(b"\x80\x02\x05")

        ident = classify_soul(str(foreign))

        assert ident.exists is True
        assert ident.spelling == "foreign"
        assert ident.format == "safetensors"
        assert ident.resolved == str(foreign)

    def test_real_container_reads_as_soul(self, tmp_path):
        canonical = _write_checkpoint(tmp_path, "real", "real")

        assert classify_soul(str(canonical)).format == "soul"

    def test_plain_text_artifact_is_not_soul_despite_the_suffix(self, tmp_path):
        # The name claims soul, the magic bytes disagree — call it what it is
        # rather than rounding the extension up into a claim of identity.
        fake = tmp_path / "fake.soul"
        fake.write_text("this is not a soul container")

        assert classify_soul(str(fake)).format == "not-soul"

    @pytest.mark.parametrize(
        ("spelling", "expected"),
        [
            ("ck.soul", "canonical"),
            ("ck.soul.soul", "legacy-double"),
            ("ck", "bare"),
            ("ck.sou", "alias"),
        ],
    )
    def test_spelling_is_reported_per_input(self, tmp_path, spelling, expected):
        _write_checkpoint(tmp_path, "ck", "spelled")

        assert classify_soul(str(tmp_path / spelling)).spelling == expected


class TestTheNamedSpellingWins:
    """Naming a checkpoint must return THAT checkpoint.

    Discovery now lists ``<stem>_trained.soul.soul`` as its own row, so a user
    can see it and ask for it. Loading has to honour the request: before this,
    both spellings returned the canonical neighbour, leaving the listed
    checkpoint unaddressable.
    """

    @staticmethod
    def _write_two_runs(tmp_path) -> tuple[Path, Path]:
        canonical = _write_checkpoint(tmp_path, "pair", "canonical-run")

        save_soul(
            None,
            str(tmp_path / "_legacy"),
            soul_profile=SloProfile(name="legacy-run"),
            weights_only=True,
        )
        legacy = tmp_path / "pair.soul.soul"
        os.rename(tmp_path / "_legacy.soul", legacy)
        os.rename(tmp_path / "_legacy.soul.meta.json", tmp_path / "pair.soul.soul.meta.json")
        return canonical, legacy

    def test_the_pair_really_is_two_different_checkpoints(self, tmp_path):
        canonical, legacy = self._write_two_runs(tmp_path)

        assert classify_soul(str(canonical)).shadowed is True
        assert canonical.read_bytes() != legacy.read_bytes()

    def test_naming_the_legacy_spelling_returns_it(self, tmp_path):
        _canonical, legacy = self._write_two_runs(tmp_path)

        soul, _state = load_soul(str(legacy))

        assert soul.name == "legacy-run"

    def test_naming_the_canonical_spelling_still_returns_it(self, tmp_path):
        canonical, _legacy = self._write_two_runs(tmp_path)

        soul, _state = load_soul(str(canonical))

        assert soul.name == "canonical-run"

    def test_canonical_name_falls_back_when_only_the_legacy_file_exists(self, tmp_path):
        # The fallback direction must not break: three real checkpoints live
        # ONLY under the legacy spelling, and callers ask for them canonically.
        save_soul(
            None,
            str(tmp_path / "_only"),
            soul_profile=SloProfile(name="only-legacy"),
            weights_only=True,
        )
        os.rename(tmp_path / "_only.soul", tmp_path / "lonely.soul.soul")
        os.rename(tmp_path / "_only.soul.meta.json", tmp_path / "lonely.soul.soul.meta.json")

        soul, _state = load_soul(str(tmp_path / "lonely.soul"))

        assert soul.name == "only-legacy"


class TestBloatedSidecar:
    """``metadata.training_state`` — the optimizer state training resumes from
    (read by `train_pipeline`, pinned by its tests) — reaches 40MB inside a
    69MB sidecar on disk. Classification needs these scalars from its head.

    What is pinned here: the head-parse returns the SAME identity a full parse
    would. Getting it wrong shows up as a checkpoint reported hashless, not
    as an exception — so the values are compared, not just the absence of a
    crash. Parsing the blob instead cost 3.4s per file, 82s per models/ dir.
    """

    # Declared identity axes live ABOVE `metadata` in the document, so the
    # head-parse owes them the same answer a full parse gives — a bloated
    # sidecar must not quietly lose the very fields it declares.
    _IDENTITY = {
        "born_at": "2026-09-20T05:27:55.490339Z",
        "final_train_loss": 3.2909,
        "integrity_hash": "d9676fd9ddc9cb22",
        "tier": "canonical",
        "provenance": "training",
    }

    @staticmethod
    def _write(tmp_path: Path, name: str, blob_chars: int) -> Path:
        soul = tmp_path / name
        soul.write_bytes(b"identity comes from the sidecar, not these bytes")
        meta = dict(TestBloatedSidecar._IDENTITY)
        # The real layout: small metadata keys FIRST, resume blob LAST. The
        # head-parse closes the document at the blob, so the fixture has to
        # match that ordering — otherwise it tests a shape no file has.
        meta["metadata"] = {
            "vocab_size": 31,
            "config": {"n_embed": 128, "n_layer": 4},
            "training_state": {"step": 13, "optimizer": "x" * blob_chars},
        }
        Path(f"{soul}.meta.json").write_text(json.dumps(meta), encoding="utf-8")
        return soul

    @pytest.mark.parametrize("blob_chars", [0, 5 * 1024 * 1024])
    def test_identity_survives_the_training_blob(self, tmp_path, blob_chars):
        # 0 bytes takes the full-parse path, 5MB the head path — same answer.
        soul = self._write(tmp_path, "big.soul", blob_chars)

        ident = classify_soul(str(soul))

        assert ident.integrity_hash == self._IDENTITY["integrity_hash"]
        assert ident.born_at == self._IDENTITY["born_at"]
        assert ident.final_train_loss == pytest.approx(self._IDENTITY["final_train_loss"])
        # The declared axes must survive the blob too — at 5MB this runs the
        # head path, so a regression there shows up as tier/provenance
        # silently blanking on the exact files that motivated head-parsing.
        assert ident.tier == self._IDENTITY["tier"]
        assert ident.provenance == self._IDENTITY["provenance"]

    def test_both_paths_return_the_same_document(self, tmp_path):
        # The invariant that matters: which branch ran must be invisible to
        # callers. Checkpoint rows and identity both read metadata (vocab_size
        # for a row, integrity_hash for a fingerprint), so metadata has to
        # survive the head-parse, not just the identity keys.
        documents = []
        pair = (
            self._write(tmp_path, "big.soul", 5 * 1024 * 1024),
            self._write(tmp_path, "small.soul", 0),
        )
        for soul in pair:
            got = read_sidecar(str(soul))
            assert set(got) == set(self._IDENTITY) | {"metadata"}
            assert (got.get("metadata") or {}).get("vocab_size") == 31
            documents.append(json.dumps(got, sort_keys=True))

        assert documents[0] == documents[1]

    @pytest.mark.parametrize("blob_chars", [0, 5 * 1024 * 1024])
    def test_the_training_blob_is_never_returned(self, tmp_path, blob_chars):
        # If the blob comes back, the size guard failed — and both
        # classification and GET /training/checkpoints get slow again.
        soul = self._write(tmp_path, "big.soul", blob_chars)

        meta = read_sidecar(str(soul))

        assert "training_state" not in (meta.get("metadata") or {})


class TestTierAndProvenance:
    """Level 3 — tier and provenance are DECLARED, never inferred.

    A tier decided by parsing flips the moment a reader changes; provenance
    decided later is a guess. Both exist only at write time, which is the one
    moment the answer is actually known.
    """

    def test_policy_covers_every_read_suffix(self):
        assert set(SOUL_SUFFIXES) <= set(SOUL_TIER_POLICY)
        assert SOUL_TIER_POLICY[".slnc"] == "runtime"

    def test_save_declares_the_tier_of_the_file_it_writes(self, tmp_path):
        out = save_soul(None, str(tmp_path / "demo.soul"), weights_only=True)

        assert read_sidecar(out)["tier"] == "canonical"
        assert classify_soul(out).tier == "canonical"

    def test_record_reaches_header_and_sidecar_from_one_write(self, tmp_path):
        out = save_soul(None, str(tmp_path / "demo.soul"), weights_only=True, record="training")

        # Two readers, one declaration: the embedded header is what a
        # sidecar-less load parses, the sidecar is what a listing reads.
        assert _read_header(Path(out))["provenance"] == "training"
        assert read_sidecar(out)["provenance"] == "training"
        assert classify_soul(out).provenance == "training"

    def test_absent_provenance_stays_unknown_rather_than_guessed(self, tmp_path):
        out = save_soul(None, str(tmp_path / "demo.soul"), weights_only=True)

        assert read_sidecar(out)["provenance"] == ""
        assert classify_soul(out).provenance == ""

    def test_tier_falls_back_to_the_policy_for_preexisting_files(self, tmp_path):
        # Everything on disk written before this feature has neither field.
        # The suffix can honestly answer tier; nothing can answer provenance,
        # so it stays unknown rather than being back-filled from the name.
        out = save_soul(None, str(tmp_path / "legacy.soul"), weights_only=True)
        meta = read_sidecar(out)
        meta.pop("tier", None)
        meta.pop("provenance", None)
        Path(f"{out}.meta.json").write_text(json.dumps(meta), encoding="utf-8")

        ident = classify_soul(out)
        assert ident.tier == "canonical"
        assert ident.provenance == ""

    def test_a_declared_tier_wins_over_the_suffix(self, tmp_path):
        out = save_soul(None, str(tmp_path / "demo.soul"), weights_only=True)
        meta = read_sidecar(out)
        meta["tier"] = "interchange"
        Path(f"{out}.meta.json").write_text(json.dumps(meta), encoding="utf-8")

        assert classify_soul(out).tier == "interchange"

    def test_a_mislabeled_file_gets_no_tier_fallback(self, tmp_path):
        # The name claims soul and the header disagrees. Falling back to the
        # suffix here would stamp a broken artifact "canonical" — laundering
        # the very claim that proved false.
        fake = tmp_path / "fake.soul"
        fake.write_text("not a soul container at all")
        fake.with_name("fake.soul.meta.json").write_text("{}", encoding="utf-8")

        ident = classify_soul(str(fake))
        assert ident.format == "not-soul"
        assert ident.tier == ""

    def test_an_undeclared_provenance_is_refused_at_the_boundary(self, tmp_path):
        # Provenance is stamped into the header and sidecar and never
        # rewritten, so a typo would become permanent identity metadata.
        # Refuse at the boundary rather than record a value no reader can fix.
        with pytest.raises(ValueError, match="not a declared provenance"):
            save_soul(None, str(tmp_path / "bad.soul"), weights_only=True, record="trianing")

        # Nothing was written for the rejected call.
        assert not (tmp_path / "bad.soul").exists()
        assert not (tmp_path / "bad.soul.meta.json").exists()

    def test_declared_fields_never_move_the_identity_hash(self):
        # integrity_hash is content-derived — the uniqueness key this whole
        # work rests on. If `record=` or the tier leaked into it, identical
        # weights saved under different declarations would hash apart and stop
        # matching as the same checkpoint.
        plain = SloProfile(name="same", born_at="2026-01-01T00:00:00Z")
        declared = SloProfile(
            name="same",
            born_at="2026-01-01T00:00:00Z",
            tier="interchange",
            provenance="export",
        )

        assert plain.compute_hash() == declared.compute_hash()

        # ...while real content still moves it, so this isn't a no-op.
        changed = SloProfile(name="same", born_at="2026-01-01T00:00:00Z", final_train_loss=1.0)
        assert plain.compute_hash() != changed.compute_hash()
