"""Structure-aware .soul filename grammar — one owner, canonical write, tolerant read."""

from __future__ import annotations

import os

import pytest

from domain.inference import (
    SloProfile,
    load_soul,
    save_soul,
    soul_path,
    soul_read_candidates,
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

    def test_legacy_double_append_collapses_then_probes_itself(self):
        assert soul_read_candidates("m/x.soul.soul") == ["m/x.soul", "m/x.soul.soul"]

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
