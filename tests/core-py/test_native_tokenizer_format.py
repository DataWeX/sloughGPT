"""Core native tokenizer format (card c9f217ee).

Backward-compat contract for the NativeTokenizer unification:
- char/tree behavior stays byte-identical to the legacy wrappers, asserted
  against INDEPENDENT inline expectations (not the legacy class names — those
  are now aliases of the same code and would be vacuous);
- the legacy names stay importable (test_slonet_provider.py imports them);
- the optional tokenizer_format tag round-trips and never overrides content;
- save-side and load-side literals agree (test-as-contract, no new module).
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from domain.inference._internal.slonet_provider import (
    NATIVE_FORMAT_CHAR,
    NATIVE_FORMAT_TREE,
    NativeTokenizer,
    _CharTokenizer,
    _TreeTokenizer,
    native_tokenizer_from_metadata,
)

# Realistic char vocab: JSON round-trips int keys as strings for itos,
# char stoi keeps str keys — both as a real .soul carries them.
CHAR_STOI = {"\n": 0, " ": 1, "a": 2, "b": 3, "c": 4, "d": 5, "e": 6, "!": 7, "A": 8}
CHAR_ITOS = {str(v): k for k, v in CHAR_STOI.items()}

CORPUS = ["abc", "abc abc", "a b c!", "A B", "abcabc!\nabc", ""]


def _legacy_char_encode(stoi, text):
    """The legacy _CharTokenizer.encode body, frozen here as the oracle."""
    return [int(stoi.get(c, 0)) for c in text]


def _legacy_char_decode(itos, token_ids):
    """The legacy _CharTokenizer.decode body, frozen here as the oracle."""
    itos_n = {int(k): v for k, v in itos.items()}
    return "".join(itos_n.get(int(tid), "") for tid in token_ids)


class TestCharFormatBitIdentity:
    """native/char must behave exactly like the legacy _CharTokenizer."""

    @pytest.fixture
    def tok(self):
        return native_tokenizer_from_metadata({"stoi": CHAR_STOI, "itos": CHAR_ITOS})

    def test_scheme_derived_from_content_when_tag_absent(self, tok):
        # ONE scheme per family (user decision): family + variant + pretokenizer.
        assert tok.scheme == "native"
        assert tok.variant == "char"
        assert tok.pretokenizer is None

    def test_chars_live_on_the_tree(self, tok):
        # The vocab rides a TokenTree carrier: dense vocab view + verbatim maps.
        assert tok.tree is not None
        assert tok.tree.vocab == [CHAR_ITOS[str(i)] for i in range(len(CHAR_STOI))]
        assert tok.tree.stoi == CHAR_STOI
        assert tok.tree.itos == {int(k): v for k, v in CHAR_ITOS.items()}

    @pytest.mark.parametrize("text", CORPUS)
    def test_encode_byte_identical(self, tok, text):
        assert tok.encode(text) == _legacy_char_encode(CHAR_STOI, text)

    @pytest.mark.parametrize("text", CORPUS)
    def test_decode_roundtrip_byte_identical(self, tok, text):
        ids = tok.encode(text)
        assert tok.decode(ids) == _legacy_char_decode(CHAR_ITOS, ids)

    def test_decode_arbitrary_ids_byte_identical(self, tok):
        # Gaps and unknown ids: decode answers "" — exactly as legacy did.
        ids = [0, 999, 3, -1, 7, 8]
        assert tok.decode(ids) == _legacy_char_decode(CHAR_ITOS, ids)

    def test_oov_char_maps_to_zero(self, tok):
        # Legacy: stoi.get(c, 0) — an unknown char becomes id 0 ('\n').
        assert tok.encode("z") == [0]
        assert tok.encode("xyz") == [0, 0, 0]

    def test_case_is_preserved(self, tok):
        # No lowercase normalization in char scheme (unlike token-tree).
        assert tok.encode("A") == [CHAR_STOI["A"]]
        assert tok.encode("a") == [CHAR_STOI["a"]]

    def test_attribute_semantics_match_legacy(self, tok):
        assert tok.eos_token_id == CHAR_STOI.get("\n", 0) == 0
        assert tok.pad_token_id == 0
        assert tok.vocab_size == max(int(k) for k in CHAR_ITOS) + 1

    def test_apply_chat_template_identical(self, tok):
        msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]
        assert tok.apply_chat_template(msgs) == "system: sys\nuser: hi\nassistant:"

    def test_origin_passthrough(self):
        tok = native_tokenizer_from_metadata(
            {"stoi": CHAR_STOI, "itos": CHAR_ITOS}, origin="checkpoint:/x/m.soul"
        )
        assert tok.origin == "checkpoint:/x/m.soul"


@pytest.fixture(scope="module")
def tree_meta():
    from domain.training._internal.token_tree import TokenTree

    tree = TokenTree().train(
        ["the cat sat on the mat", "the dog sat on the log"],
        vocab_size=48,
        min_frequency=1,
    )
    return tree.to_dict()


class TestTokenTreeFormat:
    """native/token-tree must behave exactly like the legacy _TreeTokenizer."""

    @pytest.fixture
    def tok(self, tree_meta):
        return native_tokenizer_from_metadata(
            {"tokenizer": {"type": "token_tree", "tree": tree_meta}}
        )

    def test_scheme_derived_from_content_when_tag_absent(self, tok):
        assert tok.scheme == "native"
        assert tok.variant == "tree"
        assert tok.pretokenizer in ("gpt2", "whitespace")

    def test_delegates_encode_decode_to_tree(self, tok):
        # The tree itself is the oracle for this format (its semantics are
        # locked by test_token_tree.py); the wrapper must not mangle them.
        text = "the cat sat"
        assert tok.encode(text) == tok.tree.encode(text)
        ids = tok.encode(text)
        assert tok.decode(ids) == tok.tree.decode(ids)

    def test_lowercase_normalization_active(self, tok):
        # Token-tree encode lowercases — part of THIS format's contract.
        assert tok.encode("THE CAT") == tok.encode("the cat")

    def test_attribute_semantics_match_legacy(self, tok):
        assert tok.eos_token_id == int(tok.tree.eos_id)
        assert tok.pad_token_id == int(tok.tree.pad_id)
        # Legacy tree wrapper read len(tree.vocab) live via property.
        assert tok.vocab_size == tok.tree.vocab_size
        tok.tree.vocab.append("extra</w>")
        assert tok.vocab_size == tok.tree.vocab_size == len(tok.tree.vocab)

    def test_apply_chat_template_identical(self, tok):
        msgs = [{"role": "user", "content": "hi"}]
        assert tok.apply_chat_template(msgs) == "user: hi\nassistant:"


class TestFormatTag:
    """The tag is a declaration; content is the ground truth."""

    def test_tag_consistent_char_honored_without_warning(self, caplog):
        with caplog.at_level("WARNING"):
            tok = native_tokenizer_from_metadata(
                {"stoi": CHAR_STOI, "itos": CHAR_ITOS, "tokenizer_format": NATIVE_FORMAT_CHAR}
            )
        assert tok.scheme == "native" and tok.variant == "char"
        assert not [r for r in caplog.records if "tokenizer_format" in r.getMessage()]

    def test_tag_mismatch_content_wins_char(self, caplog):
        # Tag claims token-tree, content is char vocab: char wins, logged.
        with caplog.at_level("WARNING"):
            tok = native_tokenizer_from_metadata(
                {"stoi": CHAR_STOI, "itos": CHAR_ITOS, "tokenizer_format": NATIVE_FORMAT_TREE}
            )
        assert tok.scheme == "native" and tok.variant == "char"
        assert any("tokenizer_format" in r.getMessage() for r in caplog.records)

    def test_tag_mismatch_content_wins_tree(self, caplog):
        from domain.training._internal.token_tree import TokenTree

        tree = TokenTree().train(["hello world hello"], vocab_size=32, min_frequency=1)
        meta = {
            "tokenizer": {"type": "token_tree", "tree": tree.to_dict()},
            "tokenizer_format": NATIVE_FORMAT_CHAR,
        }
        with caplog.at_level("WARNING"):
            tok = native_tokenizer_from_metadata(meta)
        assert tok.scheme == "native" and tok.variant == "tree"
        assert any("tokenizer_format" in r.getMessage() for r in caplog.records)

    def test_no_tokenizer_at_all_returns_none(self):
        assert native_tokenizer_from_metadata({}) is None
        assert native_tokenizer_from_metadata({"tokenizer_format": NATIVE_FORMAT_CHAR}) is None


class TestLegacySurface:
    """Old names stay importable and constructible (existing test suites)."""

    def test_subclass_relationship(self):
        assert issubclass(_CharTokenizer, NativeTokenizer)
        assert issubclass(_TreeTokenizer, NativeTokenizer)

    def test_legacy_char_constructor(self):
        tok = _CharTokenizer(CHAR_STOI, CHAR_ITOS)
        assert tok.scheme == "native" and tok.variant == "char"
        assert tok.encode("abc") == _legacy_char_encode(CHAR_STOI, "abc")

    def test_legacy_tree_constructor(self):
        from domain.training._internal.token_tree import TokenTree

        tree = TokenTree().train(["hello world"], vocab_size=32, min_frequency=1)
        tok = _TreeTokenizer(tree)
        assert tok.scheme == "native" and tok.variant == "tree"
        assert tok.tree is tree and tok._tree is tree

    def test_unknown_scheme_rejected(self):
        from domain.training._internal.token_tree import TokenTree

        with pytest.raises(ValueError, match="Unknown native tokenizer variant"):
            NativeTokenizer(TokenTree(), "nonsense")

    def test_embedded_loader_raises_when_missing(self):
        with pytest.raises(ValueError, match="Embedded tokenizer missing"):
            from domain.inference._internal.slonet_provider import SloNetChatProvider

            SloNetChatProvider._tokenizer_from_embedded({})


class TestSaveLoadContract:
    """Save-side stamps and load-side constants must speak the same strings."""

    def test_format_literals_agree_between_save_and_load(self):
        import domain.inference._internal.slonet_provider as sp
        import domain.training._internal.train_pipeline as tp

        load_src = Path(inspect.getsourcefile(sp)).read_text()
        save_src = Path(inspect.getsourcefile(tp)).read_text()
        for literal in ("native/char", "native/token-tree"):
            assert f'"{literal}"' in load_src, f"load side missing {literal}"
            assert f'"{literal}"' in save_src, f"save side missing {literal}"

    def test_save_stamp_prefers_tree_then_char(self):
        import domain.training._internal.train_pipeline as tp

        src = inspect.getsource(tp)
        # The stamp must be ordered: tokenizer (tree) branch first, char second.
        tree_pos = src.find('soul.metadata["tokenizer_format"] = "native/token-tree"')
        char_pos = src.find('soul.metadata["tokenizer_format"] = "native/char"')
        assert tree_pos != -1 and char_pos != -1 and tree_pos < char_pos


class TestProvenanceTaxonomy:
    """Scheme names separate tokenizer families at a glance (card 85dedb7e)."""

    def test_one_scheme_per_family_and_uniform_attrs(self):
        from domain.infrastructure._internal.morph_tokenizer import MorphTokenizer
        from domain.training._internal.tokenizer import SloBPE, SloUnigram

        native = NativeTokenizer.from_char_vocab({"a": 0}, {0: "a"})
        morph = MorphTokenizer({}, [], byte_level=True)
        bpe = SloBPE()
        uni = SloUnigram()

        # ONE scheme per family — all four distinct, no scheme variants.
        schemes = {native.scheme, morph.scheme, bpe.scheme, uni.scheme}
        assert schemes == {"native", "hf", "training/bpe", "training/unigram"}

        # Uniform provenance surface: every tokenizer answers the same fields.
        for tok in (native, morph, bpe, uni):
            assert isinstance(tok.scheme, str)
            assert hasattr(tok, "origin")
            assert hasattr(tok, "pretokenizer")

        # The details live in the uniform attributes, not the scheme.
        assert native.variant == "char" and native.pretokenizer is None
        assert morph.variant == "byte-level" and morph.pretokenizer is None
        assert bpe.pretokenizer == "gpt2"
        assert uni.pretokenizer == "gpt2"
