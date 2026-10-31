"""Tests for the AML ModalitySequence @stream format."""

import pytest
from aml.stream import (
    MODALITIES,
    ModalitySequence,
    aml_to_sequences,
    sequences_to_aml,
)


class TestModalitySequence:
    def test_text_sequence(self):
        seq = ModalitySequence(modality="text", payload="hello world")
        assert seq.modality == "text"
        assert seq.payload == "hello world"
        assert seq.pair_id is None

    def test_int_sequence(self):
        seq = ModalitySequence(modality="voice", payload=[1, 2, 3], pair_id="p1")
        assert seq.payload == [1, 2, 3]
        assert seq.pair_id == "p1"

    def test_unknown_modality_rejected(self):
        with pytest.raises(ValueError, match="unknown modality"):
            ModalitySequence(modality="video", payload="x")

    def test_bad_payload_type_rejected(self):
        with pytest.raises(TypeError):
            ModalitySequence(modality="text", payload=42)  # type: ignore[arg-type]

    def test_int_list_only(self):
        with pytest.raises(TypeError, match="only ints"):
            ModalitySequence(modality="binary", payload=[1, "two"])  # type: ignore[list-item]

    def test_all_modalities_accepted(self):
        for mod in MODALITIES:
            seq = ModalitySequence(modality=mod, payload="x")
            assert seq.modality == mod


class TestRoundTrip:
    def test_text_roundtrip(self):
        seq = ModalitySequence(
            modality="text",
            payload="The quick brown fox",
            pair_id="c1",
            metadata={"source": "wiki"},
        )
        text = sequences_to_aml([seq])
        out = aml_to_sequences(text)
        assert len(out) == 1
        assert out[0].modality == "text"
        assert out[0].payload == "The quick brown fox"
        assert out[0].pair_id == "c1"
        assert out[0].metadata["source"] == "wiki"

    def test_int_list_roundtrip(self):
        seq = ModalitySequence(modality="voice", payload=[759, 121, 118], pair_id="c1")
        out = aml_to_sequences(sequences_to_aml([seq]))
        assert out[0].payload == [759, 121, 118]

    def test_paired_modalities(self):
        text_seq = ModalitySequence(modality="text", payload="transcript", pair_id="pair-9")
        voice_seq = ModalitySequence(modality="voice", payload=[1, 2], pair_id="pair-9")
        out = aml_to_sequences(sequences_to_aml([text_seq, voice_seq]))
        assert [s.modality for s in out] == ["text", "voice"]
        assert out[0].pair_id == out[1].pair_id == "pair-9"

    def test_multiline_payload_rejected(self):
        seq = ModalitySequence(modality="text", payload="line1\nline2")
        with pytest.raises(ValueError, match="single-line"):
            sequences_to_aml([seq])

    def test_multiple_sequences(self):
        seqs = [
            ModalitySequence(modality="text", payload=f"doc {i}", pair_id=f"p{i}") for i in range(5)
        ]
        out = aml_to_sequences(sequences_to_aml(seqs))
        assert len(out) == 5
        assert [s.payload for s in out] == [f"doc {i}" for i in range(5)]
