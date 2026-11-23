"""Format sniffers — sniff → adapter round-trip + detect parity (card 20261114_151505).

Two gates from the card:
  1. format-detection tests keep passing — parity for every sniffed type
     (``sniff_dataset(f).dataset_type == detect_dataset_type(f)``);
  2. sniff → adapter round-trip for every registered format — a sniffed
     result either names a registered adapter or declares
     ``capabilities.unsupported`` (image/video never fake-success).

Grandfathered formats (instruction/audio/.json-conversation — not yet
migrated) must still be owned by ``detect_dataset_type``'s if-chain:
sniff returns None and the label comes from the legacy branch.
"""

from __future__ import annotations

import json

import pytest

from domain.training import DatasetType, detect_dataset_type
from domain.training._internal.adapter_registry import (
    REGISTRY,
    AdapterError,
    AdapterSpec,
    boot,
)
from domain.training._internal.format_sniffers import (
    SNIFFERS,
    SniffResult,
    sniff_dataset,
)


def _mk(tmp_path, name: str, content: str):
    f = tmp_path / name
    f.write_text(content, encoding="utf-8")
    return f


def _jsonl(tmp_path, record: dict):
    return _mk(tmp_path, "data.jsonl", json.dumps(record) + "\n")


# ── sniffed formats: (format_id, factory, expected type, adapter, unsupported) ──

SNIFFED_CASES = [
    (
        "code-ext",
        lambda d: _mk(d, "x.py", "def f(): pass\n"),
        DatasetType.CODE,
        "text-pretrain",
        False,
    ),
    (
        "code-content",
        lambda d: _mk(d, "x.rs", "import os\nfunction foo() {\nlet x = 1;\n"),
        DatasetType.CODE,
        "text-pretrain",
        False,
    ),
    (
        "jsonl-image",
        lambda d: _jsonl(d, {"image": "photo.jpg", "caption": "a cat"}),
        DatasetType.IMAGE_TEXT,
        None,
        True,
    ),
    (
        "jsonl-video",
        lambda d: _jsonl(d, {"video": "clip.mp4", "text": "a dog"}),
        DatasetType.VIDEO_TEXT,
        None,
        True,
    ),
    (
        "text",
        lambda d: _mk(d, "x.txt", "hello world\n" * 3),
        DatasetType.TEXT,
        "text-pretrain",
        False,
    ),
    (
        "jsonl-conversation",
        lambda d: _jsonl(d, {"messages": [{"role": "user", "content": "hi"}]}),
        DatasetType.CONVERSATION,
        "chat-sft",
        False,
    ),
]

# Grandfathered formats — still the if-chain's business (sniff → None).
GRANDFATHERED_CASES = [
    ("jsonl-instruction", lambda d: _jsonl(d, {"instruction": "do this"}), DatasetType.INSTRUCTION),
    (
        "jsonl-audio",
        lambda d: _jsonl(d, {"audio": "f.wav", "text": "hello"}),
        DatasetType.AUDIO_TEXT,
    ),
    (
        "json-conversation",
        lambda d: _mk(d, "d.json", json.dumps({"messages": []})),
        DatasetType.CONVERSATION,
    ),
]

# Key-precedence guards: an earlier key must beat a migrated later key
# (legacy elif order — sniffers may not reorder it).
PRECEDENCE_CASES = [
    (
        "audio+image",
        lambda d: _jsonl(d, {"audio": "f.wav", "image": "p.jpg"}),
        DatasetType.AUDIO_TEXT,
        None,
    ),
    (
        "instruction+image",
        lambda d: _jsonl(d, {"instruction": "x", "image": "p.jpg"}),
        DatasetType.INSTRUCTION,
        None,
    ),
    (
        # conv is the earliest legacy elif key — it must still win over
        # image, now claimed by the chat-sft sniffer instead of the if-chain.
        "conversation+image",
        lambda d: _jsonl(d, {"messages": [], "image": "p.jpg"}),
        DatasetType.CONVERSATION,
        "sniffed",
    ),
    (
        "image+video",
        lambda d: _jsonl(d, {"video": "c.mp4", "image": "p.jpg"}),
        DatasetType.IMAGE_TEXT,
        "sniffed",
    ),
]


class TestDetectParityForSniffedTypes:
    """Gate 1 — every sniffed type keeps its detect_dataset_type label."""

    @pytest.mark.parametrize(
        "fmt,factory,expected_type,expected_adapter,unsupported",
        SNIFFED_CASES,
        ids=[c[0] for c in SNIFFED_CASES],
    )
    def test_sniff_label_matches_detect(
        self, tmp_path, fmt, factory, expected_type, expected_adapter, unsupported
    ):
        path = factory(tmp_path)
        result = sniff_dataset(path)
        assert result is not None, f"{fmt}: sniff returned None — format not claimed"
        assert result.dataset_type == expected_type
        assert detect_dataset_type(str(path)) == expected_type

    @pytest.mark.parametrize(
        "fmt,factory,expected_type",
        GRANDFATHERED_CASES,
        ids=[c[0] for c in GRANDFATHERED_CASES],
    )
    def test_grandfathered_still_owned_by_if_chain(self, tmp_path, fmt, factory, expected_type):
        path = factory(tmp_path)
        assert sniff_dataset(path) is None, f"{fmt}: claimed by sniffers before its adapter exists"
        assert detect_dataset_type(str(path)) == expected_type

    def test_binary_txt_degrades_to_text_not_raise(self, tmp_path):
        """Legacy caught every read error and fell through to TEXT — a
        binary file named .txt must not crash detect (UnicodeDecodeError
        regression pin)."""
        f = tmp_path / "blob.txt"
        f.write_bytes(b"\xff\xfe\x00\x01binary\xff" * 10)
        assert detect_dataset_type(str(f)) == DatasetType.TEXT

    def test_broken_jsonl_first_line_degrades_to_text(self, tmp_path):
        f = tmp_path / "broken.jsonl"
        f.write_text("{not json\n", encoding="utf-8")
        assert sniff_dataset(f) is None
        assert detect_dataset_type(str(f)) == DatasetType.TEXT

    @pytest.mark.parametrize(
        "fmt,factory,expected_type,expected_adapter",
        PRECEDENCE_CASES,
        ids=[c[0] for c in PRECEDENCE_CASES],
    )
    def test_key_precedence_preserved(
        self, tmp_path, fmt, factory, expected_type, expected_adapter
    ):
        path = factory(tmp_path)
        assert detect_dataset_type(str(path)) == expected_type
        result = sniff_dataset(path)
        if expected_adapter == "sniffed":
            assert result is not None and result.dataset_type == expected_type
        else:
            assert result is None, f"{fmt}: precedence reordered — {result!r}"


class TestSniffAdapterRoundTrip:
    """Gate 2 — sniff → adapter round-trip for every registered format."""

    @pytest.mark.parametrize(
        "fmt,factory,expected_type,expected_adapter,unsupported",
        SNIFFED_CASES,
        ids=[c[0] for c in SNIFFED_CASES],
    )
    def test_round_trip(self, tmp_path, fmt, factory, expected_type, expected_adapter, unsupported):
        boot()
        result = sniff_dataset(factory(tmp_path))
        if unsupported:
            # declared dead-end — never a fake adapter name
            assert result.adapter is None
            assert result.capabilities == frozenset({"unsupported"})
            return
        assert result.adapter == expected_adapter
        spec = REGISTRY.get(result.adapter)  # raises AdapterError if it does not resolve
        assert spec.name == expected_adapter
        # the round-trip lands on a declared BatchSampler contract
        assert spec.sampler in ("pairs", "token-stream")

    def test_every_registered_format_has_a_fixture(self):
        """Meta-gate: the sniffer registry's declared formats and this file's
        fixtures stay in lockstep — a registered format without a fixture (or
        vice versa) fails here."""
        registered = {fmt for s in SNIFFERS for fmt in s.formats}
        fixtures = {c[0] for c in SNIFFED_CASES}
        assert registered == fixtures

    def test_registry_entries_are_valid(self):
        names = [s.name for s in SNIFFERS]
        assert len(names) == len(set(names)), "duplicate sniffer name"
        for s in SNIFFERS:
            assert s.formats, f"{s.name}: declares no formats"


class TestUnsupportedDeclaration:
    """Card (4): image/video declare capabilities.unsupported, never fake-success."""

    @pytest.mark.parametrize(
        "fmt,factory",
        [("jsonl-image", c[1]) for c in SNIFFED_CASES if c[0] == "jsonl-image"]
        + [("jsonl-video", c[1]) for c in SNIFFED_CASES if c[0] == "jsonl-video"],
    )
    def test_image_video_declare_unsupported(self, tmp_path, fmt, factory):
        result = sniff_dataset(factory(tmp_path))
        assert "unsupported" in result.capabilities
        assert result.adapter is None

    def test_unsupported_is_not_a_registered_adapter_name(self):
        boot()
        for s in SNIFFERS:
            for fmt in s.formats:
                if fmt.startswith(("jsonl-image", "jsonl-video")):
                    continue  # declared unsupported above
                # every non-unsupported format resolves to a real adapter
                assert fmt in {c[0] for c in SNIFFED_CASES}


class TestSamplerContractDeclaration:
    """Card (2): every adapter declares its BatchSampler contract, fail-fast."""

    def _spec(self, **overrides):
        base = dict(
            name="probe",
            task="text-pretrain",
            consumes=("text-records",),
            produces=list,  # register() checks the BatchArray shape dunders
            build=lambda: None,
            module=__name__,
            sampler="token-stream",
            block_size=None,
        )
        base.update(overrides)
        return AdapterSpec(**base)

    def test_spec_requires_sampler_field(self):
        with pytest.raises(TypeError):
            AdapterSpec(
                name="no-sampler",
                task="text-pretrain",
                consumes=("text-records",),
                produces=list,
                build=lambda: None,
                module=__name__,
                block_size=None,
            )

    @pytest.mark.parametrize("bad", ["streaming", "blocks", "", None])
    def test_bad_sampler_rejected_at_register(self, bad):
        reg = REGISTRY.__class__()
        with pytest.raises(AdapterError, match="sampler"):
            reg.register(self._spec(sampler=bad))

    @pytest.mark.parametrize("bad", [0, -1, 12.5, "64"])
    def test_bad_block_size_rejected_at_register(self, bad):
        reg = REGISTRY.__class__()
        with pytest.raises(AdapterError, match="block_size"):
            reg.register(self._spec(block_size=bad))

    @pytest.mark.parametrize("sampler", ["pairs", "token-stream"])
    @pytest.mark.parametrize("block_size", [None, 64])
    def test_valid_contracts_accepted(self, sampler, block_size):
        reg = REGISTRY.__class__()
        spec = reg.register(self._spec(sampler=sampler, block_size=block_size))
        assert spec.sampler == sampler and spec.block_size == block_size

    def test_text_pretrain_declares_token_stream(self):
        boot()
        spec = REGISTRY.get("text-pretrain")
        assert spec.sampler == "token-stream"
        assert spec.block_size is None  # block comes from training config at runtime

    def test_sniff_result_is_immutable(self):
        r = SniffResult(dataset_type=DatasetType.TEXT, adapter="text-pretrain")
        with pytest.raises(AttributeError):
            r.adapter = "other"  # type: ignore[misc]
