"""Chat Phase 2 intent-routing tools: image_gen + file_qa (card 20261116_090332).

Exactly two new ToolSpecs join the model-facing chat registry — internal tier
(model sees these; site utilities never enter this registry). image_gen wraps
the SAME Pillow generators as POST /images/generate (domain.multimodal.gallery),
file_qa reads a specific file or falls back to the knowledge/docs search.
"""

from __future__ import annotations

import base64

import pytest

from domain.agents._internal.tools import ToolRegistry

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


class TestRegistryProjection:
    def test_two_new_tools_registered(self) -> None:
        names = {t["name"] for t in ToolRegistry().list_tools()}
        assert {"image_gen", "file_qa"} <= names
        # the eight defaults stay — nothing was replaced
        assert {"calculator", "current_time", "web_search", "run_code"} <= names

    def test_image_gen_contract(self) -> None:
        spec = ToolRegistry().get("image_gen")
        assert spec is not None
        params = {p.name: p for p in spec.parameters}
        assert set(params) == {"prompt", "style"}
        assert params["prompt"].required is True
        assert params["style"].required is False
        # routing is model-driven — no pattern-table regex on the spec
        assert spec.pattern is None

    def test_file_qa_contract(self) -> None:
        spec = ToolRegistry().get("file_qa")
        assert spec is not None
        params = {p.name: p for p in spec.parameters}
        assert set(params) == {"query", "path"}
        assert params["query"].required is True
        assert params["path"].required is False
        assert spec.pattern is None

    def test_params_json_schema_projection(self) -> None:
        spec = ToolRegistry().get("image_gen")
        assert spec.params["required"] == ["prompt"]
        assert "prompt" in spec.params["properties"]

    def test_image_gen_prompt_declares_query_alias(self) -> None:
        # walk_img11: the model writes file_qa's input key (`query`) for
        # image_gen — the alias is declared ON the param (one source of
        # truth) and maps to canonical at the loop + execute seams.
        spec = ToolRegistry().get("image_gen")
        params = {p.name: p for p in spec.parameters}
        assert params["prompt"].aliases == ("query",)
        # the JSON Schema keeps teaching the canonical name only — aliases
        # are silent repair (like the trailing-comma rung), not contract.
        assert "query" not in spec.params["properties"]


class TestImageGenExecutor:
    @pytest.fixture()
    def registry(self):
        return ToolRegistry()

    @pytest.fixture(autouse=True)
    def gallery_to_tmp(self, tmp_path, monkeypatch):
        """Keep test art out of the real data/gallery."""
        import domain.multimodal as mm

        real_save = mm.save_image

        def _save(image_bytes, style="", gallery_dir=None):
            return real_save(image_bytes, style, gallery_dir=tmp_path)

        monkeypatch.setattr(mm, "save_image", _save)

    async def test_generates_png_and_carries_attachment(self, registry) -> None:
        res = await registry.execute("image_gen", {"prompt": "a blue sky", "style": "watercolor"})
        assert res.success, res.error
        att = res.metadata["attachment"]
        assert att["mime"] == "image/png"
        assert att["path"].startswith("/data/gallery/generated_")
        assert att["alt"] == "a blue sky"
        assert base64.b64decode(att["b64"])[:8] == PNG_MAGIC
        # the model-facing output must be SMALL — the payload never leaks to the model
        assert att["b64"] not in res.output
        assert len(res.output) < 300

    async def test_unknown_style_normalizes_to_realistic(self, registry) -> None:
        res = await registry.execute("image_gen", {"prompt": "x", "style": "bogus"})
        assert res.success, res.error
        assert res.metadata["style"] == "realistic"
        assert "realistic" in res.output

    async def test_empty_prompt_rejected(self, registry) -> None:
        res = await registry.execute("image_gen", {"prompt": "   "})
        assert not res.success
        assert res.error

    async def test_query_alias_accepted_at_execute_seam(self, registry) -> None:
        # walk_img11: the model called image_gen with file_qa's input key
        # (`query`) — declared aliases map it to `prompt` at the seam, so the
        # tool runs instead of failing validation.
        res = await registry.execute("image_gen", {"query": "a blue circle"})
        assert res.success, res.error
        assert res.metadata["attachment"]["alt"] == "a blue circle"

    async def test_canonical_beats_alias_when_both_present(self, registry) -> None:
        res = await registry.execute("image_gen", {"prompt": "canonical", "query": "ignored"})
        assert res.success, res.error
        assert res.metadata["attachment"]["alt"] == "canonical"


class TestFileQAExecutor:
    @pytest.fixture()
    def registry(self):
        return ToolRegistry()

    async def test_path_returns_matching_excerpts_not_full_dump(self, registry, tmp_path) -> None:
        doc = tmp_path / "notes.md"
        lines = [f"line {i} filler" for i in range(200)]
        lines[42] = "the launch window is Tuesday"
        doc.write_text("\n".join(lines))
        res = await registry.execute("file_qa", {"query": "launch window", "path": str(doc)})
        assert res.success, res.error
        assert "the launch window is Tuesday" in res.output
        assert "line 0 filler" not in res.output  # excerpts, not the whole file

    async def test_path_with_no_hits_returns_head(self, registry, tmp_path) -> None:
        doc = tmp_path / "doc.txt"
        doc.write_text("alpha\nbeta\ngamma")
        res = await registry.execute("file_qa", {"query": "zebra", "path": str(doc)})
        assert res.success, res.error
        assert "alpha" in res.output

    async def test_missing_path_is_an_error(self, registry, tmp_path) -> None:
        res = await registry.execute("file_qa", {"query": "x", "path": str(tmp_path / "nope.md")})
        assert not res.success
        assert res.error

    async def test_without_path_delegates_to_knowledge_search(self, registry, monkeypatch) -> None:
        called = {}

        async def fake_knowledge(query):
            called["query"] = query
            return {"output": "found docs", "engine": "files", "count": 1}

        monkeypatch.setattr(registry, "_run_knowledge_retrieval", fake_knowledge)
        res = await registry.execute("file_qa", {"query": "roadmap ideas"})
        assert res.success, res.error
        assert called["query"] == "roadmap ideas"
        assert res.output == "found docs"
