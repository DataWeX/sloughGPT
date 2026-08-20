"""Tests for domains.models.provider — ModelCapabilities, ToolDef; domains.multimodal.vision — ImageCaption, VisualObject; domains.shell.surface — RenderLine, TextSurface, strip_ansi, clip."""

from domains.models.provider import ModelCapabilities, ToolDef
from domains.multimodal.vision import ImageCaption, VisualObject
from domains.shell.surface import RenderLine, TextSurface, strip_ansi, clip


class TestModelCapabilities:
    def test_defaults(self):
        mc = ModelCapabilities()
        assert mc.chat is False
        assert mc.streaming is False
        assert mc.embedding is False
        assert mc.vision is False
        assert mc.functions is False

    def test_custom(self):
        mc = ModelCapabilities(chat=True, vision=True)
        assert mc.chat is True
        assert mc.vision is True
        assert mc.streaming is False


class TestToolDef:
    def test_fields(self):
        td = ToolDef(name="describe_image", provider_name="multimodal", description="desc")
        assert td.name == "describe_image"
        assert td.provider_name == "multimodal"


class TestImageCaption:
    def test_fields(self):
        ic = ImageCaption(text="a cat", confidence=0.95, tags=["cat"])
        assert ic.text == "a cat"
        assert ic.confidence == 0.95


class TestVisualObject:
    def test_fields(self):
        vo = VisualObject(label="cat", bbox=[0, 0, 100, 100], confidence=0.9)
        assert vo.label == "cat"
        assert vo.bbox == [0, 0, 100, 100]


class TestRenderLine:
    def test_fields(self):
        rl = RenderLine(text="hello", style="bold")
        assert rl.text == "hello"
        assert rl.style == "bold"


class TestStripAnsi:
    def test_strip_codes(self):
        assert strip_ansi("\x1b[31mred\x1b[0m") == "red"

    def test_no_ansi(self):
        assert strip_ansi("plain") == "plain"


class TestClip:
    def test_shorter_than_width(self):
        assert clip("hello", 10) == "hello"

    def test_longer_than_width(self):
        result = clip("hello world", 5)
        assert len(result) == 5

    def test_exact_width(self):
        assert clip("hello", 5) == "hello"


class TestTextSurface:
    def test_write_and_capture(self):
        ts = TextSurface()
        ts.write("hello")
        lines = ts.capture
        assert len(lines) >= 1
        assert "hello" in lines[-1]

    def test_write_custom_end(self):
        ts = TextSurface()
        ts.write("hello", end="")
        ts.write("world", end="")
        lines = ts.capture
        assert any("helloworld" in l for l in lines)

    def test_clear(self):
        ts = TextSurface()
        ts.write("test")
        ts.clear()
        lines = ts.capture
        assert lines == []

    def test_render(self):
        ts = TextSurface()
        ts.set_width(40)
        ts.write("hello")
        lines = ts.render(5)
        assert len(lines) >= 1

    def test_write_long_line_stored(self):
        ts = TextSurface()
        ts.set_width(10)
        ts.write("a" * 20)
        lines = ts.capture
        assert len(lines) >= 1
        assert len(lines[-1]) == 20
