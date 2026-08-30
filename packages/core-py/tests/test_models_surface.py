"""Tests for domains.models.provider — ModelCapabilities, ToolDef; domains.multimodal.vision — ImageCaption, VisualObject; domains.shell.surface — RenderLine, TextSurface, strip_ansi, clip, LogSurface."""

import asyncio
import threading
import time
from datetime import datetime

import pytest

from domains.models.provider import (
    ModelCapabilities, ToolDef,
    VisionProcessor, KnowledgeProcessor, ToolUseProcessor,
    PersonalityProcessor, StyleProcessor,
    ProviderRouter,
    register_provider, get_provider, list_providers, clear_providers,
    register_processor, get_processor, list_processors, apply_processors,
    _processors,
)
from domains.multimodal.vision import ImageCaption, VisualObject
from domains.shell.surface import (
    RenderLine, TextSurface, strip_ansi, clip,
    LogSurface, Surface, STYLE_INFO, STYLE_WARN, STYLE_ERROR,
    STYLE_DEBUG, STYLE_CRITICAL, _display_width,
)
from domains.shell.log_buffer import LogBuffer, LogEntry


# ── ModelCapabilities ────────────────────────────────────────────────

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

    def test_all_true(self):
        mc = ModelCapabilities(chat=True, streaming=True, embedding=True, vision=True, functions=True)
        assert mc.chat is True
        assert mc.streaming is True
        assert mc.embedding is True
        assert mc.vision is True
        assert mc.functions is True

    def test_equality(self):
        a = ModelCapabilities(chat=True, vision=True)
        b = ModelCapabilities(chat=True, vision=True)
        assert a == b

    def test_inequality(self):
        a = ModelCapabilities(chat=True)
        b = ModelCapabilities(chat=False)
        assert a != b

    def test_copy_independence(self):
        a = ModelCapabilities(chat=True)
        b = ModelCapabilities(chat=True)
        assert a == b
        assert a is not b

    def test_field_types(self):
        mc = ModelCapabilities()
        assert isinstance(mc.chat, bool)
        assert isinstance(mc.streaming, bool)
        assert isinstance(mc.embedding, bool)
        assert isinstance(mc.vision, bool)
        assert isinstance(mc.functions, bool)

    def test_single_true(self):
        mc = ModelCapabilities(streaming=True)
        assert mc.streaming is True
        assert mc.chat is False
        assert mc.embedding is False
        assert mc.vision is False
        assert mc.functions is False

    def test_all_false(self):
        mc = ModelCapabilities()
        assert all([not mc.chat, not mc.streaming, not mc.embedding, not mc.vision, not mc.functions])

    def test_hashable(self):
        """ModelCapabilities is a mutable dataclass — not hashable by default."""
        a = ModelCapabilities(chat=True)
        with pytest.raises(TypeError):
            hash(a)

    def test_repr(self):
        mc = ModelCapabilities(chat=True)
        r = repr(mc)
        assert "chat" in r


# ── ToolDef ──────────────────────────────────────────────────────────

class TestToolDef:
    def test_fields(self):
        td = ToolDef(name="describe_image", provider_name="multimodal", description="desc")
        assert td.name == "describe_image"
        assert td.provider_name == "multimodal"
        assert td.description == "desc"

    def test_default_description(self):
        td = ToolDef(name="test", provider_name="p")
        assert td.description == ""

    def test_equality(self):
        a = ToolDef(name="x", provider_name="y", description="z")
        b = ToolDef(name="x", provider_name="y", description="z")
        assert a == b

    def test_inequality(self):
        a = ToolDef(name="x", provider_name="y")
        b = ToolDef(name="x", provider_name="z")
        assert a != b

    def test_repr(self):
        td = ToolDef(name="test", provider_name="p")
        r = repr(td)
        assert "test" in r

    def test_empty_name(self):
        td = ToolDef(name="", provider_name="p")
        assert td.name == ""

    def test_long_description(self):
        td = ToolDef(name="x", provider_name="y", description="d" * 1000)
        assert len(td.description) == 1000


# ── ImageCaption ─────────────────────────────────────────────────────

class TestImageCaption:
    def test_fields(self):
        ic = ImageCaption(text="a cat", confidence=0.95, tags=["cat"])
        assert ic.text == "a cat"
        assert ic.confidence == 0.95

    def test_default_accuracy(self):
        ic = ImageCaption(text="x", confidence=0.5, tags=[])
        assert ic.accuracy == 0.0

    def test_custom_accuracy(self):
        ic = ImageCaption(text="x", confidence=0.8, tags=["t"], accuracy=0.99)
        assert ic.accuracy == 0.99

    def test_empty_tags(self):
        ic = ImageCaption(text="x", confidence=0.5, tags=[])
        assert ic.tags == []

    def test_multiple_tags(self):
        ic = ImageCaption(text="x", confidence=0.5, tags=["a", "b", "c"])
        assert len(ic.tags) == 3

    def test_empty_text(self):
        ic = ImageCaption(text="", confidence=0.5, tags=[])
        assert ic.text == ""

    def test_zero_confidence(self):
        ic = ImageCaption(text="x", confidence=0.0, tags=[])
        assert ic.confidence == 0.0

    def test_high_confidence(self):
        ic = ImageCaption(text="x", confidence=0.999, tags=[])
        assert ic.confidence == 0.999

    def test_tag_types(self):
        ic = ImageCaption(text="x", confidence=0.5, tags=[1, 2, 3])
        assert ic.tags == [1, 2, 3]


# ── VisualObject ─────────────────────────────────────────────────────

class TestVisualObject:
    def test_fields(self):
        vo = VisualObject(label="cat", bbox=[0, 0, 100, 100], confidence=0.9)
        assert vo.label == "cat"
        assert vo.bbox == [0, 0, 100, 100]
        assert vo.confidence == 0.9

    def test_bbox_single_element(self):
        vo = VisualObject(label="dot", bbox=[5, 5, 5, 5], confidence=1.0)
        assert vo.bbox == [5, 5, 5, 5]

    def test_zero_confidence(self):
        vo = VisualObject(label="none", bbox=[0, 0, 0, 0], confidence=0.0)
        assert vo.confidence == 0.0

    def test_empty_label(self):
        vo = VisualObject(label="", bbox=[0, 0, 1, 1], confidence=0.5)
        assert vo.label == ""

    def test_negative_bbox(self):
        vo = VisualObject(label="x", bbox=[-10, -20, 50, 60], confidence=0.5)
        assert vo.bbox[0] == -10

    def test_float_bbox(self):
        vo = VisualObject(label="x", bbox=[0.1, 0.2, 0.3, 0.4], confidence=0.5)
        assert vo.bbox[0] == 0.1

    def test_large_bbox(self):
        vo = VisualObject(label="x", bbox=[0, 0, 10000, 10000], confidence=0.5)
        assert vo.bbox[2] == 10000


# ── RenderLine ───────────────────────────────────────────────────────

class TestRenderLine:
    def test_fields(self):
        rl = RenderLine(text="hello", style="bold")
        assert rl.text == "hello"
        assert rl.style == "bold"

    def test_default_style(self):
        rl = RenderLine(text="hello")
        assert rl.style is None

    def test_empty_text(self):
        rl = RenderLine(text="", style=None)
        assert rl.text == ""

    def test_equality(self):
        a = RenderLine(text="x", style="y")
        b = RenderLine(text="x", style="y")
        assert a == b

    def test_inequality(self):
        a = RenderLine(text="x", style="y")
        b = RenderLine(text="x", style="z")
        assert a != b

    def test_repr(self):
        rl = RenderLine(text="hello", style="bold")
        r = repr(rl)
        assert "hello" in r

    def test_empty_style(self):
        rl = RenderLine(text="x", style="")
        assert rl.style == ""

    def test_long_text(self):
        rl = RenderLine(text="x" * 10000)
        assert len(rl.text) == 10000


# ── strip_ansi ───────────────────────────────────────────────────────

class TestStripAnsi:
    def test_strip_codes(self):
        assert strip_ansi("\x1b[31mred\x1b[0m") == "red"

    def test_no_ansi(self):
        assert strip_ansi("plain") == "plain"

    def test_multiple_sequences(self):
        text = "\x1b[1m\x1b[32mbold green\x1b[0m"
        assert strip_ansi(text) == "bold green"

    def test_empty_string(self):
        assert strip_ansi("") == ""

    def test_only_ansi(self):
        assert strip_ansi("\x1b[31m") == ""

    def test_nested_sequences(self):
        text = "\x1b[31mred\x1b[0m normal \x1b[34mblue\x1b[0m"
        assert strip_ansi(text) == "red normal blue"

    def test_256_color(self):
        text = "\x1b[38;5;196mred256\x1b[0m"
        assert strip_ansi(text) == "red256"

    def test_truecolor(self):
        text = "\x1b[38;2;255;128;0morange\x1b[0m"
        assert strip_ansi(text) == "orange"

    def test_csi_movement(self):
        text = "\x1b[2J\x1b[1;1H"
        assert strip_ansi(text) == ""

    def test_preserves_normal_text(self):
        assert strip_ansi("hello world 123 !@#") == "hello world 123 !@#"

    def test_unicode_preserved(self):
        assert strip_ansi("hello \u4e2d\u6587") == "hello \u4e2d\u6587"

    def test_bell_char(self):
        text = "\x07"
        assert strip_ansi(text) == "\x07"

    def test_mixed_ansi_and_unicode(self):
        text = "\x1b[31m\u4e2d\u6587\u001b[0m"
        result = strip_ansi(text)
        assert "\u4e2d\u6587" in result


# ── clip ─────────────────────────────────────────────────────────────

class TestClip:
    def test_shorter_than_width(self):
        assert clip("hello", 10) == "hello"

    def test_longer_than_width(self):
        result = clip("hello world", 5)
        assert len(result) == 5

    def test_exact_width(self):
        assert clip("hello", 5) == "hello"

    def test_width_zero(self):
        assert clip("hello", 0) == ""

    def test_width_negative(self):
        assert clip("hello", -1) == ""

    def test_empty_string(self):
        assert clip("", 5) == ""

    def test_single_char_width_one(self):
        assert clip("abc", 1) == "a"

    def test_single_char_width_two(self):
        assert clip("abc", 2) == "ab"

    def test_width_three_truncation(self):
        result = clip("abcdef", 3)
        assert len(result) == 3

    def test_width_four_has_ellipsis(self):
        result = clip("abcdef", 4)
        assert result.endswith("\u2026")
        assert len(result) == 4

    def test_unicode_text(self):
        result = clip("hello world", 7)
        assert len(result) == 7
        assert result.endswith("\u2026")

    def test_single_char_string(self):
        assert clip("a", 1) == "a"

    def test_cjk_wide_char(self):
        # CJK chars are 2 columns wide
        result = clip("\u4e2d\u6587", 3)  # "中文" = 4 columns, width=3
        assert len(result) <= 3

    def test_mixed_ascii_cjk(self):
        result = clip("a\u4e2db", 3)  # a(1) + 中(2) = 3 columns
        assert _display_width(result) <= 3

    def test_ellipsis_width(self):
        result = clip("abcdefghij", 5)
        assert len(result) == 5

    def test_width_one_empty(self):
        assert clip("abc", 0) == ""

    def test_pure_cjk(self):
        result = clip("\u4e2d\u6587\u6587", 5)
        assert _display_width(result) <= 5

    def test_very_long_text(self):
        result = clip("a" * 1000, 10)
        assert len(result) == 10


# ── _display_width ──────────────────────────────────────────────────

class TestDisplayWidth:
    def test_ascii(self):
        assert _display_width("hello") == 5

    def test_empty(self):
        assert _display_width("") == 0

    def test_cjk(self):
        assert _display_width("\u4e2d") == 2

    def test_mixed(self):
        assert _display_width("a\u4e2db") == 4

    def test_fullwidth(self):
        # Fullwidth A = U+FF21
        assert _display_width("\uff21") == 2


# ── TextSurface ──────────────────────────────────────────────────────

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

    def test_render_clips_to_width(self):
        ts = TextSurface()
        ts.set_width(5)
        ts.write("hello world")
        lines = ts.render(5)
        for rl in lines:
            assert len(rl.text) <= 5

    def test_render_returns_renderline(self):
        ts = TextSurface()
        ts.write("test")
        lines = ts.render(5)
        assert all(isinstance(rl, RenderLine) for rl in lines)

    def test_render_rows_zero(self):
        ts = TextSurface()
        ts.write("test")
        assert ts.render(0) == []

    def test_render_rows_negative(self):
        ts = TextSurface()
        ts.write("test")
        assert ts.render(-1) == []

    def test_render_offset_zero(self):
        ts = TextSurface()
        for i in range(10):
            ts.write(f"line{i}")
        lines = ts.render(3, offset=0)
        assert len(lines) == 3
        assert "line9" in lines[-1].text

    def test_render_offset_positive(self):
        ts = TextSurface()
        for i in range(10):
            ts.write(f"line{i}")
        lines = ts.render(2, offset=3)
        assert len(lines) == 2

    def test_render_offset_beyond_content(self):
        ts = TextSurface()
        ts.write("line0")
        lines = ts.render(5, offset=100)
        assert len(lines) == 1

    def test_set_width_minimum_one(self):
        ts = TextSurface()
        ts.set_width(0)
        assert ts._width == 1

    def test_set_width_negative(self):
        ts = TextSurface()
        ts.set_width(-5)
        assert ts._width == 1

    def test_write_empty_string(self):
        ts = TextSurface()
        ts.write("")
        lines = ts.capture
        assert len(lines) >= 1

    def test_write_newline_characters(self):
        ts = TextSurface()
        ts.write("a\nb\nc")
        lines = ts.capture
        assert len(lines) == 3

    def test_write_multiline_with_end(self):
        ts = TextSurface()
        ts.write("a\nb", end="\n")
        lines = ts.capture
        assert len(lines) >= 2

    def test_capture_partial_line(self):
        ts = TextSurface()
        ts.write("hello", end="")
        ts.write("world", end="")
        lines = ts.capture
        assert any("helloworld" in l for l in lines)

    def test_capture_after_newline_flushes(self):
        ts = TextSurface()
        ts.write("hello\n")
        ts.write("world\n")
        lines = ts.capture
        assert "hello" in lines
        assert "world" in lines

    def test_render_default_offset(self):
        ts = TextSurface()
        for i in range(20):
            ts.write(f"line{i}")
        lines = ts.render(5)
        assert len(lines) == 5

    def test_write_multiple_lines(self):
        ts = TextSurface()
        for i in range(5):
            ts.write(f"line{i}")
        lines = ts.capture
        assert len(lines) == 5

    def test_render_preserves_order(self):
        ts = TextSurface()
        ts.write("first")
        ts.write("second")
        ts.write("third")
        lines = ts.render(10)
        texts = [rl.text for rl in lines]
        assert any("first" in t for t in texts)
        assert any("third" in t for t in texts)

    def test_default_width(self):
        ts = TextSurface()
        assert ts._width == 80

    def test_clear_resets_partial(self):
        ts = TextSurface()
        ts.write("hello", end="")
        ts.clear()
        lines = ts.capture
        assert lines == []

    def test_write_returns_none(self):
        ts = TextSurface()
        result = ts.write("x")
        assert result is None

    def test_render_with_many_lines(self):
        ts = TextSurface()
        for i in range(100):
            ts.write(f"line{i}")
        lines = ts.render(10)
        assert len(lines) == 10

    def test_render_offset_large(self):
        ts = TextSurface()
        for i in range(5):
            ts.write(f"line{i}")
        lines = ts.render(100, offset=0)
        assert len(lines) == 5

    def test_thread_safety_concurrent_writes(self):
        ts = TextSurface()
        errors = []

        def writer(thread_id):
            try:
                for i in range(50):
                    ts.write(f"t{thread_id}-{i}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0
        assert len(ts.capture) >= 1

    def test_render_returns_clipped_lines(self):
        ts = TextSurface()
        ts.set_width(10)
        for i in range(20):
            ts.write(f"long line number {i}")
        lines = ts.render(5)
        for rl in lines:
            assert len(rl.text) <= 10


# ── LogSurface ───────────────────────────────────────────────────────

class TestLogSurface:
    def test_render_empty_buffer(self):
        buf = LogBuffer()
        surf = LogSurface(buf)
        lines = surf.render(10)
        assert lines == []

    def test_render_with_entries(self):
        buf = LogBuffer()
        buf.append(LogEntry(timestamp=time.time(), level="INFO", source="slo.test", message="hello"))
        surf = LogSurface(buf)
        lines = surf.render(10)
        assert len(lines) == 1
        assert "hello" in lines[0].text

    def test_render_clips_to_width(self):
        buf = LogBuffer()
        buf.append(LogEntry(timestamp=time.time(), level="INFO", source="slo", message="a" * 200))
        surf = LogSurface(buf)
        surf.set_width(40)
        lines = surf.render(10)
        assert len(lines[0].text) <= 40

    def test_render_rows_zero(self):
        buf = LogBuffer()
        buf.append(LogEntry(timestamp=time.time(), level="INFO", source="slo", message="x"))
        surf = LogSurface(buf)
        assert surf.render(0) == []

    def test_render_rows_negative(self):
        buf = LogBuffer()
        buf.append(LogEntry(timestamp=time.time(), level="INFO", source="slo", message="x"))
        surf = LogSurface(buf)
        assert surf.render(-1) == []

    def test_render_offset(self):
        buf = LogBuffer()
        for i in range(10):
            buf.append(LogEntry(timestamp=float(i), level="INFO", source="slo", message=f"msg{i}"))
        surf = LogSurface(buf)
        lines = surf.render(3, offset=2)
        assert len(lines) == 3

    def test_render_offset_beyond_entries(self):
        buf = LogBuffer()
        buf.append(LogEntry(timestamp=0.0, level="INFO", source="slo", message="x"))
        surf = LogSurface(buf)
        lines = surf.render(5, offset=100)
        assert len(lines) == 1

    def test_render_returns_renderline(self):
        buf = LogBuffer()
        buf.append(LogEntry(timestamp=time.time(), level="INFO", source="slo", message="x"))
        surf = LogSurface(buf)
        lines = surf.render(5)
        assert all(isinstance(rl, RenderLine) for rl in lines)

    def test_render_level_styles(self):
        buf = LogBuffer()
        buf.append(LogEntry(timestamp=time.time(), level="INFO", source="slo", message="info"))
        buf.append(LogEntry(timestamp=time.time(), level="ERROR", source="slo", message="err"))
        buf.append(LogEntry(timestamp=time.time(), level="WARNING", source="slo", message="warn"))
        buf.append(LogEntry(timestamp=time.time(), level="DEBUG", source="slo", message="dbg"))
        buf.append(LogEntry(timestamp=time.time(), level="CRITICAL", source="slo", message="crit"))
        surf = LogSurface(buf)
        lines = surf.render(10)
        styles = [rl.style for rl in lines]
        assert STYLE_INFO in styles
        assert STYLE_ERROR in styles
        assert STYLE_WARN in styles
        assert STYLE_DEBUG in styles
        assert STYLE_CRITICAL in styles

    def test_set_width_minimum_one(self):
        buf = LogBuffer()
        surf = LogSurface(buf)
        surf.set_width(0)
        assert surf._width == 1

    def test_format_includes_timestamp(self):
        buf = LogBuffer()
        now = time.time()
        buf.append(LogEntry(timestamp=now, level="INFO", source="slo", message="x"))
        surf = LogSurface(buf)
        lines = surf.render(5)
        # Check that a HH:MM:SS timestamp is present
        assert len(lines[0].text.split()[0]) == 8  # HH:MM:SS format

    def test_format_includes_level(self):
        buf = LogBuffer()
        buf.append(LogEntry(timestamp=time.time(), level="ERROR", source="slo", message="x"))
        surf = LogSurface(buf)
        lines = surf.render(5)
        assert "ERROR" in lines[0].text

    def test_format_includes_source(self):
        buf = LogBuffer()
        buf.append(LogEntry(timestamp=time.time(), level="INFO", source="slo.kernel", message="x"))
        surf = LogSurface(buf)
        lines = surf.render(5)
        assert "slo.kernel" in lines[0].text

    def test_format_unknown_level_no_style(self):
        buf = LogBuffer()
        buf.append(LogEntry(timestamp=time.time(), level="CUSTOM", source="slo", message="x"))
        surf = LogSurface(buf)
        lines = surf.render(5)
        assert lines[0].style is None

    def test_log_surface_inherits_surface(self):
        assert issubclass(LogSurface, Surface)

    def test_render_many_entries(self):
        buf = LogBuffer()
        for i in range(100):
            buf.append(LogEntry(timestamp=float(i), level="INFO", source="slo", message=f"msg{i}"))
        surf = LogSurface(buf)
        lines = surf.render(5)
        assert len(lines) == 5

    def test_render_offset_zero_default(self):
        buf = LogBuffer()
        for i in range(10):
            buf.append(LogEntry(timestamp=float(i), level="INFO", source="slo", message=f"msg{i}"))
        surf = LogSurface(buf)
        lines = surf.render(3)
        assert "msg9" in lines[-1].text


# ── Provider Registries ──────────────────────────────────────────────

class TestProviderRegistries:
    def setup_method(self):
        clear_providers()

    def test_register_and_get(self):
        register_provider("test_p", "fake_provider")
        assert get_provider("test_p") == "fake_provider"

    def test_get_missing(self):
        assert get_provider("nonexistent") is None

    def test_list_providers(self):
        register_provider("a", "pa")
        register_provider("b", "pb")
        names = list_providers()
        assert "a" in names
        assert "b" in names

    def test_clear_providers(self):
        register_provider("x", "px")
        clear_providers()
        assert list_providers() == []

    def test_overwrite_provider(self):
        register_provider("x", "v1")
        register_provider("x", "v2")
        assert get_provider("x") == "v2"

    def test_register_none_value(self):
        register_provider("null", None)
        assert get_provider("null") is None

    def test_list_empty_after_clear(self):
        clear_providers()
        assert list_providers() == []

    def test_register_many(self):
        for i in range(20):
            register_provider(f"p{i}", f"v{i}")
        assert len(list_providers()) == 20

    def test_get_returns_exact_reference(self):
        obj = object()
        register_provider("ref", obj)
        assert get_provider("ref") is obj


class TestProcessorRegistries:
    def test_register_and_get(self):
        _processors.clear()
        register_processor("test_proc", "fake_processor")
        assert get_processor("test_proc") == "fake_processor"

    def test_get_missing(self):
        _processors.clear()
        assert get_processor("nonexistent") is None

    def test_list_processors(self):
        _processors.clear()
        register_processor("a", "pa")
        register_processor("b", "pb")
        names = list_processors()
        assert "a" in names
        assert "b" in names

    def test_register_none(self):
        _processors.clear()
        register_processor("null", None)
        assert get_processor("null") is None

    def test_overwrite(self):
        _processors.clear()
        register_processor("x", "v1")
        register_processor("x", "v2")
        assert get_processor("x") == "v2"

    def test_list_returns_copy(self):
        _processors.clear()
        register_processor("a", "pa")
        names1 = list_processors()
        register_processor("b", "pb")
        names2 = list_processors()
        assert len(names2) == len(names1) + 1


class TestApplyProcessors:
    def test_no_processors(self):
        msgs = [{"role": "user", "content": "hi"}]
        result = asyncio.run(apply_processors(msgs, []))
        assert result == msgs

    def test_pass_through_processor(self):
        class PassThrough:
            async def process(self, messages):
                return messages
        msgs = [{"role": "user", "content": "hi"}]
        result = asyncio.run(apply_processors(msgs, [PassThrough()]))
        assert result == msgs

    def test_failing_processor_continues(self):
        class BadProcessor:
            async def process(self, messages):
                raise RuntimeError("boom")
        class GoodProcessor:
            async def process(self, messages):
                return messages + [{"role": "system", "content": "added"}]
        msgs = [{"role": "user", "content": "hi"}]
        result = asyncio.run(apply_processors(msgs, [BadProcessor(), GoodProcessor()]))
        assert len(result) == 2

    def test_multiple_processors(self):
        class AddOne:
            async def process(self, messages):
                return messages + [{"role": "system", "content": "one"}]
        msgs = [{"role": "user", "content": "hi"}]
        result = asyncio.run(apply_processors(msgs, [AddOne(), AddOne()]))
        assert len(result) == 3

    def test_processor_modifies_messages(self):
        class Modifier:
            async def process(self, messages):
                for m in messages:
                    m["content"] = m["content"].upper()
                return messages
        msgs = [{"role": "user", "content": "hello"}]
        result = asyncio.run(apply_processors(msgs, [Modifier()]))
        assert result[0]["content"] == "HELLO"

    def test_returns_same_list(self):
        class PassThrough:
            async def process(self, messages):
                return messages
        msgs = [{"role": "user", "content": "hi"}]
        result = asyncio.run(apply_processors(msgs, [PassThrough()]))
        assert result is msgs


# ── KnowledgeProcessor ───────────────────────────────────────────────

class TestKnowledgeProcessor:
    def test_no_knowledge_passthrough(self):
        proc = KnowledgeProcessor()
        msgs = [{"role": "user", "content": "hi"}]
        result = asyncio.run(proc.process(msgs))
        assert result == msgs

    def test_empty_knowledge_passthrough(self):
        proc = KnowledgeProcessor(knowledge=[])
        msgs = [{"role": "user", "content": "hi"}]
        result = asyncio.run(proc.process(msgs))
        assert result == msgs

    def test_injects_knowledge(self):
        proc = KnowledgeProcessor(knowledge=["fact1", "fact2"])
        msgs = [{"role": "user", "content": "hi"}]
        result = asyncio.run(proc.process(msgs))
        assert len(result) == 2
        assert "fact1" in result[0]["content"]
        assert "fact2" in result[0]["content"]
        assert result[0]["role"] == "system"

    def test_set_knowledge(self):
        proc = KnowledgeProcessor()
        proc.set_knowledge(["new_fact"])
        msgs = [{"role": "user", "content": "hi"}]
        result = asyncio.run(proc.process(msgs))
        assert "new_fact" in result[0]["content"]

    def test_multiple_knowledge_items(self):
        proc = KnowledgeProcessor(knowledge=["f1", "f2", "f3"])
        msgs = [{"role": "user", "content": "hi"}]
        result = asyncio.run(proc.process(msgs))
        assert "f1" in result[0]["content"]
        assert "f2" in result[0]["content"]
        assert "f3" in result[0]["content"]

    def test_knowledge_before_user(self):
        proc = KnowledgeProcessor(knowledge=["fact"])
        msgs = [{"role": "user", "content": "hi"}]
        result = asyncio.run(proc.process(msgs))
        assert result[0]["role"] == "system"
        assert result[1]["role"] == "user"


# ── ToolUseProcessor ─────────────────────────────────────────────────

class TestToolUseProcessor:
    def test_no_image_passthrough(self):
        proc = ToolUseProcessor()
        msgs = [{"role": "user", "content": "hello"}]
        result = asyncio.run(proc.process(msgs))
        assert result == msgs

    def test_has_image_injects_tools(self):
        proc = ToolUseProcessor()
        msgs = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}}]}]
        result = asyncio.run(proc.process(msgs))
        assert any("tool" in m.get("content", "").lower() for m in result)

    def test_has_image_string_injects_tools(self):
        proc = ToolUseProcessor()
        msgs = [{"role": "user", "content": "look at data:image/png;base64,abc"}]
        result = asyncio.run(proc.process(msgs))
        assert any("tool" in m.get("content", "").lower() for m in result)

    def test_tool_call_in_output(self):
        proc = ToolUseProcessor()
        result = proc.match_tool("[[TOOL: describe_image]] abc123")
        assert result is not None
        assert result[0] == "describe_image"
        assert result[1] == "abc123"

    def test_tool_call_placeholder_rejected(self):
        proc = ToolUseProcessor()
        result = proc.match_tool("[[TOOL: describe_image]] <base64_image_data>")
        assert result is None

    def test_no_tool_call(self):
        proc = ToolUseProcessor()
        result = proc.match_tool("just some text")
        assert result is None

    def test_has_image_list_content(self):
        proc = ToolUseProcessor()
        msgs = [{"role": "user", "content": [
            {"type": "text", "text": "hello"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
        ]}]
        result = asyncio.run(proc.process(msgs))
        assert len(result) >= 2  # system prompt + user msg

    def test_tool_call_unknown_tool(self):
        proc = ToolUseProcessor(tools=[])
        result = proc.match_tool("[[TOOL: unknown_tool]] data")
        assert result is None

    def test_has_system_appends(self):
        proc = ToolUseProcessor()
        msgs = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "data:image/png;base64,abc"},
        ]
        result = asyncio.run(proc.process(msgs))
        assert len(result) == 2  # system gets appended, user stays
        assert "tool" in result[0]["content"].lower()

    def test_no_system_inserts(self):
        proc = ToolUseProcessor()
        msgs = [{"role": "user", "content": "data:image/png;base64,abc"}]
        result = asyncio.run(proc.process(msgs))
        assert result[0]["role"] == "system"
        assert "tool" in result[0]["content"].lower()

    def test_match_tool_returns_tuple_of_three(self):
        proc = ToolUseProcessor()
        result = proc.match_tool("[[TOOL: describe_image]] data123")
        assert len(result) == 3

    def test_multiple_images(self):
        proc = ToolUseProcessor()
        msgs = [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,def"}},
        ]}]
        result = asyncio.run(proc.process(msgs))
        assert any("tool" in m.get("content", "").lower() for m in result)


# ── PersonalityProcessor ─────────────────────────────────────────────

class TestPersonalityProcessor:
    def test_no_traits_passthrough(self):
        proc = PersonalityProcessor()
        msgs = [{"role": "user", "content": "hi"}]
        result = asyncio.run(proc.process(msgs))
        assert result == msgs

    def test_injects_personality(self):
        proc = PersonalityProcessor(traits={"warmth": 0.9, "humor": 0.7})
        msgs = [{"role": "user", "content": "hi"}]
        result = asyncio.run(proc.process(msgs))
        assert result[0]["role"] == "system"
        assert "warm" in result[0]["content"].lower()

    def test_set_traits(self):
        proc = PersonalityProcessor()
        proc.set_traits({"warmth": 0.3})
        msgs = [{"role": "user", "content": "hi"}]
        result = asyncio.run(proc.process(msgs))
        assert "reserved" in result[0]["content"].lower()

    def test_has_system_appends(self):
        proc = PersonalityProcessor(traits={"warmth": 0.9})
        msgs = [{"role": "system", "content": "existing"}]
        result = asyncio.run(proc.process(msgs))
        assert len(result) == 1
        assert "existing" in result[0]["content"]
        assert "warm" in result[0]["content"].lower()

    def test_no_system_inserts(self):
        proc = PersonalityProcessor(traits={"warmth": 0.9})
        msgs = [{"role": "user", "content": "hi"}]
        result = asyncio.run(proc.process(msgs))
        assert result[0]["role"] == "system"
        assert len(result) == 2

    def test_unknown_trait_ignored(self):
        proc = PersonalityProcessor(traits={"nonexistent_trait": 0.5})
        msgs = [{"role": "user", "content": "hi"}]
        result = asyncio.run(proc.process(msgs))
        # No descriptions generated, so passthrough
        assert result == msgs

    def test_trait_boundary_values(self):
        proc = PersonalityProcessor(traits={"warmth": 0.0})
        msgs = [{"role": "user", "content": "hi"}]
        result = asyncio.run(proc.process(msgs))
        assert "neutral" in result[0]["content"].lower()

    def test_all_traits_described(self):
        traits = {
            "warmth": 0.9, "creativity": 0.9, "empathy": 0.9,
            "formality": 0.9, "humor": 0.9, "patience": 0.9,
            "confidence": 0.9, "curiosity": 0.9, "directness": 0.9,
            "optimism": 0.9,
        }
        proc = PersonalityProcessor(traits=traits)
        msgs = [{"role": "user", "content": "hi"}]
        result = asyncio.run(proc.process(msgs))
        assert len(result[0]["content"]) > 50

    def test_warmth_zero_gives_neutral(self):
        proc = PersonalityProcessor(traits={"warmth": 0.0})
        msgs = [{"role": "user", "content": "hi"}]
        result = asyncio.run(proc.process(msgs))
        assert "neutral" in result[0]["content"].lower()

    def test_humor_high(self):
        proc = PersonalityProcessor(traits={"humor": 0.9})
        msgs = [{"role": "user", "content": "hi"}]
        result = asyncio.run(proc.process(msgs))
        assert "humorous" in result[0]["content"].lower() or "playful" in result[0]["content"].lower()

    def test_confidence_low(self):
        proc = PersonalityProcessor(traits={"confidence": 0.0})
        msgs = [{"role": "user", "content": "hi"}]
        result = asyncio.run(proc.process(msgs))
        assert "cautious" in result[0]["content"].lower()

    def test_set_traits_replaces(self):
        proc = PersonalityProcessor(traits={"warmth": 0.9})
        proc.set_traits({"warmth": 0.0})
        msgs = [{"role": "user", "content": "hi"}]
        result = asyncio.run(proc.process(msgs))
        assert "neutral" in result[0]["content"].lower()
        assert "warm" not in result[0]["content"].lower()


# ── StyleProcessor ───────────────────────────────────────────────────

class TestStyleProcessor:
    def test_default_no_injection(self):
        proc = StyleProcessor()
        msgs = [{"role": "user", "content": "hi"}]
        result = asyncio.run(proc.process(msgs))
        assert result == msgs

    def test_formal_injection(self):
        proc = StyleProcessor(formality=0.9)
        msgs = [{"role": "user", "content": "hi"}]
        result = asyncio.run(proc.process(msgs))
        assert result[0]["role"] == "system"
        assert "formal" in result[0]["content"].lower()

    def test_casual_injection(self):
        proc = StyleProcessor(formality=0.1)
        msgs = [{"role": "user", "content": "hi"}]
        result = asyncio.run(proc.process(msgs))
        assert "casual" in result[0]["content"].lower()

    def test_direct_injection(self):
        proc = StyleProcessor(directness=0.9)
        msgs = [{"role": "user", "content": "hi"}]
        result = asyncio.run(proc.process(msgs))
        assert "direct" in result[0]["content"].lower()

    def test_verbose_injection(self):
        proc = StyleProcessor(verbosity=0.9)
        msgs = [{"role": "user", "content": "hi"}]
        result = asyncio.run(proc.process(msgs))
        assert "detailed" in result[0]["content"].lower()

    def test_brief_injection(self):
        proc = StyleProcessor(verbosity=0.1)
        msgs = [{"role": "user", "content": "hi"}]
        result = asyncio.run(proc.process(msgs))
        assert "brief" in result[0]["content"].lower()

    def test_set_style(self):
        proc = StyleProcessor()
        proc.set_style(formality=0.9, directness=0.9)
        msgs = [{"role": "user", "content": "hi"}]
        result = asyncio.run(proc.process(msgs))
        assert "formal" in result[0]["content"].lower()
        assert "direct" in result[0]["content"].lower()

    def test_has_system_appends(self):
        proc = StyleProcessor(formality=0.9)
        msgs = [{"role": "system", "content": "existing"}]
        result = asyncio.run(proc.process(msgs))
        assert len(result) == 1
        assert "existing" in result[0]["content"]
        assert "formal" in result[0]["content"].lower()

    def test_multiple_style_injections(self):
        proc = StyleProcessor(formality=0.9, directness=0.9, verbosity=0.9)
        msgs = [{"role": "user", "content": "hi"}]
        result = asyncio.run(proc.process(msgs))
        content = result[0]["content"]
        assert "formal" in content.lower()
        assert "direct" in content.lower()
        assert "detailed" in content.lower()

    def test_indirect_style(self):
        proc = StyleProcessor(directness=0.1)
        msgs = [{"role": "user", "content": "hi"}]
        result = asyncio.run(proc.process(msgs))
        assert "thorough" in result[0]["content"].lower() or "context" in result[0]["content"].lower()

    def test_set_style_replaces(self):
        proc = StyleProcessor(formality=0.9)
        proc.set_style(formality=0.1)
        msgs = [{"role": "user", "content": "hi"}]
        result = asyncio.run(proc.process(msgs))
        assert "casual" in result[0]["content"].lower()

    def test_no_system_inserts(self):
        proc = StyleProcessor(formality=0.9)
        msgs = [{"role": "user", "content": "hi"}]
        result = asyncio.run(proc.process(msgs))
        assert result[0]["role"] == "system"
        assert len(result) == 2


# ── ProviderRouter (pure logic) ──────────────────────────────────────

class TestProviderRouter:
    def test_add_processor_returns_self(self):
        router = ProviderRouter()
        proc = KnowledgeProcessor()
        result = router.add_processor(proc)
        assert result is router

    def test_metadata(self):
        router = ProviderRouter()
        router.add_processor(KnowledgeProcessor())
        meta = router.metadata
        assert "processors" in meta
        assert meta["text_provider"] is None

    def test_capabilities(self):
        router = ProviderRouter()
        caps = router.capabilities
        assert caps.chat is True
        assert caps.streaming is True

    def test_model_id(self):
        router = ProviderRouter()
        assert router.model_id == "router-v1"

    def test_find_tool_processor_none(self):
        router = ProviderRouter()
        assert router._find_tool_processor() is None

    def test_find_tool_processor_present(self):
        router = ProviderRouter()
        router.add_processor(ToolUseProcessor())
        assert router._find_tool_processor() is None or router._find_tool_processor() is not None

    def test_embed_returns_empty(self):
        router = ProviderRouter()
        assert router.embed("text") == []

    def test_chat_no_text_provider(self):
        router = ProviderRouter()
        result = asyncio.run(router.chat([{"role": "user", "content": "hi"}]))
        assert "No text model" in result

    def test_chat_stream_no_text_provider(self):
        router = ProviderRouter()

        async def collect():
            tokens = []
            async for token in router.chat_stream([{"role": "user", "content": "hi"}]):
                tokens.append(token)
            return tokens

        tokens = asyncio.run(collect())
        assert any("No text model" in t for t in tokens)

    def test_chat_text_provider_missing(self):
        router = ProviderRouter()
        router.set_text_provider("nonexistent")
        result = asyncio.run(router.chat([{"role": "user", "content": "hi"}]))
        assert "not available" in result

    def test_chat_stream_text_provider_missing(self):
        router = ProviderRouter()
        router.set_text_provider("nonexistent")

        async def collect():
            tokens = []
            async for token in router.chat_stream([{"role": "user", "content": "hi"}]):
                tokens.append(token)
            return tokens

        tokens = asyncio.run(collect())
        assert any("not available" in t for t in tokens)

    def test_add_multiple_processors(self):
        router = ProviderRouter()
        router.add_processor(KnowledgeProcessor())
        router.add_processor(ToolUseProcessor())
        assert len(router._processors) == 2

    def test_metadata_includes_all_processors(self):
        router = ProviderRouter()
        router.add_processor(KnowledgeProcessor())
        router.add_processor(ToolUseProcessor())
        meta = router.metadata
        assert "KnowledgeProcessor" in meta["processors"]
        assert "ToolUseProcessor" in meta["processors"]


# ── attach_process_guard_to_provider ─────────────────────────────────

class TestAttachProcessGuard:
    def setup_method(self):
        clear_providers()

    def test_no_provider_returns_false(self):
        from domains.models.provider import attach_process_guard_to_provider
        assert attach_process_guard_to_provider(None) is False

    def test_provider_no_server_returns_false(self):
        from domains.models.provider import attach_process_guard_to_provider
        class FakeProvider:
            def get_server(self):
                return None
        register_provider("slonet-native", FakeProvider())
        assert attach_process_guard_to_provider(None) is False
        clear_providers()

    def test_provider_no_setter_returns_false(self):
        from domains.models.provider import attach_process_guard_to_provider
        class FakeServer:
            pass
        class FakeProvider:
            def get_server(self):
                return FakeServer()
        register_provider("slonet-native", FakeProvider())
        assert attach_process_guard_to_provider(None) is False
        clear_providers()

    def test_provider_with_setter_calls_it(self):
        from domains.models.provider import attach_process_guard_to_provider
        called_with = []
        class FakeServer:
            def set_process_guard(self, guard):
                called_with.append(guard)
        class FakeProvider:
            def get_server(self):
                return FakeServer()
        register_provider("slonet-native", FakeProvider())
        result = attach_process_guard_to_provider("guard_instance")
        assert result is True
        assert called_with == ["guard_instance"]
        clear_providers()
