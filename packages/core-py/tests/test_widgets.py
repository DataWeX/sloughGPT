"""Tests for widgets.py — text-only reactive widget system."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from domains.shell.widgets import (
    App, Widget, Container, Panel, Text, Button, Input,
    List, Menu, Tabs, Dialog, Separator, ProgressBar,
    Spinner, Box, EventBus, KeyEvent, ResizeEvent, FocusEvent,
)


# ── Helpers ───────────────────────────────────────────────────────────

def render_widget(w: Widget, rows: int = 10, cols: int = 40) -> list[str]:
    """Compute layout and render a widget, returning lines."""
    w.compute(rows, cols)
    return w.render()


def assert_lines(lines: list[str], rows: int, cols: int) -> None:
    """Assert all lines are exactly cols wide and there are exactly rows lines."""
    assert len(lines) == rows, f"expected {rows} lines, got {len(lines)}: {lines}"
    for i, line in enumerate(lines):
        assert len(line) == cols, f"line {i} expected {cols} cols, got {len(line)}: {line!r}"


def assert_all_same_width(lines: list[str]) -> None:
    """Assert all lines have the same width."""
    if not lines:
        return
    widths = {len(l) for l in lines}
    assert len(widths) == 1, f"inconsistent widths: {widths} in {lines}"


# ── Box characters ────────────────────────────────────────────────────

def test_box_characters_are_single_width():
    assert len(Box.TL) == 1
    assert len(Box.TR) == 1
    assert len(Box.BL) == 1
    assert len(Box.BR) == 1
    assert len(Box.H) == 1
    assert len(Box.V) == 1


# ── Widget base ───────────────────────────────────────────────────────

def test_widget_base_compute_and_render():
    w = Widget(name="test")
    used = w.compute(5, 20)
    assert used == (5, 20)
    lines = w.render()
    assert len(lines) == 1
    assert lines[0] == ""


def test_widget_invalidate():
    w = Widget()
    assert w._dirty is True
    w.mark_clean()
    assert w._dirty is False
    w.invalidate()
    assert w._dirty is True


def test_widget_pad_line():
    w = Widget()
    assert w._pad_line("hi", 5) == "hi   "
    assert w._pad_line("hello world", 5) == "hello"  # truncates
    assert w._pad_line("", 3) == "   "


def test_widget_center():
    w = Widget()
    assert w._center("hi", 6) == "  hi  "
    assert w._center("hi", 5) == " hi  "  # left-biased
    assert w._center("hello", 5) == "hello"


def test_widget_hline():
    w = Widget()
    assert w._hline(5) == "\u2500" * 5
    assert w._hline(0) == ""


def test_widget_truncate():
    w = Widget()
    assert w._truncate("hello", 10) == "hello"
    assert w._truncate("hello world", 5) == "hell\u2026"  # ellipsis
    assert w._truncate("hello", 4) == "hel\u2026"  # ellipsis
    assert w._truncate("hello", 3) == "hel"  # max_width <= 3, no ellipsis
    assert w._truncate("hi", 2) == "hi"


# ── Text ──────────────────────────────────────────────────────────────

def test_text_simple():
    t = Text(content="Hello")
    lines = render_widget(t, 3, 10)
    assert len(lines) == 3
    assert lines[0] == "Hello     "
    assert lines[1] == "          "
    assert lines[2] == "          "


def test_text_center():
    t = Text(content="Hi", align="center")
    lines = render_widget(t, 1, 10)
    assert lines[0] == "    Hi    "


def test_text_right():
    t = Text(content="Hi", align="right")
    lines = render_widget(t, 1, 10)
    assert lines[0] == "        Hi"


def test_text_multiline():
    t = Text(content="Line1\nLine2")
    lines = render_widget(t, 3, 10)
    assert lines[0] == "Line1     "
    assert lines[1] == "Line2     "
    assert lines[2] == "          "


def test_text_reactive():
    t = Text(content="old")
    render_widget(t, 1, 10)
    assert t._dirty is False
    t.content = "new"
    assert t._dirty is True
    lines = render_widget(t, 1, 10)
    assert lines[0] == "new       "


def test_text_empty():
    t = Text(content="")
    lines = render_widget(t, 2, 5)
    assert lines == ["     ", "     "]


def test_text_longer_than_width():
    t = Text(content="hello world")
    lines = render_widget(t, 1, 5)
    assert lines[0] == "hello world"[:5]


# ── Button ────────────────────────────────────────────────────────────

def test_button_default():
    b = Button(label="OK")
    lines = render_widget(b, 1, 20)
    assert "[ OK ]" in lines[0]
    assert len(lines[0]) == 20


def test_button_focused():
    b = Button(label="OK")
    b.compute(1, 20)
    b.focus()
    lines = b.render()
    assert "> OK <" in lines[0]


def test_button_enter():
    clicked = []
    b = Button(label="Go", on_click=lambda: clicked.append(True))
    b.compute(1, 20)
    b.handle(KeyEvent(key="enter"))
    assert clicked == [True]


# ── Input ─────────────────────────────────────────────────────────────

def test_input_default():
    inp = Input(prompt="Name:")
    lines = render_widget(inp, 1, 30)
    assert lines[0].startswith("Name: ")
    assert len(lines[0]) == 30


def test_input_with_value():
    inp = Input(prompt=">", default="hello")
    lines = render_widget(inp, 1, 20)
    assert "hello" in lines[0]


def test_input_type_char():
    inp = Input(prompt=">")
    inp.compute(1, 20)
    inp.handle(KeyEvent(key="a"))
    inp.handle(KeyEvent(key="b"))
    assert inp.value == "ab"


def test_input_backspace():
    inp = Input(prompt=">", default="abc")
    inp.compute(1, 20)
    inp.handle(KeyEvent(key="backspace"))
    assert inp.value == "ab"


def test_input_delete():
    inp = Input(prompt=">", default="abc")
    inp.compute(1, 20)
    inp.handle(KeyEvent(key="left"))
    inp.handle(KeyEvent(key="left"))
    inp.handle(KeyEvent(key="delete"))
    assert inp.value == "ac"


def test_input_cursor_movement():
    inp = Input(prompt=">", default="abc")
    inp.compute(1, 20)
    # cursor starts at 3 (end), left twice → position 1
    inp.handle(KeyEvent(key="left"))
    inp.handle(KeyEvent(key="left"))
    inp.handle(KeyEvent(key="d"))
    # insert "d" at position 1: "a" + "d" + "bc"
    assert inp.value == "adbc"
    assert inp._cursor == 2


def test_input_home_end():
    inp = Input(prompt=">", default="abc")
    inp.compute(1, 20)
    inp.handle(KeyEvent(key="home"))
    assert inp._cursor == 0
    inp.handle(KeyEvent(key="end"))
    assert inp._cursor == 3


def test_input_submit():
    submitted = []
    inp = Input(prompt=">", on_submit=lambda v: submitted.append(v))
    inp.compute(1, 20)
    inp.handle(KeyEvent(key="a"))
    inp.handle(KeyEvent(key="b"))
    inp.handle(KeyEvent(key="enter"))
    assert submitted == ["ab"]


def test_input_password():
    inp = Input(prompt="Pass:", password=True)
    inp.compute(1, 20)
    inp.handle(KeyEvent(key="s"))
    inp.handle(KeyEvent(key="e"))
    inp.handle(KeyEvent(key="c"))
    lines = inp.render()
    assert "\u2022" in lines[0]  # bullet char
    assert "sec" not in lines[0]


def test_input_empty():
    inp = Input(prompt=">")
    lines = render_widget(inp, 1, 15)
    assert lines[0] == ">              "


def test_input_long_value_scroll():
    inp = Input(prompt=">", default="a" * 50)
    lines = render_widget(inp, 1, 20)
    assert len(lines[0]) == 20


# ── Container ─────────────────────────────────────────────────────────

def test_container_vertical():
    c = Container(direction="vertical")
    t1 = Text(content="A")
    t2 = Text(content="B")
    c.add(t1)
    c.add(t2)
    lines = render_widget(c, 4, 10)
    assert len(lines) == 4
    assert "A" in lines[0]
    assert "B" in lines[1]


def test_container_horizontal():
    c = Container(direction="horizontal")
    t1 = Text(content="A")
    t2 = Text(content="B")
    c.add(t1)
    c.add(t2)
    c.compute(1, 20)
    lines = c.render()
    assert len(lines) == 1
    # Both A and B should appear on the same line
    assert "A" in lines[0]
    assert "B" in lines[0]


def test_container_gap():
    c = Container(direction="vertical", gap=1)
    t1 = Text(content="X")
    t2 = Text(content="Y")
    c.add(t1)
    c.add(t2)
    lines = render_widget(c, 5, 10)
    assert "X" in lines[0]
    assert lines[1] == "          "  # gap
    assert "Y" in lines[2]


def test_container_hidden_child():
    c = Container(direction="vertical")
    t1 = Text(content="A")
    t2 = Text(content="B", visible=False)
    c.add(t1)
    c.add(t2)
    lines = render_widget(c, 2, 10)
    assert "A" in lines[0]
    assert "B" not in "\n".join(lines)


# ── Panel ─────────────────────────────────────────────────────────────

def test_panel_border():
    p = Panel(title="Hi", child=Text(content="X"), border=True)
    lines = render_widget(p, 5, 20)
    assert len(lines) == 5
    # Top border has TL and TR
    assert lines[0][0] == Box.TL
    assert lines[0][-1] == Box.TR
    # Bottom border
    assert lines[-1][0] == Box.BL
    assert lines[-1][-1] == Box.BR
    # Side borders
    for line in lines[1:-1]:
        assert line[0] == Box.V
        assert line[-1] == Box.V


def test_panel_title():
    p = Panel(title="Test", child=Text(content=""), border=True)
    lines = render_widget(p, 3, 30)
    assert "Test" in lines[0]


def test_panel_no_border():
    p = Panel(child=Text(content="Hi"), border=False)
    lines = render_widget(p, 2, 10)
    # No border chars
    assert lines[0][0] != Box.TL
    assert "Hi" in lines[0]


def test_panel_content_inside_border():
    p = Panel(title="", child=Text(content="Hello"), border=True)
    lines = render_widget(p, 5, 20)
    # Content should be in the middle rows, inside borders
    found = False
    for line in lines[1:-1]:
        if "Hello" in line:
            found = True
    assert found


def test_panel_alignment():
    p = Panel(title="Align", child=Text(content="X"), border=True)
    lines = render_widget(p, 5, 20)
    # All lines should be same width
    assert_all_same_width(lines)


def test_panel_with_padding():
    p = Panel(title="", child=Text(content="X"), border=True, padding=0)
    lines_0 = render_widget(p, 5, 20)
    p2 = Panel(title="", child=Text(content="X"), border=True, padding=2)
    lines_2 = render_widget(p2, 7, 20)
    # Both should render without error
    assert len(lines_0) == 5
    assert len(lines_2) == 7


# ── List ──────────────────────────────────────────────────────────────

def test_list_basic():
    li = List(items=["a", "b", "c"])
    lines = render_widget(li, 5, 20)
    assert len(lines) == 5
    # First item should have arrow
    assert "a" in lines[0]


def test_list_selection():
    li = List(items=["a", "b", "c"])
    li.compute(3, 20)
    li.handle(KeyEvent(key="down"))
    assert li._selected == 1
    lines = li.render()
    assert "b" in lines[1]  # arrow on second item


def test_list_filter():
    li = List(items=["apple", "banana", "cherry"])
    li.compute(5, 20)
    li.handle(KeyEvent(key="b"))
    assert li.filtered_items == ["banana"]


def test_list_empty():
    li = List(items=[])
    lines = render_widget(li, 3, 20)
    assert len(lines) == 3


def test_list_page_up_down():
    items = [str(i) for i in range(50)]
    li = List(items=items)
    li.compute(5, 20)
    li.handle(KeyEvent(key="page_down"))
    assert li._selected > 0
    li.handle(KeyEvent(key="page_up"))
    assert li._selected == 0


# ── Menu ──────────────────────────────────────────────────────────────

def test_menu_closed():
    m = Menu(title="File", options=["New", "Open", "Save"])
    lines = render_widget(m, 1, 20)
    assert "File" in lines[0]


def test_menu_open():
    m = Menu(title="File", options=["New", "Open", "Save"])
    m.compute(5, 20)
    m.handle(KeyEvent(key="enter"))
    lines = m.render()
    assert len(lines) >= 3  # header + separator + options


def test_menu_select():
    selected = []
    m = Menu(title="File", options=["New", "Open"], on_select=lambda v: selected.append(v))
    m.compute(5, 20)
    m.handle(KeyEvent(key="enter"))  # open
    m.handle(KeyEvent(key="down"))   # move to Open
    m.handle(KeyEvent(key="enter"))  # select
    assert selected == ["Open"]


def test_menu_escape():
    m = Menu(title="File", options=["New"])
    m.compute(3, 20)
    m.handle(KeyEvent(key="enter"))  # open
    assert m._open is True
    m.handle(KeyEvent(key="esc"))
    assert m._open is False


# ── Tabs ──────────────────────────────────────────────────────────────

def test_tabs_basic():
    t = Tabs(tabs=[("A", Text(content="Content A")), ("B", Text(content="Content B"))])
    lines = render_widget(t, 5, 30)
    assert len(lines) == 5
    assert "A" in lines[0] or "B" in lines[0]


def test_tabs_switch():
    t = Tabs(tabs=[("A", Text(content="Alpha")), ("B", Text(content="Beta"))])
    t.compute(5, 30)
    assert t.active_tab == "A"
    t.handle(KeyEvent(key="right"))
    assert t.active_tab == "B"
    lines = t.render()
    assert "Beta" in "\n".join(lines)


# ── Dialog ────────────────────────────────────────────────────────────

def test_dialog_closed():
    d = Dialog(title="Confirm", child=Text(content="Sure?"))
    lines = render_widget(d, 10, 40)
    # Closed dialog renders nothing
    assert all(l.strip() == "" for l in lines)


def test_dialog_open():
    d = Dialog(title="Confirm", child=Text(content="Are you sure?"))
    d.compute(10, 40)
    d.open()
    lines = d.render()
    assert len(lines) == 10
    # Should contain the title
    found_title = any("Confirm" in l for l in lines)
    assert found_title


def test_dialog_close():
    d = Dialog(title="Test", child=Text(content="X"))
    d.compute(10, 40)
    d.open()
    d.handle(KeyEvent(key="esc"))
    assert d._visible is False


# ── Separator ─────────────────────────────────────────────────────────

def test_separator_horizontal():
    s = Separator(direction="horizontal")
    lines = render_widget(s, 1, 10)
    assert lines[0] == Box.H * 10


def test_separator_vertical():
    s = Separator(direction="vertical")
    lines = render_widget(s, 3, 1)
    assert len(lines) == 3
    for line in lines:
        assert line == Box.V


# ── ProgressBar ───────────────────────────────────────────────────────

def test_progress_bar_zero():
    pb = ProgressBar(value=0.0, label="Test")
    lines = render_widget(pb, 1, 30)
    assert "0%" in lines[0]


def test_progress_bar_full():
    pb = ProgressBar(value=1.0, label="Done")
    lines = render_widget(pb, 1, 30)
    assert "100%" in lines[0]


def test_progress_bar_reactive():
    pb = ProgressBar(value=0.0)
    render_widget(pb, 1, 30)
    assert pb._dirty is False
    pb.value = 0.5
    assert pb._dirty is True


# ── Spinner ───────────────────────────────────────────────────────────

def test_spinner_tick():
    sp = Spinner(text="Loading")
    lines = render_widget(sp, 1, 30)
    assert "Loading" in lines[0]
    frame1 = sp._frame
    sp.tick()
    assert sp._frame != frame1


# ── EventBus ──────────────────────────────────────────────────────────

def test_event_bus_subscribe_publish():
    bus = EventBus()
    received = []
    bus.subscribe(KeyEvent, lambda e: received.append(e))
    bus.publish(KeyEvent(key="x"))
    assert len(received) == 1
    assert received[0].key == "x"


def test_event_bus_unsubscribe():
    bus = EventBus()
    received = []
    cb = lambda e: received.append(e)
    bus.subscribe(KeyEvent, cb)
    bus.unsubscribe(KeyEvent, cb)
    bus.publish(KeyEvent(key="x"))
    assert len(received) == 0


# ── Focus management ──────────────────────────────────────────────────

def test_focus_blur():
    w = Widget()
    w.focus()
    assert w._focused is True
    w.blur()
    assert w._focused is False


def test_app_focus_next():
    app = App()
    root = Container()
    b1 = Button(label="1")
    b2 = Button(label="2")
    b3 = Button(label="3")
    root.add(b1)
    root.add(b2)
    root.add(b3)
    app.set_root(root)
    root.compute(3, 20)

    app.focus_first()
    assert b1._focused is True

    app.focus_next()
    assert b1._focused is False
    assert b2._focused is True

    app.focus_next()
    assert b2._focused is False
    assert b3._focused is True

    # Wrap around
    app.focus_next()
    assert b3._focused is False
    assert b1._focused is True


def test_app_focus_prev():
    app = App()
    root = Container()
    b1 = Button(label="1")
    b2 = Button(label="2")
    root.add(b1)
    root.add(b2)
    app.set_root(root)
    root.compute(2, 20)

    app.focus_first()
    assert b1._focused is True

    app.focus_prev()
    assert b1._focused is False
    assert b2._focused is True


# ── Edge cases: zero size ─────────────────────────────────────────────

def test_text_zero_size():
    t = Text(content="Hi")
    lines = render_widget(t, 0, 0)
    assert lines == []


def test_button_zero_size():
    b = Button(label="OK")
    lines = render_widget(b, 0, 0)
    assert lines == []


def test_container_empty():
    c = Container()
    lines = render_widget(c, 5, 10)
    assert lines == []


def test_panel_zero_size():
    p = Panel(child=Text(content="X"))
    lines = render_widget(p, 0, 0)
    assert lines == []


# ── Edge cases: very small sizes ──────────────────────────────────────

def test_panel_minimum_size():
    p = Panel(title="T", child=Text(content="X"), border=True)
    lines = render_widget(p, 3, 5)
    assert len(lines) == 3
    assert_all_same_width(lines)


def test_input_single_char():
    inp = Input(prompt=">")
    lines = render_widget(inp, 1, 5)
    assert len(lines[0]) == 5


def test_list_one_item():
    li = List(items=["only"])
    lines = render_widget(li, 2, 10)
    assert "only" in "\n".join(lines)


# ── Edge cases: very long content ─────────────────────────────────────

def test_text_very_long():
    t = Text(content="x" * 200)
    lines = render_widget(t, 1, 10)
    assert len(lines[0]) == 10


def test_button_very_long_label():
    b = Button(label="a" * 50)
    lines = render_widget(b, 1, 10)
    assert len(lines[0]) == 10


# ── Nested layout ─────────────────────────────────────────────────────

def test_nested_containers():
    outer = Container(direction="vertical")
    inner = Container(direction="horizontal")
    inner.add(Text(content="L"))
    inner.add(Text(content="R"))
    outer.add(inner)
    outer.add(Text(content="Bottom"))
    lines = render_widget(outer, 4, 20)
    assert_all_same_width(lines)
    assert len(lines) == 4


def test_panel_with_container_child():
    inner = Container(direction="horizontal")
    inner.add(Text(content="A"))
    inner.add(Text(content="B"))
    p = Panel(title="Test", child=inner, border=True)
    lines = render_widget(p, 5, 30)
    assert_all_same_width(lines)
    assert "A" in "\n".join(lines)
    assert "B" in "\n".join(lines)


def test_deeply_nested():
    app = App()
    root = Container()
    panel = Panel(title="Deep", border=True)
    inner = Container(direction="horizontal")
    inner.add(Text(content="X"))
    inner.add(Text(content="Y"))
    panel.set_child(inner)
    root.add(panel)
    app.set_root(root)
    lines = render_widget(root, 8, 40)
    assert_all_same_width(lines)
    assert "Deep" in lines[0]


# ── All lines same width ──────────────────────────────────────────────

def test_all_widgets_same_width():
    """Every widget should produce lines of consistent width."""
    widgets = [
        Text(content="Hi"),
        Button(label="OK"),
        Input(prompt=">"),
        List(items=["a", "b"]),
        Separator(direction="horizontal"),
        ProgressBar(value=0.5, label="Test"),
        Spinner(text="Loading"),
    ]
    for w in widgets:
        lines = render_widget(w, 3, 25)
        assert_all_same_width(lines), f"{type(w).__name__} failed"
