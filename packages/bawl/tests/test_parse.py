"""Tests for bawl.parse — edge cases, nesting, malformed HTML, br, img."""

from bawl.parse import parse_html, Page


def test_title():
    p = parse_html("<html><head><title>Hello</title></head></html>")
    assert p.title == "Hello"


def test_text_block_separator():
    p = parse_html("<p>First</p><p>Second</p>")
    assert p.text == "First\nSecond"


def test_links():
    p = parse_html('<a href="/page">click</a>')
    assert len(p.links) == 1
    assert p.links[0]["href"] == "/page"
    assert p.links[0]["text"] == "click"


def test_links_ignore_javascript():
    p = parse_html('<a href="javascript:void(0)">no</a>')
    assert p.links == []


def test_br_inserts_newline():
    p = parse_html("<p>line1<br>line2</p>")
    assert p.text == "line1\nline2"


def test_hr_creates_separator():
    p = parse_html("<p>top</p><hr><p>bottom</p>")
    assert "top" in p.text
    assert "bottom" in p.text


def test_img_alt_included():
    p = parse_html('<img alt="a photo">')
    assert "a photo" in p.text


def test_img_no_alt_skipped():
    p = parse_html('<img src="x.jpg">')
    assert p.text == ""


def test_meta():
    p = parse_html('<meta name="author" content="Alice">')
    assert p.meta.get("author") == "Alice"


def test_meta_og():
    p = parse_html('<meta property="og:title" content="Hello">')
    assert p.meta.get("og:title") == "Hello"


def test_script_removed():
    p = parse_html("<p>ok</p><script>alert('x')</script><p>fine</p>")
    assert "alert" not in p.text
    assert "ok" in p.text
    assert "fine" in p.text


def test_style_removed():
    p = parse_html("<p>text</p><style>.x{color:red}</style>")
    assert ".x" not in p.text
    assert "text" in p.text


def test_nested_inline_formatting():
    html = "<p><b>bold</b> and <i>italic</i></p>"
    p = parse_html(html)
    assert p.text == "bold and italic"


def test_deeply_nested_links():
    html = '<p><b><a href="/x"><i>nested</i></a></b></p>'
    p = parse_html(html)
    assert len(p.links) == 1
    assert p.links[0]["href"] == "/x"
    assert p.links[0]["text"] == "nested"


def test_tables():
    html = """<table><caption>Data</caption>
<tr><th>A</th><th>B</th></tr>
<tr><td>1</td><td>2</td></tr></table>"""
    p = parse_html(html)
    assert len(p.tables) == 1
    t = p.tables[0]
    assert t["caption"] == "Data"
    assert t["headers"] == ["A", "B"]
    assert t["rows"] == [["1", "2"]]


def test_lists():
    html = "<ul><li>A</li><li>B</li></ul><ol><li>1</li></ol>"
    p = parse_html(html)
    tags = {l["tag"] for l in p.lists}
    assert "ul" in tags
    assert "ol" in tags
    for l in p.lists:
        if l["tag"] == "ul":
            assert l["items"] == ["A", "B"]
        if l["tag"] == "ol":
            assert l["items"] == ["1"]


def test_code():
    html = '<pre><code class="language-py">print(1)</code></pre><code>inline</code>'
    p = parse_html(html)
    langs = {c["lang"] for c in p.code}
    assert "py" in langs
    assert any(c["body"] == "print(1)" for c in p.code)


def test_nested_tags_malformed():
    html = "<p><b><i>both</b></i></p>"
    p = parse_html(html)
    assert p.text == "both"


def test_empty_html():
    p = parse_html("")
    assert p.title == ""
    assert p.text == ""


def test_no_text():
    p = parse_html("<div><img alt='pic'></div>")
    assert p.text == "pic"


def test_unknown_tag_transparent():
    p = parse_html("<p>hello <custom>world</custom></p>")
    assert p.text == "hello world"


def test_heading_h1_h6():
    html = "<h1>Big</h1><h3>Medium</h3><h6>Small</h6>"
    p = parse_html(html)
    assert "Big" in p.text
    assert "Medium" in p.text
    assert "Small" in p.text


def test_blockquote():
    p = parse_html("<blockquote>cited text</blockquote>")
    assert "cited text" in p.text


def test_to_dict_roundtrip():
    p = parse_html("<html><head><title>T</title></head><body><p>X</p></body></html>")
    d = p.to_dict()
    assert d["title"] == "T"
    p2 = Page.from_dict(d)
    assert p2.title == "T"
    assert p2.text == "X"


def test_from_dict_defaults():
    p = Page.from_dict({})
    assert p.title == ""
    assert p.text == ""
    assert p.links == []
    assert not p.meta  # empty dict or list


def test_multiline_text():
    p = parse_html("<p>line1</p><p>line2</p><p>line3</p>")
    assert p.text == "line1\nline2\nline3"


def test_footer_ignored():
    p = parse_html("<p>main</p><footer>skip</footer>")
    assert "main" in p.text
    assert "skip" not in p.text


def test_nav_ignored():
    p = parse_html("<nav><a href='/x'>nav</a></nav><p>content</p>")
    assert "nav" not in p.text
    assert "content" in p.text
    assert p.links == []


def test_form_ignored_with_inputs():
    p = parse_html("<form><input name='x'></form><p>ok</p>")
    assert "ok" in p.text
    assert p.links == []


def test_paragraph_with_code():
    p = parse_html("<p>Use <code>fetch()</code> to get data.</p>")
    assert "fetch()" in p.text


def test_complex_real_world_html():
    html = """<!DOCTYPE html>
<html><head><title>Example</title>
<meta name="desc" content="test">
</head><body>
<header><nav><a href="/">Home</a></nav></header>
<main>
<h1>Title</h1>
<p>First paragraph with <strong>bold</strong> and <em>italic</em>.</p>
<img alt="Photo of a cat" src="cat.jpg">
<br>
<p>Second paragraph.</p>
<table><tr><th>Name</th><td>Alice</td></tr></table>
<ul><li>Red</li><li>Green</li></ul>
<pre><code class="language-js">console.log("hi")</code></pre>
</main>
<footer>Footer text</footer>
</body></html>"""
    p = parse_html(html)
    assert p.title == "Example"
    assert p.meta.get("desc") == "test"
    assert "First paragraph" in p.text
    assert "bold" in p.text
    assert "italic" in p.text
    assert "Photo of a cat" in p.text
    assert "Second paragraph" in p.text
    assert "Footer" not in p.text
    assert "Home" not in p.text
    assert len(p.tables) == 1
    assert len(p.lists) == 1
    assert p.lists[0]["items"] == ["Red", "Green"]
    assert any("console.log" in c["body"] for c in p.code)


def test_classes():
    from bawl import (fetch, parse, parse_html, save, load, dumps, loads,
                         dumps_json_array, save_json_array, crawl, crawl_urls,
                         parse_sitemap)
    assert callable(fetch)
    assert callable(parse)
    assert callable(parse_html)
    assert callable(save)
    assert callable(load)
    assert callable(dumps)
    assert callable(loads)
    assert callable(dumps_json_array)
    assert callable(save_json_array)
    assert callable(crawl)
    assert callable(crawl_urls)
    assert callable(parse_sitemap)
