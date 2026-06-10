"""
parse — extract text from HTML. Zero deps (stdlib html.parser only).

Uses a stack-based approach for correct nested element handling.
Handles <br>, <img> alt, inline formatting, tables, lists, code.
"""

from html.parser import HTMLParser
from typing import Optional


def parse_html(html: str) -> "Page":
    """Parse HTML string into a structured Page.

    Args:
        html: Raw HTML.

    Returns:
        Page with .title, .text, .links, .tables, .lists, .code, .meta.
    """
    p = _Parser()
    p.feed(html)
    p.close()
    texts = "\n".join(t for t in p.texts if t)
    return Page(
        title=p.title,
        text=texts,
        links=p.links,
        tables=p.tables,
        lists=p.lists,
        code=p.code,
        meta=p.meta,
    )


def parse(url: str, *, timeout: int = 15, rate: float = 0.5) -> Optional["Page"]:
    """Shorthand: fetch a URL and parse it.

    Args:
        url: URL to fetch and parse.
        timeout: Request timeout in seconds.
        rate: Rate limit in seconds per domain.

    Returns:
        Page, or None if fetch failed.
    """
    from .fetch import fetch as _fetch
    html = _fetch(url, timeout=timeout, rate=rate)
    if html is None:
        return None
    page = parse_html(html)
    page.url = url
    return page


class Page:
    """Structured content from a single crawled page.

    All fields are plain attributes.

    Attributes:
        url: Source URL.
        title: Page title.
        text: All visible text, block-separated by newlines.
        links: List of {"href": str, "text": str}.
        tables: List of {"caption": str, "headers": list, "rows": list[list]}.
        lists: List of {"tag": "ul"|"ol", "items": list[str]}.
        code: List of {"lang": str, "body": str}.
        meta: Dict of meta name/OG property -> content.
    """

    __slots__ = ("url", "title", "text", "links", "tables", "lists", "code", "meta")

    def __init__(self, **kw):
        defaults = {"url": "", "title": "", "text": "", "meta": {}}
        for k in self.__slots__:
            setattr(self, k, kw.get(k, defaults.get(k, [])))

    def to_dict(self) -> dict:
        """Serialize Page to a plain dict (JSON-compatible)."""
        return {k: getattr(self, k) for k in self.__slots__}

    @classmethod
    def from_dict(cls, d: dict) -> "Page":
        """Deserialize a dict back into a Page.

        Args:
            d: Dict with keys matching __slots__.

        Returns:
            New Page instance.
        """
        defaults = {"url": "", "title": "", "text": "", "meta": {}}
        kwargs = {}
        for k in cls.__slots__:
            kwargs[k] = d.get(k, defaults.get(k, []))
        return cls(**kwargs)


class _Parser(HTMLParser):
    """Stack-based HTML parser. Correctly handles nesting, <br>, <img>, tables, lists, code."""

    BLOCK = {"p", "div", "section", "article", "main", "header", "h1", "h2", "h3",
             "h4", "h5", "h6", "blockquote", "pre", "td", "th", "li", "caption"}
    IGNORE = {"script", "style", "nav", "footer", "aside", "noscript", "form", "svg"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.title = ""
        self.texts: list[str] = []
        self.links: list[dict] = []
        self.tables: list[dict] = []
        self.lists: list[dict] = []
        self.code: list[dict] = []
        self.meta: dict[str, str] = {}
        self._stack: list[dict] = []
        self._root_buf: list[str] = []

    def _push(self, tag: str, extra: Optional[dict] = None) -> dict:
        ctx = {"tag": tag.lower(), "buf": [], "ignore": False}
        if extra:
            ctx.update(extra)
        self._stack.append(ctx)
        return ctx

    def _pop(self) -> dict:
        return self._stack.pop()

    @property
    def _ctx(self):
        return self._stack[-1] if self._stack else None

    def _ignored(self) -> bool:
        return any(s.get("ignore") for s in self._stack)

    def _write(self, text: str) -> None:
        """Write text to current context, or root if no context (outside ignored)."""
        if not self._ignored():
            ctx = self._ctx
            if ctx:
                ctx["buf"].append(text)
            else:
                self._root_buf.append(text)

    def _on_tag(self, tag: str) -> bool:
        """Check if the top of stack has a given tag name."""
        return bool(self._stack and self._stack[-1]["tag"] == tag)

    # --- tag handlers ---

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        a = dict(attrs)

        if tag in self.IGNORE:
            self._push(tag, {"ignore": True})
            return

        if self._ignored():
            return

        if tag == "title":
            self._push(tag)

        elif tag == "br":
            self._write("\n")

        elif tag == "hr":
            self.texts.append("─" * 40)

        elif tag == "img":
            alt = a.get("alt", "").strip()
            if alt:
                self._write(alt)

        elif tag == "meta":
            name = a.get("name") or a.get("property") or ""
            content = a.get("content", "")
            if name and content:
                self.meta[name] = content

        elif tag == "a":
            self._push("a", {"href": a.get("href", "")})

        elif tag == "table":
            self._push("table", {"caption": "", "headers": [], "rows": [], "noflush": True})

        elif tag in ("th", "td"):
            for s in reversed(self._stack):
                if s["tag"] == "tr":
                    s["is_header"] = tag == "th"
                    break
            self._push("cell", {"noflush": True})

        elif tag == "tr":
            self._push("tr", {"cells": [], "is_header": False})

        elif tag in ("ul", "ol"):
            self._push("list", {"ltype": tag, "items": [], "noflush": True})

        elif tag == "li":
            self._push("li", {"noflush": True})

        elif tag == "pre":
            self._push("pre", {"lang": "", "noflush": True})

        elif tag == "code":
            if self._on_tag("pre"):
                lang = next((v.replace("language-", "").strip() for k, v in attrs if k == "class"), "")
                self._push("code", {"lang": lang, "in_pre": True, "noflush": True})

        elif tag == "caption":
            self._push("caption")

        elif tag in self.BLOCK:
            self._push(tag)

    def handle_endtag(self, tag):
        tag = tag.lower()

        if tag == "title":
            if not self._on_tag("title"):
                return
            ctx = self._pop()
            self.title = "".join(ctx["buf"]).strip()
            return

        if tag in self.IGNORE:
            self._pop()
            return

        if self._ignored():
            return

        if tag == "a" and self._on_tag("a"):
            ctx = self._pop()
            href = ctx.get("href", "")
            text = "".join(ctx["buf"]).strip()
            if href and not href.startswith(("#", "javascript:", "mailto:")):
                self.links.append({"href": href, "text": text or href})
            return

        if tag in ("th", "td") and self._on_tag("cell"):
            ctx = self._pop()
            cell = "".join(ctx["buf"]).strip()
            for s in reversed(self._stack):
                if s["tag"] == "tr":
                    s["cells"].append(cell)
                    break
            return

        if tag == "tr" and self._on_tag("tr"):
            ctx = self._pop()
            cells = ctx["cells"]
            is_header = ctx["is_header"]
            for s in reversed(self._stack):
                if s["tag"] == "table":
                    if is_header and not s["headers"]:
                        s["headers"] = cells
                    else:
                        s["rows"].append(cells)
                    break
            return

        if tag == "table" and self._on_tag("table"):
            ctx = self._pop()
            cap = ctx.get("caption", "")
            headers = ctx.get("headers", [])
            rows = ctx.get("rows", [])
            if headers or rows:
                self.tables.append({"caption": cap or "", "headers": headers, "rows": rows})
            return

        if tag in ("ul", "ol") and self._on_tag("list"):
            ctx = self._pop()
            items = ctx.get("items", [])
            if items:
                self.lists.append({"tag": ctx.get("ltype", tag), "items": items})
            return

        if tag == "li" and self._on_tag("li"):
            ctx = self._pop()
            text = "".join(ctx["buf"]).strip()
            if text:
                for s in reversed(self._stack):
                    if s["tag"] == "list":
                        s["items"].append(text)
                        break
            return

        if tag == "pre" and self._on_tag("pre"):
            ctx = self._pop()
            body = "".join(ctx["buf"]).strip()
            if body:
                self.code.append({"lang": "", "body": body})
            return

        if tag == "code" and self._on_tag("code"):
            ctx = self._pop()
            body = "".join(ctx["buf"]).strip()
            if body:
                lang = ctx.get("lang", "")
                for c in reversed(self.code):
                    if c["lang"] == "" and c["body"] == body:
                        c["lang"] = lang
                        return
                self.code.append({"lang": lang, "body": body})
            return

        if tag == "caption" and self._on_tag("caption"):
            ctx = self._pop()
            text = "".join(ctx["buf"]).strip()
            for s in reversed(self._stack):
                if s["tag"] == "table":
                    s["caption"] = text
                    break
            return

        if tag in self.BLOCK and self._on_tag(tag):
            ctx = self._pop()
            text = "".join(ctx["buf"]).strip()
            if text:
                self.texts.append(text)
            return

    def handle_data(self, data):
        self._write(data)

    def close(self):
        super().close()
        if self._root_buf:
            text = "".join(self._root_buf).strip()
            if text:
                self.texts.append(text)
