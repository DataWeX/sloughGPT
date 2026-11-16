#!/usr/bin/env python3
"""Algorithmized book-source discovery + acquisition (card 20261116_095846).

Pipeline: discover -> rank/filter -> acquire -> validate -> atomic write -> report.

  discover : paginated search with two sources — Gutendex API primary
             (rich metadata: formats/languages/subjects), gutenberg.org
             HTML search fallback when Gutendex is down (its API was
             observed hanging in bursts on 2026-11-16); language for HTML
             candidates is enriched lazily from the book page
  rank     : plain-text format present, language, title/subject overlap with
             the query, slug dedupe against existing data/ dirs; unknown
             metadata (None) defers the check instead of rejecting
  acquire  : reuses ISBNImporter._download_gutenberg_text — one conveyor,
             no second download implementation
  validate : min-chars + printable-ratio gate BEFORE anything touches data/
  write    : tmp dir -> os.replace, so a killed run never leaves a half
             dataset behind (the 0-B frankenstein abort proved the need)
  report   : concise accept/reject table — for data acquisition the report
             IS the measurement (the standing benchmark rule covers
             injection/curriculum changes, not sourcing)

Usage:
  scripts/discover_books.py -q "philosophy" -q "science" --dry-run
  scripts/discover_books.py -q "philosophy science" --max-books 5
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from html import unescape
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from domain.training import ISBNImporter  # noqa: E402

GUTENDEX_SEARCH = "https://gutendex.com/books/?search={query}&page={page}"
PG_SEARCH = "https://www.gutenberg.org/ebooks/search/?query={query}{start}"
PG_BOOK_PAGE = "https://www.gutenberg.org/ebooks/{book_id}"
PG_PAGE_SIZE = 25
USER_AGENT = "SloughGPT/1.0"
HTTP_RETRIES = 3
HTTP_TIMEOUT = 10

_importer: ISBNImporter | None = None


def _http_get_bytes(url: str) -> bytes:
    """Fetch a document with retry/backoff. Tests monkeypatch urlopen here.
    Gutendex intermittently hangs mid-request (observed 2026-11-16), so a
    single timeout must not kill a whole discovery run."""
    last_exc: BaseException | None = None
    for attempt in range(HTTP_RETRIES):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            return urllib.request.urlopen(req, timeout=HTTP_TIMEOUT).read()
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            last_exc = e
            if attempt < HTTP_RETRIES - 1:
                time.sleep(1.5 * (attempt + 1))
    raise last_exc  # type: ignore[misc]


def _http_get_json(url: str) -> dict:
    """Fetch + parse a JSON document. JSON errors propagate (they are not
    transport faults); discover() falls back to the HTML source on any error."""
    return json.loads(_http_get_bytes(url).decode())


def _http_get_text(url: str) -> str:
    """Fetch a document as text (tolerant decode for HTML pages)."""
    return _http_get_bytes(url).decode("utf-8", errors="replace")


def _download_text(gutenberg_id: int) -> str | None:
    """Download a book's plain text via the existing importer conveyor."""
    global _importer
    if _importer is None:
        _importer = ISBNImporter(output_dir=str(_REPO_ROOT / "data"))
    return _importer._download_gutenberg_text(gutenberg_id)


def slugify(title: str) -> str:
    """Dataset dir name for a title: lowercase ascii, non-alnum -> hyphens."""
    ascii_title = unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode()
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_title.lower()).strip("-")
    return slug[:80]


def _discover_gutendex(query: str, max_pages: int) -> list[dict]:
    """Paginated Gutendex search — rich metadata (formats/languages/subjects)."""
    results: list[dict] = []
    url = GUTENDEX_SEARCH.format(query=urllib.parse.quote(query, safe=""), page=1)
    pages = 0
    while url and pages < max_pages:
        data = _http_get_json(url)
        results.extend(data.get("results", []))
        url = data.get("next")
        pages += 1
    return results


def _parse_booklinks(html: str) -> list[dict]:
    """Parse gutenberg.org search-result booklinks into candidate dicts.

    Same keys rank() expects; languages/formats stay None (unknown) until
    enrich_from_book_page() resolves them from the book page.
    """
    books: list[dict] = []
    for block in html.split('<li class="booklink">')[1:]:
        id_m = re.search(r'href="/ebooks/(\d+)"', block)
        title_m = re.search(r'<span class="title">(.*?)</span>', block, re.DOTALL)
        if not id_m or not title_m:
            continue
        title = unescape(re.sub(r"<[^>]+>", "", title_m.group(1))).strip()
        author_m = re.search(r'<span class="subtitle">(.*?)</span>', block, re.DOTALL)
        author = unescape(re.sub(r"<[^>]+>", "", author_m.group(1))).strip() if author_m else ""
        dl_m = re.search(r'<span class="extra">([\d,]+)\s*downloads?', block)
        books.append(
            {
                "id": int(id_m.group(1)),
                "title": title,
                "authors": [{"name": author}] if author else [],
                "subjects": [],
                "languages": None,  # unknown until enriched from the book page
                "formats": None,  # ditto; acquisition enforces plain text anyway
                "download_count": int(dl_m.group(1).replace(",", "")) if dl_m else 0,
            }
        )
    return books


def _discover_gutenberg_html(query: str, max_pages: int) -> list[dict]:
    """Paginated gutenberg.org HTML search — the fallback source.

    Works when Gutendex is down and carries book ids, so acquisition is
    unchanged. Page 1 has no start_index; page N starts at 1 + 25*(N-1).
    """
    results: list[dict] = []
    for page in range(1, max_pages + 1):
        start = "" if page == 1 else f"&start_index={1 + PG_PAGE_SIZE * (page - 1)}"
        url = PG_SEARCH.format(query=urllib.parse.quote(query, safe=""), start=start)
        books = _parse_booklinks(_http_get_text(url))
        results.extend(books)
        if len(books) < PG_PAGE_SIZE:
            break
    return results


def discover(query: str, max_pages: int = 3, source: str = "auto") -> tuple[list[dict], str]:
    """Discover candidates. Returns (candidates, source_name).

    'auto' tries Gutendex first and falls back to gutenberg.org HTML search
    on ANY failure — Gutendex was observed hanging in bursts (2026-11-16),
    and a dead discovery API must not take the whole tool down with it.
    """
    if source in ("auto", "gutendex"):
        try:
            return _discover_gutendex(query, max_pages), "gutendex"
        except Exception as e:
            if source == "gutendex":
                raise
            print(
                f"note: gutendex unavailable ({type(e).__name__}); "
                "falling back to gutenberg.org search",
                file=sys.stderr,
            )
    if source in ("auto", "gutenberg"):
        return _discover_gutenberg_html(query, max_pages), "gutenberg"
    raise ValueError(f"unknown source: {source}")


def enrich_from_book_page(book: dict) -> None:
    """Resolve unknown metadata (languages/plain-text) from the book page.

    Best-effort: on a fetch failure the keys stay None and the deferred
    checks stay deferred — acquisition + validation are the hard gate
    either way. A page that loads but has no plain-text link yields an
    explicit empty formats dict (that absence IS meaningful).
    """
    book_id = int(book["id"])
    if book.get("languages") is not None and book.get("formats") is not None:
        return  # already known (e.g. Gutendex metadata) — no fetch needed
    try:
        page = _http_get_text(PG_BOOK_PAGE.format(book_id=book_id))
    except Exception:
        return
    if book.get("languages") is None:
        lang_m = re.search(r'dcterms:language[^>]*content="([^"]+)"', page)
        if lang_m:
            book["languages"] = [lang_m.group(1)]
    if book.get("formats") is None:
        if re.search(rf'href="/ebooks/{book_id}\.txt', page):
            book["formats"] = {
                "text/plain": (f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt")
            }
        else:
            book["formats"] = {}


def _metadata_reject_reason(book: dict, lang: str) -> str | None:
    """Format/language rejection reason; None when passing OR unknown.

    Unknown (None) metadata defers the check: HTML-source candidates are
    enriched from their book page before acquisition, and if enrichment
    fails the download + validation gate still stands.
    """
    formats = book.get("formats")
    if formats is not None and not _plain_text_url(formats):
        return "no plain-text format"
    if lang:
        languages = book.get("languages")
        if languages is not None and lang not in languages:
            return f"language not {lang}"
    return None


def _plain_text_url(formats: dict) -> str | None:
    for key, value in formats.items():
        if key.startswith("text/plain") and isinstance(value, str):
            return value
    return None


def rank(
    candidates: list[dict],
    *,
    query: str,
    lang: str,
    existing_slugs: set[str],
) -> tuple[list[tuple[int, dict, str]], list[tuple[str, str]]]:
    """Filter + score candidates.

    Returns (ranked, rejected): ranked is (score, book, slug) sorted best
    first; rejected is (title, reason) in discovery order.
    """
    tokens = {t for t in re.findall(r"[a-z0-9]+", query.lower()) if len(t) > 2}
    ranked: list[tuple[int, dict, str]] = []
    rejected: list[tuple[str, str]] = []
    seen_ids: set[int] = set()

    for book in candidates:
        title = str(book.get("title") or "").strip()
        book_id = book.get("id")
        if book_id in seen_ids:
            rejected.append((title, "duplicate id"))
            continue
        seen_ids.add(book_id)
        if not title:
            rejected.append((f"#{book_id}", "no title"))
            continue
        reason = _metadata_reject_reason(book, lang)
        if reason:
            rejected.append((title, reason))
            continue
        slug = slugify(title)
        if not slug:
            rejected.append((title, "empty slug"))
            continue
        if slug in existing_slugs:
            rejected.append((title, f"already in data/{slug}"))
            continue

        haystack = (title + " " + " ".join(book.get("subjects") or [])).lower()
        overlap = sum(1 for t in tokens if t in haystack)
        score = overlap * 1000 + int(book.get("download_count") or 0)
        ranked.append((score, book, slug))

    ranked.sort(key=lambda item: item[0], reverse=True)
    return ranked, rejected


def validate_text(text: str, *, min_chars: int, min_printable: float) -> str | None:
    """Return a rejection reason, or None when the text is acceptable."""
    if len(text) < min_chars:
        return f"too short ({len(text)} < {min_chars} chars)"
    printable = sum(1 for c in text if c.isprintable() or c in "\n\r\t")
    ratio = printable / len(text) if text else 0.0
    if ratio < min_printable:
        return f"printable ratio {ratio:.2f} < {min_printable}"
    return None


def write_dataset(data_dir: Path, slug: str, text: str, meta: dict) -> Path:
    """Atomic dataset write: build in a hidden tmp dir, then os.replace."""
    target = data_dir / slug
    if target.exists():
        raise FileExistsError(f"data/{slug} already exists")
    tmp = data_dir / f".{slug}.tmp"
    try:
        tmp.mkdir(parents=True, exist_ok=True)
        (tmp / f"{slug}.txt").write_text(text, encoding="utf-8")
        (tmp / "metadata.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        os.replace(tmp, target)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return target


def run(args: argparse.Namespace) -> int:
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    existing_slugs = {
        p.name for p in data_dir.iterdir() if p.is_dir() and not p.name.startswith(".")
    }

    query = " ".join(args.query)
    candidates: list[dict] = []
    per_query: dict[str, int] = {}
    source_used = "gutendex"
    source = args.source
    for q in args.query:
        found, used = discover(q, max_pages=args.max_pages, source=source)
        if used == "gutenberg" and source == "auto":
            source = "gutenberg"  # gutendex already failed once — don't re-probe per query
        per_query[q] = len(found)
        source_used = used
        candidates.extend(found)

    ranked, rejected = rank(
        candidates,
        query=query,
        lang=args.lang,
        existing_slugs=existing_slugs,
    )

    accepted: list[tuple[str, int, int, str]] = []  # (slug, id, chars, url)
    for _score, book, slug in ranked:
        if len(accepted) >= args.max_books:
            break
        if source_used == "gutenberg":
            # Lazy: resolve language/plain-text only for pages we may
            # actually acquire — never all candidates (rate-limit safety)
            enrich_from_book_page(book)
        meta_reason = _metadata_reject_reason(book, args.lang)
        if meta_reason:
            rejected.append((str(book.get("title") or f"#{book.get('id')}"), meta_reason))
            continue
        book_id = int(book["id"])
        text = _download_text(book_id)
        if not text:
            rejected.append((str(book.get("title") or f"#{book_id}"), "download failed"))
            continue
        reason = validate_text(text, min_chars=args.min_chars, min_printable=args.min_printable)
        if reason:
            rejected.append((str(book.get("title") or f"#{book_id}"), reason))
            continue
        formats = book.get("formats") or {}
        plain_url = _plain_text_url(formats) or ""
        if not args.dry_run:
            meta = {
                "source": f"gutenberg:{book_id}",
                "source_type": "gutenberg",
                "title": book.get("title"),
                "authors": [a.get("name") for a in book.get("authors") or []],
                "subjects": book.get("subjects") or [],
                "languages": book.get("languages") or [],
                "gutenberg_id": book_id,
                "text_url": plain_url,
                "text_chars": len(text),
                "query": query,
            }
            write_dataset(data_dir, slug, text, meta)
        accepted.append((slug, book_id, len(text), plain_url))

    # ── report (concise; raw detail stays one flag away) ──
    print(
        f"discovered: {len(candidates)} candidates via {source_used} "
        f"({len(args.query)} queries: " + ", ".join(f"{q}={n}" for q, n in per_query.items()) + ")"
    )
    print(f"ranked    : {len(ranked)} eligible, {len(rejected)} rejected")
    if accepted:
        print("acquired  :")
        for slug, book_id, chars, url in accepted:
            print(f"  + {slug:<36} #{book_id:<7} {chars:>8,} chars  {url}")
    else:
        print("acquired  : none")
    if rejected:
        print("rejected  :")
        for title, reason in rejected[: args.max_rejects]:
            print(f"  - {title[:52]:<52} {reason}")
        if len(rejected) > args.max_rejects:
            print(f"  ... {len(rejected) - args.max_rejects} more")
    else:
        print("rejected  : none")
    dest = " (dry-run — nothing written)" if args.dry_run else f" -> {data_dir}"
    total = sum(chars for _, _, chars, _ in accepted)
    print(f"total     : {len(accepted)} dataset(s), {total:,} chars{dest}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Discover and download free book texts into data/ datasets."
    )
    parser.add_argument(
        "-q",
        "--query",
        action="append",
        required=True,
        help="Search query (repeatable), e.g. -q philosophy -q science",
    )
    parser.add_argument("--max-pages", type=int, default=3, help="Search pages per query")
    parser.add_argument(
        "--source",
        choices=("auto", "gutendex", "gutenberg"),
        default="auto",
        help="Discovery source (auto: Gutendex, falling back to gutenberg.org)",
    )
    parser.add_argument("--max-books", type=int, default=10, help="Max books to acquire")
    parser.add_argument("--lang", default="en", help="Required language (empty = any)")
    parser.add_argument("--min-chars", type=int, default=5000, help="Min corpus size")
    parser.add_argument(
        "--min-printable",
        type=float,
        default=0.90,
        help="Min printable-char ratio (rejects binary/garbled downloads)",
    )
    parser.add_argument(
        "--data-dir",
        default=str(_REPO_ROOT / "data"),
        help="Dataset root (default: repo data/)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Discover + rank only, no writes")
    parser.add_argument("--max-rejects", type=int, default=15, help="Rejection lines to print")
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
