# bawl

**Tiny, zero-dependency crawler.** Fetch, parse, crawl, sitemap, store, GUI. All stdlib.

```bash
pip install bawl

bawl https://example.com              # shorthand — JSONL to stdout
bawl page https://x.com -o data       # to file
bawl page https://x.com -f text       # plain text
bawl page https://x.com -f json       # JSON array
bawl crawl https://site.com --depth 2 # recursive (concurrent)
bawl crawl @urls.txt                  # URLs from file
bawl crawl --dedup                    # skip duplicate text content
bawl crawl --include '*.html'         # only crawl matching URLs
bawl crawl --exclude '*print*'        # skip matching URLs
bawl crawl --progress                 # live terminal status
bawl sitemap https://site.com/xml     # from sitemap
bawl gui                              # graphical interface
bawl cat < data.jsonl                 # read + print text
```

## Library

```python
from bawl import (
    fetch, parse, parse_html,           # single page
    save, load, dumps, loads,           # JSONL
    dumps_json_array, save_json_array,  # JSON array
    crawl, crawl_urls,                  # crawling
    parse_sitemap,                      # sitemaps
)

# single page
page = parse("https://example.com")
print(page.title, len(page.text))

# recursive crawl (concurrent, configurable workers)
pages = crawl("https://docs.python.org/3/", depth=2, max_pages=10, workers=8)

# content dedup, include/exclude filters, live progress
pages = crawl("https://site.com", depth=2, dedup=True,
              include=["*.html"], exclude=["*print*"],
              on_page=lambda p: ...)

# fetch URL list concurrently
pages = crawl_urls(["https://a.com", "https://b.com"], workers=5)

# from sitemap
urls = parse_sitemap("https://site.com/sitemap.xml")
for url in urls[:10]:
    save(parse(url), path="crawl.jsonl")

# JSON array output
save_json_array(pages, path="output.json")

# pipe-friendly
save(page)                                  # → stdout JSONL
for p in load("data.jsonl"):
    print(p.text[:200])
```

### Page fields

| Field | Type | Content |
|-------|------|---------|
| `.url` | `str` | Source URL |
| `.title` | `str` | `<title>` text |
| `.text` | `str` | Visible page text (block-separated by newlines) |
| `.links` | `list[dict]` | `{"href": str, "text": str}` |
| `.tables` | `list[dict]` | `{"caption": str, "headers": list, "rows": list[list]}` |
| `.lists` | `list[dict]` | `{"tag": "ul"|"ol", "items": list[str]}` |
| `.code` | `list[dict]` | `{"lang": str, "body": str}` |
| `.meta` | `dict` | Meta name/OG property → content |

## CLI

```
bawl https://example.com              # shorthand JSONL to stdout
bawl page https://x.com -o data       # to file
bawl page https://x.com -f text       # plain text
bawl page https://x.com -f json       # JSON array
bawl crawl https://site.com --depth 2 --workers 10
bawl crawl @urls.txt                  # read URLs from file
bawl sitemap https://site.com/xml
bawl gui                              # tkinter GUI
bawl cat < data.jsonl                 # read + print
bawl completion bash|zsh              # shell completion
bawl --version                        # → bawl 0.3.0

Options: --rate SEC, --timeout SEC, --workers N, --dedup,
         --include PATTERN, --exclude PATTERN, --progress
```

## Why bawl

- **Zero dependencies** — stdlib only (urllib, html.parser, json, xml, tkinter, concurrent)
- **~18KB wheel** — installs in <1 second
- **Concurrent** — thread pool crawl speeds up multi-page fetches
- **97 tests** — CLI, parser edge cases, concurrent crawl, URL normalization, dedup, filters, progress
- **Modular** — import only what you need
- **Composable** — stdin/stdout JSONL/JSON, works with any pipeline
- **GUI included** — `bawl gui` for interactive browsing

## License

MIT
