"""cli — thin dispatch for `bawl` commands. Lazy-imports modules on demand.

Usage:
    bawl https://example.com       # shorthand → page
    bawl page https://x.com -o f   # page -> file
    bawl crawl https://x.com        # recursive
    bawl sitemap https://x.com/xml  # from sitemap
    bawl gui                        # graphical interface
    bawl cat < data.jsonl           # read back
"""

import sys
from typing import Optional


def entry() -> int:
    args = sys.argv[1:]
    if not args or args[0] in ("-h", "--help", "help"):
        _help()
        return 0
    if args[0] in ("--version", "-V"):
        print("bawl 0.3.0")
        return 0

    rate = 0.5
    timeout = 15
    cmd_args = []
    i = 0
    while i < len(args):
        if args[i] == "--rate" and i + 1 < len(args):
            rate = float(args[i + 1])
            i += 2
        elif args[i] == "--timeout" and i + 1 < len(args):
            timeout = int(args[i + 1])
            i += 2
        else:
            cmd_args.append(args[i])
            i += 1

    cmd = cmd_args[0] if cmd_args else ""

    if cmd in ("completion",):
        shell = cmd_args[1] if len(cmd_args) > 1 else ""
        return _completion(shell)

    # shorthand: bawl https://...  →  page
    if cmd.startswith("http://") or cmd.startswith("https://"):
        return _page(cmd, rate=rate, timeout=timeout)

    if cmd == "page" and len(cmd_args) >= 2:
        fmt = "jsonl"
        out = "-"
        url = cmd_args[1]
        rest = cmd_args[2:]
        for j, a in enumerate(rest):
            if a in ("-o", "--output") and j + 1 < len(rest):
                out = rest[j + 1]
            if a in ("-f", "--format") and j + 1 < len(rest):
                fmt = rest[j + 1]
        return _page(url, fmt=fmt, out=out, rate=rate, timeout=timeout)

    if cmd == "crawl" and len(cmd_args) >= 2:
        depth = 1
        mp = 50
        out = "-"
        workers = 5
        fmt = "jsonl"
        dedup = False
        progress = False
        include = []
        exclude = []
        url = cmd_args[1]
        rest = cmd_args[2:]
        for j, a in enumerate(rest):
            if a == "--depth" and j + 1 < len(rest):
                depth = int(rest[j + 1])
            if a in ("--max", "--max-pages") and j + 1 < len(rest):
                mp = int(rest[j + 1])
            if a in ("-o", "--output") and j + 1 < len(rest):
                out = rest[j + 1]
            if a == "--workers" and j + 1 < len(rest):
                workers = int(rest[j + 1])
            if a in ("-f", "--format") and j + 1 < len(rest):
                fmt = rest[j + 1]
            if a == "--dedup":
                dedup = True
            if a == "--progress":
                progress = True
            if a == "--include" and j + 1 < len(rest):
                include.append(rest[j + 1])
            if a == "--exclude" and j + 1 < len(rest):
                exclude.append(rest[j + 1])
        return _crawl(url, depth=depth, max_pages=mp, out=out,
                      rate=rate, timeout=timeout, workers=workers, fmt=fmt,
                      dedup=dedup, include=include, exclude=exclude,
                      progress=progress)

    if cmd == "gui":
        return _gui()

    if cmd == "sitemap" and len(cmd_args) >= 2:
        mp = 500
        out = "-"
        url = cmd_args[1]
        rest = cmd_args[2:]
        for j, a in enumerate(rest):
            if a in ("--max", "--max-pages") and j + 1 < len(rest):
                mp = int(rest[j + 1])
            if a in ("-o", "--output") and j + 1 < len(rest):
                out = rest[j + 1]
        return _sitemap(url, max_pages=mp, out=out, rate=rate, timeout=timeout)

    if cmd == "cat":
        return _cat()

    print(f"bawl: unknown command '{cmd}' — try 'bawl help'", file=sys.stderr)
    return 1


def _page(url: str, fmt: str = "jsonl", out: str = "-",
          rate: float = 0.5, timeout: int = 15) -> int:
    from .parse import parse
    from .store import save, save_json_array

    page = parse(url, timeout=timeout, rate=rate)
    if page is None:
        print(f"bawl: failed {url}", file=sys.stderr)
        return 1
    if fmt == "text":
        import json
        data = json.dumps({"text": page.text, "url": page.url, "title": page.title},
                          ensure_ascii=False) + "\n"
        _write(data, out)
    elif fmt == "json":
        save_json_array([page], path=out)
    else:
        save(page, path=out)
    return 0


def _crawl(url_or_file: str, depth: int = 1, max_pages: int = 50,
           out: str = "-", rate: float = 0.5, timeout: int = 15,
           workers: int = 5, fmt: str = "jsonl", dedup: bool = False,
           include: Optional[list[str]] = None,
           exclude: Optional[list[str]] = None,
           progress: bool = False) -> int:
    from .crawl import crawl, crawl_urls, ProgressTracker
    from .store import save, save_json_array

    tracker = ProgressTracker(total_depth=depth) if progress else None

    def _on_page(p) -> None:
        if tracker:
            tracker.inc()
            sys.stderr.write("\r" + tracker.status())
            sys.stderr.flush()

    # @file syntax: read URLs from file
    if url_or_file.startswith("@"):
        path = url_or_file[1:]
        try:
            with open(path) as f:
                urls = [line.strip() for line in f if line.strip() and not line.startswith("#")]
        except OSError as e:
            print(f"bawl: can't read {path}: {e}", file=sys.stderr)
            return 1
        pages = crawl_urls(urls, rate=rate, timeout=timeout, workers=workers, dedup=dedup,
                          include=include, exclude=exclude,
                          on_page=_on_page if progress else None)
    else:
        pages = crawl(url_or_file, depth=depth, max_pages=max_pages,
                      rate=rate, timeout=timeout, same_domain=True, workers=workers,
                      dedup=dedup, include=include, exclude=exclude,
                      on_page=_on_page if progress else None)

    if fmt == "json":
        save_json_array(pages, path=out)
    else:
        for page in pages:
            save(page, path=out)
    if progress:
        sys.stderr.write("\r" + tracker.status() + "\n")
        sys.stderr.flush()
    if out != "-":
        label = url_or_file.replace("://", "/").replace("/", "_")[:40]
        print(f"bawl: {len(pages)} pages → {out}", file=sys.stderr)
    return 0


def _sitemap(url: str, max_pages: int = 500, out: str = "-",
             rate: float = 0.5, timeout: int = 15) -> int:
    from .sitemap import parse as parse_sitemap
    from .store import save

    urls = parse_sitemap(url, timeout=timeout)
    print(f"bawl: sitemap has {len(urls)} URLs", file=sys.stderr)
    count = 0
    for u in urls[:max_pages]:
        from .parse import parse as _parse
        page = _parse(u, timeout=timeout, rate=rate)
        if page:
            save(page, path=out)
            count += 1
    if out != "-":
        print(f"bawl: crawled {count}/{min(len(urls), max_pages)} pages → {out}", file=sys.stderr)
    return 0


def _completion(shell: str) -> int:
    if shell == "bash":
        print("""_bawl() {
    local cur=${COMP_WORDS[COMP_CWORD]}
    local prev=${COMP_WORDS[COMP_CWORD-1]}
    local cmds="page crawl sitemap gui cat help completion --version --rate --timeout"
    case $prev in
        page|crawl|sitemap) COMPREPLY=($(compgen -A file -- "$cur")) ;;
        *) COMPREPLY=($(compgen -W "$cmds" -- "$cur")) ;;
    esac
}
complete -F _bawl bawl""")
    elif shell == "zsh":
        print("""#compdef bawl
_bawl() {
    local -a cmds
    cmds=('page:fetch single page' 'crawl:recursive crawl' 'sitemap:parse sitemap'
          'gui:graphical interface' 'cat:read back jsonl' 'help:show help'
          'completion:generate shell completion')
    _describe 'command' cmds
}
compdef _bawl bawl""")
    else:
        print("Usage: bawl completion bash|zsh", file=sys.stderr)
        return 1
    return 0


def _gui() -> int:
    import tkinter
    from .gui import App
    app = App()
    return app.run()


def _cat() -> int:
    from .store import load
    for page in load("-"):
        print(page.text[:2000])
        print("---")
    return 0


def _write(data: str, path: str) -> None:
    if path == "-":
        sys.stdout.write(data)
    else:
        with open(path, "w") as f:
            f.write(data)


def _help() -> None:
    print("""\
bawl — tiny crawler, works with your infra.

  bawl https://example.com          # shorthand JSONL to stdout
  bawl page https://x.com -o data   # to file
  bawl page https://x.com -f text   # plain text
  bawl page https://x.com -f json   # JSON array
  bawl crawl https://site.com --depth 2 --workers 10
  bawl crawl @urls.txt              # read URLs from file
  bawl sitemap https://site.com/sitemap.xml
  bawl gui                         # graphical interface
  bawl cat < data.jsonl            # read + print text
  bawl completion bash             # shell completion (bash/zsh)

  # pipe into any app:
  bawl https://x.com | your-app

Options: --rate SEC, --timeout SEC, --workers N, --version, -h
""")
