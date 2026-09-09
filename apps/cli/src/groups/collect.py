"""
Collect command group — collect data from files, URLs, RSS feeds, and APIs.
"""

from core.framework import click
from core.helpers import ns as _ns
from domains.logging import get_global
log = get_global()


def register(cli):
    """Register collect commands with the CLI group."""

    @cli.group(help="Collect data from files, URLs, RSS feeds, and APIs")
    def collect():
        pass

    @collect.command("file", help="Collect data from a local file")
    @click.argument("path")
    @click.option("--output", "-o", default=None, help="Output JSONL file")
    @click.option("--min-length", default=10, type=int, help="Min record length")
    @click.option("--dedup/--no-dedup", default=True, help="Deduplicate records")
    def collect_file(path, output, min_length, dedup):
        from domains.collections import FileSource, MemoryStore, FileStore, Collector
        from domains.collections import LengthFilter, DedupFilter
        source = FileSource(path)
        store = FileStore(output) if output else MemoryStore()
        filters = []
        if min_length > 0:
            filters.append(LengthFilter(min_length=min_length))
        if dedup:
            filters.append(DedupFilter())
        collector = Collector(source, store, filters=filters)
        count = collector.collect()
        log.success(f"Collected {count} records from {path}")
        if output:
            log.info(f"Output: {output}")

    @collect.command("url", help="Collect data from a URL")
    @click.argument("url")
    @click.option("--output", "-o", default=None, help="Output JSONL file")
    @click.option("--min-length", default=10, type=int, help="Min record length")
    def collect_url(url, output, min_length):
        from domains.collections import UrlSource, MemoryStore, FileStore, Collector
        from domains.collections import LengthFilter
        source = UrlSource(url)
        store = FileStore(output) if output else MemoryStore()
        filters = [LengthFilter(min_length=min_length)] if min_length > 0 else []
        collector = Collector(source, store, filters=filters)
        count = collector.collect()
        log.success(f"Collected {count} records from {url}")
        if output:
            log.info(f"Output: {output}")

    @collect.command("rss", help="Collect data from an RSS/Atom feed")
    @click.argument("url")
    @click.option("--output", "-o", default=None, help="Output JSONL file")
    def collect_rss(url, output):
        from domains.collections import RssSource, MemoryStore, FileStore, Collector
        source = RssSource(url)
        store = FileStore(output) if output else MemoryStore()
        collector = Collector(source, store)
        count = collector.collect()
        log.success(f"Collected {count} records from RSS feed")
        if output:
            log.info(f"Output: {output}")

    @collect.command("merge", help="Merge multiple JSONL files into one")
    @click.argument("inputs", nargs=-1, required=True)
    @click.option("--output", "-o", required=True, help="Output JSONL file")
    def collect_merge(inputs, output):
        import json
        from pathlib import Path
        count = 0
        with open(output, "w") as out_f:
            for input_path in inputs:
                p = Path(input_path)
                if not p.exists():
                    log.warning(f"Skipping {input_path} (not found)")
                    continue
                with open(p) as in_f:
                    for line in in_f:
                        line = line.strip()
                        if line:
                            out_f.write(line + "\n")
                            count += 1
        log.success(f"Merged {len(inputs)} files -> {output} ({count} records)")

    @collect.command("stats", help="Show collection statistics")
    @click.argument("path")
    def collect_stats(path):
        import json
        from pathlib import Path
        p = Path(path)
        if not p.exists():
            log.error(f"File not found: {path}")
            return
        count = 0
        total_bytes = 0
        with open(p) as f:
            for line in f:
                line = line.strip()
                if line:
                    count += 1
                    total_bytes += len(line)
        log.header(f"Collection Stats: {p.name}")
        log.key_value("Records", str(count))
        log.key_value("Total Size", f"{total_bytes:,} bytes")
        log.key_value("Avg Size", f"{total_bytes // max(count, 1):,} bytes")

    return collect
