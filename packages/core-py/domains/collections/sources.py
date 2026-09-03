from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@dataclass
class Record:
    content: str
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if "timestamp" not in self.metadata:
            self.metadata["timestamp"] = time.time()

    def to_dict(self) -> dict:
        return {"content": self.content, "metadata": dict(self.metadata)}


@runtime_checkable
class Source(Protocol):
    name: str

    def read(self) -> Iterator[Record]: ...


class FileSource:
    def __init__(self, path: str, name: str = ""):
        self.path = Path(path)
        self.name = name or f"file:{self.path.name}"

    def read(self) -> Iterator[Record]:
        if not self.path.exists():
            return
        suffix = self.path.suffix.lower()
        if suffix == ".jsonl":
            yield from self._read_jsonl()
        elif suffix == ".json":
            yield from self._read_json()
        elif suffix == ".csv":
            yield from self._read_csv()
        else:
            yield from self._read_text()

    def _read_jsonl(self) -> Iterator[Record]:
        with open(self.path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    content = data.pop("content", "") if isinstance(data, dict) else str(data)
                    yield Record(content=content, metadata={"source": self.name, "line": i, **(data if isinstance(data, dict) else {})})
                except json.JSONDecodeError as e:
                    logger.debug("Skipping malformed JSONL line %d: %s", i, e)
                    yield Record(content=line, metadata={"source": self.name, "line": i})

    def _read_json(self) -> Iterator[Record]:
        with open(self.path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            for i, item in enumerate(data):
                if isinstance(item, str):
                    yield Record(content=item, metadata={"source": self.name, "index": i})
                elif isinstance(item, dict):
                    content = item.pop("content", "") if isinstance(item, dict) else str(item)
                    yield Record(content=content, metadata={"source": self.name, "index": i, **item})
        elif isinstance(data, dict):
            content = data.pop("content", "") if isinstance(data, dict) else str(data)
            yield Record(content=content, metadata={"source": self.name, **data})

    def _read_csv(self) -> Iterator[Record]:
        import csv
        with open(self.path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                content = row.pop("content", "") if "content" in row else json.dumps(row)
                yield Record(content=content, metadata={"source": self.name, "row": i, **row})

    def _read_text(self) -> Iterator[Record]:
        with open(self.path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.rstrip("\n")
                if line:
                    yield Record(content=line, metadata={"source": self.name, "line": i})


class UrlSource:
    def __init__(self, url: str, name: str = "", timeout: int = 30):
        self.url = url
        self.name = name or f"url:{url[:60]}"
        self.timeout = timeout

    def read(self) -> Iterator[Record]:
        import urllib.request
        import urllib.error
        try:
            req = urllib.request.Request(self.url, headers={"User-Agent": "sloughgpt-culler/1.0"})
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = resp.read().decode("utf-8", errors="replace")
            content_type = resp.headers.get("Content-Type", "")
            if "json" in content_type:
                yield from self._parse_json(body)
            elif "xml" in content_type or "rss" in content_type:
                yield from self._parse_rss(body)
            else:
                for i, line in enumerate(body.splitlines()):
                    if line.strip():
                        yield Record(content=line.strip(), metadata={"source": self.name, "line": i, "url": self.url})
        except (urllib.error.URLError, OSError) as e:
            yield Record(content="", metadata={"source": self.name, "error": str(e), "url": self.url})

    def _parse_json(self, body: str) -> Iterator[Record]:
        data = json.loads(body)
        if isinstance(data, list):
            for i, item in enumerate(data):
                content = item.pop("content", "") if isinstance(item, dict) else str(item)
                yield Record(content=content, metadata={"source": self.name, "index": i, "url": self.url})
        elif isinstance(data, dict):
            content = data.pop("content", "") if isinstance(data, dict) else str(data)
            yield Record(content=content, metadata={"source": self.name, "url": self.url, **data})

    def _parse_rss(self, body: str) -> Iterator[Record]:
        try:
            import feedparser
            feed = feedparser.parse(body)
            for i, entry in enumerate(feed.entries):
                title = getattr(entry, "title", "")
                summary = getattr(entry, "summary", "")
                content = f"{title}\n{summary}" if title and summary else (title or summary or "")
                link = getattr(entry, "link", "")
                yield Record(content=content.strip(), metadata={"source": self.name, "index": i, "url": link or self.url, "title": title})
        except ImportError:
            logger.debug("feedparser not installed, falling back to line-based RSS parsing")
            for i, line in enumerate(body.splitlines()):
                if line.strip():
                    yield Record(content=line.strip(), metadata={"source": self.name, "line": i, "url": self.url})


class RssSource:
    def __init__(self, feed_url: str, name: str = "", timeout: int = 30):
        self.feed_url = feed_url
        self.name = name or f"rss:{feed_url[:60]}"
        self.timeout = timeout
        self._seen: set[str] = set()

    def read(self) -> Iterator[Record]:
        import urllib.request
        import feedparser
        try:
            req = urllib.request.Request(self.feed_url, headers={"User-Agent": "sloughgpt-culler/1.0"})
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = resp.read().decode("utf-8", errors="replace")
        except (urllib.error.URLError, OSError) as e:
            logger.warning("Failed to fetch RSS feed %s: %s", self.feed_url, e)
            return
        except ImportError:
            logger.debug("feedparser not installed, skipping RSS source %s", self.name)
            return

        feed = feedparser.parse(body)
        for entry in feed.entries:
            link = getattr(entry, "link", "")
            if link and link in self._seen:
                continue
            if link:
                self._seen.add(link)

            title = getattr(entry, "title", "")
            summary = getattr(entry, "summary", "")
            content = f"{title}\n{summary}" if title and summary else (title or summary or "")
            published = getattr(entry, "published", "")
            yield Record(
                content=content.strip(),
                metadata={"source": self.name, "url": link, "title": title, "published": published},
            )

    def reset(self):
        self._seen.clear()


class ApiSource:
    def __init__(self, url: str, name: str = "", headers: dict | None = None, poll_interval: float = 60.0, timeout: int = 30):
        self.url = url
        self.name = name or f"api:{url[:60]}"
        self.headers = headers or {}
        self.poll_interval = poll_interval
        self.timeout = timeout
        self._last_id: str | None = None

    def read(self) -> Iterator[Record]:
        import urllib.request
        import urllib.error
        try:
            req = urllib.request.Request(self.url, headers={"User-Agent": "sloughgpt-collections/1.0", **self.headers})
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = resp.read().decode("utf-8", errors="replace")
            data = json.loads(body)
            items = data if isinstance(data, list) else [data]
            for i, item in enumerate(items):
                item_id = str(item.get("id", i))
                if item_id == self._last_id:
                    break
                content = item.pop("content", "") if isinstance(item, dict) else str(item)
                yield Record(content=content, metadata={"source": self.name, "index": i, "url": self.url, **(item if isinstance(item, dict) else {})})
            if items:
                self._last_id = str(items[0].get("id", 0))
        except (urllib.error.URLError, OSError, json.JSONDecodeError) as e:
            logger.warning("Failed to fetch API source %s: %s", self.url, e)
            return


class SseSource:
    def __init__(self, url: str, name: str = "", timeout: int = 30):
        self.url = url
        self.name = name or f"sse:{url[:60]}"
        self.timeout = timeout

    def read(self) -> Iterator[Record]:
        import urllib.request
        import urllib.error
        try:
            req = urllib.request.Request(self.url, headers={
                "User-Agent": "sloughgpt-collections/1.0",
                "Accept": "text/event-stream",
            })
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                event_type = ""
                event_data = []
                for raw_line in resp:
                    line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
                    if line.startswith("event:"):
                        event_type = line[6:].strip()
                    elif line.startswith("data:"):
                        event_data.append(line[5:].strip())
                    elif line == "":
                        if event_data:
                            content = "\n".join(event_data)
                            yield Record(content=content, metadata={"source": self.name, "event": event_type, "url": self.url})
                            event_type = ""
                            event_data = []
        except (urllib.error.URLError, OSError) as e:
            logger.warning("Failed to fetch SSE stream %s: %s", self.url, e)
            return


class WatchSource:
    def __init__(self, path: str, name: str = "", poll_interval: float = 1.0, patterns: list[str] | None = None):
        self.path = Path(path)
        self.name = name or f"watch:{self.path.name}"
        self.poll_interval = poll_interval
        self.patterns = patterns or ["*"]
        self._seen_mtimes: dict[str, float] = {}

    def read(self) -> Iterator[Record]:
        import fnmatch
        for pattern in self.patterns:
            for file_path in self.path.glob(pattern):
                if not file_path.is_file():
                    continue
                mtime = file_path.stat().st_mtime
                key = str(file_path)
                last_mtime = self._seen_mtimes.get(key, 0.0)
                if mtime > last_mtime:
                    self._seen_mtimes[key] = mtime
                    try:
                        content = file_path.read_text(encoding="utf-8", errors="replace")
                        if content.strip():
                            yield Record(
                                content=content.strip(),
                                metadata={"source": self.name, "path": str(file_path), "mtime": mtime},
                            )
                    except OSError as e:
                        logger.debug("Failed to read file %s: %s", file_path, e)
                        continue

    def reset(self):
        self._seen_mtimes.clear()


class GeneratorSource:
    def __init__(self, generator_fn, name: str = "generator"):
        self._generator_fn = generator_fn
        self.name = name

    def read(self) -> Iterator[Record]:
        for item in self._generator_fn():
            if isinstance(item, Record):
                yield item
            elif isinstance(item, str):
                yield Record(content=item, metadata={"source": self.name})
            elif isinstance(item, dict):
                content = item.pop("content", "") if isinstance(item, dict) else str(item)
                yield Record(content=content, metadata={"source": self.name, **(item if isinstance(item, dict) else {})})
