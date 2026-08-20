from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Protocol, runtime_checkable


@dataclass
class Record:
    content: str
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if "timestamp" not in self.metadata:
            self.metadata["timestamp"] = time.time()


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
                except json.JSONDecodeError:
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
        try:
            req = urllib.request.Request(self.feed_url, headers={"User-Agent": "sloughgpt-culler/1.0"})
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = resp.read().decode("utf-8", errors="replace")
        except (urllib.error.URLError, OSError):
            return

        try:
            import feedparser
        except ImportError:
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
            req = urllib.request.Request(self.url, headers={"User-Agent": "sloughgpt-culler/1.0", **self.headers})
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
        except (urllib.error.URLError, OSError, json.JSONDecodeError):
            return
