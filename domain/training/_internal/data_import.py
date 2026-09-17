"""
Data Import Tools
Import datasets from various sources: GitHub, HuggingFace, URLs, local files.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import urllib.request
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("slo.data_import")


def _retry(
    fn, retries=2, delay=1.0, exceptions=(urllib.error.URLError, ConnectionError, TimeoutError)
):
    """Retry a callable on transient network errors."""
    import time

    last_exc = None
    for attempt in range(retries + 1):
        try:
            return fn()
        except exceptions as e:
            last_exc = e
            if attempt < retries:
                logger.warning(
                    "Retry %s/%s after %s: %s",
                    attempt + 1,
                    retries,
                    type(e).__name__,
                    e,
                    extra={"tag": "TRAIN"},
                )
                time.sleep(delay * (attempt + 1))
    raise last_exc


DEFAULT_IGNORES: set[str] = {
    ".git",
    ".svn",
    ".hg",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
    "dist",
    "build",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "*.egg-info",
    ".tox",
}


@dataclass
class ImportResult:
    """Result of a data import operation."""

    success: bool
    name: str
    source: str
    files_imported: int
    total_chars: int
    output_path: str
    error: str | None = None


class RepoImporter:
    """Import code from Git repositories."""

    def __init__(self, cache_dir: str = "runs/repos"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def clone_repo(
        self,
        url: str,
        branch: str | None = None,
        depth: int | None = 1,
    ) -> Path:
        """Clone a git repository."""
        # Validate URL protocol — reject file:// and other local protocols
        _allowed_protocols = ("https://", "git://", "git@", "ssh://", "http://")
        if not any(url.startswith(p) for p in _allowed_protocols):
            raise ValueError(f"URL must use one of {_allowed_protocols}, got: {url[:60]}")

        repo_name = url.split("/")[-1].replace(".git", "")
        # Sanitize repo_name — reject path traversal
        repo_name = re.sub(r"[^\w\-]", "_", repo_name).strip("_") or "repo"
        target = self.cache_dir / repo_name

        if target.exists():
            logger.info(
                "Repo already exists: %s",
                target,
                extra={"tag": "TRAIN"},
            )
            return target

        if branch and not re.match(r"^[a-zA-Z0-9_\-/.]+$", branch):
            raise ValueError(f"Invalid branch name: {branch}")

        cmd = ["git", "clone", url, str(target)]
        if branch:
            cmd.extend(["-b", branch])
        if depth:
            cmd.extend(["--depth", str(depth)])

        logger.info(
            "Cloning %s...",
            url,
            extra={"tag": "TRAIN"},
        )
        subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        return target

    def export_to_corpus(
        self,
        repo_path: Path,
        output_path: str,
        ignores: set[str] | None = None,
        max_files: int | None = None,
        max_bytes: int = 200_000,
        extensions: list[str] | None = None,
    ) -> int:
        """Export repository files to JSONL corpus."""
        ignores = ignores or DEFAULT_IGNORES
        extensions = extensions or [".py", ".js", ".ts", ".md", ".txt", ".json", ".yaml", ".yml"]

        count = 0
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        with output.open("w", encoding="utf-8") as f:
            for file_path in self._iter_files(repo_path, ignores, extensions):
                if max_files and count >= max_files:
                    break

                try:
                    content = file_path.read_text(encoding="utf-8")
                except (UnicodeDecodeError, OSError):
                    continue

                if len(content) > max_bytes:
                    content = content[:max_bytes]

                record = {
                    "path": str(file_path.relative_to(repo_path)),
                    "content": content,
                    "size": len(content),
                    "language": self._detect_language(file_path, content),
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1

        return count

    def _iter_files(self, root: Path, ignores: set[str], extensions: list[str]):
        """Iterate over files in repository."""
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in ignores and not d.startswith(".")]

            for name in filenames:
                if name in ignores or name.startswith("."):
                    continue

                file_path = Path(dirpath) / name

                if extensions:
                    if file_path.suffix.lower() not in extensions:
                        continue

                yield file_path

    def _detect_language(self, path: Path, content: str = "") -> str:
        """Detect programming language using MIME type, extension, and content analysis."""
        import mimetypes

        # Initialize mimetypes
        mimetypes.init()

        # Step 1: MIME type detection
        mime_type, _ = mimetypes.guess_type(str(path))
        if mime_type:
            mime_to_lang = {
                "text/x-python": "python",
                "application/x-python-code": "python",
                "text/javascript": "javascript",
                "application/javascript": "javascript",
                "text/typescript": "typescript",
                "application/typescript": "typescript",
                "text/html": "html",
                "text/css": "css",
                "text/markdown": "markdown",
                "application/json": "json",
                "application/x-yaml": "yaml",
                "text/x-yaml": "yaml",
                "text/x-sh": "shell",
                "application/x-sh": "shell",
                "text/x-rust": "rust",
                "text/x-go": "go",
                "text/x-java-source": "java",
                "text/x-c": "c",
                "text/x-c++": "cpp",
                "text/x-csharp": "csharp",
                "text/x-ruby": "ruby",
                "text/x-php": "php",
                "application/x-httpd-php": "php",
                "text/x-sql": "sql",
                "application/sql": "sql",
                "text/x-perl": "perl",
                "text/x-lua": "lua",
                "text/x-r": "r",
                "text/x-swift": "swift",
                "text/x-kotlin": "kotlin",
                "text/x-scala": "scala",
                "text/xml": "xml",
                "application/xml": "xml",
            }
            if mime_type in mime_to_lang:
                return mime_to_lang[mime_type]

        # Step 2: Extension-based detection (expanded)
        ext_map = {
            ".py": "python",
            ".pyw": "python",
            ".pyi": "python",
            ".js": "javascript",
            ".jsx": "javascript",
            ".mjs": "javascript",
            ".cjs": "javascript",
            ".ts": "typescript",
            ".tsx": "typescript",
            ".mts": "typescript",
            ".md": "markdown",
            ".markdown": "markdown",
            ".json": "json",
            ".jsonl": "json",
            ".yaml": "yaml",
            ".yml": "yaml",
            ".sh": "shell",
            ".bash": "shell",
            ".zsh": "shell",
            ".rs": "rust",
            ".go": "go",
            ".java": "java",
            ".cpp": "cpp",
            ".cc": "cpp",
            ".cxx": "cpp",
            ".c": "c",
            ".h": "c",
            ".hpp": "cpp",
            ".hxx": "cpp",
            ".cs": "csharp",
            ".rb": "ruby",
            ".php": "php",
            ".sql": "sql",
            ".pl": "perl",
            ".pm": "perl",
            ".lua": "lua",
            ".r": "r",
            ".R": "r",
            ".swift": "swift",
            ".kt": "kotlin",
            ".kts": "kotlin",
            ".scala": "scala",
            ".sc": "scala",
            ".xml": "xml",
            ".html": "html",
            ".htm": "html",
            ".xhtml": "html",
            ".css": "css",
            ".scss": "scss",
            ".sass": "sass",
            ".less": "less",
            ".vue": "vue",
            ".svelte": "svelte",
            ".ex": "elixir",
            ".exs": "elixir",
            ".erl": "erlang",
            ".hs": "haskell",
            ".lhs": "haskell",
            ".clj": "clojure",
            ".cljs": "clojure",
            ".ml": "ocaml",
            ".mli": "ocaml",
            ".fs": "fsharp",
            ".fsi": "fsharp",
            ".dart": "dart",
            ".jl": "julia",
            ".nim": "nim",
            ".cr": "crystal",
            ".d": "d",
            ".asm": "assembly",
            ".s": "assembly",
            ".toml": "toml",
            ".ini": "ini",
            ".cfg": "ini",
            ".conf": "config",
            ".env": "env",
            ".txt": "text",
            ".tex": "latex",
            ".rst": "restructuredtext",
            ".org": "org",
            ".csv": "csv",
            ".tsv": "tsv",
            ".dockerfile": "dockerfile",
            "Dockerfile": "dockerfile",
            "Makefile": "makefile",
        }

        ext_result = ext_map.get(path.suffix.lower(), "")

        # Check for special files without extensions
        if not ext_result:
            name_lower = path.name.lower()
            if name_lower in ("dockerfile", "makefile", "rakefile", "gemfile", "pipfile"):
                ext_result = name_lower
            elif name_lower.startswith("readme"):
                ext_result = "markdown"
            elif name_lower.startswith("license"):
                ext_result = "text"
            elif name_lower.startswith(".env"):
                ext_result = "env"

        if ext_result:
            return ext_result

        # Step 3: Content-based detection
        if content:
            return self._detect_from_content(content)

        return "text"

    def _detect_from_content(self, content: str) -> str:
        """Detect language from file content using pattern matching."""
        if not content:
            return "text"

        # Get first 2000 chars for analysis
        sample = content[:2000].strip()
        first_line = sample.split("\n")[0] if sample else ""

        # Python patterns
        python_patterns = [
            "def ",
            "class ",
            "import ",
            "from ",
            "if __name__",
            "print(",
            "self.",
            "__init__",
            "lambda ",
            "async def",
            "await ",
            "except ",
            "raise ",
            "with open(",
            "import os",
        ]
        if any(p in sample for p in python_patterns[:6]):
            python_score = sum(1 for p in python_patterns if p in sample)
            if python_score >= 3:
                return "python"

        # JavaScript/TypeScript patterns
        js_patterns = [
            "function ",
            "const ",
            "let ",
            "var ",
            "=> {",
            "export ",
            "import {",
            "require(",
            "console.log",
            "async function",
            "await ",
            "Promise.",
            ".then(",
        ]
        ts_patterns = [
            "interface ",
            "type ",
            ": string",
            ": number",
            ": boolean",
            "<T>",
            "extends ",
            "implements ",
            "as ",
            ": void",
            ": string[]",
            ": number[]",
            "readonly ",
            "Record<",
        ]

        js_score = sum(1 for p in js_patterns if p in sample)
        ts_score = sum(1 for p in ts_patterns if p in sample)

        # TypeScript is a superset of JS, so if TS patterns are present, it's TS
        if ts_score >= 2:
            return "typescript"
        if js_score >= 3:
            return "javascript"

        # HTML patterns
        if sample.startswith("<!DOCTYPE") or sample.startswith("<html"):
            return "html"
        html_tags = ["<div", "<span", "<p>", "<body", "<head", "<script", "<style"]
        if sum(1 for t in html_tags if t in sample.lower()) >= 2:
            return "html"

        # CSS patterns
        css_selectors = [
            ":hover",
            ":focus",
            ":active",
            "::before",
            "::after",
            "body {",
            "html {",
            ".class",
            "#id",
            "div {",
            "span {",
        ]
        css_properties = [
            "margin:",
            "padding:",
            "color:",
            "background:",
            "font-",
            "display:",
            "position:",
            "width:",
            "height:",
            "flex",
            "border:",
            "text-",
            "overflow:",
            "z-index:",
            "opacity:",
        ]

        has_selector = any(s in sample for s in css_selectors)
        has_props = sum(1 for p in css_properties if p in sample)

        if has_selector or has_props >= 2:
            return "css"
        if sample.startswith("@") and any(
            p in sample for p in ["@media", "@keyframes", "@font-face"]
        ):
            return "css"

        # JSON patterns
        if sample.startswith("{") or sample.startswith("["):
            try:
                json.loads(sample)
                return "json"
            except Exception:
                logger.warning(
                    "Failed to parse JSON for sample",
                    exc_info=True,
                    extra={"tag": "TRAIN"},
                )

        # YAML patterns
        yaml_indicators = ["---\n", ": ", "\n  ", "\n    "]
        if sample.startswith("---") or first_line.endswith(":"):
            if "---" in sample or sum(1 for i in yaml_indicators if i in sample) >= 2:
                return "yaml"

        # Shell/Bash patterns
        shell_patterns = [
            "#!/bin/bash",
            "#!/bin/sh",
            "#!/usr/bin/env bash",
            "echo ",
            "export ",
            "source ",
            "cd ",
            "mkdir ",
            "chmod ",
            "sudo ",
            "apt ",
            "yum ",
            "brew ",
        ]
        if any(sample.startswith(p) for p in ["#!/bin", "#!/usr/bin/env"]):
            return "shell"
        if sum(1 for p in shell_patterns if p in sample) >= 3:
            return "shell"

        # SQL patterns
        sql_keywords = [
            "SELECT ",
            "FROM ",
            "WHERE ",
            "INSERT ",
            "UPDATE ",
            "DELETE ",
            "CREATE TABLE",
            "ALTER TABLE",
        ]
        if any(k in sample.upper() for k in sql_keywords[:3]):
            return "sql"

        # Go patterns
        go_patterns = ["package main", "func main()", "import (", "fmt.", "go func", "goroutine"]
        if any(p in sample for p in go_patterns[:3]):
            return "go"

        # Rust patterns
        rust_patterns = [
            "fn ",
            "let mut",
            "impl ",
            "pub fn",
            "use std::",
            "struct ",
            "enum ",
            "mod ",
            "crate ",
            "::<",
            "println!(",
            "vec![",
            "Some(",
            "None",
            "Ok(",
            "Err(",
        ]
        if sum(1 for p in rust_patterns if p in sample) >= 2:
            return "rust"

        # Java patterns
        java_patterns = [
            "public class",
            "private ",
            "public static void main",
            "System.out.println",
            "import java.",
        ]
        if sum(1 for p in java_patterns if p in sample) >= 2:
            return "java"

        # C/C++ patterns
        c_patterns = ["#include <", "int main(", "printf(", "scanf(", "void *", "malloc(", "free("]
        cpp_patterns = ["#include <", "std::", "cout <<", "cin >>", "class ", "namespace "]

        if any(p in sample for p in cpp_patterns[1:]):
            return "cpp"
        if any(p in sample for p in c_patterns):
            return "c"

        # Markdown patterns
        md_patterns = ["# ", "## ", "### ", "- ", "* ", "```", "[", "](", "**", "__"]
        if sum(1 for p in md_patterns if p in sample) >= 3:
            return "markdown"

        # XML patterns
        if sample.startswith("<?xml") or sample.startswith("<"):
            if "</" in sample and ">" in sample:
                return "xml"

        return "text"

    def import_from_github(
        self,
        url: str,
        dataset_name: str,
        output_dir: str = "datasets",
        extensions: list[str] | None = None,
        max_files: int | None = None,
    ) -> ImportResult:
        """Import dataset from GitHub repository."""
        try:
            repo_path = self.clone_repo(url)

            output_path = f"{output_dir}/{dataset_name}/corpus.jsonl"
            count = self.export_to_corpus(
                repo_path,
                output_path,
                extensions=extensions,
                max_files=max_files,
            )

            total_chars = 0
            if Path(output_path).exists():
                with open(output_path) as f:
                    for line_num, line in enumerate(f, 1):
                        try:
                            data = json.loads(line)
                        except (json.JSONDecodeError, ValueError):
                            logger.warning("Skipping malformed JSONL line %d", line_num)
                            continue
                        total_chars += data.get("size", 0)

            return ImportResult(
                success=True,
                name=dataset_name,
                source=url,
                files_imported=count,
                total_chars=total_chars,
                output_path=output_path,
            )
        except Exception as e:
            logger.error(
                "Failed to import from GitHub: %s",
                e,
                extra={"tag": "TRAIN"},
            )
            return ImportResult(
                success=False,
                name=dataset_name,
                source=url,
                files_imported=0,
                total_chars=0,
                output_path="",
                error=str(e),
            )


class BooksSearch:
    """Search books by title or ISBN via Open Library API."""

    def _sanitize_query(self, query: str) -> str:
        import urllib.parse

        return urllib.parse.quote(query.strip(), safe="")

    def _is_isbn(self, query: str) -> str | None:
        digits = re.sub(r"[-_\s]", "", query)
        if len(digits) == 10 and self._valid_isbn10(digits):
            return digits
        if len(digits) == 13 and self._valid_isbn13(digits):
            return digits
        return None

    def _valid_isbn10(self, isbn: str) -> bool:
        if not isbn[:9].isdigit():
            return False
        check = isbn[9].upper()
        if not (check.isdigit() or check == "X"):
            return False
        total = sum(int(isbn[i]) * (10 - i) for i in range(9))
        total += 11 if check == "X" else int(check)
        return total % 11 == 0

    def _valid_isbn13(self, isbn: str) -> bool:
        if not isbn.isdigit():
            return False
        total = sum(int(isbn[i]) * (1 if i % 2 == 0 else 3) for i in range(12))
        check = (10 - (total % 10)) % 10
        return check == int(isbn[12])

    def get_by_isbn(self, isbn: str) -> dict | None:
        """Fetch a book directly by ISBN from Open Library."""
        clean = re.sub(r"[-_\s]", "", isbn)
        url = f"https://openlibrary.org/isbn/{clean}.json"
        try:
            req = urllib.request.Request(url)
            req.add_header("User-Agent", "SloughGPT/1.0")
            data = _retry(
                lambda: json.loads(urllib.request.urlopen(req, timeout=30).read().decode())
            )
            return self._normalize_ol_book(data, clean)
        except Exception:
            return None

    def _normalize_ol_book(self, data: dict, isbn: str) -> dict:
        authors = []
        for a in data.get("authors", []):
            if isinstance(a, dict) and "key" in a:
                authors.append(a["key"].split("/")[-1])
            elif isinstance(a, str):
                authors.append(a)

        return {
            "key": data.get("key", ""),
            "title": data.get("title", ""),
            "subtitle": data.get("subtitle", ""),
            "author": ", ".join(authors) if authors else "Unknown",
            "isbn": isbn,
            "year": data.get("first_publish_year"),
            "cover": None,
            "subjects": data.get("subjects", [])[:20],
            "subject_places": [s for s in data.get("subject_places", [])[:10]],
            "subject_times": [s for s in data.get("subject_times", [])[:10]],
            "subject_people": [s for s in data.get("subject_people", [])[:10]],
            "publishers": data.get("publishers", []),
            "publish_date": data.get("publish_date", ""),
            "number_of_pages": data.get("number_of_pages"),
            "physical_format": data.get("physical_format", ""),
            "weight": data.get("weight", ""),
            "description": "",
        }

    def search(self, query: str, limit: int = 10) -> list[dict]:
        """Search books by title or ISBN."""
        isbn_query = self._is_isbn(query)
        if isbn_query:
            book = self.get_by_isbn(isbn_query)
            return [book] if book else []

        q = f"title:{self._sanitize_query(query)}"
        url = f"https://openlibrary.org/search.json?q={q}&limit={limit}&fields=title,author_name,first_publish_year,cover_i,isbn,key,subject,publisher,physical_format,number_of_pages_median"
        try:
            req = urllib.request.Request(url)
            req.add_header("User-Agent", "SloughGPT/1.0")

            data = _retry(
                lambda: json.loads(urllib.request.urlopen(req, timeout=30).read().decode())
            )
            return [
                {
                    "key": r.get("key", ""),
                    "title": r.get("title", ""),
                    "author": r.get("author_name", [""])[0],
                    "isbn": r.get("isbn", [""])[0],
                    "year": r.get("first_publish_year"),
                    "cover": r.get("cover_i"),
                    "subjects": r.get("subject", [])[:20],
                    "publishers": r.get("publisher", []),
                    "physical_format": r.get("physical_format", ""),
                    "number_of_pages": r.get("number_of_pages_median"),
                }
                for r in data.get("docs", [])[:limit]
                if r.get("title")
            ]
        except Exception as e:
            logger.error(
                "Books search failed: %s",
                e,
                extra={"tag": "TRAIN"},
            )
            return []


class HuggingFaceImporter:
    """Import datasets from HuggingFace Hub."""

    def __init__(self):
        self._hf_available = self._check_hf()

    def _check_hf(self) -> bool:
        """Whether the ``datasets`` package is installed (needed to download).

        Search uses plain HTTP against the Hub REST API and needs no
        third-party package, so it always works regardless of this flag.
        """
        try:
            from datasets import load_dataset  # noqa: F401

            return True
        except ImportError:
            return False

    def search_datasets(self, query: str, limit: int = 10) -> list[dict]:
        """Search HuggingFace datasets via the Hub REST API."""
        try:
            from domain.infrastructure._internal.hf_hub import fetch_dataset_search
        except ImportError:
            logger.warning(
                "downcraft not available — dataset search disabled",
                extra={"tag": "TRAIN"},
            )
            return []

        try:
            return fetch_dataset_search(query, limit=limit)
        except Exception as e:
            logger.error(
                "Search failed: %s",
                e,
                extra={"tag": "TRAIN"},
            )
            return []

    def download_dataset(
        self,
        dataset_id: str,
        output_dir: str = "datasets",
        name: str | None = None,
        config: str | None = None,
        split: str | None = None,
    ) -> ImportResult:
        """Download dataset from HuggingFace."""
        if not self._hf_available:
            return ImportResult(
                success=False,
                name=name or dataset_id,
                source=f"huggingface:{dataset_id}",
                files_imported=0,
                total_chars=0,
                output_path="",
                error="HuggingFace dataset download requires the 'datasets' package. Install: pip install datasets",
            )

        try:
            from datasets import load_dataset

            name = name or dataset_id.split("/")[-1]
            output_path = Path(output_dir) / name
            output_path.mkdir(parents=True, exist_ok=True)

            logger.info(
                "Downloading %s%s...",
                dataset_id,
                f" (config={config})" if config else "",
                extra={"tag": "TRAIN"},
            )

            load_kwargs: dict = {}
            if config:
                load_kwargs["name"] = config
            if split:
                load_kwargs["split"] = split

            try:
                dataset = load_dataset(dataset_id, **load_kwargs)
            except Exception as config_err:
                error_msg = str(config_err)
                if (
                    "scripts are no longer supported" in error_msg
                    or "trust_remote_code" in error_msg
                ):
                    logger.warning(
                        "Script-based dataset detected, retrying with trust_remote_code: %s",
                        dataset_id,
                        extra={"tag": "TRAIN"},
                    )
                    try:
                        dataset = load_dataset(dataset_id, trust_remote_code=True, **load_kwargs)
                    except Exception:
                        fallback_kwargs = {k: v for k, v in load_kwargs.items() if k != "split"}
                        dataset = load_dataset(
                            dataset_id, split="train", trust_remote_code=True, **fallback_kwargs
                        )
                elif "Config name is missing" in error_msg:
                    import re

                    configs_match = re.search(r"available configs: \[(.+?)\]", error_msg)
                    available_configs = []
                    if configs_match:
                        available_configs = [
                            c.strip().strip("'") for c in configs_match.group(1).split(",")
                        ]
                    if available_configs:
                        auto_config = available_configs[0]
                        logger.info(
                            "Auto-selecting config '%s' for dataset %s",
                            auto_config,
                            dataset_id,
                            extra={"tag": "TRAIN"},
                        )
                        fallback_kwargs = {k: v for k, v in load_kwargs.items() if k != "split"}
                        dataset = load_dataset(
                            dataset_id, name=auto_config, split="train", **fallback_kwargs
                        )
                    else:
                        return ImportResult(
                            success=False,
                            name=name or dataset_id,
                            source=f"huggingface:{dataset_id}",
                            files_imported=0,
                            total_chars=0,
                            output_path="",
                            error=f"Dataset '{dataset_id}' requires a config name. Please specify a config.",
                        )
                else:
                    logger.warning(
                        "Full load failed, trying default config: %s",
                        config_err,
                        extra={"tag": "TRAIN"},
                    )
                    fallback_kwargs = {k: v for k, v in load_kwargs.items() if k != "split"}
                    dataset = load_dataset(dataset_id, split="train", **fallback_kwargs)

            # Handle both DatasetDict and Dataset
            total_chars = 0
            corpus_file = output_path / "corpus.jsonl"
            files_imported = 0

            with open(corpus_file, "w", encoding="utf-8") as f:
                # Check if dataset is a dict (DatasetDict) or single dataset
                if hasattr(dataset, "keys"):
                    # It's a DatasetDict - iterate over splits
                    for split in dataset.keys():
                        for item in dataset[split]:
                            text = item.get("text") or item.get("content") or str(item)
                            record = {"content": text, "split": split}
                            f.write(json.dumps(record, ensure_ascii=False) + "\n")
                            total_chars += len(text)
                            files_imported += 1
                else:
                    # Single dataset
                    for item in dataset:
                        text = item.get("text") or item.get("content") or str(item)
                        record = {"content": text, "split": "train"}
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")
                        total_chars += len(text)
                        files_imported += 1

            return ImportResult(
                success=True,
                name=name,
                source=f"huggingface:{dataset_id}",
                files_imported=files_imported,
                total_chars=total_chars,
                output_path=str(corpus_file),
            )
        except Exception as e:
            logger.error(
                "Download failed: %s",
                e,
                extra={"tag": "TRAIN"},
            )
            return ImportResult(
                success=False,
                name=name or dataset_id,
                source=f"huggingface:{dataset_id}",
                files_imported=0,
                total_chars=0,
                output_path="",
                error=str(e),
            )


class URLImporter:
    """Import data from URLs."""

    def __init__(self, cache_dir: str = "runs/downloads"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def download_file(self, url: str) -> Path:
        """Download file from URL."""
        import urllib.request

        filename = url.split("/")[-1] or "download"
        if "." not in filename:
            filename = "download.txt"

        output = self.cache_dir / filename
        urllib.request.urlretrieve(url, output)
        return output

    def import_from_url(
        self,
        url: str,
        dataset_name: str,
        output_dir: str = "datasets",
    ) -> ImportResult:
        """Import dataset from URL."""
        try:
            file_path = self.download_file(url)

            output_path = Path(output_dir) / dataset_name
            output_path.mkdir(parents=True, exist_ok=True)

            corpus_file = output_path / "corpus.jsonl"
            content = file_path.read_text(encoding="utf-8")

            with open(corpus_file, "w", encoding="utf-8") as f:
                record = {"content": content, "source": url}
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

            return ImportResult(
                success=True,
                name=dataset_name,
                source=url,
                files_imported=1,
                total_chars=len(content),
                output_path=str(corpus_file),
            )
        except Exception as e:
            logger.error(
                "URL import failed: %s",
                e,
                extra={"tag": "TRAIN"},
            )
            return ImportResult(
                success=False,
                name=dataset_name,
                source=url,
                files_imported=0,
                total_chars=0,
                output_path="",
                error=str(e),
            )


class GitHubSearch:
    """Search GitHub repositories."""

    def _sanitize_query(self, query: str) -> str:
        """Convert natural language to GitHub search query format.

        User types: "python code in here"
        URL becomes: q=python+code+in+here
        """
        import urllib.parse

        normalized = query.strip().lower()
        return urllib.parse.quote(normalized, safe="")

    def search_repos(self, query: str, limit: int = 10) -> list[dict]:
        """Search GitHub repos via API.

        Args:
            query: Natural language like "python code" or "machine learning examples"
        """
        try:
            q = self._sanitize_query(query)
            url = f"https://api.github.com/search/repositories?q={q}&per_page={limit}"
            req = urllib.request.Request(url)
            req.add_header("Accept", "application/vnd.github.v3+json")
            req.add_header("User-Agent", "SloughGPT")

            data = _retry(
                lambda: json.loads(urllib.request.urlopen(req, timeout=30).read().decode())
            )
            return [
                {
                    "full_name": r["full_name"],
                    "description": r.get("description", ""),
                    "html_url": r["html_url"],
                    "stargazers_count": r.get("stargazers_count", 0),
                    "forks_count": r.get("forks_count", 0),
                    "language": r.get("language"),
                }
                for r in data.get("items", [])
            ]
        except Exception as e:
            logger.error(
                "GitHub search failed: %s",
                e,
                extra={"tag": "TRAIN"},
            )
            return []


class ISBNImporter:
    """Import books by ISBN. Fetches full text from Project Gutenberg if available."""

    def __init__(self, output_dir: str = "datasets"):
        self.output_dir = output_dir
        self._books_search = BooksSearch()

    def import_from_isbn(self, isbn: str, name: str) -> ImportResult:
        """Import a book by ISBN.

        Looks up the book via Open Library (direct ISBN endpoint), then tries
        to download the full text from Project Gutenberg. Falls back to
        saving metadata only.

        Args:
            isbn: ISBN-10 or ISBN-13
            name: Dataset name

        Returns:
            ImportResult with success status
        """
        clean_isbn = re.sub(r"[-_\s]", "", isbn)

        book = self._books_search.get_by_isbn(clean_isbn)
        if not book:
            books = self._books_search.search(isbn, limit=1)
            if books:
                book = books[0]

        if not book:
            error_msg = f"Book not found for ISBN: {isbn}"
            logger.error(error_msg, extra={"tag": "TRAIN"})
            return ImportResult(
                success=False,
                name=name,
                source=f"isbn:{clean_isbn}",
                files_imported=0,
                total_chars=0,
                output_path="",
                error=error_msg,
            )

        output_dir = Path(self.output_dir) / name
        output_dir.mkdir(parents=True, exist_ok=True)

        title = book.get("title", "Unknown")
        author = book.get("author", "Unknown")

        text_content = self._fetch_gutenberg_text(title, author, clean_isbn)

        meta_content = {
            "source": f"isbn:{clean_isbn}",
            "title": title,
            "subtitle": book.get("subtitle", ""),
            "author": author,
            "isbn": clean_isbn,
            "year": book.get("year"),
            "publish_date": book.get("publish_date", ""),
            "publishers": book.get("publishers", []),
            "subjects": book.get("subjects", []),
            "subject_places": book.get("subject_places", []),
            "subject_times": book.get("subject_times", []),
            "subject_people": book.get("subject_people", []),
            "number_of_pages": book.get("number_of_pages"),
            "physical_format": book.get("physical_format", ""),
            "description": book.get("description", ""),
        }

        if text_content:
            text_path = output_dir / f"{name}.txt"
            text_path.write_text(text_content, encoding="utf-8")

            meta_content["source_type"] = "gutenberg"
            meta_content["text_chars"] = len(text_content)
            meta_path = output_dir / "metadata.json"
            meta_path.write_text(
                json.dumps(meta_content, indent=2, ensure_ascii=False), encoding="utf-8"
            )

            logger.info(
                "Imported '%s' from Gutenberg (%d chars)",
                title,
                len(text_content),
                extra={"tag": "TRAIN"},
            )
            return ImportResult(
                success=True,
                name=name,
                source=f"isbn:{clean_isbn}",
                files_imported=1,
                total_chars=len(text_content),
                output_path=str(output_dir),
            )

        meta_content["source_type"] = "metadata_only"
        meta_content["note"] = "Full text not available on Project Gutenberg"
        meta_path = output_dir / "metadata.json"
        meta_path.write_text(
            json.dumps(meta_content, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        book_info = (
            f"Title: {title}\nAuthor: {author}\nISBN: {clean_isbn}\nYear: {book.get('year', '')}\n"
        )
        if book.get("subtitle"):
            book_info += f"Subtitle: {book['subtitle']}\n"
        if book.get("publishers"):
            book_info += f"Publishers: {', '.join(book['publishers'])}\n"
        if book.get("subjects"):
            book_info += f"Subjects: {', '.join(book['subjects'][:10])}\n"
        book_info += "\nFull text not available on Project Gutenberg.\n"

        info_path = output_dir / f"{name}_info.txt"
        info_path.write_text(book_info, encoding="utf-8")

        return ImportResult(
            success=True,
            name=name,
            source=f"isbn:{clean_isbn}",
            files_imported=1,
            total_chars=len(book_info),
            output_path=str(output_dir),
        )

    def _fetch_gutenberg_text(self, title: str, author: str, isbn: str) -> str | None:
        """Search Project Gutenberg via Gutendex API and download full text.

        First tries to match by ISBN, then falls back to title+author search.
        """
        import urllib.parse

        book_id = self._find_gutenberg_id_by_isbn(isbn)
        if not book_id:
            search_terms = f"{title} {author}".strip()
            if not search_terms:
                return None
            q = urllib.parse.quote(search_terms[:200], safe="")
            book_id = self._find_gutenberg_id_by_search(q)

        if not book_id:
            return None

        return self._download_gutenberg_text(book_id)

    def _find_gutenberg_id_by_isbn(self, isbn: str) -> int | None:
        """Try to find a Gutenberg book by ISBN via Gutendex."""
        import urllib.parse

        q = urllib.parse.quote(f"isbn:{isbn}", safe="")
        url = f"https://gutendex.com/books?search={q}"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "SloughGPT/1.0"})
            data = _retry(
                lambda: json.loads(urllib.request.urlopen(req, timeout=30).read().decode())
            )
            items = data.get("results", [])
            if items:
                return items[0]["id"]
        except Exception:
            pass
        return None

    def _find_gutenberg_id_by_search(self, query: str) -> int | None:
        """Find a Gutenberg book by title+author search."""
        url = f"https://gutendex.com/books?search={query}"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "SloughGPT/1.0"})
            data = _retry(
                lambda: json.loads(urllib.request.urlopen(req, timeout=30).read().decode())
            )
            items = data.get("results", [])
            if items:
                return items[0]["id"]
        except Exception as e:
            logger.warning("Gutenberg search failed: %s", e, extra={"tag": "TRAIN"})
        return None

    def _download_gutenberg_text(self, gutenberg_id: int) -> str | None:
        """Download full text from Project Gutenberg by book ID."""
        text_url = f"https://www.gutenberg.org/cache/epub/{gutenberg_id}/pg{gutenberg_id}.txt"
        try:
            text_req = urllib.request.Request(text_url, headers={"User-Agent": "SloughGPT/1.0"})
            text = _retry(
                lambda: (
                    urllib.request.urlopen(text_req, timeout=60)
                    .read()
                    .decode("utf-8", errors="replace")
                )
            )
            logger.info(
                "Downloaded %s chars from Gutenberg #%s",
                len(text),
                gutenberg_id,
                extra={"tag": "TRAIN"},
            )
            return text
        except Exception as e:
            logger.warning(
                "Gutenberg download failed for #%s: %s",
                gutenberg_id,
                e,
                extra={"tag": "TRAIN"},
            )
            return None


class DataImporter:
    """Unified data importer for various sources."""

    def __init__(self, output_dir: str = "datasets"):
        self.output_dir = output_dir
        self.repo_importer = RepoImporter()
        self.hf_importer = HuggingFaceImporter()
        self.url_importer = URLImporter()

    def import_from_github(
        self,
        url: str,
        name: str,
        extensions: list[str] | None = None,
        max_files: int | None = None,
    ) -> ImportResult:
        """Import from GitHub repository."""
        return self.repo_importer.import_from_github(
            url, name, self.output_dir, extensions, max_files
        )

    def import_from_huggingface(
        self,
        dataset_id: str,
        name: str | None = None,
    ) -> ImportResult:
        """Import from HuggingFace Hub."""
        return self.hf_importer.download_dataset(dataset_id, self.output_dir, name)

    def import_from_url(self, url: str, name: str) -> ImportResult:
        """Import from URL."""
        return self.url_importer.import_from_url(url, name, self.output_dir)

    @staticmethod
    def _extract_pdf_text(file_path: Path) -> str:
        """Extract text from a PDF file using PyMuPDF (fitz) or PyPDF2.

        Args:
            file_path: Path to the PDF file.

        Returns:
            Extracted text content, or empty string if extraction fails.
        """
        try:
            import fitz

            doc = fitz.open(str(file_path))
            pages = [page.get_text() for page in doc]
            doc.close()
            return "\n".join(pages)
        except ImportError:
            pass
        try:
            import PyPDF2

            with open(file_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                return "\n".join(page.extract_text() or "" for page in reader.pages)
        except ImportError:
            pass
        return ""

    def import_from_local(
        self,
        path: str,
        name: str,
        extensions: list[str] | None = None,
    ) -> ImportResult:
        """Import from local file or directory.

        Supports text files (.py, .js, .ts, .md, .txt, .json, .yaml, .csv)
        and PDF files (.pdf) with automatic text extraction via PyMuPDF/PyPDF2.
        """
        source = Path(path)
        output_path = Path(self.output_dir) / name

        extensions = extensions or [".py", ".js", ".ts", ".md", ".txt", ".json", ".pdf"]

        try:
            output_path.mkdir(parents=True, exist_ok=True)

            corpus_file = output_path / "corpus.jsonl"
            total_chars = 0
            files_count = 0

            with open(corpus_file, "w", encoding="utf-8") as f:
                if source.is_file():
                    if source.suffix.lower() == ".pdf":
                        content = self._extract_pdf_text(source)
                    else:
                        content = source.read_text(encoding="utf-8")
                    record = {"content": content, "path": str(source)}
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    total_chars = len(content)
                    files_count = 1
                else:
                    for ext in extensions:
                        for file_path in source.rglob(f"*{ext}"):
                            try:
                                if file_path.suffix.lower() == ".pdf":
                                    content = self._extract_pdf_text(file_path)
                                else:
                                    content = file_path.read_text(encoding="utf-8")
                                record = {
                                    "content": content,
                                    "path": str(file_path.relative_to(source)),
                                }
                                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                                total_chars += len(content)
                                files_count += 1
                            except (UnicodeDecodeError, OSError):
                                continue

            return ImportResult(
                success=True,
                name=name,
                source=path,
                files_imported=files_count,
                total_chars=total_chars,
                output_path=str(corpus_file),
            )
        except Exception as e:
            return ImportResult(
                success=False,
                name=name,
                source=path,
                files_imported=0,
                total_chars=0,
                output_path="",
                error=str(e),
            )


def import_data(
    source: str,
    name: str,
    *,
    source_type: str | None = None,
    output_dir: str = "runs/imports",
    **kwargs,
) -> ImportResult:
    """Convenience wrapper: auto-detect source type and dispatch to the right importer.

    source_type can be "github", "huggingface", "url", or "local" to force
    a specific importer. If omitted, the type is detected from the source string.
    """
    imp: object
    if source_type == "local" or (
        not source_type
        and not source.startswith(("http://", "https://", "git@"))
        and "/" not in source.split(":")[-1]
    ):
        imp = DataImporter(output_dir=output_dir)
        return imp.import_from_local(source, name, **kwargs)  # type: ignore[union-attr]
    elif source_type == "huggingface" or (
        not source_type
        and not source.startswith(("http://", "https://", "git@"))
        and ":" not in source
    ):
        return HuggingFaceImporter().download_dataset(
            source, output_dir=output_dir, name=name, **kwargs
        )  # type: ignore[union-attr]
    elif source_type == "url" or (
        not source_type
        and source.startswith(("http://", "https://"))
        and "github.com" not in source
    ):
        return URLImporter().import_from_url(source, name, output_dir=output_dir, **kwargs)  # type: ignore[union-attr]
    else:
        return RepoImporter().import_from_github(source, name, output_dir=output_dir, **kwargs)  # type: ignore[union-attr]


__all__ = [
    "DataImporter",
    "RepoImporter",
    "HuggingFaceImporter",
    "URLImporter",
    "GitHubSearch",
    "BooksSearch",
    "ISBNImporter",
    "ImportResult",
    "import_data",
]
