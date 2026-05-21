"""
Data Import Tools
Import datasets from various sources: GitHub, HuggingFace, URLs, local files.
"""

import json
import os
import subprocess
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set
from dataclasses import dataclass

logger = logging.getLogger("sloughgpt.data_import")

DEFAULT_IGNORES: Set[str] = {
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
    error: Optional[str] = None


class RepoImporter:
    """Import code from Git repositories."""

    def __init__(self, cache_dir: str = "runs/repos"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def clone_repo(
        self,
        url: str,
        branch: Optional[str] = None,
        depth: Optional[int] = 1,
    ) -> Path:
        """Clone a git repository."""
        repo_name = url.split("/")[-1].replace(".git", "")
        target = self.cache_dir / repo_name

        if target.exists():
            logger.info(f"Repo already exists: {target}")
            return target

        cmd = ["git", "clone", url, str(target)]
        if branch:
            cmd.extend(["-b", branch])
        if depth:
            cmd.extend(["--depth", str(depth)])

        logger.info(f"Cloning {url}...")
        subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        return target

    def export_to_corpus(
        self,
        repo_path: Path,
        output_path: str,
        ignores: Optional[Set[str]] = None,
        max_files: Optional[int] = None,
        max_bytes: int = 200_000,
        extensions: Optional[List[str]] = None,
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

    def _iter_files(self, root: Path, ignores: Set[str], extensions: List[str]):
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
                import json

                json.loads(sample)
                return "json"
            except:
                pass

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
        extensions: Optional[List[str]] = None,
        max_files: Optional[int] = None,
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
                with open(output_path, "r") as f:
                    for line in f:
                        data = json.loads(line)
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
            logger.error(f"Failed to import from GitHub: {e}")
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

    def _is_isbn(self, query: str) -> Optional[str]:
        import re
        digits = re.sub(r"[-_\s]", "", query)
        if len(digits) == 10 and digits.isdigit():
            return f"isbn:{digits}"
        if len(digits) == 13 and digits.isdigit():
            return f"isbn:{digits}"
        return None

    def search(self, query: str, limit: int = 10) -> List[Dict]:
        """Search books by title or ISBN."""
        import urllib.request
        import json

        isbn_query = self._is_isbn(query)
        if isbn_query:
            q = isbn_query
        else:
            q = f"title:{self._sanitize_query(query)}"

        url = f"https://openlibrary.org/search.json?q={q}&limit={limit}"
        try:
            req = urllib.request.Request(url)
            req.add_header("User-Agent", "SloughGPT/1.0")

            with urllib.request.urlopen(req, timeout=30) as response:
                data = json.loads(response.read().decode())
                return [
                    {
                        "key": r.get("key", ""),
                        "title": r.get("title", ""),
                        "author": r.get("author_name", [""])[0],
                        "isbn": r.get("isbn", [""])[0],
                        "year": r.get("first_publish_year"),
                        "cover": r.get("cover_i"),
                    }
                    for r in data.get("docs", [])[:limit]
                    if r.get("title")
                ]
        except Exception as e:
            logger.error(f"Books search failed: {e}")
            return []


class HuggingFaceImporter:
    """Import datasets from HuggingFace Hub."""

    def __init__(self):
        self._hf_available = self._check_hf()

    def _check_hf(self) -> bool:
        try:
            from huggingface_hub import HfApi

            return True
        except ImportError:
            return False

    def search_datasets(self, query: str, limit: int = 10) -> List[Dict]:
        """Search HuggingFace datasets."""
        if not self._hf_available:
            logger.warning("HuggingFace not available. Install: pip install huggingface_hub")
            return []

        from huggingface_hub import HfApi

        api = HfApi()

        try:
            datasets = api.list_datasets(search=query, limit=limit)
            return [{"id": ds.id, "downloads": getattr(ds, "downloads", 0) or 0} for ds in datasets]
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    def download_dataset(
        self,
        dataset_id: str,
        output_dir: str = "datasets",
        name: Optional[str] = None,
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
                error="HuggingFace not available. Install: pip install huggingface_hub datasets",
            )

        try:
            from datasets import load_dataset

            name = name or dataset_id.split("/")[-1]
            output_path = Path(output_dir) / name
            output_path.mkdir(parents=True, exist_ok=True)

            logger.info(f"Downloading {dataset_id}...")
            
            # Handle datasets with subsets/config
            try:
                dataset = load_dataset(dataset_id, trust_remote_code=True)
            except Exception as config_err:
                # Try with default config for datasets with subsets
                logger.warning(f"Full load failed, trying default config: {config_err}")
                dataset = load_dataset(dataset_id, split="train", trust_remote_code=True)
            
            # Handle both DatasetDict and Dataset
            total_chars = 0
            corpus_file = output_path / "corpus.jsonl"
            files_imported = 0

            with open(corpus_file, "w", encoding="utf-8") as f:
                # Check if dataset is a dict (DatasetDict) or single dataset
                if hasattr(dataset, 'keys'):
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
            logger.error(f"Download failed: {e}")
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
            logger.error(f"URL import failed: {e}")
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

    def search_repos(self, query: str, limit: int = 10) -> List[Dict]:
        """Search GitHub repos via API.

        Args:
            query: Natural language like "python code" or "machine learning examples"
        """
        try:
            import urllib.request
            import json

            q = self._sanitize_query(query)
            url = f"https://api.github.com/search/repositories?q={q}&per_page={limit}"
            req = urllib.request.Request(url)
            req.add_header("Accept", "application/vnd.github.v3+json")
            req.add_header("User-Agent", "SloughGPT")

            with urllib.request.urlopen(req, timeout=30) as response:
                data = json.loads(response.read().decode())
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
            logger.error(f"GitHub search failed: {e}")
            return []


class ISBNImporter:
    """Import books by ISBN. Fetches full text from Project Gutenberg if available."""

    def __init__(self, output_dir: str = "datasets"):
        self.output_dir = output_dir
        self._books_search = BooksSearch()

    def import_from_isbn(self, isbn: str, name: str) -> ImportResult:
        """
        Import a book by ISBN.

        Looks up the book via Open Library, then tries to download the full text
        from Project Gutenberg. Falls back to saving metadata only.

        Args:
            isbn: ISBN-10 or ISBN-13
            name: Dataset name

        Returns:
            ImportResult with success status
        """
        import urllib.request
        import json

        output_dir = Path(self.output_dir) / name
        output_dir.mkdir(parents=True, exist_ok=True)

        books = self._books_search.search(isbn, limit=1)
        if not books:
            error_msg = f"Book not found for ISBN: {isbn}"
            logger.error(error_msg)
            return ImportResult(
                success=False,
                name=name,
                source=f"isbn:{isbn}",
                files_imported=0,
                total_chars=0,
                output_path=str(output_dir),
                error=error_msg,
            )

        book = books[0]
        title = book.get("title", "Unknown")
        author = book.get("author", "Unknown")
        metadata = json.dumps(book, indent=2)

        text_content = self._fetch_gutenberg_text(title, author)

        if text_content:
            text_path = output_dir / f"{name}.txt"
            text_path.write_text(text_content, encoding="utf-8")

            meta_path = output_dir / "metadata.json"
            meta_content = {
                "source": f"isbn:{isbn}",
                "title": title,
                "author": author,
                "isbn": isbn,
                "year": book.get("year"),
                "source_type": "gutenberg",
            }
            meta_path.write_text(json.dumps(meta_content, indent=2), encoding="utf-8")

            return ImportResult(
                success=True,
                name=name,
                source=f"isbn:{isbn}",
                files_imported=1,
                total_chars=len(text_content),
                output_path=str(output_dir),
            )
        else:
            meta_path = output_dir / "metadata.json"
            meta_content = {
                "source": f"isbn:{isbn}",
                "title": title,
                "author": author,
                "isbn": isbn,
                "year": book.get("year"),
                "source_type": "metadata_only",
                "note": "Full text not available on Project Gutenberg",
            }
            meta_path.write_text(json.dumps(meta_content, indent=2), encoding="utf-8")

            book_info = f"Title: {title}\nAuthor: {author}\nISBN: {isbn}\nYear: {book.get('year', '')}\n\n{metadata}"
            info_path = output_dir / f"{name}_info.txt"
            info_path.write_text(book_info, encoding="utf-8")

            return ImportResult(
                success=True,
                name=name,
                source=f"isbn:{isbn}",
                files_imported=1,
                total_chars=len(book_info),
                output_path=str(output_dir),
            )

    def _fetch_gutenberg_text(self, title: str, author: str) -> Optional[str]:
        """
        Search Project Gutenberg via Gutendex API and download full text.

        Args:
            title: Book title
            author: Book author

        Returns:
            Full text content if found, None otherwise
        """
        import urllib.request
        import json

        import urllib.parse
        search_terms = f"{title} {author}".strip()
        if not search_terms:
            return None
        q = urllib.parse.quote(search_terms[:200], safe="")

        try:
            url = f"http://gutendex.com/books?search={q}"
            req = urllib.request.Request(url, headers={"User-Agent": "SloughGPT/1.0"})
            with urllib.request.urlopen(req, timeout=30) as response:
                data = json.loads(response.read().decode())
                items = data.get("results", [])
                if not items:
                    logger.info(f"No Gutendex results for: {title}")
                    return None

                gutenberg_id = items[0]["id"]
                text_url = f"https://www.gutenberg.org/cache/epub/{gutenberg_id}/pg{gutenberg_id}.txt"
                text_req = urllib.request.Request(text_url, headers={"User-Agent": "SloughGPT/1.0"})
                with urllib.request.urlopen(text_req, timeout=60) as text_resp:
                    text = text_resp.read().decode("utf-8", errors="replace")
                    logger.info(f"Downloaded {len(text)} chars from Gutenberg #{gutenberg_id}")
                    return text
        except Exception as e:
            logger.warning(f"Gutenberg fetch failed for '{title}': {e}")
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
        extensions: Optional[List[str]] = None,
        max_files: Optional[int] = None,
    ) -> ImportResult:
        """Import from GitHub repository."""
        return self.repo_importer.import_from_github(
            url, name, self.output_dir, extensions, max_files
        )

    def import_from_huggingface(
        self,
        dataset_id: str,
        name: Optional[str] = None,
    ) -> ImportResult:
        """Import from HuggingFace Hub."""
        return self.hf_importer.download_dataset(dataset_id, self.output_dir, name)

    def import_from_url(self, url: str, name: str) -> ImportResult:
        """Import from URL."""
        return self.url_importer.import_from_url(url, name, self.output_dir)

    def import_from_local(
        self,
        path: str,
        name: str,
        extensions: Optional[List[str]] = None,
    ) -> ImportResult:
        """Import from local file or directory."""
        source = Path(path)
        output_path = Path(self.output_dir) / name
        output_path.mkdir(parents=True, exist_ok=True)

        extensions = extensions or [".py", ".js", ".ts", ".md", ".txt", ".json"]

        try:
            corpus_file = output_path / "corpus.jsonl"
            total_chars = 0
            files_count = 0

            with open(corpus_file, "w", encoding="utf-8") as f:
                if source.is_file():
                    content = source.read_text(encoding="utf-8")
                    record = {"content": content, "path": str(source)}
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    total_chars = len(content)
                    files_count = 1
                else:
                    for ext in extensions:
                        for file_path in source.rglob(f"*{ext}"):
                            try:
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
    source_type: str = "auto",
    output_dir: str = "datasets",
    **kwargs,
) -> ImportResult:
    """
    Import data from various sources.

    Args:
        source: URL, path, or dataset ID
        name: Dataset name
        source_type: 'github', 'huggingface', 'url', 'local', or 'auto'
        output_dir: Output directory
        **kwargs: Additional options

    Returns:
        ImportResult
    """
    importer = DataImporter(output_dir)

    if source_type == "auto":
        if source.startswith(("https://github.com", "git@github.com")):
            source_type = "github"
        elif "/" in source and not source.startswith("http"):
            source_type = "huggingface"
        elif source.startswith(("http://", "https://")):
            source_type = "url"
        else:
            source_type = "local"

    if source_type == "github":
        return importer.import_from_github(source, name, **kwargs)
    elif source_type == "huggingface":
        return importer.import_from_huggingface(source, name)
    elif source_type == "url":
        return importer.import_from_url(source, name)
    else:
        return importer.import_from_local(source, name, **kwargs)


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
