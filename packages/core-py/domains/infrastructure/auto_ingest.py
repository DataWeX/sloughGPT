"""
Auto-Ingestion Pipeline - Scans the repo and ingests all code/docs into vector store.
Run: python3 -m domains.infrastructure.auto_ingest --path . --provider chromadb
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass
import hashlib
import logging

logger = logging.getLogger("slo.auto_ingest")

DEFAULT_IGNORE_DIRS = {
    '.git', '__pycache__', 'node_modules', '.venv', 'venv', '.env',
    'dist', 'build', '.next', '.cache', 'coverage', '.pytest_cache',
    '.mypy_cache', '.ruff_cache', 'site-packages', 'lib', 'include',
    'data/backups', 'data/vector_store', 'data/training_exports',
    '.turbo', '.cursor', '.vscode', '.idea', 'sloughgpt_colab.ipynb',
}

DEFAULT_IGNORE_EXTENSIONS = {
    '.pyc', '.pyo', '.so', '.dll', '.dylib', '.exe', '.bin', '.o',
    '.a', '.lib', '.obj', '.wasm', '.min.js', '.min.css', '.map',
    '.lock', '.snap', '.腐败', '.DS_Store', 'Thumbs.db',
}

DEFAULT_IGNORE_FILES = {
    'package-lock.json', 'yarn.lock', 'pnpm-lock.yaml', 'poetry.lock',
    ' Pipfile.lock', 'requirements.txt', 'setup.py', 'pyproject.toml',
    'Makefile', 'Dockerfile', '.dockerignore', 'README.md', 'LICENSE',
    'CONTRIBUTING.md', 'AGENTS.md', 'CHANGELOG.md', '.gitignore',
    '.gitattributes', 'conftest.py', 'noxfile.py', 'tox.ini',
}

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


@dataclass
class FileChunk:
    id: str
    file_path: str
    content: str
    metadata: Dict[str, Any]
    chunk_index: int


class RepoScanner:
    """Walks the filesystem and yields relevant files."""

    def __init__(
        self,
        root_path: str = ".",
        ignore_dirs: set = None,
        ignore_exts: set = None,
        ignore_files: set = None,
        max_file_size: int = 2_000_000,  # 2MB
    ):
        self.root = Path(root_path).resolve()
        self.ignore_dirs = ignore_dirs or DEFAULT_IGNORE_DIRS
        self.ignore_exts = ignore_exts or DEFAULT_IGNORE_EXTENSIONS
        self.ignore_files = ignore_files or DEFAULT_IGNORE_FILES
        self.max_file_size = max_file_size

    def should_ignore(self, path: Path) -> bool:
        name = path.name
        if name in self.ignore_files:
            return True
        if path.suffix in self.ignore_exts:
            return True
        for part in path.parts:
            if part in self.ignore_dirs:
                return True
        return False

    def get_file_type(self, path: Path) -> str:
        """Categorize file for metadata."""
        ext = path.suffix.lower()
        if ext in {'.py'}: return 'python'
        if ext in {'.ts', '.tsx', '.js', '.jsx', '.mjs', '.cjs'}: return 'javascript'
        if ext in {'.rs'}: return 'rust'
        if ext in {'.go'}: return 'go'
        if ext in {'.md', '.mdx'}: return 'markdown'
        if ext in {'.json', '.jsonl'}: return 'json'
        if ext in {'.yaml', '.yml'}: return 'yaml'
        if ext in {'.toml'}: return 'toml'
        if ext in {'.txt', '.text'}: return 'text'
        if ext in {'.sql'}: return 'sql'
        if ext in {'.sh', '.bash'}: return 'shell'
        if ext in {'.css', '.scss', '.less'}: return 'stylesheet'
        if ext in {'.html', '.htm'}: return 'html'
        return 'unknown'

    def guess_language(self, path: Path) -> str:
        """For code block rendering."""
        ext = path.suffix.lower()
        mapping = {
            '.py': 'python', '.js': 'javascript', '.ts': 'typescript',
            '.jsx': 'jsx', '.tsx': 'tsx', '.rs': 'rust', '.go': 'go',
            '.sh': 'bash', '.yaml': 'yaml', '.yml': 'yaml',
            '.toml': 'toml', '.json': 'json', '.md': 'markdown',
            '.css': 'css', '.html': 'html', '.sql': 'sql',
        }
        return mapping.get(ext, 'text')

    def iter_files(self):
        """Yield (path, content) for all relevant files."""
        for path in self.root.rglob('*'):
            if not path.is_file():
                continue
            if self.should_ignore(path):
                continue
            try:
                size = path.stat().st_size
                if size > self.max_file_size:
                    yield path, f"[File too large: {size} bytes — skipped]"
                    continue
                try:
                    content = path.read_text(encoding='utf-8', errors='replace')
                    yield path, content
                except Exception:
                    yield path, "[Binary or unreadable file]"
            except OSError:
                continue


class CodeChunker:
    """Splits content into chunks with overlap for context."""

    def __init__(self, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_text(self, text: str, path: str, chunk_index: int) -> FileChunk:
        """Create a single chunk with metadata."""
        chunk_id = hashlib.md5(f"{path}:{chunk_index}".encode()).hexdigest()[:16]
        return FileChunk(
            id=chunk_id,
            file_path=path,
            content=text,
            metadata={},
            chunk_index=chunk_index,
        )

    def chunk_file(self, path: str, content: str) -> List[FileChunk]:
        """Split file into overlapping chunks."""
        if not content or content.startswith('[File too large') or content.startswith('[Binary'):
            return [self.chunk_text(content, path, 0)]

        relative = str(Path(path).relative_to(Path(path).parent.parent))
        lang = Path(path).suffix.lstrip('.')

        lines = content.split('\n')
        chunks: List[FileChunk] = []

        # Try to chunk by logical blocks first (for code)
        if self._is_code_file(path):
            chunks = self._chunk_code(path, content, lang)
        else:
            chunks = self._chunk_prose(path, content)

        return chunks if chunks else [self.chunk_text(content, path, 0)]

    def _is_code_file(self, path: str) -> bool:
        code_exts = {'.py', '.js', '.ts', '.jsx', '.tsx', '.rs', '.go', '.sh', '.sql'}
        return Path(path).suffix.lower() in code_exts

    def _chunk_code(self, path: str, content: str, lang: str) -> List[FileChunk]:
        """Split code by functions/classes when possible."""
        chunks = []
        chunk_content = ""
        chunk_start = 0
        lines = content.split('\n')

        for i, line in enumerate(lines):
            # Class/function definition lines
            is_def = any([
                line.strip().startswith('def '),
                line.strip().startswith('class '),
                line.strip().startswith('async def '),
                line.strip().startswith('fn '),
                line.strip().startswith('func '),
                line.strip().startswith('function '),
                line.strip().startswith('const '),
                line.strip().startswith('let '),
                line.strip().startswith('var '),
                line.strip().startswith('interface '),
                line.strip().startswith('type '),
            ])

            if is_def and len(chunk_content) > self.chunk_size:
                chunks.append(self.chunk_text(chunk_content, path, len(chunks)))
                chunk_content = ""

            chunk_content += line + "\n"

        if chunk_content.strip():
            chunks.append(self.chunk_text(chunk_content, path, len(chunks)))

        return chunks

    def _chunk_prose(self, path: str, content: str) -> List[FileChunk]:
        """Split prose/text by paragraphs."""
        chunks = []
        paragraphs = content.split('\n\n')
        current = ""

        for para in paragraphs:
            if len(current) + len(para) < self.chunk_size:
                current += para + "\n\n"
            else:
                if current.strip():
                    chunks.append(self.chunk_text(current.strip(), path, len(chunks)))
                # Keep overlap
                overlap_text = current[-self.overlap:] if len(current) > self.overlap else current
                current = overlap_text + para + "\n\n"

        if current.strip():
            chunks.append(self.chunk_text(current.strip(), path, len(chunks)))

        return chunks


def simple_embed(text: str, dim: int = 384) -> List[float]:
    """Embed text using sentence-transformers (cached singleton) with hash fallback."""
    from domains.inference.vector_store import simple_embed as vs_embed
    return vs_embed(text, dimension=dim)


class AutoIngester:
    """Main ingestion engine."""

    def __init__(
        self,
        root_path: str = ".",
        provider: str = "chromadb",
        chunk_size: int = CHUNK_SIZE,
    ):
        self.root = Path(root_path).resolve()
        self.provider = provider
        self.scanner = RepoScanner(root_path)
        self.chunker = CodeChunker(chunk_size=chunk_size)
        self.stats = {
            "files_scanned": 0,
            "files_ingested": 0,
            "chunks_created": 0,
            "errors": 0,
        }

    async def get_vector_store(self):
        """Connect to vector store."""
        try:
            from domains.inference.vector_store import create_vector_store
            kwargs = {"dimension": 384}
            if self.provider == "chromadb":
                kwargs["persist_directory"] = "data/vector_store"
            store = await create_vector_store(provider=self.provider, **kwargs)
            await store.connect()
            return store
        except ImportError:
            logger.warning("Vector store not available. Running in dry-run mode.",
                extra={"tag": "INFRA"})
            return None

    def build_metadata(self, chunk: FileChunk) -> Dict[str, Any]:
        """Build rich metadata for a chunk."""
        path = Path(chunk.file_path)
        relative = str(path.relative_to(self.root))
        ext = path.suffix.lstrip('.')

        return {
            "file": relative,
            "file_name": path.name,
            "extension": ext,
            "file_type": self.scanner.get_file_type(path),
            "language": self.scanner.guess_language(path),
            "chunk_index": chunk.chunk_index,
            "repo": str(self.root.name),
        }

    async def ingest(self, dry_run: bool = False) -> Dict[str, Any]:
        """Run full ingestion."""
        store = await self.get_vector_store()

        all_chunks: List[FileChunk] = []

        # Scan and chunk
        logger.info("Scanning %s...", self.root,
            extra={"tag": "INFRA"})
        for path, content in self.scanner.iter_files():
            self.stats["files_scanned"] += 1
            try:
                chunks = self.chunker.chunk_file(str(path), content)
                all_chunks.extend(chunks)
                self.stats["chunks_created"] += len(chunks)
            except Exception:
                self.stats["errors"] += 1

        logger.info("  %d files, %d chunks", self.stats['files_scanned'], self.stats['chunks_created'],
            extra={"tag": "INFRA"})

        if dry_run or not store:
            logger.info("  Dry run — not writing to vector store",
                extra={"tag": "INFRA"})
            return self.stats

        # Upsert to vector store
        logger.info("  Ingesting to %s...", self.provider,
            extra={"tag": "INFRA"})
        from domains.inference.vector_store import VectorEntry

        entries = []
        for chunk in all_chunks:
            embedding = simple_embed(chunk.content)
            metadata = self.build_metadata(chunk)
            entries.append(VectorEntry(
                id=chunk.id,
                text=chunk.content[:2000],  # ChromaDB has limits
                vector=embedding,
                metadata=metadata,
            ))

        count = await store.upsert(entries)
        self.stats["files_ingested"] = count

        logger.info("  %d chunks ingested", count,
            extra={"tag": "INFRA"})
        return self.stats

    async def ingest_single_file(self, file_path: str) -> int:
        """Ingest a single file."""
        path = Path(file_path)
        if not path.exists():
            return 0

        content = path.read_text(encoding='utf-8', errors='replace')
        chunks = self.chunker.chunk_file(str(path), content)
        self.stats["files_scanned"] = 1
        self.stats["chunks_created"] += len(chunks)

        store = await self.get_vector_store()
        if not store:
            return len(chunks)

        from domains.inference.vector_store import VectorEntry
        entries = [
            VectorEntry(
                id=c.id,
                text=c.content[:2000],
                vector=simple_embed(c.content),
                metadata=self.build_metadata(c),
            )
            for c in chunks
        ]
        return await store.upsert(entries)

    def query_relevant(self, query: str, top_k: int = 5) -> List[Dict]:
        """Query the ingested knowledge."""
        async def _query():
            store = await self.get_vector_store()
            if not store:
                return []
            vec = simple_embed(query)
            results = await store.query(vec, top_k=top_k)
            return [
                {
                    "file": r.metadata.get("file", r.id),
                    "score": r.score,
                    "text": r.text[:300],
                    "type": r.metadata.get("file_type", "unknown"),
                }
                for r in results
            ]

        import asyncio
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(_query())
        finally:
            loop.close()


async def main():
    import argparse
    parser = argparse.ArgumentParser(description="Auto-ingest repo into vector store")
    parser.add_argument("--path", default=".", help="Root path to scan")
    parser.add_argument("--provider", default="chromadb", choices=["chromadb", "in_memory", "pinecone"])
    parser.add_argument("--dry-run", action="store_true", help="Scan without writing")
    parser.add_argument("--file", help="Ingest single file")
    args = parser.parse_args()

    ingester = AutoIngester(root_path=args.path, provider=args.provider)

    if args.file:
        count = await ingester.ingest_single_file(args.file)
        print(f"✅ Ingested {count} chunks from {args.file}")
    else:
        stats = await ingester.ingest(dry_run=args.dry_run)
        print(f"📊 Done: {stats}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
