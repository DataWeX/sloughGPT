"""
Datasets Router - MVC View layer
"""
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from pathlib import Path
from typing import Optional
import asyncio
import json
import time

from schemas.datasets import (
    DatasetInfo, DatasetCreate, DatasetUpdate, DatasetDataRequest,
    DatasetStats, DatasetListResponse,
    GitHubImportRequest, HuggingFaceImportRequest, URLImportRequest,
    LocalImportRequest, KaggleImportRequest, CSVImportRequest,
    BatchImportRequest, ISBNImportRequest, ImportResponse,
    VersionCreateResponse, VersionListResponse, VersionRestoreResponse,
    FromChatRequest, DatasetExportRequest,
)
from schemas.common import success_response, raise_error, classify_and_raise, safe_audit_log
from controllers.datasets import get_datasets_controller

import logging
import re

logger = logging.getLogger("slo.routers.datasets")


class DatasetsRouter:
    """Router for dataset CRUD, import, export, search and versioning."""

    def __init__(self):
        self.router = APIRouter(prefix="/datasets", tags=["datasets"])
        self._DATASETS_DIR = Path(__file__).resolve().parents[4] / "datasets"
        self._import_locks: dict[str, asyncio.Lock] = {}
        self._import_locks_lock = asyncio.Lock()
        self._DATASET_ID_RE = re.compile(r'^[a-zA-Z0-9_\-]+$')
        self._register_routes()

    def _register_routes(self):
        self.router.add_api_route("", self.list_datasets, methods=["GET"], response_model=DatasetListResponse)
        self.router.add_api_route("/import/local", self.import_from_local, methods=["POST"], response_model=ImportResponse)
        self.router.add_api_route("/import/github", self.import_from_github, methods=["POST"], response_model=ImportResponse)
        self.router.add_api_route("/import/huggingface", self.import_from_huggingface, methods=["POST"], response_model=ImportResponse)
        self.router.add_api_route("/import/url", self.import_from_url, methods=["POST"], response_model=ImportResponse)
        self.router.add_api_route("/import/kaggle", self.import_from_kaggle, methods=["POST"], response_model=ImportResponse)
        self.router.add_api_route("/import/csv", self.import_from_csv, methods=["POST"], response_model=ImportResponse)
        self.router.add_api_route("/import/batch", self.batch_import, methods=["POST"])
        self.router.add_api_route("/search/books", self.search_books, methods=["GET"])
        self.router.add_api_route("/search/github", self.search_github, methods=["GET"])
        self.router.add_api_route("/import/isbn", self.import_from_isbn, methods=["POST"], response_model=ImportResponse)
        self.router.add_api_route("/search", self.search_datasets, methods=["GET"])
        self.router.add_api_route("/{dataset_id}", self.get_dataset, methods=["GET"], response_model=DatasetInfo)
        self.router.add_api_route("/{dataset_id}/stats", self.get_dataset_stats, methods=["GET"], response_model=DatasetStats)
        self.router.add_api_route("", self.create_dataset, methods=["POST"], response_model=DatasetInfo)
        self.router.add_api_route("/{dataset_id}", self.update_dataset, methods=["PATCH"], response_model=DatasetInfo)
        self.router.add_api_route("/{dataset_id}", self.delete_dataset, methods=["DELETE"])
        self.router.add_api_route("/{dataset_id}/versions", self.create_version, methods=["POST"], response_model=VersionCreateResponse)
        self.router.add_api_route("/{dataset_id}/versions", self.list_versions, methods=["GET"], response_model=VersionListResponse)
        self.router.add_api_route("/{dataset_id}/versions/{timestamp}", self.restore_version, methods=["POST"], response_model=VersionRestoreResponse)
        self.router.add_api_route("/{dataset_id}/data", self.add_dataset_data, methods=["POST"])
        self.router.add_api_route("/{dataset_id}/preview", self.preview_dataset, methods=["GET"])
        self.router.add_api_route("/{dataset_id}/export", self.export_dataset, methods=["POST"])
        self.router.add_api_route("/from-chat", self.create_dataset_from_chat, methods=["POST"])
        self.router.add_api_route("/convert-to-messages", self.convert_to_messages, methods=["POST"])

    def _validate_dataset_id(self, dataset_id: str) -> str:
        """Validate dataset_id contains only safe characters (no path traversal)."""
        if not self._DATASET_ID_RE.match(dataset_id):
            raise_error(f"Invalid dataset ID: {dataset_id!r} — only alphanumeric, hyphens, underscores allowed", "E_VAL_REQUEST")
        return dataset_id

    async def _get_import_lock(self, name: str) -> asyncio.Lock:
        """Get a per-dataset import lock to prevent concurrent imports to the same name."""
        async with self._import_locks_lock:
            if name not in self._import_locks:
                self._import_locks[name] = asyncio.Lock()
            return self._import_locks[name]

    def _get_data_importer(self):
        """Get DataImporter configured to save to the repo datasets directory."""
        from domains.training.data_import import DataImporter
        return DataImporter(output_dir=str(self._DATASETS_DIR))

    async def list_datasets(
        self,
        q: Optional[str] = Query(None, description="Search query"),
        type: Optional[str] = Query(None, description="Filter by type"),
    ) -> dict:
        """List all datasets, optionally filtered by search query and type.

        Args:
            q: Optional search string to filter datasets by name or
                description. When None, all datasets are returned.
            type: Optional type filter (e.g. "text", "image"). When None,
                all types are returned.

        Returns:
            DatasetListResponse containing a list of DatasetInfo objects
            and a count of matching datasets.

        Side effects:
            Reads from the DatasetsController which scans the datasets
            directory on disk.
        """
        ctrl = get_datasets_controller()
        datasets = await asyncio.to_thread(ctrl.list_datasets, q, type)
        return DatasetListResponse(
            datasets=[DatasetInfo(**d) for d in datasets],
            count=len(datasets),
        )

    async def import_from_local(self, request: LocalImportRequest) -> dict:
        """Import dataset from local file or directory."""
        lock = await self._get_import_lock(request.name)
        if lock.locked():
            raise_error(f"Import already in progress for '{request.name}'", "E_INFRA_BUSY")
        async with lock:
            try:
                from pathlib import Path as _P
                import_path = _P(request.path).resolve()
                _allowed = {_P.home(), _P.home() / "Documents", _P.home() / "Downloads", _P.home() / "Pictures", self._DATASETS_DIR.resolve()}
                if not any(import_path == b or str(import_path).startswith(str(b) + "/") for b in _allowed):
                    raise_error(f"Directory not in allowed paths: {request.path}", "E_AUTH_FORBIDDEN")
                importer = self._get_data_importer()
                _t0 = time.monotonic()
                result = await asyncio.to_thread(
                    importer.import_from_local,
                    path=request.path,
                    name=request.name,
                    extensions=request.extensions or None,
                )
                _elapsed_ms = (time.monotonic() - _t0) * 1000
                if result.success:
                    safe_audit_log("dataset.import", resource=request.name, detail=f"local elapsed={_elapsed_ms:.0f}ms", files=result.files_imported, chars=result.total_chars)
                    return ImportResponse(
                        success=True,
                        dataset_id=request.name,
                        message=f"Imported {result.files_imported} files ({result.total_chars} chars)",
                        output_path=result.output_path,
                    )
                raise_error(result.error or "Import failed", "E_BAD_REQUEST")
            except HTTPException:
                raise
            except Exception as e:
                logger.warning("Dataset import (local) failed: %s", e)
                classify_and_raise(e, source="dataset_import_local")

    async def import_from_github(self, request: GitHubImportRequest) -> dict:
        """Import dataset from GitHub repository."""
        lock = await self._get_import_lock(request.name)
        if lock.locked():
            raise_error(f"Import already in progress for '{request.name}'", "E_INFRA_BUSY")
        async with lock:
            try:
                from domains.training.data_import import RepoImporter
                importer = RepoImporter()
                _t0 = time.monotonic()
                result = await asyncio.to_thread(
                    importer.import_from_github,
                    url=request.url,
                    dataset_name=request.name,
                    output_dir=str(self._DATASETS_DIR),
                    extensions=request.extensions or None,
                    max_files=request.max_files,
                )
                _elapsed_ms = (time.monotonic() - _t0) * 1000
                if result.success:
                    safe_audit_log("dataset.import", resource=result.name or request.name, detail=f"github elapsed={_elapsed_ms:.0f}ms", files=result.files_imported, chars=result.total_chars)
                    return ImportResponse(
                        success=True,
                        dataset_id=result.name or request.name,
                        message=f"Imported {result.files_imported} files ({result.total_chars} chars)",
                        output_path=result.output_path,
                    )
                raise_error(result.error or "Import failed", "E_BAD_REQUEST")
            except HTTPException:
                raise
            except Exception as e:
                logger.warning("Dataset import (github) failed: %s", e)
                classify_and_raise(e, source="dataset_handler")

    async def import_from_huggingface(self, request: HuggingFaceImportRequest) -> dict:
        """Import dataset from HuggingFace Hub."""
        name = request.name or request.dataset_id.split("/")[-1]
        lock = await self._get_import_lock(name)
        if lock.locked():
            raise_error(f"Import already in progress for '{name}'", "E_INFRA_BUSY")
        async with lock:
            try:
                from domains.training.data_import import HuggingFaceImporter
                importer = HuggingFaceImporter()
                _t0 = time.monotonic()
                result = await asyncio.to_thread(
                    importer.downloadDataset,
                    dataset_id=request.dataset_id,
                    name=name,
                    output_dir=str(self._DATASETS_DIR),
                )
                _elapsed_ms = (time.monotonic() - _t0) * 1000
                if result.success:
                    safe_audit_log("dataset.import", resource=name, detail=f"huggingface elapsed={_elapsed_ms:.0f}ms", dataset_id=request.dataset_id, files=result.files_imported, chars=result.total_chars)
                    return ImportResponse(
                        success=True,
                        dataset_id=name,
                        message=f"Downloaded {result.files_imported} splits ({result.total_chars} chars)",
                        output_path=result.output_path,
                    )
                raise_error(result.error or "Download failed", "E_BAD_REQUEST")
            except HTTPException:
                raise
            except Exception as e:
                logger.warning("Dataset import (huggingface) failed: %s", e)
                classify_and_raise(e, source="dataset_handler")

    async def import_from_url(self, request: URLImportRequest) -> dict:
        """Download and import a dataset from a remote URL.

        Args:
            request: URLImportRequest with url (the file URL to download)
                and name (the dataset name to create).

        Returns:
            ImportResponse with success status, dataset_id, message
            containing character count, and output_path.

        Side effects:
            Downloads the file via urllib, saves it under the datasets
            directory with the given name.
            Uses a per-name lock to prevent concurrent imports of the
            same dataset.
            Logs an audit entry on success.
            Raises 429 if an import is already in progress for this name.
        """
        lock = await self._get_import_lock(request.name)
        if lock.locked():
            raise_error(f"Import already in progress for '{request.name}'", "E_INFRA_BUSY")
        async with lock:
            try:
                from domains.training.data_import import URLImporter
                importer = URLImporter()
                _t0 = time.monotonic()
                result = await asyncio.to_thread(
                    importer.import_from_url,
                    url=request.url,
                    dataset_name=request.name,
                    output_dir=str(self._DATASETS_DIR),
                )
                _elapsed_ms = (time.monotonic() - _t0) * 1000
                if result.success:
                    safe_audit_log("dataset.import", resource=request.name, detail=f"url elapsed={_elapsed_ms:.0f}ms", url=request.url, chars=result.total_chars)
                    return ImportResponse(
                        success=True,
                        dataset_id=request.name,
                        message=f"Downloaded {result.total_chars} chars",
                        output_path=result.output_path,
                    )
                raise_error(result.error or "Download failed", "E_BAD_REQUEST")
            except HTTPException:
                raise
            except Exception as e:
                logger.warning("Dataset import (url) failed: %s", e)
                classify_and_raise(e, source="dataset_handler")

    async def import_from_kaggle(self, request: KaggleImportRequest) -> dict:
        """Download and import a dataset from Kaggle using the Kaggle CLI.

        Args:
            request: KaggleImportRequest with dataset (the Kaggle dataset
                slug in "owner/dataset" format) and an optional name.

        Returns:
            ImportResponse with success status, dataset_id, message
            containing file count and total size in MB, and output_path.

        Side effects:
            Runs the `kaggle datasets download` command as a subprocess.
            Unzips the downloaded archive into the datasets directory.
            Raises 400 if the Kaggle CLI is not installed or the download
            fails.
        """
        import asyncio
        import shutil
        try:
            name = request.name or request.dataset.replace("/", "_")
            output_dir = self._DATASETS_DIR / name
            await asyncio.to_thread(output_dir.mkdir, parents=True, exist_ok=True)

            _t0 = time.monotonic()
            proc = await asyncio.create_subprocess_exec(
                "kaggle", "datasets", "download", "-d", request.dataset, "-p", str(output_dir), "--unzip",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=300)
            _elapsed_ms = (time.monotonic() - _t0) * 1000
            if proc.returncode != 0:
                raise_error(f"Kaggle import failed: {stderr.decode()}", "E_BAD_REQUEST")

            temp_dir = output_dir / request.dataset.replace("/", "_")

            def _organize():
                if temp_dir.exists():
                    for item in temp_dir.iterdir():
                        shutil.move(str(item), str(output_dir / item.name))
                    temp_dir.rmdir()
                file_count = sum(1 for _ in output_dir.rglob("*") if _.is_file())
                total_size = sum(f.stat().st_size for f in output_dir.rglob("*") if f.is_file())
                return file_count, total_size

            file_count, total_size = await asyncio.to_thread(_organize)

            safe_audit_log("dataset.import", resource=name, detail=f"kaggle elapsed={_elapsed_ms:.0f}ms", dataset=request.dataset, files=file_count)
            return ImportResponse(
                success=True,
                dataset_id=name,
                message=f"Downloaded {file_count} files ({total_size / 1024 / 1024:.1f} MB) from Kaggle",
                output_path=str(output_dir),
            )
        except HTTPException:
            raise
        except FileNotFoundError:
            raise_error("Kaggle CLI not found. Install with: pip install kaggle", "E_BAD_REQUEST")
        except Exception as e:
            logger.warning("Dataset import (kaggle) failed: %s", e)
            classify_and_raise(e, source="dataset_handler")

    async def import_from_csv(self, request: CSVImportRequest) -> dict:
        """Import dataset from CSV URL."""
        import csv
        import asyncio
        import urllib.request
        try:
            name = request.name
            output_dir = self._DATASETS_DIR / name
            await asyncio.to_thread(output_dir.mkdir, parents=True, exist_ok=True)

            def _fetch_and_parse():
                req = urllib.request.Request(request.url, headers={"User-Agent": "SloughGPT"})
                with urllib.request.urlopen(req, timeout=30) as resp:
                    content = resp.read().decode(request.encoding or "utf-8")
                lines = content.strip().split("\n")
                if not lines:
                    return None, None, 0
                dialect = csv.Sniffer().sniff(lines[0][:1000], delimiters=",;\t")
                reader = csv.reader(lines, dialect=dialect)
                headers = next(reader)
                rows = list(reader)
                return headers, rows, len(rows)

            headers, rows, row_count = await asyncio.to_thread(_fetch_and_parse)
            if headers is None:
                raise_error("CSV file is empty", "E_BAD_REQUEST")

            jsonl_path = output_dir / f"{name}.jsonl"
            meta_path = output_dir / "metadata.json"

            def _write_csv():
                with open(jsonl_path, "w", encoding="utf-8") as f:
                    for row in rows:
                        obj = {headers[i]: row[i] if i < len(row) else "" for i in range(len(headers))}
                        f.write(json.dumps(obj) + "\n")
                with open(meta_path, "w", encoding="utf-8") as f:
                    json.dump({"source": request.url, "columns": headers, "rows": len(rows)}, f, indent=2)

            await asyncio.to_thread(_write_csv)

            safe_audit_log("dataset.import", resource=name, detail="csv", url=request.url, rows=row_count, columns=len(headers))
            return ImportResponse(
                success=True,
                dataset_id=name,
                message=f"Imported CSV with {row_count} rows, {len(headers)} columns",
                output_path=str(output_dir),
            )
        except HTTPException:
            raise
        except Exception as e:
            logger.warning("Dataset import (csv) failed: %s", e)
            classify_and_raise(e, source="dataset_handler")

    async def batch_import(self, request: BatchImportRequest) -> dict:
        """Import multiple datasets in one request."""
        results = []
        errors = []
        cm_op = None
        _batch_t0 = time.monotonic()
        try:
            from domains.infrastructure.cancel_manager import get_cancel_manager, OpType
            import threading
            _cancel_event = threading.Event()
            _cm = get_cancel_manager()
            cm_op = _cm.register(
                op_type=OpType.IMPORT,
                label=f"batch-import({len(request.sources)} sources)",
                cancel_fn=lambda: _cancel_event.set(),
            )
            _cm.start(cm_op)
        except Exception as e:
            logger.warning("CancelManager registration failed for batch import (may be unkillable): %s", e)

        try:
            for i, source in enumerate(request.sources[:20]):
                name = source.name or f"batch_{i}"
                try:
                    if source.type == "url" and source.url:
                        from domains.training.data_import import URLImporter
                        importer = URLImporter()
                        result = importer.import_from_url(url=source.url, dataset_name=name, output_dir=str(self._DATASETS_DIR))
                    elif source.type == "local" and source.path:
                        importer = self._get_data_importer()
                        result = importer.import_from_local(path=source.path, name=name, extensions=source.extensions)
                    elif source.type == "github" and source.url:
                        from domains.training.data_import import RepoImporter
                        importer = RepoImporter()
                        result = importer.import_from_github(url=source.url, dataset_name=name, output_dir=str(self._DATASETS_DIR))
                    elif source.type == "huggingface" and source.dataset_id:
                        from domains.training.data_import import HuggingFaceImporter
                        importer = HuggingFaceImporter()
                        result = importer.download_dataset(dataset_id=source.dataset_id, name=name, output_dir=str(self._DATASETS_DIR))
                    else:
                        errors.append({"index": i, "error": f"Unsupported source type: {source.type}"})
                        continue

                    if result.success:
                        results.append({"index": i, "name": name, "files": result.files_imported, "chars": result.total_chars})
                    else:
                        errors.append({"index": i, "error": result.error or "Import failed"})
                except Exception as e:
                    errors.append({"index": i, "error": str(e)})

            if cm_op:
                try:
                    from domains.infrastructure.cancel_manager import get_cancel_manager
                    get_cancel_manager().finish(cm_op)
                except Exception as exc:
                    logger.warning("CancelManager.finish failed for batch import: %s", exc)
        except Exception as e:
            if cm_op:
                try:
                    from domains.infrastructure.cancel_manager import get_cancel_manager
                    get_cancel_manager().finish(cm_op, error=str(e))
                except Exception as exc:
                    logger.warning("CancelManager.finish failed for batch import error: %s", exc)

        _batch_elapsed_ms = (time.monotonic() - _batch_t0) * 1000
        safe_audit_log("dataset.import", resource=f"batch({len(request.sources)})", detail=f"batch elapsed={_batch_elapsed_ms:.0f}ms", imported=len(results), errors=len(errors))
        return success_response(data={"imported": len(results), "errors": errors, "elapsed_ms": round(_batch_elapsed_ms, 1)})

    async def search_books(self, q: str = Query(..., description="Search by title or ISBN"), limit: int = Query(10, ge=1, le=50)) -> dict:
        """Search books by title or ISBN via Open Library."""
        try:
            from domains.training.data_import import BooksSearch
            searcher = BooksSearch()
            results = searcher.search(q, limit)
            return success_response(data={"books": results})
        except Exception as e:
            classify_and_raise(e, source="search_books")

    async def search_github(self, q: str = Query(..., description="Search query"), limit: int = Query(10, ge=1, le=50)) -> dict:
        """Search GitHub for repositories matching a query string.

        Args:
            q: Search query string (e.g. "machine learning", "nlp corpus").
            limit: Maximum number of results to return (1-50, default 10).

        Returns:
            Success envelope containing a repos array. Each repo has id,
            name, full_name, description, stars, url, and language fields.

        Side effects:
            Calls the GitHub Search API via GitHubSearch.search_repos().
        """
        try:
            from domains.training.data_import import GitHubSearch
            searcher = GitHubSearch()
            items = searcher.search_repos(q, limit)
            repos = [
                {
                    "id": item["full_name"],
                    "name": item["full_name"].split("/")[1] if "/" in item["full_name"] else item["full_name"],
                    "full_name": item["full_name"],
                    "description": item.get("description"),
                    "stars": item.get("stargazers_count", 0),
                    "url": item.get("html_url", ""),
                    "language": item.get("language"),
                }
                for item in items
            ]
            return success_response(data={"repos": repos})
        except Exception as e:
            classify_and_raise(e, source="search_github")

    async def import_from_isbn(self, request: ISBNImportRequest) -> dict:
        """Import book by ISBN. Fetches full text if available on Project Gutenberg."""
        try:
            from domains.training.data_import import ISBNImporter
            importer = ISBNImporter(output_dir=str(self._DATASETS_DIR))
            _t0 = time.monotonic()
            result = await asyncio.to_thread(
                importer.import_from_isbn,
                isbn=request.isbn,
                name=request.name or f"book_{request.isbn}",
            )
            _elapsed_ms = (time.monotonic() - _t0) * 1000
            if result.success:
                safe_audit_log("dataset.import", resource=result.name or request.name or f"book_{request.isbn}", detail=f"isbn elapsed={_elapsed_ms:.0f}ms", isbn=request.isbn, files=result.files_imported)
                return ImportResponse(
                    success=True,
                    dataset_id=result.name or request.name or f"book_{request.isbn}",
                    message=f"Imported book: {result.files_imported} files ({result.total_chars} chars)"
                    if result.files_imported
                    else "Book metadata saved",
                    output_path=result.output_path,
                )
            raise_error(result.error or "Import failed", "E_BAD_REQUEST")
        except HTTPException:
            raise
        except Exception as e:
            logger.warning("Dataset import (isbn) failed: %s", e)
            classify_and_raise(e, source="dataset_handler")

    async def search_datasets(self, q: str = Query(..., min_length=1, max_length=500, description="Search query")) -> dict:
        """Search datasets by name using a substring or fuzzy match.

        Args:
            q: Search string (1 to 500 characters). Matched against
                dataset names and descriptions.

        Returns:
            Success envelope containing a results array of matching
            dataset summaries and a count of matches.

        Side effects:
            Delegates to DatasetsController.search_datasets which
            performs a case-insensitive substring match.
        """
        ctrl = get_datasets_controller()
        results = ctrl.search_datasets(q)
        return success_response(data={"results": results, "count": len(results)})

    async def get_dataset(self, dataset_id: str) -> dict:
        """Return full details for a single dataset by its ID.

        Args:
            dataset_id: The unique dataset identifier (alphanumeric,
                hyphens, underscores only).

        Returns:
            DatasetInfo with id, name, description, created_at,
            updated_at, row_count, and other metadata fields.

        Side effects:
            Validates dataset_id format (raises 422 on invalid chars).
            Raises 404 if no dataset with the given ID is found.
        """
        self._validate_dataset_id(dataset_id)
        ctrl = get_datasets_controller()
        dataset = ctrl.get_dataset(dataset_id)
        if not dataset:
            raise_error("Dataset not found", "E_NOT_FOUND")
        return DatasetInfo(**dataset)

    async def get_dataset_stats(self, dataset_id: str) -> dict:
        """Return aggregate statistics for a dataset.

        Args:
            dataset_id: The unique dataset identifier (alphanumeric,
                hyphens, underscores only).

        Returns:
            DatasetStats with fields like format, row_count,
            avg_length, total_chars, and import_method.

        Side effects:
            Validates dataset_id format (raises 422 on invalid chars).
            Reads the dataset's input file from disk to compute stats.
            Raises 404 if the dataset is not found or has no data.
        """
        self._validate_dataset_id(dataset_id)
        ctrl = get_datasets_controller()
        stats = ctrl.get_dataset_stats(dataset_id)
        if not stats:
            raise_error("Dataset not found", "E_NOT_FOUND")
        return DatasetStats(**stats)

    async def create_dataset(self, req: DatasetCreate) -> dict:
        """Create a new empty dataset with the given name and description.

        Args:
            req: DatasetCreate with name (used as the directory name and
                ID slug) and an optional description string.

        Returns:
            DatasetInfo with the newly created dataset's id, name,
            description, and timestamp fields.

        Side effects:
            Creates a new directory under the datasets root directory.
            Persists dataset metadata via DatasetsController.
            Logs an audit entry for dataset creation.
        """
        ctrl = get_datasets_controller()
        dataset = ctrl.create_dataset(req.name, req.description)
        safe_audit_log("dataset.create", resource=req.name, detail=req.description or "")
        return DatasetInfo(**dataset)

    async def update_dataset(self, dataset_id: str, req: DatasetUpdate) -> dict:
        """Update a dataset's metadata fields (name, description, etc).

        Args:
            dataset_id: The unique dataset identifier to update.
            req: DatasetUpdate with optional name and description fields.
                Only non-None fields are applied.

        Returns:
            DatasetInfo with the updated metadata.

        Side effects:
            Validates dataset_id format (raises 422 on invalid chars).
            Modifies the dataset's metadata.json on disk.
            Logs an audit entry for the update.
            Raises 404 if the dataset is not found.
        """
        self._validate_dataset_id(dataset_id)
        ctrl = get_datasets_controller()
        dataset = ctrl.update_dataset(dataset_id, req.model_dump(exclude_none=True))
        if not dataset:
            raise_error("Dataset not found", "E_NOT_FOUND")
        safe_audit_log("dataset.update", resource=dataset_id, detail=str(req.model_dump(exclude_none=True)))
        return DatasetInfo(**dataset)

    async def delete_dataset(self, dataset_id: str) -> dict:
        """Delete a dataset and all its files from disk.

        Args:
            dataset_id: The unique dataset identifier to delete.

        Returns:
            Success response with status "deleted" and the dataset_id.

        Side effects:
            Validates dataset_id format (raises 422 on invalid chars).
            Removes the dataset directory and all contained files.
            Logs an audit entry for the deletion.
            Raises 404 if the dataset is not found.
        """
        self._validate_dataset_id(dataset_id)
        ctrl = get_datasets_controller()
        ok = ctrl.delete_dataset(dataset_id)
        if not ok:
            raise_error("Dataset not found", "E_NOT_FOUND")
        safe_audit_log("dataset.delete", resource=dataset_id)
        return success_response(data={"status": "deleted", "dataset_id": dataset_id})

    async def create_version(self, dataset_id: str) -> dict:
        """Create a timestamped snapshot of a dataset."""
        self._validate_dataset_id(dataset_id)
        ctrl = get_datasets_controller()
        timestamp = ctrl.create_version_snapshot(dataset_id)
        if not timestamp:
            raise_error("Dataset not found", "E_NOT_FOUND")
        safe_audit_log("dataset.version", resource=dataset_id, detail=str(timestamp))
        return VersionCreateResponse(timestamp=timestamp, message="Version created")

    async def list_versions(self, dataset_id: str) -> dict:
        """List all version timestamps for a dataset."""
        self._validate_dataset_id(dataset_id)
        ctrl = get_datasets_controller()
        versions = ctrl.list_versions(dataset_id)
        return VersionListResponse(versions=versions, count=len(versions))

    async def restore_version(self, dataset_id: str, timestamp: str) -> dict:
        """Restore a dataset to a specific version snapshot."""
        self._validate_dataset_id(dataset_id)
        ctrl = get_datasets_controller()
        ok = ctrl.restore_version(dataset_id, timestamp)
        if not ok:
            raise_error("Dataset or version not found", "E_NOT_FOUND")
        safe_audit_log("dataset.version.restore", resource=dataset_id, detail=timestamp)
        return VersionRestoreResponse(success=True, message="Version restored")

    async def add_dataset_data(self, dataset_id: str, req: DatasetDataRequest) -> dict:
        """Append data rows to an existing dataset's input file.

        Args:
            dataset_id: The unique dataset identifier to append to.
            req: DatasetDataRequest with data (list of string rows to
                append to the dataset's input file).

        Returns:
            Success response with status "appended" and rows_added count.

        Side effects:
            Validates dataset_id format (raises 422 on invalid chars).
            Appends rows to the dataset's input.jsonl file on disk.
            Logs an audit entry with the row count.
            Raises 404 if the dataset is not found.
        """
        self._validate_dataset_id(dataset_id)
        ctrl = get_datasets_controller()
        result = ctrl.add_data(dataset_id, req.data)
        if result is None:
            raise_error("Dataset not found", "E_NOT_FOUND")
        safe_audit_log("dataset.data.append", resource=dataset_id, detail=f"rows={result}")
        return success_response(data={"status": "appended", "rows_added": result})

    async def preview_dataset(self, dataset_id: str, limit: int = Query(10, ge=1, le=1000, description="Number of samples")):
        """Return the first N rows of a dataset for preview.

        Args:
            dataset_id: The unique dataset identifier to preview.
            limit: Number of rows to return (1-1000, default 10).

        Returns:
            Preview data from the dataset's input file (format depends
            on the dataset's file type).

        Side effects:
            Validates dataset_id format (raises 422 on invalid chars).
            Reads up to limit rows from the dataset file on disk.
            Raises 404 if the dataset is not found or empty.
        """
        self._validate_dataset_id(dataset_id)
        ctrl = get_datasets_controller()
        preview = await asyncio.to_thread(ctrl.preview_dataset, dataset_id, limit)
        if not preview:
            raise_error("Dataset not found or empty", "E_NOT_FOUND")
        return preview

    async def export_dataset(self, dataset_id: str, request: DatasetExportRequest) -> dict:
        """Export a dataset as a downloadable file in the requested format.

        Args:
            dataset_id: The unique dataset identifier to export.
            request: DatasetExportRequest with format (e.g. "jsonl",
                "csv", "txt") specifying the output file format.

        Returns:
            FileResponse with the exported file, Content-Disposition
            header set to the dataset filename, and media type
            application/octet-stream.

        Side effects:
            Validates dataset_id format (raises 422 on invalid chars).
            Reads the dataset from disk and writes the exported file.
            Raises 404 if the dataset is not found or empty.
        """
        self._validate_dataset_id(dataset_id)
        ctrl = get_datasets_controller()
        export_path = await asyncio.to_thread(ctrl.export_dataset, dataset_id, request.format)
        if not export_path:
            raise_error("Dataset not found or empty", "E_NOT_FOUND")
        return FileResponse(
            path=str(export_path),
            filename=f"{dataset_id}.{request.format}",
            media_type="application/octet-stream",
        )

    async def create_dataset_from_chat(self, req: FromChatRequest) -> dict:
        """Create a training dataset from a chat conversation.

        Accepts validated { messages: [{role, content}...], name?: string }.
        Saves as JSONL in the datasets directory and returns the dataset ID.
        """
        messages = req.messages
        name = req.name

        if not messages:
            raise_error("No messages provided", "E_BAD_REQUEST")

        ctrl = get_datasets_controller()
        dataset = ctrl.create_dataset(name, description=f"Exported from chat ({len(messages)} messages)")

        dataset_dir = self._DATASETS_DIR / dataset["id"]
        await asyncio.to_thread(dataset_dir.mkdir, parents=True, exist_ok=True)

        jsonl_path = dataset_dir / "input.jsonl"

        def _write_chat():
            with open(jsonl_path, "w") as f:
                for msg in messages:
                    if msg.role in ("user", "assistant") and msg.content:
                        f.write(json.dumps({"messages": [{"role": msg.role, "content": msg.content}]}) + "\n")

        await asyncio.to_thread(_write_chat)

        safe_audit_log("dataset.create", resource=name, detail=f"from-chat ({len(messages)} messages)", messages=len(messages))
        return success_response(data={
            "status": "created",
            "dataset_id": dataset["id"],
            "name": name,
            "messages_exported": len([m for m in messages if m.role in ("user", "assistant") and m.content]),
        })

    async def convert_to_messages(self, dataset_id: str, system_prompt: str = "You are a helpful assistant.") -> dict:
        """Convert a dataset to chat message format for fine-tuning.

        Reads the dataset's input.jsonl, wraps each entry in a
        system/user/assistant message structure, and saves as a new dataset.
        """
        self._validate_dataset_id(dataset_id)
        ctrl = get_datasets_controller()
        datasets = ctrl.list_datasets()
        source = None
        for ds in datasets:
            if ds["id"] == dataset_id:
                source = ds
                break
        if not source:
            raise_error(f"Dataset {dataset_id} not found", "E_NOT_FOUND")

        source_dir = self._DATASETS_DIR / dataset_id
        jsonl_path = source_dir / "input.jsonl"
        if not await asyncio.to_thread(jsonl_path.exists):
            raise_error("Dataset has no input.jsonl", "E_NOT_FOUND")

        def _read_source():
            msgs = []
            with open(jsonl_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if "text" in row:
                        msgs.append({"messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": row["text"]},
                            {"role": "assistant", "content": row["text"]},
                        ]})
                    elif "messages" in row:
                        m = row["messages"]
                        if m and m[0].get("role") != "system":
                            m = [{"role": "system", "content": system_prompt}] + m
                        msgs.append({"messages": m})
            return msgs

        messages_out = await asyncio.to_thread(_read_source)

        new_ds = ctrl.create_dataset(
            name=f"{source['name']}-messages",
            description=f"Converted from {source['name']} ({len(messages_out)} conversations)",
        )
        new_dir = self._DATASETS_DIR / new_ds["id"]
        await asyncio.to_thread(new_dir.mkdir, parents=True, exist_ok=True)
        out_path = new_dir / "input.jsonl"

        def _write_output():
            with open(out_path, "w") as f:
                for entry in messages_out:
                    f.write(json.dumps(entry) + "\n")

        await asyncio.to_thread(_write_output)

        safe_audit_log("dataset.convert", resource=dataset_id, detail=str(new_ds["id"]), conversations=len(messages_out))
        return success_response(data={
            "status": "converted",
            "new_dataset_id": new_ds["id"],
            "total_conversations": len(messages_out),
        })


router = DatasetsRouter().router
