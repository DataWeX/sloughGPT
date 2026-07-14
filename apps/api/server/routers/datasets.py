"""
Datasets Router - MVC View layer
"""
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from pathlib import Path
from typing import Optional, List
import json

from schemas.datasets import (
    DatasetInfo, DatasetCreate, DatasetUpdate, DatasetDataRequest,
    DatasetStats, DatasetListResponse,
    GitHubImportRequest, HuggingFaceImportRequest, URLImportRequest,
    LocalImportRequest, KaggleImportRequest, CSVImportRequest,
    BatchImportRequest, ISBNImportRequest, ImportResponse,
    VersionCreateResponse, VersionListResponse, VersionRestoreResponse,
)
from schemas.common import success_response
from controllers.datasets import get_datasets_controller

router = APIRouter(prefix="/datasets", tags=["datasets"])

_DATASETS_DIR = Path(__file__).resolve().parents[4] / "datasets"


def _get_data_importer():
    """Get DataImporter configured to save to the repo datasets directory."""
    from domains.training.data_import import DataImporter
    return DataImporter(output_dir=str(_DATASETS_DIR))


@router.get("", response_model=DatasetListResponse)
async def list_datasets(
    q: Optional[str] = Query(None, description="Search query"),
    type: Optional[str] = Query(None, description="Filter by type"),
):
    """List available datasets"""
    ctrl = get_datasets_controller()
    datasets = ctrl.list_datasets(q, type)
    return DatasetListResponse(
        datasets=[DatasetInfo(**d) for d in datasets],
        count=len(datasets),
    )


@router.post("/import/local", response_model=ImportResponse)
async def import_from_local(request: LocalImportRequest):
    """Import dataset from local file or directory."""
    try:
        importer = _get_data_importer()
        result = importer.import_from_local(
            path=request.path,
            name=request.name,
            extensions=request.extensions or None,
        )
        if result.success:
            return ImportResponse(
                success=True,
                dataset_id=request.name,
                message=f"Imported {result.files_imported} files ({result.total_chars} chars)",
                output_path=result.output_path,
            )
        raise HTTPException(status_code=400, detail=result.error or "Import failed")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/import/github", response_model=ImportResponse)
async def import_from_github(request: GitHubImportRequest):
    """Import dataset from GitHub repository."""
    try:
        from domains.training.data_import import RepoImporter
        importer = RepoImporter()
        result = importer.import_from_github(
            url=request.url,
            dataset_name=request.name,
            output_dir=str(_DATASETS_DIR),
            extensions=request.extensions or None,
            max_files=request.max_files,
        )
        if result.success:
            return ImportResponse(
                success=True,
                dataset_id=result.name or request.name,
                message=f"Imported {result.files_imported} files ({result.total_chars} chars)",
                output_path=result.output_path,
            )
        raise HTTPException(status_code=400, detail=result.error or "Import failed")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/import/huggingface", response_model=ImportResponse)
async def import_from_huggingface(request: HuggingFaceImportRequest):
    """Import dataset from HuggingFace Hub."""
    try:
        from domains.training.data_import import HuggingFaceImporter
        importer = HuggingFaceImporter()
        name = request.name or request.dataset_id.split("/")[-1]
        result = importer.download_dataset(
            dataset_id=request.dataset_id,
            name=name,
            output_dir=str(_DATASETS_DIR),
        )
        if result.success:
            return ImportResponse(
                success=True,
                dataset_id=name,
                message=f"Downloaded {result.files_imported} splits ({result.total_chars} chars)",
                output_path=result.output_path,
            )
        raise HTTPException(status_code=400, detail=result.error or "Download failed")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/import/url", response_model=ImportResponse)
async def import_from_url(request: URLImportRequest):
    """Import dataset from URL."""
    try:
        from domains.training.data_import import URLImporter
        importer = URLImporter()
        result = importer.import_from_url(
            url=request.url,
            dataset_name=request.name,
            output_dir=str(_DATASETS_DIR),
        )
        if result.success:
            return ImportResponse(
                success=True,
                dataset_id=request.name,
                message=f"Downloaded {result.total_chars} chars",
                output_path=result.output_path,
            )
        raise HTTPException(status_code=400, detail=result.error or "Download failed")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/import/kaggle", response_model=ImportResponse)
async def import_from_kaggle(request: KaggleImportRequest):
    """Import dataset from Kaggle."""
    import asyncio
    import shutil
    try:
        name = request.name or request.dataset.replace("/", "_")
        output_dir = _DATASETS_DIR / name
        output_dir.mkdir(parents=True, exist_ok=True)

        proc = await asyncio.create_subprocess_exec(
            "kaggle", "datasets", "download", "-d", request.dataset, "-p", str(output_dir), "--unzip",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=300)
        if proc.returncode != 0:
            raise HTTPException(status_code=400, detail=f"Kaggle import failed: {stderr.decode()}")

        temp_dir = output_dir / request.dataset.replace("/", "_")
        if temp_dir.exists():
            for item in temp_dir.iterdir():
                shutil.move(str(item), str(output_dir / item.name))
            temp_dir.rmdir()

        file_count = sum(1 for _ in output_dir.rglob("*") if _.is_file())
        total_size = sum(f.stat().st_size for f in output_dir.rglob("*") if f.is_file())

        return ImportResponse(
            success=True,
            dataset_id=name,
            message=f"Downloaded {file_count} files ({total_size / 1024 / 1024:.1f} MB) from Kaggle",
            output_path=str(output_dir),
        )
    except HTTPException:
        raise
    except FileNotFoundError:
        raise HTTPException(status_code=400, detail="Kaggle CLI not found. Install with: pip install kaggle")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/import/csv", response_model=ImportResponse)
async def import_from_csv(request: CSVImportRequest):
    """Import dataset from CSV URL."""
    import csv
    import asyncio
    import urllib.request
    try:
        name = request.name
        output_dir = _DATASETS_DIR / name
        output_dir.mkdir(parents=True, exist_ok=True)

        req = urllib.request.Request(request.url, headers={"User-Agent": "SloughGPT"})
        raw = await asyncio.to_thread(urllib.request.urlopen, req, 30)
        content = raw.read().decode(request.encoding or "utf-8")

        lines = content.strip().split("\n")
        if not lines:
            raise HTTPException(status_code=400, detail="CSV file is empty")

        dialect = csv.Sniffer().sniff(lines[0][:1000], delimiters=",;\t")
        reader = csv.reader(lines, dialect=dialect)
        headers = next(reader)
        rows = list(reader)

        jsonl_path = output_dir / f"{name}.jsonl"
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for row in rows:
                obj = {headers[i]: row[i] if i < len(row) else "" for i in range(len(headers))}
                f.write(json.dumps(obj) + "\n")

        meta_path = output_dir / "metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({"source": request.url, "columns": headers, "rows": len(rows)}, f, indent=2)

        return ImportResponse(
            success=True,
            dataset_id=name,
            message=f"Imported CSV with {len(rows)} rows, {len(headers)} columns",
            output_path=str(output_dir),
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/import/batch")
async def batch_import(request: BatchImportRequest):
    """Import multiple datasets in one request."""
    results = []
    errors = []

    for i, source in enumerate(request.sources[:20]):
        name = source.name or f"batch_{i}"
        try:
            if source.type == "url" and source.url:
                from domains.training.data_import import URLImporter
                importer = URLImporter()
                result = importer.import_from_url(url=source.url, dataset_name=name, output_dir=str(_DATASETS_DIR))
            elif source.type == "local" and source.path:
                importer = _get_data_importer()
                result = importer.import_from_local(path=source.path, name=name, extensions=source.extensions)
            elif source.type == "github" and source.url:
                from domains.training.data_import import RepoImporter
                importer = RepoImporter()
                result = importer.import_from_github(url=source.url, dataset_name=name, output_dir=str(_DATASETS_DIR))
            elif source.type == "huggingface" and source.dataset_id:
                from domains.training.data_import import HuggingFaceImporter
                importer = HuggingFaceImporter()
                result = importer.download_dataset(dataset_id=source.dataset_id, name=name, output_dir=str(_DATASETS_DIR))
            else:
                errors.append({"index": i, "error": f"Unsupported source type: {source.type}"})
                continue

            if result.success:
                results.append({"index": i, "name": name, "files": result.files_imported, "chars": result.total_chars})
            else:
                errors.append({"index": i, "error": result.error or "Import failed"})
        except Exception as e:
            errors.append({"index": i, "error": str(e)})

    return success_response(data={"imported": len(results), "errors": errors})


@router.get("/search/books")
async def search_books(q: str = Query(..., description="Search by title or ISBN"), limit: int = Query(10, ge=1, le=50)):
    """Search books by title or ISBN via Open Library."""
    from domains.training.data_import import BooksSearch
    searcher = BooksSearch()
    results = searcher.search(q, limit)
    return success_response(data={"books": results})


@router.get("/search/github")
async def search_github(q: str = Query(..., description="Search query"), limit: int = Query(10, ge=1, le=50)):
    """Search GitHub repositories."""
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


@router.post("/import/isbn", response_model=ImportResponse)
async def import_from_isbn(request: ISBNImportRequest):
    """Import book by ISBN. Fetches full text if available on Project Gutenberg."""
    try:
        from domains.training.data_import import ISBNImporter
        importer = ISBNImporter(output_dir=str(_DATASETS_DIR))
        result = importer.import_from_isbn(
            isbn=request.isbn,
            name=request.name or f"book_{request.isbn}",
        )
        if result.success:
            return ImportResponse(
                success=True,
                dataset_id=result.name or request.name or f"book_{request.isbn}",
                message=f"Imported book: {result.files_imported} files ({result.total_chars} chars)"
                if result.files_imported
                else "Book metadata saved",
                output_path=result.output_path,
            )
        raise HTTPException(status_code=400, detail=result.error or "Import failed")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/search")
async def search_datasets(q: str = Query(..., description="Search query")):
    """Search datasets by name"""
    ctrl = get_datasets_controller()
    results = ctrl.search_datasets(q)
    return success_response(data={"results": results, "count": len(results)})


@router.get("/{dataset_id}", response_model=DatasetInfo)
async def get_dataset(dataset_id: str):
    """Get dataset details"""
    ctrl = get_datasets_controller()
    dataset = ctrl.get_dataset(dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return DatasetInfo(**dataset)


@router.get("/{dataset_id}/stats", response_model=DatasetStats)
async def get_dataset_stats(dataset_id: str):
    """Get dataset statistics"""
    ctrl = get_datasets_controller()
    stats = ctrl.get_dataset_stats(dataset_id)
    if not stats:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return DatasetStats(**stats)


@router.post("", response_model=DatasetInfo)
async def create_dataset(req: DatasetCreate):
    """Create a new dataset"""
    ctrl = get_datasets_controller()
    dataset = ctrl.create_dataset(req.name, req.description)
    return DatasetInfo(**dataset)


@router.patch("/{dataset_id}", response_model=DatasetInfo)
async def update_dataset(dataset_id: str, req: DatasetUpdate):
    """Update dataset metadata"""
    ctrl = get_datasets_controller()
    dataset = ctrl.update_dataset(dataset_id, req.model_dump(exclude_none=True))
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return DatasetInfo(**dataset)


@router.delete("/{dataset_id}")
async def delete_dataset(dataset_id: str):
    """Delete a dataset"""
    ctrl = get_datasets_controller()
    ok = ctrl.delete_dataset(dataset_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return success_response(data={"status": "deleted", "dataset_id": dataset_id})


@router.post("/{dataset_id}/versions", response_model=VersionCreateResponse)
async def create_version(dataset_id: str):
    """Create a timestamped snapshot of a dataset."""
    ctrl = get_datasets_controller()
    timestamp = ctrl.create_version_snapshot(dataset_id)
    if not timestamp:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return VersionCreateResponse(timestamp=timestamp, message="Version created")


@router.get("/{dataset_id}/versions", response_model=VersionListResponse)
async def list_versions(dataset_id: str):
    """List all version timestamps for a dataset."""
    ctrl = get_datasets_controller()
    versions = ctrl.list_versions(dataset_id)
    return VersionListResponse(versions=versions, count=len(versions))


@router.post("/{dataset_id}/versions/{timestamp}", response_model=VersionRestoreResponse)
async def restore_version(dataset_id: str, timestamp: str):
    """Restore a dataset to a specific version snapshot."""
    ctrl = get_datasets_controller()
    ok = ctrl.restore_version(dataset_id, timestamp)
    if not ok:
        raise HTTPException(status_code=404, detail="Dataset or version not found")
    return VersionRestoreResponse(success=True, message="Version restored")




@router.post("/{dataset_id}/data")
async def add_dataset_data(dataset_id: str, req: DatasetDataRequest):
    """Add data rows to a dataset"""
    ctrl = get_datasets_controller()
    result = ctrl.add_data(dataset_id, req.data)
    if result is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return success_response(data={"status": "appended", "rows_added": result})



@router.get("/{dataset_id}/preview")
async def preview_dataset(dataset_id: str, limit: int = Query(10, description="Number of samples")):
    """Preview dataset contents (first N rows)"""
    ctrl = get_datasets_controller()
    preview = ctrl.preview_dataset(dataset_id, limit)
    if not preview:
        raise HTTPException(status_code=404, detail="Dataset not found or empty")
    return preview


@router.post("/{dataset_id}/export")
async def export_dataset(dataset_id: str, format: str = Query("jsonl", description="Export format")):
    """Export a dataset as a downloadable file"""
    ctrl = get_datasets_controller()
    export_path = ctrl.export_dataset(dataset_id, format)
    if not export_path:
        raise HTTPException(status_code=404, detail="Dataset not found or empty")
    return FileResponse(
        path=str(export_path),
        filename=f"{dataset_id}.{format}",
        media_type="application/octet-stream",
    )


@router.post("/from-chat")
async def create_dataset_from_chat(req: dict):
    """Create a training dataset from a chat conversation.

    Accepts { messages: [{role, content}...], name?: string }.
    Saves as JSONL in the datasets directory and returns the dataset ID.
    """
    messages = req.get("messages", [])
    name = req.get("name", "chat-export")

    if not messages:
        raise HTTPException(status_code=400, detail="No messages provided")

    ctrl = get_datasets_controller()
    dataset = ctrl.create_dataset(name, description=f"Exported from chat ({len(messages)} messages)")

    dataset_dir = _DATASETS_DIR / dataset["id"]
    dataset_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = dataset_dir / "input.jsonl"
    with open(jsonl_path, "w") as f:
        for msg in messages:
            if msg.get("role") in ("user", "assistant") and msg.get("content"):
                f.write(json.dumps({"messages": [{"role": msg["role"], "content": msg["content"]}]}) + "\n")

    return {
        "status": "created",
        "dataset_id": dataset["id"],
        "name": name,
        "messages_exported": len([m for m in messages if m.get("role") in ("user", "assistant") and m.get("content")]),
    }


@router.post("/convert-to-messages")
async def convert_to_messages(dataset_id: str, system_prompt: str = "You are a helpful assistant."):
    """Convert a dataset to chat message format for fine-tuning.

    Reads the dataset's input.jsonl, wraps each entry in a
    system/user/assistant message structure, and saves as a new dataset.
    """
    ctrl = get_datasets_controller()
    datasets = ctrl.list_datasets()
    source = None
    for ds in datasets:
        if ds["id"] == dataset_id:
            source = ds
            break
    if not source:
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")

    source_dir = _DATASETS_DIR / dataset_id
    jsonl_path = source_dir / "input.jsonl"
    if not jsonl_path.exists():
        raise HTTPException(status_code=404, detail="Dataset has no input.jsonl")

    messages_out = []
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
                messages_out.append({"messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": row["text"]},
                    {"role": "assistant", "content": row["text"]},
                ]})
            elif "messages" in row:
                msgs = row["messages"]
                if msgs and msgs[0].get("role") != "system":
                    msgs = [{"role": "system", "content": system_prompt}] + msgs
                messages_out.append({"messages": msgs})

    new_ds = ctrl.create_dataset(
        name=f"{source['name']}-messages",
        description=f"Converted from {source['name']} ({len(messages_out)} conversations)",
    )
    new_dir = _DATASETS_DIR / new_ds["id"]
    new_dir.mkdir(parents=True, exist_ok=True)
    out_path = new_dir / "input.jsonl"
    with open(out_path, "w") as f:
        for entry in messages_out:
            f.write(json.dumps(entry) + "\n")

    return {
        "status": "converted",
        "new_dataset_id": new_ds["id"],
        "total_conversations": len(messages_out),
    }
