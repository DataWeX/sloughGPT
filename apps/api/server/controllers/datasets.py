"""
Datasets Controller - Business logic for dataset management
"""
from typing import Optional, List, Dict, Any
from pathlib import Path
import json
import shutil


class DatasetsController:
    """Controller for dataset management"""

    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self.data_dir = repo_root / "data" / "features"
        self.datasets_dir = repo_root / "data"

    def list_datasets(self, q: Optional[str] = None, dataset_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """List available datasets"""
        # Use datasets/ directory primarily
        datasets_dir = self.datasets_dir
        if not datasets_dir.exists():
            datasets_dir = self.data_dir
        if not datasets_dir.exists():
            return []

        datasets = []
        for d in datasets_dir.iterdir():
            if not d.is_dir():
                continue

            input_file = d / "input.txt"
            corpus_file = d / "corpus.jsonl"

            has_corpus = corpus_file.exists()
            size = corpus_file.stat().st_size if has_corpus else (input_file.stat().st_size if input_file.exists() else 0)
            num_samples = 0
            if has_corpus and size < 1_000_000:
                try:
                    with open(corpus_file) as f:
                        num_samples = sum(1 for _ in f)
                except Exception:
                    pass

            # Detect Visual dataset from metadata marker
            visual_meta_path = d / ".visual_metadata.json"
            if visual_meta_path.exists():
                try:
                    json.loads(visual_meta_path.read_text())
                    dataset_type = "visual"
                except Exception:
                    dataset_type = "corpus" if has_corpus else "text"
            else:
                dataset_type = "corpus" if has_corpus else "text"

            dataset = {
                "id": d.name,
                "name": d.name.replace("_", " ").title(),
                "path": str(d),
                "type": dataset_type,
                "size_bytes": size,
                "size_formatted": f"{size / 1024:.1f} KB" if size > 0 else "Empty",
                "size": size,
                "num_samples": num_samples,
                "samples": num_samples,
                "description": self._describe_dataset(d, [], size) if (corpus_file.exists() or input_file.exists()) else "",
            }

            # Attach Visual metadata if present
            if visual_meta_path.exists() and dataset_type == "visual":
                try:
                    dataset["visual_metadata"] = json.loads(visual_meta_path.read_text())
                except Exception:
                    pass

            # Filters
            if q and q.lower() not in d.name.lower() and q.lower() not in dataset["name"].lower():
                continue
            if dataset_type and dataset["type"] != dataset_type:
                continue

            datasets.append(dataset)

        return datasets

    def get_dataset(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        """Get dataset details"""
        path = self.datasets_dir / dataset_id
        if not path.exists():
            return None

        return {
            "id": dataset_id,
            "name": path.name,
            "path": str(path),
            "exists": True,
        }

    def get_dataset_stats(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        """Get dataset statistics matching the frontend DatasetStats interface."""
        path = self.datasets_dir / dataset_id
        if not path.exists():
            return None

        files = list(path.glob("*.jsonl"))
        total_size = sum(f.stat().st_size for f in files)

        input_file = path / "input.txt"
        if input_file.exists() and not files:
            total_size = input_file.stat().st_size
            files = [input_file]

        sample_file = path / "corpus.jsonl"
        if not sample_file.exists():
            sample_file = path / "input.txt"

        lines_list: list[str] = []
        total_chars = 0
        is_jsonl = False
        is_messages = False
        has_dialogue = False
        dialogue_markers = ["user:", "assistant:", "human:", "<|user|>", "<|assistant|>"]
        sample_preview: list[str] = []

        if sample_file.exists():
            try:
                with open(sample_file, "r", encoding="utf-8", errors="replace") as f:
                    raw = f.read()
                lines_list = [l.strip() for l in raw.split("\n") if l.strip()]
                total_chars = len(raw)
                sample_preview = lines_list[:5]

                for line in lines_list[:20]:
                    if line.startswith("{"):
                        is_jsonl = True
                        try:
                            import json
                            obj = json.loads(line)
                            if "messages" in obj or "conversations" in obj:
                                is_messages = True
                        except Exception:
                            pass
                    lower = line.lower()
                    if any(lower.startswith(m) or f" {m}" in lower for m in dialogue_markers):
                        has_dialogue = True
            except Exception:
                pass

        num_lines = len(lines_list)
        avg_length = (total_chars / num_lines) if num_lines > 0 else 0

        if is_messages:
            fmt = "messages"
            suggested = "distill"
        elif is_jsonl:
            fmt = "jsonl"
            suggested = "finetune"
        elif has_dialogue:
            fmt = "dialogue"
            suggested = "distill"
        else:
            fmt = "text"
            suggested = "distill"

        if total_size > 1024 * 1024:
            suggested = "finetune"

        return {
            "format": fmt,
            "samples": num_lines,
            "chars": total_chars,
            "avg_length": avg_length,
            "has_messages": is_messages,
            "sample_preview": sample_preview,
            "lines": num_lines,
            "suggested_method": suggested,
            "file_type": "jsonl" if is_jsonl else "txt",
        }

    def _describe_dataset(self, path: Path, files: list, total_size: int) -> str:
        """Generate a plain-language description of a dataset."""
        size_kb = total_size / 1024
        if size_kb < 1:
            size_str = f"{total_size} bytes"
        elif size_kb < 1024:
            size_str = f"{size_kb:.0f} KB"
        else:
            size_str = f"{size_kb / 1024:.1f} MB"

        # Try to read a sample to detect format and content
        sample_file = path / "corpus.jsonl"
        if not sample_file.exists():
            sample_file = path / "input.txt"
        if not sample_file.exists():
            return f"Dataset with {size_str} of data."

        try:
            with open(sample_file, "r", encoding="utf-8", errors="replace") as f:
                sample = f.read(2000)
        except Exception:
            return f"Dataset with {size_str} of data."

        lines = [l.strip() for l in sample.split("\n") if l.strip()]
        word_count = len(sample.split())

        # Detect format
        is_messages = False
        has_dialogue = False
        dialogue_markers = ["user:", "assistant:", "human:", "<|user|>", "<|assistant|>"]

        for line in lines[:10]:
            if line.startswith("{"):
                try:
                    import json
                    obj = json.loads(line)
                    if "messages" in obj or "conversations" in obj:
                        is_messages = True
                except Exception:
                    pass
            lower = line.lower()
            if any(lower.startswith(m) or f" {m}" in lower for m in dialogue_markers):
                has_dialogue = True

        parts = []
        if is_messages:
            parts.append(f"Conversational data with {len(lines)} turns")
        elif has_dialogue:
            parts.append(f"Dialogue text with {word_count} words")
        else:
            parts.append(f"Text with {word_count} words")

        parts.append(f"({size_str})")

        if word_count < 1000:
            parts.append("— small dataset, good for quick experiments")
        elif word_count > 100000:
            parts.append("— large dataset, training will take longer but learn more")

        return " ".join(parts) + "."

    def search_datasets(self, q: str) -> List[Dict[str, Any]]:
        """Search datasets by name — returns full dataset summaries.

        Args:
            q: case-insensitive substring matched against the directory name
               and the humanized title.

        Returns:
            List of dataset summary dicts (same shape as ``list_datasets``)
            whose name or title contains ``q``.

        Side effects:
            - reads dataset directories to build summaries
        """
        return self.list_datasets(q=q)

    def create_dataset(self, name: str, description: Optional[str] = None) -> Dict[str, Any]:
        """Create a new dataset"""
        path = self.datasets_dir / name
        path.mkdir(parents=True, exist_ok=True)

        return {
            "id": name,
            "name": name,
            "description": description,
            "created": True,
            "path": str(path),
        }

    def update_dataset(self, dataset_id: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Update dataset metadata. Renames directory if name changes, persists description."""
        path = self.datasets_dir / dataset_id
        if not path.exists():
            return None

        new_name = updates.get("name")
        new_desc = updates.get("description")

        # Rename directory if name changed
        if new_name and new_name != dataset_id:
            new_path = self.datasets_dir / new_name
            if new_path.exists():
                return None  # target name already taken
            path.rename(new_path)
            dataset_id = new_name
            path = new_path

        # Persist description to metadata file
        if new_desc is not None:
            meta_path = path / ".metadata.json"
            meta = {}
            if meta_path.exists():
                try:
                    import json as _json
                    with open(meta_path, "r") as f:
                        meta = _json.load(f)
                except Exception:
                    pass
            meta["description"] = new_desc
            import json as _json
            with open(meta_path, "w") as f:
                _json.dump(meta, f, indent=2)

        return {
            "id": dataset_id,
            "name": dataset_id,
            "description": new_desc or "",
            "path": str(path),
        }

    def delete_dataset(self, dataset_id: str) -> bool:
        """Delete a dataset directory"""
        path = self.datasets_dir / dataset_id
        if not path.exists():
            return False
        shutil.rmtree(path)
        return True

    def add_data(self, dataset_id: str, data: List[str]) -> Optional[int]:
        """Append data rows to a dataset's corpus file"""
        # Existing implementation unchanged (kept for context)
        path = self.datasets_dir / dataset_id
        if not path.exists():
            return None
        corpus_file = path / "corpus.jsonl"
        count = 0
        with open(corpus_file, "a") as f:
            for line in data:
                f.write(json.dumps({"text": line}) + "\n")
                count += 1
        return count

    # --- Versioning helpers -------------------------------------------------
    def _ensure_versions_dir(self, dataset_id: str) -> Path:
        """Return the versions directory for a dataset, creating it if needed."""
        versions_dir = self.datasets_dir / dataset_id / "versions"
        versions_dir.mkdir(parents=True, exist_ok=True)
        return versions_dir

    def create_version_snapshot(self, dataset_id: str) -> Optional[str]:
        """Create a timestamped snapshot of the current dataset files.

        Returns the version name (timestamp) or None if the dataset does not exist.
        """
        from datetime import datetime, timezone
        path = self.datasets_dir / dataset_id
        if not path.exists():
            return None
        versions_dir = self._ensure_versions_dir(dataset_id)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        version_path = versions_dir / timestamp
        version_path.mkdir(parents=True, exist_ok=True)
        # Copy relevant files (corpus.jsonl or input.txt) into the version folder
        for fname in ["corpus.jsonl", "input.txt"]:
            src = path / fname
            if src.exists():
                shutil.copy2(src, version_path / fname)
        return timestamp

    def list_versions(self, dataset_id: str) -> List[str]:
        """List all version timestamps for a dataset, newest first."""
        versions_dir = self._ensure_versions_dir(dataset_id)
        if not versions_dir.exists():
            return []
        versions = [p.name for p in sorted(versions_dir.iterdir(), reverse=True) if p.is_dir()]
        return versions

    def restore_version(self, dataset_id: str, version: str) -> bool:
        """Restore a specific version snapshot to the main dataset directory.

        Returns True on success, False if the version or dataset does not exist.
        """
        path = self.datasets_dir / dataset_id
        version_path = self._ensure_versions_dir(dataset_id) / version
        if not path.exists() or not version_path.exists():
            return False
        # Overwrite main corpus/input files with the snapshot copies
        for fname in ["corpus.jsonl", "input.txt"]:
            src = version_path / fname
            if src.exists():
                shutil.copy2(src, path / fname)
        return True

    def preview_dataset(self, dataset_id: str, limit: int = 10) -> Optional[dict]:
        """Return a preview of dataset contents (first N rows)."""
        path = self.datasets_dir / dataset_id
        if not path.exists():
            return None
        corpus_file = path / "corpus.jsonl"
        data_file = corpus_file if corpus_file.exists() else path / "input.txt"
        if not data_file.exists():
            return None

        # Check for Visual metadata
        visual_meta_path = path / ".visual_metadata.json"
        is_visual = visual_meta_path.exists()

        samples = []
        total_count = 0
        with open(data_file) as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                total_count += 1
                if len(samples) >= limit:
                    continue
                try:
                    obj = json.loads(line) if corpus_file.exists() else {"text": line}
                except json.JSONDecodeError:
                    obj = {"text": line}

                if is_visual:
                    # Visual entries have image_path + conversations
                    img_path = obj.get("image_path", "")
                    convs = obj.get("conversations", [])
                    human = next((c["value"] for c in convs if c.get("from") == "human"), "")
                    gpt = next((c["value"] for c in convs if c.get("from") == "gpt"), "")
                    samples.append({
                        "path": img_path,
                        "language": "visual",
                        "content": f"[IMG: {img_path}] Q: {human} A: {gpt[:120]}" + ("..." if len(gpt) > 120 else ""),
                        "size": len(gpt),
                    })
                else:
                    text = obj.get("text", obj.get("content", ""))
                    samples.append({
                        "path": "",
                        "language": "text",
                        "content": text[:200] + ("..." if len(text) > 200 else ""),
                        "size": len(text),
                    })

        return {
            "dataset_id": dataset_id,
            "samples": samples,
            "total_samples": total_count,
            "total_chars": sum(s.get("size", 0) for s in samples),
            "languages": {"visual": total_count} if is_visual else {"text": total_count},
        }

    def export_dataset(self, dataset_id: str, format: str = "jsonl") -> Optional[Path]:
        """Export a dataset as a file. Returns path to export file or None."""
        path = self.datasets_dir / dataset_id
        if not path.exists():
            return None
        corpus_file = path / "corpus.jsonl"
        if not corpus_file.exists():
            input_file = path / "input.txt"
            if input_file.exists():
                export_path = path / f"{dataset_id}.{format}"
                with open(input_file) as src, open(export_path, "w") as dst:
                    for line in src:
                        dst.write(json.dumps({"text": line.rstrip()}) + "\n")
                return export_path
            return None
        export_path = path / f"{dataset_id}.{format}"
        shutil.copy2(corpus_file, export_path)
        return export_path


_datasets_controller: Optional[DatasetsController] = None


def get_datasets_controller() -> DatasetsController:
    """Get datasets controller instance"""
    global _datasets_controller
    if _datasets_controller is None:
        repo_root = Path(__file__).resolve().parent.parent.parent.parent.parent
        _datasets_controller = DatasetsController(repo_root)
    return _datasets_controller
