"""
Datasets Controller - Business logic for dataset management
"""
from typing import Optional, List, Dict, Any
from pathlib import Path
import json


class DatasetsController:
    """Controller for dataset management"""
    
    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self.data_dir = repo_root / "data" / "features"
        self.datasets_dir = repo_root / "datasets"
    
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
            if has_corpus:
                try:
                    with open(corpus_file) as f:
                        num_samples = sum(1 for _ in f)
                except Exception:
                    pass
            
            dataset = {
                "id": d.name,
                "name": d.name.replace("_", " ").title(),
                "path": str(d),
                "type": "corpus" if has_corpus else "text",
                "size_bytes": size,
                "size_formatted": f"{size / 1024:.1f} KB" if size > 0 else "Empty",
                "num_samples": num_samples,
            }
            
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
        """Get dataset statistics"""
        path = self.datasets_dir / dataset_id
        if not path.exists():
            return None
        
        files = list(path.glob("*.jsonl"))
        total_size = sum(f.stat().st_size for f in files)
        
        return {
            "dataset_id": dataset_id,
            "files": len(files),
            "size_bytes": total_size,
        }
    
    def search_datasets(self, q: str) -> List[str]:
        """Search datasets by name"""
        if not self.datasets_dir.exists():
            return []
        
        return [d.name for d in self.datasets_dir.iterdir() if d.is_dir() and q.lower() in d.name.lower()]
    
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
        """Update dataset metadata (name is the directory name)"""
        path = self.datasets_dir / dataset_id
        if not path.exists():
            return None
        return {
            "id": dataset_id,
            "name": updates.get("name", dataset_id),
            "description": updates.get("description", ""),
            "path": str(path),
        }
    
    def delete_dataset(self, dataset_id: str) -> bool:
        """Delete a dataset directory"""
        path = self.datasets_dir / dataset_id
        if not path.exists():
            return False
        import shutil
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
        from datetime import datetime
        path = self.datasets_dir / dataset_id
        if not path.exists():
            return None
        versions_dir = self._ensure_versions_dir(dataset_id)
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
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
        """Append data rows to a dataset's corpus file"""
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

    def preview_dataset(self, dataset_id: str, limit: int = 10) -> Optional[dict]:
        """Return a preview of dataset contents (first N rows)."""
        path = self.datasets_dir / dataset_id
        if not path.exists():
            return None
        corpus_file = path / "corpus.jsonl"
        data_file = corpus_file if corpus_file.exists() else path / "input.txt"
        if not data_file.exists():
            return None
        samples = []
        with open(data_file) as f:
            for i, line in enumerate(f):
                if i >= limit:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line) if corpus_file.exists() else {"text": line}
                except json.JSONDecodeError:
                    obj = {"text": line}
                samples.append(obj)
        return {
            "dataset_id": dataset_id,
            "samples": samples,
            "total_samples": len(samples),
            "total_chars": sum(len(s.get("text", "")) for s in samples),
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
        import shutil
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