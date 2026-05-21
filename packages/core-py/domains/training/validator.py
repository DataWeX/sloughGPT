"""
Dataset Validator - Ported from recovered advanced_dataset_features.py
Enterprise-level dataset validation and quality control
"""

import json
import hashlib
import pickle
from pathlib import Path
from typing import Dict, List
from datetime import datetime


class DatasetValidator:
    """Validate dataset quality and integrity."""

    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        self.validation_results: Dict = {}

    def validate_structure(self) -> Dict:
        """Validate dataset file structure."""
        required_files = ["input.txt", "prepare.py"]
        optional_files = ["train.bin", "val.bin", "meta.pkl"]

        results = {"required_files": {}, "optional_files": {}, "structure_valid": True}

        for file_name in required_files:
            file_path = self.dataset_path / file_name
            exists = file_path.exists()
            results["required_files"][file_name] = {"exists": exists, "path": str(file_path)}
            if not exists:
                results["structure_valid"] = False

        for file_name in optional_files:
            file_path = self.dataset_path / file_name
            results["optional_files"][file_name] = {
                "exists": file_path.exists(),
                "path": str(file_path),
            }

        return results

    def validate_checksums(self) -> Dict:
        """Validate dataset file checksums."""
        results = {"files": {}, "valid": True}

        meta_file = self.dataset_path / "meta.pkl"
        if not meta_file.exists():
            return results

        try:
            with open(meta_file, "rb") as f:
                meta = pickle.load(f)

            checksums = meta.get("checksums", {})
            for filename, expected_hash in checksums.items():
                file_path = self.dataset_path / filename
                if not file_path.exists():
                    results["files"][filename] = {"valid": False, "error": "File missing"}
                    results["valid"] = False
                    continue

                actual_hash = self._compute_checksum(file_path)
                valid = actual_hash == expected_hash
                results["files"][filename] = {
                    "valid": valid,
                    "expected": expected_hash,
                    "actual": actual_hash,
                }
                if not valid:
                    results["valid"] = False
        except Exception as e:
            results["error"] = str(e)

        return results

    def _compute_checksum(self, file_path: Path) -> str:
        """Compute SHA256 checksum of a file."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def validate(self) -> Dict:
        """Run all validations."""
        self.validation_results = {
            "timestamp": datetime.now().isoformat(),
            "dataset_path": str(self.dataset_path),
            "structure": self.validate_structure(),
            "checksums": self.validate_checksums(),
        }
        return self.validation_results


class DatasetVersion:
    """Version tracking for datasets."""

    def __init__(self, version_dir: str = "datasets/versions"):
        self.version_dir = Path(version_dir)
        self.version_dir.mkdir(parents=True, exist_ok=True)

    def create_version(self, dataset_path: str, version: str, metadata: Dict) -> Path:
        """Create a new version of a dataset."""
        version_file = self.version_dir / f"{version}.json"

        version_data = {
            "version": version,
            "dataset_path": str(dataset_path),
            "created_at": datetime.now().isoformat(),
            "metadata": metadata,
        }

        with open(version_file, "w") as f:
            json.dump(version_data, f, indent=2)

        return version_file

    def list_versions(self) -> List[Dict]:
        """List all dataset versions."""
        versions = []
        for version_file in self.version_dir.glob("*.json"):
            with open(version_file, "r") as f:
                versions.append(json.load(f))
        return sorted(versions, key=lambda x: x.get("created_at", ""), reverse=True)


# DatasetQualityScorer is now in quality_scorer.py
from domains.training.quality_scorer import DatasetQualityScorer

__all__ = ["DatasetValidator", "DatasetVersion", "DatasetQualityScorer"]
