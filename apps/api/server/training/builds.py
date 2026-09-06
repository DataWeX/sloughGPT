"""Builds listing route — builds.

Extracted from execution.py to keep each module focused.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from domains.shared import find_repo_root
from fastapi import APIRouter

from .jobs import training_jobs

router = APIRouter(tags=["training-builds"])


@router.get("/training/builds")
async def list_builds():
    """List all training builds (checkpoints + fine-tuned models + LoRA adapters)."""
    from domains.training.service import load_lora_soul, load_soul

    _repo_root = find_repo_root(Path(__file__).resolve())
    _checkpoints_dir = _repo_root / "models" / "auto-training"
    _lora_dir = _repo_root / "data" / "user_adapters"
    _hf_finetuned_dir = _repo_root / "models" / "hf-finetuned"

    builds = []

    # 1. Auto-train checkpoints (.soul / .npz)
    seen = set()
    for ext in ("*.soul", "*.npz"):
        for f in sorted(_checkpoints_dir.glob(ext), key=lambda p: p.stat().st_mtime, reverse=True):
            if f.name in seen:
                continue
            seen.add(f.name)
            info = load_soul(f.name)
            if info:
                info["build_type"] = "auto-train"
                builds.append(info)

    # 2. LoRA .soul files
    for npz in sorted(_lora_dir.glob("*.soul"), key=lambda p: p.stat().st_mtime, reverse=True):
        if npz.name in seen:
            continue
        seen.add(npz.name)
        info = load_lora_soul(npz.name)
        if info:
            info["build_type"] = "lora"
            builds.append(info)

    # 3. Completed HF fine-tune jobs
    for jid, job in training_jobs.items():
        if job.get("status") == "completed":
            model_path = (
                job.get("result", {}).get("model_path", "")
                if isinstance(job.get("result"), dict)
                else ""
            )
            builds.append(
                {
                    "name": job.get("name") or jid,
                    "build_type": "hf-finetune",
                    "job_id": jid,
                    "model": job.get("model", ""),
                    "dataset": job.get("dataset", ""),
                    "loss": job.get("loss"),
                    "epochs": job.get("epochs"),
                    "model_path": model_path,
                    "created_at": job.get("started_at", ""),
                    "finished_at": job.get("completed_at", ""),
                }
            )

    # 4. HF fine-tuned model directories on disk (for builds not tracked in memory)
    if _hf_finetuned_dir.is_dir():
        for d in sorted(_hf_finetuned_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
            if d.is_dir() and d.name not in seen:
                seen.add(d.name)
                size_mb = sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) / (1024 * 1024)
                builds.append(
                    {
                        "name": d.name,
                        "build_type": "hf-finetuned-dir",
                        "model_path": str(d),
                        "size_mb": round(size_mb, 1),
                        "created_at": datetime.fromtimestamp(d.stat().st_mtime).isoformat(),
                        "model": d.name.split("_")[0].replace("--", "/"),
                        "dataset": d.name.split("_")[1] if "_" in d.name else "",
                    }
                )

    return {"builds": builds}
