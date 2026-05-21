"""SloughGPT CLI - Forwarding stub (delegates to modular CLI via absolute import)."""
import sys
from pathlib import Path

_repo_root = str(Path(__file__).resolve().parents[1])
_src = str(Path(__file__).resolve().parent / "src")
for _p in (_repo_root, _src):
    if _p not in sys.path:
        sys.path.insert(0, _p)

if __name__ == "__main__":
    import runpy
    runpy.run_path(str(Path(_src) / "cli.py"), run_name="__main__")
