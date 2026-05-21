"""
Convert ZIP-format .slo checkpoints to v3 binary for WebGPU inference.

Usage:
    PYTHONPATH=packages/core-py python scripts/convert_checkpoint.py path/to/model.slo [output.slo]
"""

from __future__ import annotations
import sys
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "packages", "core-py"))

import struct
import json

from domains.training.slonet import import_from_sou
from domains.inference import write_v3_sou


def convert(inpath: str, outpath: str | None = None) -> str:
    """Load a .slo checkpoint (any format) and write as v3 binary.

    Args:
        inpath: path to any .slo file (JSON, ZIP/PyTorch, or v3)
        outpath: destination path (default: ``<basename>.bin.slo``)

    Returns:
        path to the written v3 file
    """
    if outpath is None:
        base, ext = os.path.splitext(inpath)
        outpath = f"{base}.bin{ext}"

    # Load via SloNet's import_from_sou (handles all formats)
    net = import_from_sou(inpath)

    # Build metadata
    metadata = {
        "version": 3,
        "soul_name": net.soul_name,
        "soul_traits": net.soul_traits,
        "lineage": net.lineage,
        "system_prompt": net.system_prompt,
        "metadata": net.metadata,
    }

    # Extract flat parameter arrays in order
    raw = net._sd if (hasattr(net, '_sd') and net._sd) else net.state_dict()
    state_dict = {f"p{i}": v.flatten() for i, (k, v) in enumerate(raw.items())}

    write_v3_sou(outpath, metadata, state_dict)

    in_size = os.path.getsize(inpath)
    out_size = os.path.getsize(outpath)
    pct = (1 - out_size / in_size) * 100
    print(f"Converted {inpath} ({in_size/1024:.0f} KB)")
    print(f"  → {outpath} ({out_size/1024:.0f} KB, {pct:.0f}% smaller)")
    return outpath


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    convert(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
