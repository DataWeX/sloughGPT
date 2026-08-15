"""Resumable per-file pytest driver with per-file timeouts.

Runs each test_*.py in packages/core-py/tests separately so a slow or
hanging file never blocks the others, and writes incremental PASS/FAIL
results to test_results.txt so the run survives interruptions and can be
resumed by skipping already-finished files.
"""

import pathlib
import subprocess
import time

TESTS_DIR = pathlib.Path("packages/core-py/tests")
RESULTS = pathlib.Path("test_results.txt")
PYTHON = ".venv/bin/python"
TIMEOUT_S = 900

files = sorted(TESTS_DIR.glob("test_*.py"))

done = set()
if RESULTS.exists():
    for line in RESULTS.read_text().splitlines():
        done.add(line.split(" ", 1)[0])

for f in files:
    name = f.name
    if name in done:
        print(f"{name} SKIP (already done)", flush=True)
        continue
    t0 = time.time()
    try:
        r = subprocess.run(
            [PYTHON, "-m", "pytest", str(f), "-p", "no:cacheprovider", "-q"],
            capture_output=True, text=True, timeout=TIMEOUT_S,
        )
        if r.returncode == 5:
            status = "NONE"
        elif r.returncode == 0:
            status = "PASS"
        else:
            status = "FAIL"
        tail = ""
        if r.stdout.strip():
            tail = r.stdout.strip().splitlines()[-1][:160]
    except subprocess.TimeoutExpired:
        status = "TIMEOUT"
        tail = ""
    elapsed = time.time() - t0
    with RESULTS.open("a") as fh:
        fh.write(f"{name} {status} {elapsed:.1f}s {tail}\n")
    print(f"{name} {status} {elapsed:.1f}s", flush=True)
