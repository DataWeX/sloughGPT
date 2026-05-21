# Data structure for codebase ingestion

We store codebase data in JSONL so it can be streamed efficiently and turned
into `train.bin` / `val.bin` later.

**Note:** A **`repo_obtainer.py`** driver is **not** checked into this repository; the commands below are the **intended** interface once that tool exists or is vendored. Use **`python3`** on Unix/macOS.

## JSONL schema (corpus export)
```json
{
  "path": "relative/path/to/file.py",
  "size": 12345,
  "mtime": 1700000000,
  "content": "file contents (utf-8, truncated)"
}
```

## How to generate
```sh
python3 repo_obtainer.py export --source https://github.com/user/repo.git
```

This writes to `runs/repo_corpus/<repo>.jsonl`.

## Map schema (LLM-friendly)
```json
{
  "path": "src/foo/bar.py",
  "language": "python",
  "size": 12345,
  "mtime": 1700000000,
  "summary": "Module docstring (if any)",
  "exports": ["ClassA", "func_b"],
  "dependencies": ["torch", "numpy"],
  "chunks": [
    {"start_line": 1, "end_line": 200, "text": "..."}
  ]
}
```

Generate:
```sh
python3 repo_obtainer.py export --source https://github.com/user/repo.git --format map
```

This writes to `runs/repo_map/<repo>.jsonl`.

## State bins (signed)
When building datasets from a corpus, we also emit `train_state.bin` and
`val_state.bin` using **int16** where special tokens map to signed state values:

- `<TRUE>` → 8
- `<FAULT>` → 4
- `<FALSE>` → -8
- `<GOT>` → 2

All non-special tokens map to 0.

## Rendering state values at inference
If `SLO_STATE_OUTPUT=1` (default), decoding replaces state tokens with their
numeric values (e.g. `<TRUE>` → `8`).

## Next step
Convert JSONL to a training dataset:
1. Concatenate `content` fields.
2. Tokenize.
3. Write `train.bin` / `val.bin`.
