# Standalone tooling (`scripts/tools/`)

These scripts are **not** part of the installable `domains` package. Run them from the **repository root** (after `python3 -m pip install -e ".[dev]"` or the deps each script needs):

| Script | Purpose |
|--------|---------|
| `benchmark.py` | Benchmark harness |
| `convert_to_gguf.py` | Convert checkpoints to GGUF (see also `scripts/export_to_gguf.py`) |
| `download_model.py` | Download models for local use |
| `export_for_cloud.py` | Export bundles for cloud deployment |
| `live_dataset.py` | Live dataset helpers |
| `lora.py` | LoRA utility / experiments |
| `model_server.py` | Simple model server |
| `performance_test.py` | Latency/throughput checks (used in CI smoke) |

Example:

```bash
python3 scripts/tools/performance_test.py --test latency --runs 5 --tokens 20
```

Core training flows remain **`python3 cli.py train`** and **`python3 train_sloughgpt.py`** at the repo root.
