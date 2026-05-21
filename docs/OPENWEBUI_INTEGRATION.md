# OpenWebUI wrapper plan

This plan uses the OpenWebUI source as the UI layer and connects it to sloughGPT.

Repository: https://github.com/iamtowbee/webui

## Goals
- Train/infer from a single UI
- Swap datasets and checkpoints
- Centralize settings and tools

## Integration approach
1. **Run sloughGPT API** (from this monorepo’s root; see **QUICKSTART.md**)
   ```sh
   python3 apps/api/server/main.py
   # or: cd apps/api/server && python3 -m uvicorn main:app --host 0.0.0.0 --port 8000
   ```
2. **Run OpenWebUI**
   - Use the OpenWebUI source repo as the UI shell.
3. **Connect model provider**
   - Set OpenWebUI OpenAI-compatible endpoint to:
     - `http://localhost:8000/v1`
4. **Add tools**
   - Load tools from:
     - `http://localhost:8000/openapi.json`
   - Expose dataset update endpoints:
     - `POST /dataset/update_text`
     - `POST /dataset/upload`
5. **Dataset + checkpoint swapping**
   - Add a small admin page in OpenWebUI to:
     - change `out_dir`
     - restart API process
     - trigger `prepare.py` on dataset changes

## What we need to build
- A small OpenWebUI plugin/pipeline panel:
  - dataset picker
  - checkpoint picker
  - training start/stop
- A lightweight control API in sloughGPT:
  - reload checkpoint
  - list available datasets
  - list available checkpoints

## Next steps
- Decide if we fork OpenWebUI or build a thin wrapper app around it.
- Add a simple settings page for model/dataset selection.
