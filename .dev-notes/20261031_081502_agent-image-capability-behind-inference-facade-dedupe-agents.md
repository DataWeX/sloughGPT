---
id: 20261031_081502_agent-image-capability-behind-inference-facade-dedupe-agents
status: done
board: 20261031_081502_agent-image-capability-behind-inference-facade-dedupe-agents
title: "agent image capability behind inference facade — dedupe agents.py/tools.py VisionCNN copies"
tags: [inference, agents, facade, image-capability]
sprint: domain-hygiene
gh: null
---
# agent image capability behind inference facade (landed `4b97f92c2` — jolly-wizard lane)

## Disposition

- **DONE by jolly-wizard lane** after expert review (user go-ahead x2):
  `4b97f92c2 feat(inference): image-capability facade — dedupe agents.py/
  tools.py VisionCNN copies`; their close commit `0566d9f7f` recorded it.
- **Why this note exists:** the repo-side close was silently reverted to
  `todo` by later canonical->repo syncs — the card had no canonical note,
  and union-merge gives shared-id rows to canonical. This note restores the
  closed state the work actually has (see commit 0566d9f7f).
- **Hazard carded** (repo-side closes without canonical notes lose to
  canonical-wins union) — pipeline follow-up, not this card.

## What landed (per AGENT_SYNC, jolly-wizard entry)

- Public `domain.inference.analyze_image(path) -> ImageAnalysis` in
  `_internal/image_capability.py` (lazy export, error taxonomy
  `not_found|unavailable|failed`, formatter-free — callers keep their own
  presentation shims, output byte-identical before/after).
- `agents.py` + `tools.py` deduped onto the contract; `VisionCNN` import
  stays function-local (honesty tests stub `sys.modules`); `multimodal/`
  untouched; path validation stays with callers (`_path_violation`).
- Seams the provider swap for when `pdf_vlm` works (`_get_vlm()` returns
  None today).
