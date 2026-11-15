---
id: 20261115_081635_uxdatasets-failedempty-imports-must-not-render-as-100-ready
status: wip
board: 20261115_081635_uxdatasets-failedempty-imports-must-not-render-as-100-ready
title: "ux(datasets): failed/empty imports must not render as 100% ready (221 B stubs)"
tags: ["frontend", "datasets", "ux"]
sprint: ""
gh: null
---

# ux(datasets): failed/empty imports must not render as 100% ready (221 B stubs)

User-approved fix (2026-11-14). data/J.R.R. Tolkien's The hobbit/ = 221-byte bibliographic stub (Gutenberg: full text unavailable) yet the /training dataset tooltip shows 'Size 221 B' and the chip shows 100%. Two stubs in data/. Fix: dataset-chip.tsx tooltip has no emptiness check; locate the 100% chip rendering; surface empty/failed import state (no samples, tiny size, info.txt-only) as 'empty / no text retrieved', not ready. Backend: datasets controller knows sample counts — consider an 'empty' status. Also decide stub cleanup (ask user before deleting).
