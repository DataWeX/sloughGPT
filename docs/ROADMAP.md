# sloughGPT roadmap (brief)

## Current state
- Text-only GPT for conversational inference.
- Training via dataset folders that produce `train.bin`/`val.bin`.
- Multiple datasets per training run (--datasets flag with --ratios).
- Evaluation metrics (loss, perplexity) shown in UI.
- Model info panel with vocab/block/params on hover.

## Near-term goals (next 1–3 iterations)
1. **Multi-dataset training** ✅ Done
   - Allow selecting multiple datasets at once.
   - Support dataset mixing ratios (--ratios flag).
   - Produce combined binaries or stream batches from multiple bins.

2. **Dataset management** ✅ Done
   - Standardize dataset metadata (`meta.pkl` via DatasetRegistry).
   - Add a simple registry (name, path, vocab, size).
   - Make dataset preparation consistent for each dataset type.

3. **Better inference UX** ✅ Done
   - Stable chat UI and API serving.
   - Quick switches between checkpoints/datasets.
   - Clear model info panel (dataset, vocab size, block size). Hover to see details.

4. **Learning loop (safe)** ✅ Done (basic)
   - Log chats to buffer (JSONL) via feedback-export command.
   - Periodic batch fine-tune via feedback-train command.
   - Hold-out eval to prevent skew.

## Longer-term goals
- **Adaptive learning loop**: opt-in logging of conversations, periodic fine-tuning ✅ Done (autotrain CLI).
- **Evaluation pipeline**: basic metrics per dataset (loss, perplexity) ✅ Done (shown in UI).
- **Model variants**: allow larger configs for accuracy, smaller for speed ✅ Done (--preset flag).

## Open decisions
- ~~How to merge vocabularies across datasets~~ (resolved: shared vocab via combined text).
- ~~Whether to build a combined bin or sample from multiple bins at runtime~~ (resolved: combined text with ratios).
