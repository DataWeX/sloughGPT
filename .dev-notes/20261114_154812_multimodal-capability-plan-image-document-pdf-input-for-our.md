---
id: 20261114_154812_multimodal-capability-plan-image-document-pdf-input-for-our
status: todo
board: 20261114_154812_multimodal-capability-plan-image-document-pdf-input-for-our
title: "Multimodal capability plan: image + document (PDF) input for our model (design card)"
tags: ["training", "model", "roadmap"]
sprint: ""
gh: null
---

# Multimodal capability plan: image + document (PDF) input for our model (design card)

Text-only today: the trainer/model consume tokens only; detect_dataset_type labels IMAGE_TEXT/VIDEO_TEXT but no encoder exists to consume pixels. This is a modality decision + new encoder for OUR architecture (SloNet), not a config flag. Scope to design first: (1) vision encoder prototype on SloNet (patch-embed + projection into LM hidden space, ViT-style); (2) training stages for the encoder — stage A alignment on paired image-text (CLIP-style contrastive OR captioning through the one loop, freeze-encoder first), stage B joint unfreeze with image+text interleaved through SloughGPTTrainer at lower LR; good embeddings come from staged training, not from the raw placeholder trick; (3) fusion point — image tokens enter the token stream before SloughGPTTrainer (ONE training loop, no second modality loop); (4) dataset path — adapter registry maps image+caption/instruction JSONL to interleaved token sequences; (5) DOCUMENTS/PDFs — page images + extracted text + layout as an interleaved source (PDFs are NOT covered today): pdf -> page raster or text/figure segmentation -> same image+text adapter; treat as first-class corpus alongside plain image-caption pairs; (6) data sourcing: paired image-text corpora + document corpora; (7) benchmark gate (new script) for multimodal runs. Out of scope: video/audio, external frameworks (PyTorch/HF stay out — build-from-scratch rule), any change to the current text path. Depends on loss-modes + adapter-registry cards.
