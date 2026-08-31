---
id: 20260808_034851_20260808-anchor-knowledge-dir-to-repo-root
title: 20260808 Anchor KNOWLEDGE_DIR to repo root
status: done
tags: 
created: 2026-08-08T03:48:51.671287+00:00
---

20260808 Anchor KNOWLEDGE_DIR to repo root

Root-cause fix for the CWD-relative data path hazard that started the /chat investigation. KNOWLEDGE_DIR was Path('data/knowledge') (CWD-relative); arbitrary-CWD processes silently read/wrote the wrong data dir (that is how the stray workspace data/ was created and why probes from packages/core-py vs repo root showed different fact sets: quantum facts vs ML facts). knowledge.py now anchors via _REPO_ROOT = Path(__file__).resolve().parents[4], matching pair_extractor.py / datasets.py precedent. Made test_knowledge_memory.py hermetic (autouse fixture patches the 4 path constants to tmp_path, matching test_knowledge_ingest). Verified: from /tmp CWD KNOWLEDGE_DIR resolves to repo root; 253 knowledge tests + 183 learner-adjacent tests pass; repo data intact; server restarted (pid 40294) and /chat returns correct ML fact and clean sky answer.