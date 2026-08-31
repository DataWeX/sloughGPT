---
id: 20260808_034017_20260808-knowledge-topical-overlap-gate
title: 20260808 Knowledge topical-overlap gate
status: done
tags: 
created: 2026-08-08T03:40:17.036900+00:00
---

20260808 Knowledge topical-overlap gate

Second layer on the knowledge-enrichment fix. Residual bug: score-passing but topically-unrelated facts injected (e.g. 'What color is the sky?' matched Paris fact at 0.224). Added _content_tokens (len>=4, no stopword list) + _topically_related to knowledge_augmenter.py; _relevant() now requires a shared content token before injection, applied to memory and web paths. 11 new tests (7 unit + 4 injection) -> 40 total in test_knowledge_augmenter.py. Live-verified: sky question -> clean answer, France question -> correct fact. Regression: 167 core-py + 27 server tests pass. Server restarted (pid 27596) with fix.