---
id: 20260731_125048_infrastructure-test-coverage-round-2
title: Infrastructure test coverage round 2
status: done
tags: infrastructure,tests
created: 2026-07-31T12:50:48.205013+00:00
---

Infrastructure test coverage round 2

Round 2 complete: 6 new test files (slnc_format 39, slnc_loader 15, compression 12, embedding_service 25, hf_model_worker 6, knowledge_weight_integrator 16) all green. Found and fixed 1 real source bug: knowledge_weight_integrator._format_facts_as_text accepted max_items but caller train_knowledge_adapter passed max_facts -> TypeError on every call; renamed param to max_facts. Test-side corrections: CompressedWeight residual is 1-D (size prod(shape)), MeaningTags.names is a method, similarity returns 0.0 for unknown tags. Round-1 note (20260731_124514_infrastructure-test-coverage) also closed.