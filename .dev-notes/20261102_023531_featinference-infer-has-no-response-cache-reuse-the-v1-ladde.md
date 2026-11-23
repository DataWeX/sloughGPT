---
id: 20261102_023531_featinference-infer-has-no-response-cache-reuse-the-v1-ladde
status: todo
board: 20261102_023531_featinference-infer-has-no-response-cache-reuse-the-v1-ladde
title: "feat(inference): /infer has no response cache — reuse the /v1 ladder instead of re-deriving it"
tags: ["inference", "cache", "/infer", "area:inference"]
sprint: ""
gh: null
---

# feat(inference): /infer has no response cache — reuse the /v1 ladder instead of re-deriving it

FOUND 2026-11-02 while landing 20261102_021851 (/v1 response cache).

/v1 now replays repeated deterministic requests; /infer (routers/infer.py -> InferRouter -> the same provider seam) still does not, and neither does the chat router for anything outside ChatDomain.respond.

DiFFERENT ENVELOPE, SAME LADDER: reuse routers.llm_v1._cache_key rather than re-deriving the rule, so there is ONE ladder. What differs is the response envelope and the key surface (InferRequest's own sampling fields — include every one of them in `extra`, or two requests differing only in one of them collide).

WHY IT MATTERS: /infer is what the local UI and internal tooling call, so it is the path most likely to replay the same prompt during benchmark/eval loops — exactly the traffic the cache is for (temperature <= 0.2, non-stream).

SCOPE: no new endpoint, no new cache; projection of the existing tier. Test only the seam (provider called once for two identical deterministic requests; twice when a sampling param differs).
