---
id: 20261102_023531_productinference-route-arithmeticlookup-intents-to-a-solver
status: todo
board: 20261102_023531_productinference-route-arithmeticlookup-intents-to-a-solver
title: "product(inference): route arithmetic/lookup intents to a solver instead of caching across different prompts"
tags: ["inference", "product", "decision", "area:inference"]
sprint: ""
gh: null
---

# product(inference): route arithmetic/lookup intents to a solver instead of caching across different prompts

FOUND 2026-11-02 from the user's question: "2+3 = 5 and 4+12 = 5 — how do we resume identical-but-different-route?"

THE DECISION RECORDED: they are not the same request and must not share a cache entry. They share an ANSWER SHAPE, not an answer; and if both really equal 5, one is wrong. Answer-level reuse of a correct computation is safe ("the value is 5"); reusing the DERIVATION across different inputs is not caching at all — it is amortized reasoning.

WHAT WOULD BE CORRECT: classify the intent, route arithmetic / unit conversion / lookup to a solver, and cache the evaluator's (expression -> value) mapping, which is exact by construction. The LLM then explains a computed fact instead of recalling one. For the literal arithmetic case this removes the wrong-answer class entirely rather than making it fast.

BLOCKED ON A PRODUCT CALL, not on code: which intents may bypass the model. A wrong classification is a wrong answer, so the allowlist is the deliverable and it needs an owner. Do not build the router before the allowlist.

RELATED AND ALSO DECIDED: the semantic paraphrase tier stays OFF for serving paths. In semantic_cache.py the match is content-word Jaccard (>=80% of the query's content words) boosted by HD similarity, threshold 0.85; a single-content-word query ("continue", "yes") scores 0.95 against every variant of itself regardless of what context the cached answer depended on. Similarity may PROPOSE a candidate; it may never DECIDE one. router tests pin that a paraphrase misses (test_paraphrase_is_not_promoted).
