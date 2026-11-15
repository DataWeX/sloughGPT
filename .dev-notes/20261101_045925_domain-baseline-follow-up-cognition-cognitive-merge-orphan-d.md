---
id: 20261101_045925_domain-baseline-follow-up-cognition-cognitive-merge-orphan-d
status: done
board: 20261101_045925_domain-baseline-follow-up-cognition-cognitive-merge-orphan-d
title: "domain-baseline follow-up: cognition <-> cognitive merge (orphan dup pair, zero external consumers)"
tags: ["domain-hygiene", "ccgt", "merge"]
sprint: "domain-hygiene"
gh: null
---

# domain-baseline follow-up: cognition <-> cognitive merge (orphan dup pair, zero external consumers)

Debt 2/6 from domain-baseline.json (card 20261031_083744): cognition and cognitive are a duplicate pair; cognitive has ZERO external consumers (grep-proven). Merge into ONE dir under the CCGT mapping (both classed core-capability), update baseline hashes via scripts/domain_baseline_rehash.py, keep ratchet counts (debt 6 -> 5). User ruled staged plan (option 1): physical merges ride their own cards. Run agent suite + ruff after the move; rebase-rehash-push loop (incoming domain/ changes redden CI by design).
