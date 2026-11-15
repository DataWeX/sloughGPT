---
id: 20261101_045925_domain-baseline-follow-up-api-ops-infrastructure-overlap-res
status: todo
board: 20261101_045925_domain-baseline-follow-up-api-ops-infrastructure-overlap-res
title: "domain-baseline follow-up: api / ops / infrastructure overlap resolution (debt 1/6)"
tags: ["domain-hygiene", "ccgt", "overlap", "area:core"]
sprint: "domain-hygiene"
gh: null
---

# domain-baseline follow-up: api / ops / infrastructure overlap resolution (debt 1/6)

Debt from domain-baseline.json (card 20261031_083744): api/, ops/, infrastructure/ overlap — same capability projected across three dirs (engine-first rule: one contract dir, projections elsewhere). Scope: map each file to its true owner under CCGT facets, move (git mv), rehash baseline, shrink debt count. No behaviour change; tests must stay green. Coordinate: jolly-wizard lane lands heavily in api/ — rebase often, check AGENT_SYNC before moves.
