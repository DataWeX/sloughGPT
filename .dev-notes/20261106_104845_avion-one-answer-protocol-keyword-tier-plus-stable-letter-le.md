---
id: 20261106_104845_avion-one-answer-protocol-keyword-tier-plus-stable-letter-le
status: done
board: 20261106_104845_avion-one-answer-protocol-keyword-tier-plus-stable-letter-le
title: "avion: one answer protocol — keyword tier plus stable letter ledger"
tags: ["avion", "choice"]
sprint: ""
gh: null
---

# avion: one answer protocol — keyword tier plus stable letter ledger

choice.py now carries TWO answer parsers: parse_pick (lettered/digit fallback) and parse_action (added today 11:24 on wip/avion-observation). That is the third vocabulary, not the second, and it is the drift this card exists to stop.

Standardize on one protocol: an option's identity is a stable token bound at first grounding and never derived from list position.
- CandidateLedger assigns A/B/C once per run; withholding a card marks its entry unpickable instead of renumbering the rest (the current unavailable annotation is a workaround for exactly this).
- DONE/FAIL are keywords, not positions: resolve 'done'/'cannot' from the reply text BEFORE consulting a letter.
- parse_answer(text, ledger) -> Choice | None replaces both parsers.
- Digit support retires WITH the numbered prompt, not as a fallback beside it.
- Runtime stays tolerant; training narrows the model. Never weaken the parser to match a checkpoint.

Blocked on: the in-flight uncommitted avion edits in the primary checkout landing first.
