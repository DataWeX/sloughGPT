---
id: 20261113_075130_pin-stream-turn-response-sequence-flags-stage-order-closing
status: done
board: 20261113_075130_pin-stream-turn-response-sequence-flags-stage-order-closing
title: "Pin stream_turn response sequence — flags → stage order → closing frames (PR #311)"
tags: ["test", "inference", "area:inference"]
sprint: ""
gh: null
---

# Pin stream_turn response sequence — flags → stage order → closing frames (PR #311)

Follow-up to 20261112_140010. TestStreamTurnResponseSequence pins bare-probe minimal chain and the default full pipeline order incl. the ctx_core=None hand-off and double memory store. Affected closure 160 passed, ruff clean.
