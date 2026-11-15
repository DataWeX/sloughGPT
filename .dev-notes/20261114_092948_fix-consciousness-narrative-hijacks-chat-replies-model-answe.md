---
id: 20261114_092948_fix-consciousness-narrative-hijacks-chat-replies-model-answe
status: done
board: 20261114_092948_fix-consciousness-narrative-hijacks-chat-replies-model-answe
title: "fix: consciousness narrative hijacks chat replies — model answers the reflection system message instead of the user"
tags: ["api", "frontend", "tests"]
sprint: ""
gh: null
---

# fix: consciousness narrative hijacks chat replies — model answers the reflection system message instead of the user

Probe-proven on live server (2026-11-14): POST /chat with 'Explain like I'm 5' / 'What is 2+2?' returns replies quoting the injected consciousness narrative verbatim ('Self-Reflection Context: You have processed 129 experiences... Competence 1.00'). Mechanism: use_consciousness defaults True (apps/api/server/schemas/chat.py:49) and the web never sends the flag, so stream_turn (inference.py:2265) prepends engine.reflect() narrative as a system message via ConsciousnessProcessor; the 0.5B model latches onto it. Known class from card 20261112_140010 — that fix only gated bare probes, leaving default-on chat broken. APPROACH (user-routed): fix first (cheap: default -> false), verify live, then improve in same branch (narrative to volatile tail), re-verify, single PR. VERIFY: live probe returns a reply about the user's actual question, not the narrative; schema/contract/web gates green.
