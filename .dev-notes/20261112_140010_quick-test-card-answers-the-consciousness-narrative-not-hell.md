---
id: 20261112_140010_quick-test-card-answers-the-consciousness-narrative-not-hell
status: done
board: 20261112_140010_quick-test-card-answers-the-consciousness-narrative-not-hell
title: "Quick test card answers the consciousness narrative, not 'Hello!' (probe not bare)"
tags: ["web", "chat", "cognition"]
sprint: ""
gh: null
---

# Quick test card answers the consciousness narrative, not 'Hello!' (probe not bare)

Root cause: QuickActions 'Test model' posts /chat with no session_id (-> 'default') and no flags, so stream_turn runs the FULL production pipeline on the bare probe: _apply_consciousness_reflection prepends 'Self-reflection context: I have processed 101 experiences...' (data/consciousness level=1, self_model 100+ episodes) as a system message, plus ContextCore frame + RAG; the 0.5B model answers the narrative instead of 'Hello!'. Confirmed: narrative vocabulary (101 experiences, competence/helpfulness/creativity/accuracy/empathy, growth rates) matches self_model.reflect() + self_model.json exactly. Fix: (1) ChatRequest/TurnRequest gain use_consciousness: bool=True, gate _apply_consciousness_reflection in stream_turn; (2) honor existing use_context_core flag in stream_turn (currently only honored by non-streaming chat()); (3) chatController.stream passes flags + session_id; QuickActions sends use_consciousness=false, use_context_core=false, use_rag=false, session_id='quick-test' so the probe is bare and stops polluting the shared 'default' session history/KV/cache. Tests: backend gate test + frontend flag-passing tests; regen TS contracts.
