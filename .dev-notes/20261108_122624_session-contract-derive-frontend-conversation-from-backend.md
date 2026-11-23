---
id: 20261108_122624_session-contract-derive-frontend-conversation-from-backend
status: done
board: 20261108_122624_session-contract-derive-frontend-conversation-from-backend
title: "Session contract: derive frontend Conversation from backend"
tags: ["contract", "chat", "frontend", "backend"]
sprint: ""
gh: null
---

# Session contract: derive frontend Conversation from backend

Single source of truth for conversation/session shape: declare a Pydantic SessionModel in apps/api/server/routers/inference.py (matching _build_session_metadata_index output, extra=allow initially), register the /chat/sessions endpoint descriptor, regenerate contracts.gen.ts via scripts/gen_ts_contracts.py, then collapse the three drifted frontend Conversation interfaces (session-controller.ts, feedback-conversations-controller.ts, ConversationArchive.tsx) into generated Session type + thin frozen Conversation.from() wrapper (instanceof support). Feedback conversations stay a separate descriptor/projection. Grandfather clause: route migration to create_router when mechanically possible in same PR. Gates: gen_ts_contracts --check, test_inference_router.py, pnpm typecheck+test. No benchmark (no training/inference change). Branch feat/session-contract.
