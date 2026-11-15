---
id: 20261113_083550_persist-conversationagenttrainingjobsessioncontext-typed-wra
status: todo
board: 20261113_083550_persist-conversationagenttrainingjobsessioncontext-typed-wra
title: "persist Conversation/Agent/TrainingJob/SessionContext typed wrappers as MogDB collections per ADR-001 (salvage from closed PR #43)"
tags: ["area:api"]
sprint: ""
gh: "43"
---

# persist Conversation/Agent/TrainingJob/SessionContext typed wrappers as MogDB collections per ADR-001 (salvage from closed PR #43)

Closed PR #43 kept four typed entity wrappers with no main equivalent: ConversationRepository, AgentRepository, TrainingJobRepository, SessionContextRepository (data models ConversationData/AgentData/TrainingJobData/SessionContextData). Per ADR-001 the implementation tier must be MogDB document collections (JSON sync fallback where legacy compat matters), NOT the PR's SQLite/file factory — and it should compose with the existing Repository pattern in entity_repositories.py rather than add a second framework. Only card-worthy if a journey actually needs these entities persisted; check existing routers/controllers first (conversation list items, agent definitions, training job records may already be stored elsewhere).
