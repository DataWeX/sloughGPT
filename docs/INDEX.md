# Documentation Index

Single source of truth for navigation. Read this first.

---

## Core (read these)

| Doc | Purpose |
|-----|---------|
| **PYTHON_FIRST.md** | How we write code — import → call → result |
| **PRODUCT_ENGINEERING.md** | What we build and why |
| **ROADMAP.md** | What's done, what's next |
| **AGENT_SYNC.md** | Cross-session work sync — read first, update when you push |
| **AGENTS.md** | Rules for AI agents (root) |

## Architecture

| Doc | Purpose |
|-----|---------|
| **STRUCTURE.md** | Repo layout, domain/ map, branch lanes |
| **RAG_ARCHITECTURE.md** | RAG system design |
| **RAG_PATTERNS.md** | RAG implementation patterns |
| **TRAINING_REFACTOR_PLAN.md** | Consolidating training loops |
| **process-guard-architecture.md** | Process isolation design |
| **PRODUCER_CONSUMER_QUEUE.md** | Queue pattern |
| **PUGQEEP.md** | PGQ: points, pools, Pipe |
| **TRANSPORT_PROJECTIONS.md** | Capability → contract → route |

## Features

| Doc | Purpose |
|-----|---------|
| **FEATURES.md** | Feature list |
| **moodboards/widgets.html** | Mood board #1 — widgets (visual ideation, open in browser) |
| **NAVIGATION_AND_SOULS.md** | Soul/personality system |
| **SHELL.md** | CLI shell |
| **VM_CONSOLE.md** | VM console |
| **vm-devices-spec.md** | VM device spec |
| **WORLD_REALM.md** | World rendering |
| **CACHING.md** | LLM response cache: keys + invalidation |

## Design

| Doc | Purpose |
|-----|---------|
| **design/DESIGN_SYSTEM.md** | Noir Violet design system (LOCKED) — tokens, palettes, auras |

Color values have one source of truth: `packages/strui/tokens/palette.json`.
Edit it and run `node packages/strui/scripts/gen-tokens.mjs` (verify: `--check`);
both `globals.css` token regions, the web swatch map, the mobile color
projection, the tamagui theme overrides, and the magazine showcase data are
generated from it and guarded by `packages/strui/src/tokens/tokens.test.ts`.

## Setup & Deploy

| Doc | Purpose |
|-----|---------|
| **INSTALL.md** | Installation |
| **ENVIRONMENT.md** | Environment setup |
| **DEPLOYMENT.md** | Deployment guide |
| **DEPLOYMENT_CHECKLIST.md** | Pre-deploy checks |
| **WORKTREE_RECOVERY.md** | Worktree wipe prevention + prune/recreate recovery runbook |
| **scripts/systemd/** | systemd --user unit for the shared API (:8000) + install.sh |

## Integration

| Doc | Purpose |
|-----|---------|
| **API.md** | API reference |
| **routers.md** | Router map |
| **llama_rn_integration.md** | llama.cpp integration |
| **OPENWEBUI_INTEGRATION.md** | OpenWebUI bridge |
| **mogdb-guide.md** | MogDB usage |
| **adr-001-mogdb-json-sync.md** | MogDB ADR |

## Testing & UX

| Doc | Purpose |
|-----|---------|
| **TESTING.md** | Test guide |
| **Mole** (in TESTING.md) | The one monitoring app — watches the running stack, heals by proposing fixes (suggest-only); `/mole` surface |
| **USER_PERSONA.md** | Who we build for |
| **USER_JOURNEYS.md** | User flows |
| **UX_FLOWS.md** | UX patterns |
| **DEVELOPER_GUIDE.md** | Developer guide |
| **DOC_VS_CODE_GAP_AUDIT.md** | API doc-vs-code audit + parity script |

## Meta

| Doc | Purpose |
|-----|---------|
| **README.md** | Docs overview |
| **OPENCODE_SKILLS.md** | Skill definitions |
| **REFACTOR_PROMPT.md** | Refactoring prompt |
| **REFACTOR_SUMMARY.md** | Refactoring summary |
