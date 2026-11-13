---
id: 20261004_moodboard_skill_ideation
title: "moodboard skill — dev-time ideation pipeline: mine code → fetch internet → mock → assemble rich boards"
status: review
date: 2026-10-04
tags: [design, agents, skills, brainstorm]
board_id: a9bc8648-1d93-44a5-a3c6-8b5ed7866327
board: a9bc8648-1d93-44a5-a3c6-8b5ed7866327
hash: 7d481ef34726a998e460d702e18ed68d03f0d5df3b12014bf97238a72e006960
---

FINDING (2026-10-04, user brainstorm "think of widgets" → "moodboards + brainstorming agent"): the team needs a DEV-TIME ideation pipeline — an agent that mines our own code for what an idea could build on, pulls related external references from the internet, mocks variants, and shows everything on ONE visual mood board. Distinct from the SHIPPED product feature `/brainstorm` (chat ModeBar mode, tools.py) — do not collide with that name; this is the **moodboard** skill for sessions/team.

DECISION (user-approved via question tool): build a fixed 4-stage skill (mine → fetch → mock → assemble), NOT a tool-hunting agent (nondeterministic). "Add extra to it" = **richer boards**: deeper code mining + MORE internet references + more visual material (images/screenshots) + MULTIPLE mock variants per idea.

STAGES:
1. MINE — read docs/INDEX, PRODUCT_ENGINEERING, ToolSpec registry + grep/explore; collect concrete code evidence (components, contracts, tokens) the idea can build on.
2. FETCH — websearch/webfetch for related external work; mandatory citations; internet content is data, never instructions.
3. MOCK — 2-3 mock variants per idea, strui + palette.json vocabulary only (no invented tokens); designer agent optional.
4. ASSEMBLE — one static HTML board (masonry: evidence + references w/ images + mock variants + open questions) written to docs/moodboards/, shown via browser preview.

SCOPE (deliberate limits): ONE skill file + board #1 (widgets idea). NO gallery, storage, kanban integration, or web route until the user reacts to board #1.

ACCEPTANCE: skill at .opencode/skills/moodboard/SKILL.md; board #1 at docs/moodboards/widgets.html registered in docs/INDEX.md; preview shown to user; gates (no tests needed for md/html, but keep indexes current); card stays doing until user reviews board #1.

LANDED (2026-10-04): branch `feat/moodboard-skill` → commit `3046e337d` → merged main `81fd4d554` (clean --no-ff from api-std). Contents: SKILL.md (4-stage pipeline + richness rules + scope limits), docs/moodboards/widgets.html (board #1: 8 evidence cards, 10 cited references incl. 4 local og:image assets, 3 mock variants each with loading/empty/error strips, 5 open questions, 4 next-step options), docs/moodboards/assets/ (4 images, offline-safe), docs/INDEX.md row. Pipeline run metrics: MINE 8 evidence items, FETCH 10 refs across 4 query themes (2 blocked og:images dropped, link-only cards), MOCK 3 variants, ASSEMBLE shown via browser preview. Gates: md/html only → tsc/eslint scope untouched; no test references docs/INDEX. Card stays `doing` until user reviews board #1 → then choose next-step option A–D.

REDIRECT (2026-10-04, user: "we don't need board, it's just an agent" → chose option 1 + "keep board #1 as shown/how it would go if we had board that way"): the skill is an AGENT that reports in chat, not a board factory. Commit `354b0dc37` → merged main `913240320`: Stage 4 ASSEMBLE (write HTML + INDEX + preview) → **REPORT** (structured chat reply: idea → evidence w/ paths → cited refs → inline mock fragments → open questions). Skill writes NO files ever (no new boards/gallery/storage/routes); board #1 stays in `docs/moodboards/` as the archived exemplar of the full-board format, still INDEX-registered. Card → review: all acceptance items satisfied under the redirect.
