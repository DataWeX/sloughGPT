---
id: 20260924_043_workspace_nav_consolidate
title: Consolidate Workspace sidebar: 11 routes → 5 under /workspace/*
status: done
sprint: current
gh: ""
body: >
  Merge workspace admin nav into 4 grouped routes + shortcuts:
  /workspace (Dashboard|Usage|Audit), /workspace/members (Members|Permissions),
  /workspace/settings (Settings|API|Notifications), /workspace/data (Shared|Search).
  Redirects for old URLs. Update navigation.ts + 5 locales. Delete dead
  component dupes (workspace-api-keys).
---

# Workspace sidebar consolidation

## Scope
- [x] Consolidate 11 admin routes → 5 sidebar entries under `/workspace/*`
- [x] Section tab bars (`WorkspaceSectionTabs`) with longest-prefix active match
- [x] Legacy redirects (server 307 + client stubs + `REDIRECTS` table)
- [x] `navigation.ts` admin section + `SIDEBAR_ICONS` + 5 locales
- [x] Delete dead `components/workspace-api-keys/*` dupes
- [x] Tests: Sidebar, navigation, route-match, redirects, section tabs, all workspace pages

## Result
- Sidebar admin section: `/workspace`, `/workspace/members`, `/workspace/settings`, `/workspace/data`, `/shortcuts`.
- Route groups: `(overview)` = dashboard/usage/audit; nested members/settings/data trees.
- Longest-prefix highlight so `/workspace/members` highlights Members, not Dashboard.
- 10 legacy paths 307 via vite middleware + `redirect()` stubs under old folders.
- New: `components/workspace/{WorkspaceSectionTabs,workspace-tabs}` + unit tests.
- Gates: targeted **19 files / 148 tests** green; `tsc --noEmit` 0; eslint 0 errors on touched surface; `vite build` OK; live smoke Sidebar 200, `/members` → 307 `/workspace/members`.
