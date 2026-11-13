---
title: "Orphaned components after the 36-page prune"
status: done
date: 2026-10-24
tags: [web, cleanup, pages, page-rule]
---

# Orphaned components after the 36-page prune

Follow-up of PR #193 (feat/prune-shadowed-pages — 36 shadowed page.tsx deleted,
page.tsx 109 -> 73, plus 36 page tests and 52 error/loading companions).

## Why

The deletion removed route files only. Feature/component code that the deleted
pages consumed survives — e.g. `WorkspaceSearchPage` component + its test still
run even though `/workspace-search` is gone.

## Task

For each of the 36 deleted routes, find feature components whose only importer
was that page (grep importers across apps/web), and retire orphans
(component + colocated test together).

## Guard

Do NOT blind-delete workspace-search / docstore-named components: main's search
lane (2026-10-24, Cmd+K palette + workspace page) actively wires some of these
components into live pages. Importer check is mandatory — zero importers among
the 73 live pages + shared layers is the only retirement criterion.

## Done when

No apps/web component or component test has zero importers among live routes,
and the frontend suite is green.
