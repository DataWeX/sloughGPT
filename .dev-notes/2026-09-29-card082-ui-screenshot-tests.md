---
id: 20260929_082
board: ec2ae5c1-f57a-4eb9-aa03-deddc96810c8
title: "Card 082 — UI screenshot tests: headers + basic interactions across pages"
status: done
date: 2026-09-29
tags: [testing, frontend]
---

# UI screenshot tests: headers + basic interactions across pages

## Background

Ask 2026-08-30: 'add computer use to our component tests of basic UI interactions so we can screenshot them... run it on our headers across pages'. Add Playwright screenshot assertions to journey tests (toHaveScreenshot/page.screenshot); none exist today.

## Source
Mined from stale opencode sessions (Aug 26-31 2026) at user's request: ~/.cache/slog-journeys/session-exports/.

## Landed (2026-10-01)

- a9d3d079d test(journeys): UI screenshot tests — mobile header across
  5 pages + drawer interaction (branch feat/ui-screenshot-tests, pushed
  to feat/shared-governance; close row fd1579901).
- Full work log: 2026-10-01-card082-screenshot-tests.md
