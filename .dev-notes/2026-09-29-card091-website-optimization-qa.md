---
id: card091
title: "Card 091 — Website optimization QA: Lighthouse + bundle analysis (the 'use qa agent' ask)"
status: todo
priority: medium
created: 2026-09-29
source: card 087 investigation
---

## Ask

Card 087 confirmed the 'we still need our website optimized well, use qa
agent' ask (2026-08-28) never ran: zero Lighthouse / Web Vitals /
bundle-analysis history in notes or docs. Run the QA pass and fix top
findings.

## Scope

- Prod build (`next build`) + bundle analyzer (install nothing shared —
  check for existing analyzer first; `ANALYZE=1` pattern).
- Lighthouse (chrome-devtools tool) against the running stack — needs the
  one-process stack up (process-guard rule) and **disk**: dist/.next output
  (repo disk was 99% full on 2026-09-29 — free space first, e.g. obsolete
  `.wt-ux` worktree pending user approval).
- Measure: LCP/INP/CLS, JS payload per route (recharts pages flagged),
  report + fixes as follow-ups.

## Blocked on

Disk headroom + stack start approval.

## Evidence

_(filled in on completion)_
