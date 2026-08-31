---
id: 20260815_045619_skeleton-loading-pass-replace-loading-text-states
title: Skeleton loading pass — replace Loading... text states
status: done
tags: frontend,web,ui,loading
created: 2026-08-15T04:56:19.519713+00:00
---

Skeleton loading pass — replace Loading... text states

Replaced text-based loading states with proper design-system placeholders per PageContainer/PageSkeleton pattern: settings Process Isolation card 'Loading...' text -> Skeleton block approximating content shape; security 'Load older' and souls checkpoint 'Load' buttons swap label for inline spinner (spinner replaces button label per design system). Verified remaining pages (dataset snapshots, knowledge list, learn) already use skeletons; StatusCard 'Loading…' is a semantic model-status label, left as-is. 79/79 tests across 3 files pass, tsc --noEmit clean.