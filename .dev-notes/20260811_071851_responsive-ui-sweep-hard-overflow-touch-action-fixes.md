---
id: 20260811_071851_responsive-ui-sweep-hard-overflow-touch-action-fixes
title: Responsive UI sweep — hard-overflow + touch-action fixes
status: done
tags: ui,responsive
created: 2026-08-11T07:18:51.829716+00:00
---

Responsive UI sweep — hard-overflow + touch-action fixes

COMPLETE. 23 files, class-only changes. P1 hard overflows: OutputPanel -> calc(100vw-2rem)+max-w, training/job/[id] + export rows flex-wrap. P2 hover-only actions gained focus-within fallback across 24 sites (audit 6 + sweep 9; feedback, files, souls, knowledge x2, VoicePresetCard, ConversationSidebar x4, SystemPromptDialog, QuickPrompts, MessageActions, PersonalityProfileCard, ChatBookmarksPanel, KnowledgeTab, ConversationViewer, TrainingFormCard, LossChart). P3 cramped: DatasetQualityCard, VisionStudioDialog, auth, TrainingFormCard grids stack below sm; OutputCard log columns compress; model/[id] header wraps. Verified: viewport device-width + safe-area, no overflow-x-hidden, dialogs vw+max-w. FINAL: tsc exit 0, full suite 330/330 files / 3468 tests green. Awaiting commit approval.