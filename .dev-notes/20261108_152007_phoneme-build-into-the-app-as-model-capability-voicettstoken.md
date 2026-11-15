---
id: 20261108_152007_phoneme-build-into-the-app-as-model-capability-voicettstoken
status: done
board: 20261108_152007_phoneme-build-into-the-app-as-model-capability-voicettstoken
title: "phoneme: build into the app as MODEL capability (voice/TTS/tokenizer infra) — not a dictionary/learning UI; split from Build Order #5 2026-11-08"
tags: ["frontend", "phoneme", "architecture"]
sprint: ""
gh: null
---

# phoneme: build into the app as MODEL capability (voice/TTS/tokenizer infra) — not a dictionary/learning UI; split from Build Order #5 2026-11-08

User ruling 2026-11-08 (split off Build Order #5 card 20261101_140445_build-order-5): 'we don't need to learn how to read and pronounce — it's not a dictionary app; phoneme is for the MODEL to say and see — app features and model features.' Engineering direction (verbatim intent): build it into the app structure FROM THE GROUND UP to support the neural net — slim and powerful — layering components that are memoized and fed into the model; 'building a mansion from the inside out'. Current state: app/(app)/phoneme/ = 44-component 'Phoneme Encoder' page (12 tabs: scoring, practice, quizzes, heatmap, coach, games, TTS synthesis), grandfathered in page-baseline.json, nav-hidden since fd27316be (2026-09-20, 'remove Study IPA frontend — Alex wants Read My Files / Talk Out Loud; keep as infra for Talk/Read', backend 6 routes serve VoiceEngine TTS scoring). UX_FLOWS has zero phoneme journey (Feature 7 Talk Out Loud = voice conversation only). First move: expert review — decide what of the learning-UI layer (flashcards/journal/calendar/quiz) is deleted vs what becomes model-capability surface (encode/score/synthesize exposed through voice/chat journeys); any user-facing surface needs a UX_FLOWS journey charter before nav (Page Rule). Constraints: UX_FLOWS spec exists only after product decision; do not unhide nav without the journey; backend 6 routes = VoiceEngine infra, must not break.
