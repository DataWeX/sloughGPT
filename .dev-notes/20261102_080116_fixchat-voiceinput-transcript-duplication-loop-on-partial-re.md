---
id: 20261102_080116_fixchat-voiceinput-transcript-duplication-loop-on-partial-re
status: review
board: 20261102_080116_fixchat-voiceinput-transcript-duplication-loop-on-partial-re
title: "fix(chat): VoiceInput transcript duplication loop on partial results"
tags: ["chat", "voice", "stt", "frontend", "root-cause", "area:web"]
sprint: ""
gh: null
---

# fix(chat): VoiceInput transcript duplication loop on partial results

SHIPPED (branch fix/voice-input-duplication, commit 55394d7cf, 1 commit off origin/main).

ROOT CAUSE: apps/web/features/chat/components/input/VoiceInput.tsx onresult re-joined ALL results from index 0 on every recognition event and emitted the whole utterance; ChatInput.handleVoiceTranscript APPENDS (`${value} ${text}`) then auto-sends via pendingSendRef. Event N therefore emitted results[0..N] on top of text already holding results[0..N-1] -> multiplicative composer growth. Emitting interim fragments additionally pushed half-finished words into the append+auto-send path.

FIX: walk from event.resultIndex (added to the local SpeechRecognitionEvent interface) and keep isFinal results only -- mirrors the proven loop in features/chat/voice/speech-engine.ts:350-363, which was already the correct in-repo pattern. ChatInput's append semantic is unchanged and is now correct: each final utterance lands exactly once.

NOT DONE HERE (deferred, by design): ChatInput's stale-closure guard in handleVoiceTranscript. With isFinal-only emission there is one emit per listening session and value cannot change mid-session, so the stale capture is currently harmless -- but it is still a latent hazard if interim emission returns.

GATES: vitest 20/20 (VoiceInput 11 incl. 3 new, ChatInput 9 incl. the seam contract), tsc --noEmit 0, eslint 0 errors (3 pre-existing warnings: unused onSend arg + 2x addToast deps).

New tests: (1) interim never emitted; (2) bug repro -- 3 partial events replacing result[0] in place -> exactly 1 emission of 'hello world'; (3) new final segment emitted alone, never re-joined as 'hello there'; (4) ChatInput appends each final transcript exactly once.
