---
id: 20261031_131015_voice-warble-center-stage-linebloom-voice-visualizer-aura-7
status: done
board: 20261031_131015_voice-warble-center-stage-linebloom-voice-visualizer-aura-7
title: "voice warble - center-stage line/bloom voice visualizer (aura + 7-trace echo)"
tags: []
sprint: ""
gh: null
---

# voice warble - center-stage line/bloom voice visualizer (aura + 7-trace echo)

LANDED main 382c7ae42 (feature) + ce6e6dd4c (AGENT_SYNC). Moodboard #2 approved -> aura behind orb + resting line blooming into 7-trace echo (mic-only, 1.3s hysteresis, reduced-motion static frame). Gates: tsc clean; eslint 0 errors/warnings on touched files (baseline unchanged); targeted vitest 64/64; full suite 822 files/7951 tests green (4 shards, serial workaround for vi.mock parallel bleed); live walkthrough Talk->overlay verified (mount, error/flatline hue, reduced-motion). Files: VoiceWarble.tsx +11 tests, useVoiceChat audioFeaturesRef (fft 1024), VoiceWaveform VOICE_STATE_TONE.hue, globals.css voice-warble CSS, VoiceChatMode orb-wrapper mount; docs/moodboards/voice-warble.html + INDEX row.
