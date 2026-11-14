'use client'

import { useEffect, useRef } from 'react'
import { cn } from '@sloughgpt/strui'
import { voiceStateTone, type VoiceVisualState } from './VoiceWaveform'
import type { VoiceAudioFeatures } from '@/features/chat/hooks/useVoiceChat'

interface VoiceWarbleProps {
  /** Voice state machine — drives tone and whether the echo may bloom. */
  state: VoiceVisualState
  /** Per-frame features from useVoiceChat, sampled on this component's own rAF loop. */
  featuresRef?: React.RefObject<VoiceAudioFeatures | null>
}

/** Intensity above this counts as "a voice is present". */
const VOICE_FLOOR = 0.12
/** Hold the bloom through pauses up to ~1.3 s so mid-sentence stops don't flicker. */
const BLOOM_HOLD_MS = 1300
/** Per-second approach rates — bloom opens in ~150 ms, settles back in ~2 s. */
const BLOOM_ATTACK = 7
const BLOOM_RELEASE = 2
/** Pitch floor/ceiling for the flow tempo (Hz); unvoiced falls back to 150. */
const PITCH_MIN = 70
const PITCH_MAX = 400
const PITCH_FALLBACK = 150

/** The six echo traces that bloom beneath the resting line (i = 1…6). */
const LEAF_INDEXES = [1, 2, 3, 4, 5, 6] as const

/**
 * Center-stage voice warble: an organic aura behind the hero orb plus one fine
 * resting wave line that blooms into a 7-trace phase-lagged echo while voice
 * is actually present.
 *
 * Purely decorative (`aria-hidden`) and painted at z-index -1, so it sits
 * behind the orb and text without affecting layout or hit-testing. Signal
 * arrives through `featuresRef` and is written to CSS custom properties on
 * this component's own rAF loop — the React tree only re-renders on discrete
 * state changes, same two-channel rule as `micLevelRef`.
 */
export function VoiceWarble({ state, featuresRef }: VoiceWarbleProps) {
  const rootRef = useRef<HTMLDivElement>(null)
  const stateRef = useRef<VoiceVisualState>(state)

  useEffect(() => {
    stateRef.current = state
  }, [state])

  useEffect(() => {
    const el = rootRef.current
    if (!el) return

    // Reduced motion: publish one calm static frame — aura + resting line,
    // no bloom, no loop (globals.css also kills the animations globally).
    if (window.matchMedia?.('(prefers-reduced-motion: reduce)').matches) {
      el.style.setProperty('--voice-intensity', '0.3')
      el.style.setProperty('--voice-pitch', String(PITCH_FALLBACK))
      el.style.setProperty('--voice-timbre', '0.5')
      el.style.setProperty('--bloom', '0')
      return
    }

    let raf = 0
    let bloom = 0
    let voiceUntil = 0
    let last = performance.now()

    const tick = (now: number) => {
      // Clamp both ends: timestamps can't run backwards (stale `last` after a
      // suspended tab), and a long stall shouldn't be one giant step.
      const dt = Math.max(0, Math.min(0.05, (now - last) / 1000))
      last = now

      const f = featuresRef?.current ?? null
      const intensity = f?.intensity ?? 0
      const pitch =
        f && f.pitch > 0 ? Math.min(PITCH_MAX, Math.max(PITCH_MIN, f.pitch)) : PITCH_FALLBACK
      const timbre = f?.timbre ?? 0

      // Voice presence only counts while the mic is meant to be hearing us
      // (mic-only: TTS output can't be captured, and non-listening states
      // force the line back to rest). The hold is the hysteresis.
      if (stateRef.current === 'listening' && intensity > VOICE_FLOOR) {
        voiceUntil = now + BLOOM_HOLD_MS
      }
      const target = stateRef.current === 'listening' && now < voiceUntil ? 1 : 0
      const rate = target > bloom ? BLOOM_ATTACK : BLOOM_RELEASE
      bloom += (target - bloom) * Math.min(1, dt * rate)
      if (Math.abs(target - bloom) < 0.004) bloom = target

      el.style.setProperty('--voice-intensity', intensity.toFixed(3))
      el.style.setProperty('--voice-pitch', String(Math.round(pitch)))
      el.style.setProperty('--voice-timbre', timbre.toFixed(3))
      el.style.setProperty('--bloom', bloom.toFixed(3))
      raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
  }, [featuresRef])

  const tone = voiceStateTone(state)

  return (
    <div
      ref={rootRef}
      className={cn('voice-warble absolute inset-0')}
      data-state={state}
      style={
        {
          // Behind the orb and text, above the overlay background.
          zIndex: -1,
          '--warble-hue': tone.hue,
        } as React.CSSProperties
      }
      aria-hidden="true"
    >
      {/* Aura — breathes behind the orb */}
      <span className="voice-warble-aura" />
      {/* The stack: one resting line + six echo traces that bloom on voice */}
      <div className="voice-warble-stack">
        <div className="voice-warble-trace voice-warble-base" style={{ '--i': 0 } as React.CSSProperties}>
          <span />
        </div>
        {LEAF_INDEXES.map((i) => (
          <div
            key={i}
            className="voice-warble-trace voice-warble-leaf"
            style={{ '--i': i } as React.CSSProperties}
          >
            <span />
          </div>
        ))}
      </div>
    </div>
  )
}
