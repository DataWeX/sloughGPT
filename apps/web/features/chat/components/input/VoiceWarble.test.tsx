import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, cleanup } from '@testing-library/react'
import React from 'react'
import { VoiceWarble } from './VoiceWarble'
import type { VoiceAudioFeatures } from '@/features/chat/hooks/useVoiceChat'

/**
 * Controllable rAF: the component reschedules every frame, so `fire` drives
 * the loop deterministically with explicit timestamps (no real frames).
 */
function makeRaf() {
  let cb: FrameRequestCallback | null = null
  return {
    install() {
      cb = null // reset between tests so `scheduled()` reflects this test only
      vi.stubGlobal(
        'requestAnimationFrame',
        vi.fn((fn: FrameRequestCallback) => {
          cb = fn
          return 1
        }),
      )
      vi.stubGlobal('cancelAnimationFrame', vi.fn())
    },
    scheduled: () => cb !== null,
    fire(time: number) {
      const fn = cb
      if (!fn) throw new Error('no rAF callback scheduled')
      fn(time)
    },
    fireMany(from: number, count: number, stepMs = 16) {
      for (let i = 0; i < count; i++) this.fire(from + i * stepMs)
    },
  }
}

function features(partial: Partial<VoiceAudioFeatures> = {}): VoiceAudioFeatures {
  return { intensity: 0, pitch: 150, timbre: 0, ...partial }
}

const readVar = (el: Element, name: string) =>
  (el as HTMLElement).style.getPropertyValue(name)

describe('VoiceWarble', () => {
  const raf = makeRaf()

  beforeEach(() => {
    raf.install()
  })

  afterEach(() => {
    cleanup()
    vi.unstubAllGlobals()
  })

  it('renders the resting line plus six echo traces', () => {
    const { container } = render(<VoiceWarble state="idle" />)
    expect(container.querySelectorAll('.voice-warble-base').length).toBe(1)
    expect(container.querySelectorAll('.voice-warble-leaf').length).toBe(6)
    // Each leaf carries its echo index (drives the animation-delay stagger).
    const leaves = container.querySelectorAll('.voice-warble-leaf')
    expect(readVar(leaves[0], '--i')).toBe('1')
    expect(readVar(leaves[5], '--i')).toBe('6')
  })

  it('is decorative and carries the voice state', () => {
    const { container } = render(<VoiceWarble state="listening" />)
    const root = container.querySelector('.voice-warble')
    expect(root?.getAttribute('aria-hidden')).toBe('true')
    expect(root?.getAttribute('data-state')).toBe('listening')
  })

  it('takes its hue from VOICE_STATE_TONE, never a literal', () => {
    const { container } = render(<VoiceWarble state="processing" />)
    const root = container.querySelector('.voice-warble')
    expect(readVar(root!, '--warble-hue')).toBe('rgb(var(--warning))')
  })

  it('writes the per-frame signal to CSS vars without re-rendering', () => {
    const t0 = performance.now()
    const ref: React.RefObject<VoiceAudioFeatures | null> = {
      current: features({ intensity: 0.5, pitch: 120, timbre: 0.4 }),
    }
    const { container } = render(<VoiceWarble state="listening" featuresRef={ref} />)
    const root = container.querySelector('.voice-warble')
    raf.fire(t0 + 32)
    expect(readVar(root!, '--voice-intensity')).toBe('0.500')
    expect(readVar(root!, '--voice-pitch')).toBe('120')
    expect(readVar(root!, '--voice-timbre')).toBe('0.400')
  })

  it('clamps pitch into the flowable range and falls back when unvoiced', () => {
    const t0 = performance.now()
    const ref: React.RefObject<VoiceAudioFeatures | null> = { current: features({ pitch: 1000 }) }
    const { container } = render(<VoiceWarble state="listening" featuresRef={ref} />)
    const root = container.querySelector('.voice-warble')
    raf.fire(t0 + 32)
    expect(readVar(root!, '--voice-pitch')).toBe('400')
    ref.current = features({ pitch: 0 }) // unvoiced → calm default tempo
    raf.fire(t0 + 48)
    expect(readVar(root!, '--voice-pitch')).toBe('150')
  })

  it('blooms to the full echo while listening with voice present', () => {
    const t0 = performance.now()
    const ref: React.RefObject<VoiceAudioFeatures | null> = {
      current: features({ intensity: 0.5 }),
    }
    const { container } = render(<VoiceWarble state="listening" featuresRef={ref} />)
    const root = container.querySelector('.voice-warble')
    raf.fire(t0 + 32)
    expect(parseFloat(readVar(root!, '--bloom'))).toBeGreaterThan(0) // opens fast
    raf.fireMany(t0 + 48, 100)
    expect(readVar(root!, '--bloom')).toBe('1.000')
  })

  it('holds the bloom through a short pause (hysteresis — no flicker)', () => {
    const t0 = performance.now()
    const ref: React.RefObject<VoiceAudioFeatures | null> = {
      current: features({ intensity: 0.5 }),
    }
    const { container } = render(<VoiceWarble state="listening" featuresRef={ref} />)
    const root = container.querySelector('.voice-warble')
    raf.fireMany(t0 + 32, 80)
    expect(readVar(root!, '--bloom')).toBe('1.000')
    // Voice stops: within the 1.3 s hold the echo must stay open.
    ref.current = features({ intensity: 0 })
    raf.fireMany(t0 + 1312, 10)
    expect(readVar(root!, '--bloom')).toBe('1.000')
  })

  it('settles back to the single line once the hold expires', () => {
    const t0 = performance.now()
    const ref: React.RefObject<VoiceAudioFeatures | null> = {
      current: features({ intensity: 0.5 }),
    }
    const { container } = render(<VoiceWarble state="listening" featuresRef={ref} />)
    const root = container.querySelector('.voice-warble')
    raf.fireMany(t0 + 32, 80)
    ref.current = features({ intensity: 0 })
    raf.fireMany(t0 + 1312, 500) // ≈8 s: hold (1.3 s) + slow release
    expect(readVar(root!, '--bloom')).toBe('0.000')
  })

  it('never blooms outside listening (mic-only, other states rest)', () => {
    const t0 = performance.now()
    const ref: React.RefObject<VoiceAudioFeatures | null> = {
      current: features({ intensity: 0.9 }),
    }
    const { container } = render(<VoiceWarble state="speaking" featuresRef={ref} />)
    const root = container.querySelector('.voice-warble')
    raf.fireMany(t0 + 32, 200)
    expect(readVar(root!, '--bloom')).toBe('0.000')
  })

  it('falls back to a calm rest frame when no features arrive', () => {
    const t0 = performance.now()
    const { container } = render(<VoiceWarble state="idle" />)
    const root = container.querySelector('.voice-warble')
    raf.fireMany(t0 + 32, 50)
    expect(readVar(root!, '--voice-intensity')).toBe('0.000')
    expect(readVar(root!, '--bloom')).toBe('0.000')
    expect(readVar(root!, '--voice-pitch')).toBe('150')
  })

  it('publishes one static frame and no loop under prefers-reduced-motion', () => {
    vi.stubGlobal(
      'matchMedia',
      vi.fn(() => ({ matches: true })),
    )
    const { container } = render(<VoiceWarble state="listening" />)
    const root = container.querySelector('.voice-warble')
    expect(readVar(root!, '--voice-intensity')).toBe('0.3')
    expect(readVar(root!, '--bloom')).toBe('0')
    expect(readVar(root!, '--voice-pitch')).toBe('150')
    expect(raf.scheduled()).toBe(false) // no animation loop started
  })
})
