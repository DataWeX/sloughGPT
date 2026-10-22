// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest'
import { renderHook, act } from '@testing-library/react'
import { usePrefersReducedMotion, chartAnimation, CHART_MOTION_MS } from './chart-motion'

type MediaQueryListener = (e: { matches: boolean }) => void

function stubMatchMedia(initial: boolean) {
  const listeners = new Set<MediaQueryListener>()
  const matchMedia = vi.fn().mockImplementation((query: string) => ({
    matches: initial,
    media: query,
    addEventListener: (_type: string, cb: MediaQueryListener) => listeners.add(cb),
    removeEventListener: (_type: string, cb: MediaQueryListener) => listeners.delete(cb),
  }))
  vi.stubGlobal('matchMedia', matchMedia)
  return {
    emit(matches: boolean) {
      for (const cb of listeners) cb({ matches })
    },
  }
}

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('chartAnimation', () => {
  it('animates at the design-system duration by default', () => {
    expect(chartAnimation(false)).toEqual({
      isAnimationActive: true,
      animationDuration: CHART_MOTION_MS,
    })
    expect(CHART_MOTION_MS).toBe(150)
  })

  it('disables animation entirely under reduced motion', () => {
    expect(chartAnimation(true)).toEqual({
      isAnimationActive: false,
      animationDuration: 0,
    })
  })
})

describe('usePrefersReducedMotion', () => {
  it('reads the current OS preference on mount', () => {
    stubMatchMedia(true)
    const { result } = renderHook(() => usePrefersReducedMotion())
    expect(result.current).toBe(true)
  })

  it('defaults to animating when the preference is off', () => {
    stubMatchMedia(false)
    const { result } = renderHook(() => usePrefersReducedMotion())
    expect(result.current).toBe(false)
  })

  it('tracks live changes to the preference', () => {
    const media = stubMatchMedia(false)
    const { result } = renderHook(() => usePrefersReducedMotion())
    expect(result.current).toBe(false)
    act(() => media.emit(true))
    expect(result.current).toBe(true)
    act(() => media.emit(false))
    expect(result.current).toBe(false)
  })

  it('defaults to animating when matchMedia is unavailable', () => {
    vi.stubGlobal('matchMedia', undefined)
    const { result } = renderHook(() => usePrefersReducedMotion())
    expect(result.current).toBe(false)
  })
})
