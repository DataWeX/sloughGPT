import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { renderHook, act } from '@testing-library/react'
import { useConsciousnessLive } from './useConsciousnessLive'
import { emitConsciousness } from '@/lib/consciousness-bus'
import type { ConsciousnessEvent } from '@/lib/consciousness-bus'

function makeEvent(overrides: Partial<ConsciousnessEvent> = {}): ConsciousnessEvent {
  return {
    level: 2,
    qualia: { joy: 0.5 },
    beliefs: {},
    growth_delta: 0.1,
    self_insight: 'test',
    ...overrides,
  }
}

describe('useConsciousnessLive', () => {
  it('starts with isLive true and no lastUpdate', () => {
    const { result } = renderHook(() => useConsciousnessLive())
    expect(result.current.isLive).toBe(true)
    expect(result.current.lastUpdate).toBeNull()
    expect(result.current.latestEvent).toBeNull()
  })

  it('updates latestEvent when live', () => {
    const { result } = renderHook(() => useConsciousnessLive())

    const event = makeEvent()
    act(() => {
      emitConsciousness(event)
    })

    expect(result.current.latestEvent).toEqual(event)
    expect(result.current.lastUpdate).toBeTypeOf('number')
  })

  it('ignores events when not live', () => {
    const { result } = renderHook(() => useConsciousnessLive())

    act(() => {
      result.current.toggleLive()
    })
    expect(result.current.isLive).toBe(false)

    act(() => {
      emitConsciousness(makeEvent())
    })

    expect(result.current.latestEvent).toBeNull()
    expect(result.current.lastUpdate).toBeNull()
  })

  it('toggleLive flips isLive', () => {
    const { result } = renderHook(() => useConsciousnessLive())

    expect(result.current.isLive).toBe(true)
    act(() => { result.current.toggleLive() })
    expect(result.current.isLive).toBe(false)
    act(() => { result.current.toggleLive() })
    expect(result.current.isLive).toBe(true)
  })

  it('resumes receiving events after toggling back to live', () => {
    const { result } = renderHook(() => useConsciousnessLive())

    act(() => { result.current.toggleLive() })
    act(() => { emitConsciousness(makeEvent()) })
    expect(result.current.latestEvent).toBeNull()

    act(() => { result.current.toggleLive() })
    const event = makeEvent({ level: 5 })
    act(() => { emitConsciousness(event) })
    expect(result.current.latestEvent).toEqual(event)
  })

  it('cleans up subscription on unmount', () => {
    const { unmount } = renderHook(() => useConsciousnessLive())
    unmount()
    expect(() => emitConsciousness(makeEvent())).not.toThrow()
  })
})
