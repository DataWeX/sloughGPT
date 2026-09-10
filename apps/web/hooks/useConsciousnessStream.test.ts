import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { renderHook, act } from '@testing-library/react'
import { useConsciousnessStream } from './useConsciousnessStream'
import { emitConsciousness, onConsciousness } from '@/lib/consciousness-bus'
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

describe('useConsciousnessStream', () => {
  it('returns null latest initially', () => {
    const { result } = renderHook(() => useConsciousnessStream({ messageId: 'msg-1' }))
    expect(result.current.latest).toBeNull()
  })

  it('updates latest when event emitted with matching messageId', () => {
    const { result } = renderHook(() => useConsciousnessStream({ messageId: 'msg-1' }))

    act(() => {
      emitConsciousness(makeEvent({ messageId: 'msg-1' }))
    })

    expect(result.current.getLatest()).not.toBeNull()
    expect(result.current.getLatest()!.messageId).toBe('msg-1')
  })

  it('calls onConsciousnessUpdate callback', () => {
    const cb = vi.fn()
    renderHook(() =>
      useConsciousnessStream({ messageId: 'msg-1', onConsciousnessUpdate: cb })
    )

    const event = makeEvent({ messageId: 'msg-1' })
    act(() => {
      emitConsciousness(event)
    })

    expect(cb).toHaveBeenCalledWith(event)
  })

  it('does not subscribe when no messageId provided', () => {
    const cb = vi.fn()
    renderHook(() => useConsciousnessStream({ onConsciousnessUpdate: cb }))

    act(() => {
      emitConsciousness(makeEvent())
    })

    expect(cb).not.toHaveBeenCalled()
  })

  it('cleans up subscription on unmount', () => {
    const { unmount } = renderHook(() =>
      useConsciousnessStream({ messageId: 'msg-1' })
    )
    unmount()

    // Should not throw after unmount
    expect(() => emitConsciousness(makeEvent())).not.toThrow()
  })
})
