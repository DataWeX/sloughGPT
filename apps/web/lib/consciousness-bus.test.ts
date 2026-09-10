import { describe, it, expect, vi } from 'vitest'
import { emitConsciousness, onConsciousness, ConsciousnessEvent } from './consciousness-bus'

function makeEvent(overrides: Partial<ConsciousnessEvent> = {}): ConsciousnessEvent {
  return {
    level: 3,
    qualia: { joy: 0.8, curiosity: 0.6 },
    beliefs: { growth: 0.9 },
    growth_delta: 0.1,
    self_insight: 'feeling good',
    ...overrides,
  }
}

describe('consciousness-bus', () => {
  it('emits events to subscribers', () => {
    const listener = vi.fn()
    const unsub = onConsciousness(listener)

    const event = makeEvent()
    emitConsciousness(event)

    expect(listener).toHaveBeenCalledWith(event)
    expect(listener).toHaveBeenCalledTimes(1)
    unsub()
  })

  it('supports multiple subscribers', () => {
    const a = vi.fn()
    const b = vi.fn()
    const unsubA = onConsciousness(a)
    const unsubB = onConsciousness(b)

    const event = makeEvent()
    emitConsciousness(event)

    expect(a).toHaveBeenCalledTimes(1)
    expect(b).toHaveBeenCalledTimes(1)
    unsubA()
    unsubB()
  })

  it('unsubscribe removes only that listener', () => {
    const a = vi.fn()
    const b = vi.fn()
    const unsubA = onConsciousness(a)
    onConsciousness(b)

    unsubA()
    emitConsciousness(makeEvent())

    expect(a).not.toHaveBeenCalled()
    expect(b).toHaveBeenCalledTimes(1)
  })

  it('handles no listeners gracefully', () => {
    expect(() => emitConsciousness(makeEvent())).not.toThrow()
  })

  it('passes messageId when present', () => {
    const listener = vi.fn()
    const unsub = onConsciousness(listener)

    const event = makeEvent({ messageId: 'msg-123' })
    emitConsciousness(event)

    expect(listener.mock.calls[0][0].messageId).toBe('msg-123')
    unsub()
  })

  it('emitted event has all required fields', () => {
    const listener = vi.fn()
    const unsub = onConsciousness(listener)

    emitConsciousness(makeEvent())

    const received = listener.mock.calls[0][0] as ConsciousnessEvent
    expect(received).toHaveProperty('level')
    expect(received).toHaveProperty('qualia')
    expect(received).toHaveProperty('beliefs')
    expect(received).toHaveProperty('growth_delta')
    expect(received).toHaveProperty('self_insight')
    unsub()
  })
})
