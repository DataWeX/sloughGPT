import { describe, it, expect, vi, afterEach } from 'vitest'
import { subscribeMemoryEvents, publishMemoryEvent } from './memory-events'

afterEach(() => {
  vi.restoreAllMocks()
})

describe('memory-events pub/sub', () => {
  it('delivers the event payload to a subscriber', () => {
    const listener = vi.fn()
    subscribeMemoryEvents(listener)
    publishMemoryEvent({ stored: true })
    expect(listener).toHaveBeenCalledWith({ stored: true })
  })

  it('passes the stored fact text through', () => {
    const listener = vi.fn()
    subscribeMemoryEvents(listener)
    publishMemoryEvent({ stored: true, fact: 'The capital of France is Paris.' })
    expect(listener).toHaveBeenCalledWith({ stored: true, fact: 'The capital of France is Paris.' })
  })

  it('passes the full facts list through', () => {
    const listener = vi.fn()
    subscribeMemoryEvents(listener)
    publishMemoryEvent({ stored: true, fact: 'A.', facts: ['A.', 'B.', 'C.'] })
    expect(listener).toHaveBeenCalledWith({ stored: true, fact: 'A.', facts: ['A.', 'B.', 'C.'] })
  })

  it('notifies all subscribers', () => {
    const a = vi.fn()
    const b = vi.fn()
    subscribeMemoryEvents(a)
    subscribeMemoryEvents(b)
    publishMemoryEvent({ stored: false })
    expect(a).toHaveBeenCalledTimes(1)
    expect(b).toHaveBeenCalledTimes(1)
  })

  it('stops notifying after unsubscribe', () => {
    const listener = vi.fn()
    const unsubscribe = subscribeMemoryEvents(listener)
    publishMemoryEvent({ stored: true })
    unsubscribe()
    publishMemoryEvent({ stored: true })
    expect(listener).toHaveBeenCalledTimes(1)
  })

  it('isolates a throwing listener from the others', () => {
    const bad = vi.fn(() => { throw new Error('boom') })
    const good = vi.fn()
    subscribeMemoryEvents(bad)
    subscribeMemoryEvents(good)
    expect(() => publishMemoryEvent({ stored: true })).not.toThrow()
    expect(good).toHaveBeenCalledTimes(1)
  })

  it('swallows a throwing listener result', () => {
    const bad = vi.fn(() => { throw new Error('boom') })
    subscribeMemoryEvents(bad)
    publishMemoryEvent({ stored: true })
    expect(bad).toHaveBeenCalledTimes(1)
  })
})
