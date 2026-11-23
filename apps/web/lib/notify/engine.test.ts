import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import {
  DEFAULT_TTL_MS,
  UNDO_TTL_MS,
  MAX_VISIBLE_TOASTS,
  SWEEP_INTERVAL_MS,
  defaultTtl,
  expiryAt,
  isExpired,
  isTimed,
  makeRoom,
  registerProvider,
  resetEngine,
  resolveTag,
  stopTicker,
  syncTicker,
  type NotifyEntry,
} from './engine'

const entry = (id: string, patch: Partial<NotifyEntry> = {}): NotifyEntry => ({ id, ...patch })

beforeEach(() => {
  resetEngine()
  vi.useFakeTimers()
})

afterEach(() => {
  stopTicker()
  resetEngine()
  vi.useRealTimers()
})

describe('engine: policies', () => {
  it('resolveTag prefers an explicit key, else type:message', () => {
    expect(resolveTag('m', 'info')).toBe('info:m')
    expect(resolveTag('m', 'info', 'k')).toBe('k')
  })

  it('defaultTtl widens the window when an undo affordance is attached', () => {
    expect(defaultTtl()).toBe(DEFAULT_TTL_MS)
    expect(defaultTtl(() => {})).toBe(UNDO_TTL_MS)
    expect(defaultTtl(undefined)).toBe(DEFAULT_TTL_MS)
  })

  it('isTimed is false for sticky and for entries without a deadline', () => {
    expect(isTimed(entry('a', { dismissAt: 100 }))).toBe(true)
    expect(isTimed(entry('b', { dismissAt: 100, sticky: true }))).toBe(false)
    expect(isTimed(entry('c'))).toBe(false)
    expect(isTimed(entry('d', { sticky: true }))).toBe(false)
  })

  it('isExpired compares against the supplied clock', () => {
    expect(isExpired(entry('a', { dismissAt: 100 }), 99)).toBe(false)
    expect(isExpired(entry('a', { dismissAt: 100 }), 100)).toBe(true)
    expect(isExpired(entry('a'), 10_000)).toBe(false)
  })

  it('expiryAt is now + ttl', () => {
    expect(expiryAt(1000, 500)).toBe(1500)
  })
})

describe('engine: makeRoom (bounded stack)', () => {
  it('makes no room below the bound', () => {
    const list = [entry('1'), entry('2'), entry('3')]
    expect(makeRoom(list, 5)).toHaveLength(3)
  })

  it('evicts the oldest non-sticky entry when at capacity', () => {
    const list = [
      entry('1'),
      entry('2', { sticky: true }),
      entry('3'),
      entry('4', { sticky: true }),
      entry('5'),
    ]
    expect(makeRoom(list, 5).map((e) => e.id)).toEqual(['2', '3', '4', '5'])
  })

  it('evicts the oldest (sticky) entry when the stack is all-sticky', () => {
    const list = [
      entry('1', { sticky: true }),
      entry('2', { sticky: true }),
      entry('3', { sticky: true }),
    ]
    expect(makeRoom(list, 3).map((e) => e.id)).toEqual(['2', '3'])
  })

  it('never mutates the input list', () => {
    const list = [entry('1'), entry('2'), entry('3')]
    makeRoom(list, 3)
    expect(list).toHaveLength(3)
  })
})

describe('engine: one shared ticker', () => {
  const makeProvider = () => {
    const state = { items: [] as NotifyEntry[], sweeps: 0 }
    return {
      state,
      provider: {
        needs: () => state.items.some(isTimed),
        sweep: () => {
          state.sweeps += 1
          state.items = state.items.filter((e) => !isExpired(e, Date.now()))
        },
      },
    }
  }

  it('does not tick while no provider has timed entries', () => {
    const { state, provider } = makeProvider()
    registerProvider(provider)
    vi.advanceTimersByTime(10_000)
    expect(state.sweeps).toBe(0)
  })

  it('ticks while a provider has timed entries and stops when idle', () => {
    const { state, provider } = makeProvider()
    registerProvider(provider)
    state.items = [entry('a', { dismissAt: Date.now() + DEFAULT_TTL_MS })]
    syncTicker()
    vi.advanceTimersByTime(DEFAULT_TTL_MS + SWEEP_INTERVAL_MS)
    expect(state.items).toHaveLength(0)
    const sweepsAfterExpiry = state.sweeps
    // idle now — no further sweeps
    vi.advanceTimersByTime(10_000)
    expect(state.sweeps).toBe(sweepsAfterExpiry)
  })

  it('MAX_VISIBLE_TOASTS is the public stack bound', () => {
    expect(MAX_VISIBLE_TOASTS).toBe(5)
  })
})
