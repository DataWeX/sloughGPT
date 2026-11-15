import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { MAX_VISIBLE_TOASTS } from './engine'
import { clearToasts, dismissToast, toast, useToastStore } from './toast'

const list = () => useToastStore.getState().toasts

beforeEach(() => {
  clearToasts()
  vi.useFakeTimers()
})
afterEach(() => {
  vi.useRealTimers()
  clearToasts()
})

describe('notify toast channel', () => {
  it('starts empty', () => {
    expect(list()).toEqual([])
  })

  it('announces a toast with message and type', () => {
    const id = toast('Hello', { type: 'success' })
    const t = list().find((x) => x.id === id)
    expect(t?.message).toBe('Hello')
    expect(t?.type).toBe('success')
  })

  it('defaults type to info', () => {
    const id = toast('Hello')
    expect(list().find((x) => x.id === id)?.type).toBe('info')
  })

  it('stores verbose and undo affordances', () => {
    const undo = vi.fn()
    const id = toast('Hello', { type: 'error', verbose: 'Details here', onUndo: undo })
    const t = list().find((x) => x.id === id)
    expect(t?.verbose).toBe('Details here')
    expect(t?.onUndo).toBe(undo)
    // undo toasts get the wider default window
    expect(t?.dismissAt).toBe(Date.now() + 8000)
  })

  it('dismissToast removes a toast; clearToasts empties the channel', () => {
    const id = toast('Hello')
    expect(list()).toHaveLength(1)
    dismissToast(id)
    expect(list()).toHaveLength(0)
    toast('A')
    toast('B')
    clearToasts()
    expect(list()).toEqual([])
  })

  it('auto-removes a default toast after 6 seconds', () => {
    toast('Hello')
    expect(list()).toHaveLength(1)
    vi.advanceTimersByTime(6000)
    expect(list()).toHaveLength(0)
  })

  it('generates unique ids', () => {
    expect(toast('A')).not.toBe(toast('B'))
  })

  // ── dedupe ──────────────────────────────────────────────────

  it('dedupes identical visible message+type and returns the existing id', () => {
    const id1 = toast('Session expired')
    const id2 = toast('Session expired')
    expect(id2).toBe(id1)
    expect(list()).toHaveLength(1)
  })

  it('allows the same message again after the toast expires', () => {
    const id1 = toast('Session expired')
    vi.advanceTimersByTime(6000)
    const id2 = toast('Session expired')
    expect(id2).not.toBe(id1)
    expect(list()).toHaveLength(1)
  })

  it('dedupes by explicit key even when messages differ', () => {
    const id1 = toast('Previous training session expired — server was restarted', {
      key: 'training-session-expired',
    })
    const id2 = toast('Session expired', { key: 'training-session-expired' })
    expect(id2).toBe(id1)
    expect(list()).toHaveLength(1)
  })

  it('does not dedupe a keyed toast against an unkeyed toast with the same message', () => {
    toast('Session expired', { key: 'k-session' })
    toast('Session expired')
    expect(list()).toHaveLength(2)
  })

  it('keeps toasts with the same message but different keys distinct', () => {
    toast('Session expired', { key: 'k-a' })
    toast('Session expired', { key: 'k-b' })
    expect(list()).toHaveLength(2)
  })

  // ── sticky / timely persistence ─────────────────────────────

  it('sticky toast never auto-expires', () => {
    toast('Training paused', { sticky: true })
    vi.advanceTimersByTime(60_000)
    expect(list()).toHaveLength(1)
    expect(list()[0].sticky).toBe(true)
    expect(list()[0].dismissAt).toBeUndefined()
  })

  it('honors a custom ttlMs', () => {
    toast('Quick note', { ttlMs: 1500 })
    vi.advanceTimersByTime(1499)
    expect(list()).toHaveLength(1)
    vi.advanceTimersByTime(1)
    expect(list()).toHaveLength(0)
  })

  it('dedupe hit refreshes the expiry deadline', () => {
    toast('Progress')
    vi.advanceTimersByTime(5000)
    toast('Progress') // re-announcement resets the clock to t=5000+6000
    vi.advanceTimersByTime(5000) // t=10000 — would be dead without the refresh
    expect(list()).toHaveLength(1)
    vi.advanceTimersByTime(1500)
    expect(list()).toHaveLength(0)
  })

  it('dedupe hit upgrades an existing toast to sticky', () => {
    toast('Progress')
    toast('Progress', { sticky: true })
    vi.advanceTimersByTime(60_000)
    expect(list()).toHaveLength(1)
    expect(list()[0].sticky).toBe(true)
  })

  // ── bounded stack ───────────────────────────────────────────

  it('caps the stack, evicting the oldest non-sticky toast first', () => {
    toast('Pinned', { sticky: true })
    for (let i = 1; i <= 5; i++) toast(`Msg ${i}`)
    const visible = list()
    expect(visible).toHaveLength(MAX_VISIBLE_TOASTS)
    expect(visible.find((t) => t.message === 'Pinned')).toBeTruthy()
    expect(visible.find((t) => t.message === 'Msg 1')).toBeUndefined()
    expect(visible.find((t) => t.message === 'Msg 5')).toBeTruthy()
  })

  it('bounds an all-sticky stack by evicting the oldest', () => {
    for (let i = 1; i <= 6; i++) toast(`Sticky ${i}`, { sticky: true })
    const visible = list()
    expect(visible).toHaveLength(MAX_VISIBLE_TOASTS)
    expect(visible.find((t) => t.message === 'Sticky 1')).toBeUndefined()
    expect(visible.find((t) => t.message === 'Sticky 6')).toBeTruthy()
  })
})
