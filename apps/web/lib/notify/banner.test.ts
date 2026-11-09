import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { clearBanners, dismissBanner, banner, useBannerStore } from './banner'

const list = () => useBannerStore.getState().banners

beforeEach(() => {
  clearBanners()
  vi.useFakeTimers()
})
afterEach(() => {
  vi.useRealTimers()
  clearBanners()
})

describe('notify banner channel (single slot)', () => {
  it('shows and dismisses a banner', () => {
    const id = banner({ tone: 'info', title: 'Hello' })
    expect(list()).toHaveLength(1)
    dismissBanner(id)
    expect(list()).toHaveLength(0)
  })

  it('keeps at most one banner — newest wins regardless of key', () => {
    banner({ tone: 'info', title: 'a', key: 'x' })
    banner({ tone: 'warning', title: 'b', key: 'y' })
    expect(list()).toHaveLength(1)
    expect(list()[0].title).toBe('b')
  })

  it('dismissing a superseded banner leaves the current one showing', () => {
    const first = banner({ tone: 'info', title: 'a' })
    banner({ tone: 'warning', title: 'b' })
    dismissBanner(first)
    expect(list()).toHaveLength(1)
    expect(list()[0].title).toBe('b')
  })

  it('clears all banners', () => {
    banner({ tone: 'info', title: 'a' })
    banner({ tone: 'info', title: 'b' })
    clearBanners()
    expect(list()).toHaveLength(0)
  })

  // ── timely (ttl) banners ────────────────────────────────────

  it('default banner is sticky — no expiry without ttlMs', () => {
    banner({ tone: 'info', title: 'sticky' })
    vi.advanceTimersByTime(600_000)
    expect(list()).toHaveLength(1)
  })

  it('ttl banner self-expires after its deadline', () => {
    banner({ tone: 'info', title: 'timed', ttlMs: 3000 })
    vi.advanceTimersByTime(2999)
    expect(list()).toHaveLength(1)
    vi.advanceTimersByTime(1)
    expect(list()).toHaveLength(0)
  })

  it('a superseded banner never expires the banner that replaced it', () => {
    banner({ tone: 'info', title: 'a', ttlMs: 3000 })
    vi.advanceTimersByTime(2000)
    banner({ tone: 'warning', title: 'b' }) // slot replaced — a's deadline is gone with it
    vi.advanceTimersByTime(5000)
    expect(list()).toHaveLength(1)
    expect(list()[0].title).toBe('b')
  })

  it('dismissing a superseded id does not cancel the current banner expiry', () => {
    const first = banner({ tone: 'info', title: 'a', ttlMs: 3000 })
    banner({ tone: 'warning', title: 'b', ttlMs: 3000 })
    dismissBanner(first) // no-op for b — b carries its own deadline
    vi.advanceTimersByTime(3500)
    expect(list()).toHaveLength(0) // b expired on its own
  })

  it('dismissing a ttl banner cancels its expiry', () => {
    const id = banner({ tone: 'info', title: 'gone', ttlMs: 3000 })
    dismissBanner(id)
    vi.advanceTimersByTime(10_000)
    expect(list()).toHaveLength(0)
  })

  it('two timed banners each expire on their own deadline', () => {
    banner({ tone: 'info', title: 'first', ttlMs: 1000 })
    vi.advanceTimersByTime(500)
    banner({ tone: 'info', title: 'second', ttlMs: 3000 })
    vi.advanceTimersByTime(500) // t=1000 — first is long superseded
    expect(list()[0].title).toBe('second')
    vi.advanceTimersByTime(2500) // t=3500 > 500+3000
    expect(list()).toHaveLength(0)
  })
})
