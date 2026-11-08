import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { useBannerStore } from './banner-store'

beforeEach(() => {
  useBannerStore.getState().clearBanners()
})

afterEach(() => {
  vi.useRealTimers()
  useBannerStore.getState().clearBanners()
})

describe('banner-store', () => {
  it('shows and dismisses a banner', () => {
    const id = useBannerStore.getState().showBanner({ tone: 'info', title: 'Hello' })
    expect(useBannerStore.getState().banners).toHaveLength(1)
    useBannerStore.getState().dismissBanner(id)
    expect(useBannerStore.getState().banners).toHaveLength(0)
  })

  it('replaces banners with the same key', () => {
    useBannerStore.getState().showBanner({ tone: 'info', title: 'v1', key: 'train' })
    useBannerStore.getState().showBanner({ tone: 'warning', title: 'v2', key: 'train' })
    const banners = useBannerStore.getState().banners
    expect(banners).toHaveLength(1)
    expect(banners[0].title).toBe('v2')
  })

  it('keeps at most one banner — newest wins regardless of key', () => {
    useBannerStore.getState().showBanner({ tone: 'info', title: 'a', key: 'x' })
    useBannerStore.getState().showBanner({ tone: 'warning', title: 'b', key: 'y' })
    const banners = useBannerStore.getState().banners
    expect(banners).toHaveLength(1)
    expect(banners[0].title).toBe('b')
  })

  it('dismissing a superseded banner leaves the current one showing', () => {
    const first = useBannerStore.getState().showBanner({ tone: 'info', title: 'a' })
    useBannerStore.getState().showBanner({ tone: 'warning', title: 'b' })
    useBannerStore.getState().dismissBanner(first)
    const banners = useBannerStore.getState().banners
    expect(banners).toHaveLength(1)
    expect(banners[0].title).toBe('b')
  })

  it('clears all banners', () => {
    useBannerStore.getState().showBanner({ tone: 'info', title: 'a' })
    useBannerStore.getState().showBanner({ tone: 'info', title: 'b' })
    useBannerStore.getState().clearBanners()
    expect(useBannerStore.getState().banners).toHaveLength(0)
  })

  // ── timely (ttl) banners ────────────────────────────────────

  it('default banner is sticky — no expiry without ttlMs', () => {
    vi.useFakeTimers()
    useBannerStore.getState().showBanner({ tone: 'info', title: 'sticky' })
    vi.advanceTimersByTime(600_000)
    expect(useBannerStore.getState().banners).toHaveLength(1)
  })

  it('ttl banner self-expires after its deadline', () => {
    vi.useFakeTimers()
    useBannerStore.getState().showBanner({ tone: 'info', title: 'timed', ttlMs: 3000 })
    vi.advanceTimersByTime(2999)
    expect(useBannerStore.getState().banners).toHaveLength(1)
    vi.advanceTimersByTime(1)
    expect(useBannerStore.getState().banners).toHaveLength(0)
  })

  it('a superseded banner never expires the banner that replaced it', () => {
    vi.useFakeTimers()
    useBannerStore.getState().showBanner({ tone: 'info', title: 'a', ttlMs: 3000 })
    vi.advanceTimersByTime(2000)
    // Supersede: showBanner clears the old timer before setting the new slot
    useBannerStore.getState().showBanner({ tone: 'warning', title: 'b' })
    vi.advanceTimersByTime(5000) // old deadline (t=3000) passes while b shows
    const banners = useBannerStore.getState().banners
    expect(banners).toHaveLength(1)
    expect(banners[0].title).toBe('b')
  })

  it('dismissing a superseded id does not cancel the current banner expiry', () => {
    vi.useFakeTimers()
    const first = useBannerStore.getState().showBanner({ tone: 'info', title: 'a', ttlMs: 3000 })
    useBannerStore.getState().showBanner({ tone: 'warning', title: 'b', ttlMs: 3000 })
    useBannerStore.getState().dismissBanner(first) // must be a no-op on b's timer
    vi.advanceTimersByTime(3500)
    expect(useBannerStore.getState().banners).toHaveLength(0) // b expired on its own
  })

  it('dismissing a ttl banner cancels its expiry', () => {
    vi.useFakeTimers()
    const id = useBannerStore.getState().showBanner({ tone: 'info', title: 'gone', ttlMs: 3000 })
    useBannerStore.getState().dismissBanner(id)
    vi.advanceTimersByTime(10_000)
    expect(useBannerStore.getState().banners).toHaveLength(0)
  })
})
