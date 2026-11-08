'use client'

import { createStore } from 'zustand/vanilla'
import { useStore } from 'zustand'

export type BannerTone = 'info' | 'success' | 'warning' | 'destructive'

export interface BannerAction {
  label: string
  onAction: () => void
}

export interface Banner {
  id: string
  tone: BannerTone
  /** Headline of the banner — short, sentence case. */
  title: string
  /** Supporting detail; rendered muted after the title, em-dash separated. */
  message?: string
  /** Primary remedy — the one thing the user should be able to do next. */
  action?: BannerAction
  /** Producer's dedupe key: the banner that is showing gets replaced. */
  key?: string
}

interface BannerStore {
  /**
   * Single-slot store: at most ONE global banner exists at any time (see
   * `showBanner`). The array shape is kept for call-site/test compatibility —
   * `banners[0]` is the banner, or the array is empty.
   */
  banners: Banner[]
  showBanner: (banner: ShowBannerInput) => string
  dismissBanner: (id: string) => void
  clearBanners: () => void
}

export interface ShowBannerInput extends Omit<Banner, 'id'> {
  /** Auto-dismiss after this many ms. Omit for the default: sticky until dismissed. */
  ttlMs?: number
}

// Single-slot store, so one timer suffices. It is tagged with the id it
// belongs to: dismissing a SUPERSEDED id must never cancel the current
// banner's expiry (mirrors the no-op-dismiss rule in showBanner).
let bannerTimer: { id: string; handle: ReturnType<typeof setTimeout> } | null = null

const clearBannerTimer = () => {
  if (bannerTimer) {
    clearTimeout(bannerTimer.handle)
    bannerTimer = null
  }
}

const bannerStore = createStore<BannerStore>((set) => ({
  banners: [],

  showBanner: (banner) => {
    clearBannerTimer()
    const id = `banner_${Date.now()}_${Math.random().toString(36).slice(2, 6)}`
    const { ttlMs, ...rest } = banner
    const entry: Banner = { ...rest, id }
    // One global banner, ever. The store is a single slot: a new banner
    // displaces whatever is showing regardless of key, so two producers
    // reporting the same failure can never stack two rows. Dismissing a
    // superseded id is a no-op (it is already gone), which keeps producer
    // cleanup effects correct.
    set({ banners: [entry] })
    if (ttlMs !== undefined && ttlMs > 0) {
      bannerTimer = {
        id,
        handle: setTimeout(() => {
          // Supersede-safe: only expire the banner that is still showing.
          set((prev) => ({ banners: prev.banners.filter((b) => b.id !== id) }))
          bannerTimer = null
        }, ttlMs),
      }
    }
    return id
  },

  dismissBanner: (id) => {
    if (bannerTimer?.id === id) clearBannerTimer()
    set((prev) => ({ banners: prev.banners.filter((b) => b.id !== id) }))
  },

  clearBanners: () => {
    clearBannerTimer()
    set({ banners: [] })
  },
}))

export const useBannerStore = Object.assign(
  <T>(selector: (state: BannerStore) => T): T => useStore(bannerStore, selector),
  { getState: bannerStore.getState },
)
