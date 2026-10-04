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
  showBanner: (banner: Omit<Banner, 'id'>) => string
  dismissBanner: (id: string) => void
  clearBanners: () => void
}

const bannerStore = createStore<BannerStore>((set) => ({
  banners: [],

  showBanner: (banner) => {
    const id = `banner_${Date.now()}_${Math.random().toString(36).slice(2, 6)}`
    const entry: Banner = { ...banner, id }
    // One global banner, ever. The store is a single slot: a new banner
    // displaces whatever is showing regardless of key, so two producers
    // reporting the same failure can never stack two rows. Dismissing a
    // superseded id is a no-op (it is already gone), which keeps producer
    // cleanup effects correct.
    set({ banners: [entry] })
    return id
  },

  dismissBanner: (id) => {
    set((prev) => ({ banners: prev.banners.filter((b) => b.id !== id) }))
  },

  clearBanners: () => {
    set({ banners: [] })
  },
}))

export const useBannerStore = Object.assign(
  <T>(selector: (state: BannerStore) => T): T => useStore(bannerStore, selector),
  { getState: bannerStore.getState },
)
