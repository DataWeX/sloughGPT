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
  title: string
  message?: string
  action?: BannerAction
  /** Dedupe key: showing a banner with an existing key replaces it. */
  key?: string
}

interface BannerStore {
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
    set((prev) => ({
      banners: banner.key
        ? [...prev.banners.filter((b) => b.key !== banner.key), entry]
        : [...prev.banners, entry],
    }))
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
