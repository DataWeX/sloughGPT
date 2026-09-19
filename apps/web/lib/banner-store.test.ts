import { describe, it, expect, beforeEach } from 'vitest'
import { useBannerStore } from './banner-store'

beforeEach(() => {
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

  it('clears all banners', () => {
    useBannerStore.getState().showBanner({ tone: 'info', title: 'a' })
    useBannerStore.getState().showBanner({ tone: 'info', title: 'b' })
    useBannerStore.getState().clearBanners()
    expect(useBannerStore.getState().banners).toHaveLength(0)
  })
})
