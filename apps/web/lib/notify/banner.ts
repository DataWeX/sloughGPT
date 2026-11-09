/**
 * banner channel — single slot: at most ONE global banner, ever.
 *
 * A new banner displaces whatever is showing regardless of key (rule 16 —
 * two producers can never stack two journey-blocker rows). Timed banners
 * (`ttlMs`) carry their own `dismissAt` and ride the library's shared
 * ticker, so supersede safety is structural: the replacement's deadline is
 * the only one that exists.
 */

import { createChannel } from './channel'
import { expiryAt } from './engine'
import type { Banner, BannerInput } from './types'

export interface BannerState {
  banners: Banner[]
  dismissBanner: (id: string) => void
  clearBanners: () => void
}

const channel = createChannel<Banner, BannerState>({
  key: 'banners',
  topology: 'slot',
  dismissName: 'dismissBanner',
  clearName: 'clearBanners',
})

/** Show a banner (single slot — newest wins). Returns the banner's id. */
export function banner(input: BannerInput): string {
  const { ttlMs, ...rest } = input
  const id = `banner_${Date.now()}_${Math.random().toString(36).slice(2, 6)}`
  const timed = ttlMs !== undefined && ttlMs > 0
  channel.put({
    ...rest,
    id,
    sticky: timed ? false : undefined,
    dismissAt: timed ? expiryAt(Date.now(), ttlMs) : undefined,
  })
  return id
}

export const useBannerStore = Object.assign(channel.useChannel, { getState: channel.getState })
export const dismissBanner = channel.remove
export const clearBanners = channel.clear
