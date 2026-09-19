'use client'

import { memo } from 'react'

import { Button, StatusDot, IconX } from '@sloughgpt/strui'
import { useBannerStore, type BannerTone } from '@/lib/banner-store'

const TONE_MAP: Record<BannerTone, 'primary' | 'success' | 'warning' | 'destructive'> = {
  info: 'primary',
  success: 'success',
  warning: 'warning',
  destructive: 'destructive',
}

export const GlobalBanner = memo(function GlobalBanner() {
  const banners = useBannerStore((s) => s.banners)
  const dismissBanner = useBannerStore((s) => s.dismissBanner)
  if (banners.length === 0) return null
  return (
    <div aria-live="assertive">
      {banners.map((banner) => (
        <div key={banner.id} role="alert" className="sl-app-banner" data-tone={banner.tone}>
          <StatusDot tone={TONE_MAP[banner.tone]} pulse={banner.tone === 'warning'} />
          <span className="min-w-0 flex-1 truncate">
            <strong className="font-medium">{banner.title}</strong>
            {banner.message && <span className="opacity-80"> — {banner.message}</span>}
          </span>
          {banner.action && (
            <Button
              size="sm"
              variant="ghost"
              className="h-6 text-[11px] shrink-0"
              onClick={() => {
                banner.action?.onAction()
                dismissBanner(banner.id)
              }}
            >
              {banner.action.label}
            </Button>
          )}
          <Button
            size="sm"
            variant="ghost"
            className="h-6 w-6 shrink-0"
            onClick={() => dismissBanner(banner.id)}
            aria-label={`Dismiss: ${banner.title}`}
          >
            <IconX className="h-3 w-3" aria-hidden="true" />
          </Button>
        </div>
      ))}
    </div>
  )
})
