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
  // Single-slot store: `banners[0]` IS the banner. Rendering it explicitly
  // (instead of mapping) keeps the view aligned with the one-banner invariant.
  const banner = useBannerStore((s) => s.banners[0])
  const dismissBanner = useBannerStore((s) => s.dismissBanner)
  if (!banner) return null
  return (
    <div aria-live="assertive">
      <div role="alert" className="sl-app-banner" data-tone={banner.tone}>
        <StatusDot tone={TONE_MAP[banner.tone]} pulse={banner.tone === 'warning'} />
        {/* Centred, width-capped cluster: title + message + action read as one
            unit. Without the cap the text span grew to full bleed and shoved
            the action to the far edge — a status bar, not a message. */}
        <span className="flex min-w-0 max-w-[44rem] items-baseline gap-1.5">
          <strong className="shrink-0 font-medium">{banner.title}</strong>
          {banner.message && (
            <span className="min-w-0 truncate opacity-80"> — {banner.message}</span>
          )}
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
    </div>
  )
})
