'use client'

import { memo } from 'react'
import { cn, Button } from '@sloughgpt/strui'
import { IconAlert, IconInfo, IconCloudOff, IconX } from '@sloughgpt/strui'

export type SystemBannerType = 'offline' | 'warning' | 'info'

interface SystemBannerProps {
  type: SystemBannerType
  title: string
  message?: string
  actionLabel?: string
  onAction?: () => void
  onDismiss?: () => void
}

const STYLES: Record<SystemBannerType, string> = {
  offline: 'border-warning/30 bg-warning/5 text-warning',
  warning: 'border-warning/30 bg-warning/5 text-warning',
  info: 'border-primary/30 bg-primary/5 text-primary',
}

const ICONS: Record<SystemBannerType, React.ReactNode> = {
  offline: <IconCloudOff className="h-4 w-4" aria-hidden="true" />,
  warning: <IconAlert className="h-4 w-4" aria-hidden="true" />,
  info: <IconInfo className="h-4 w-4" aria-hidden="true" />,
}

export const SystemBanner = memo(function SystemBanner({
  type,
  title,
  message,
  actionLabel,
  onAction,
  onDismiss,
}: SystemBannerProps) {
  return (
    <div
      className={cn('mb-3 rounded-lg border p-3 text-xs', STYLES[type])}
      role="alert"
      aria-live="assertive"
    >
      <div className="flex items-start gap-2">
        <span className="shrink-0 mt-0.5">{ICONS[type]}</span>
        <div className="flex-1 min-w-0">
          <p className="font-medium">{title}</p>
          {message && <p className="mt-1 opacity-80">{message}</p>}
        </div>
        <div className="flex shrink-0 items-start gap-1">
          {actionLabel && onAction && (
            <Button variant="outline" size="sm" onClick={onAction} className="h-7 text-xs">
              {actionLabel}
            </Button>
          )}
          {onDismiss && (
            <Button
              variant="ghost"
              size="icon-sm"
              onClick={onDismiss}
              className="h-7 w-7"
              aria-label="Dismiss notification"
            >
              <IconX className="h-3.5 w-3.5" aria-hidden="true" />
            </Button>
          )}
        </div>
      </div>
    </div>
  )
})
