'use client'

import { Button } from '@sloughgpt/strui'
import { IconAlert, IconInfo, IconCloudOff } from '@sloughgpt/strui'

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

export function SystemBanner({ type, title, message, actionLabel, onAction, onDismiss }: SystemBannerProps) {
  return (
    <div
      className={`mb-3 rounded-lg border p-3 text-xs ${STYLES[type]}`}
      role="alert"
      aria-live="assertive"
    >
      <div className="flex items-start gap-2">
        <span className="shrink-0 mt-0.5">{ICONS[type]}</span>
        <div className="flex-1 min-w-0">
          <p className="font-medium">{title}</p>
          {message && <p className="mt-1 opacity-80">{message}</p>}
        </div>
        <div className="flex shrink-0 gap-2">
          {actionLabel && onAction && (
            <Button
              variant="outline"
              size="sm"
              onClick={onAction}
              className="h-7 text-xs"
            >
              {actionLabel || 'Dismiss'}
            </Button>
          )}
        </div>
      </div>
    </div>
  )
}
