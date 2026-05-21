'use client'

import { Button } from '@/components/ui/button'
import { IconAlert, IconInfo } from '@/components/ui'

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
  offline: (
    <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden>
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M18.364 5.636a9 9 0 010 12.728m0 0l-2.829-2.829m2.829 2.829L21 21M15.536 8.464a5 5 0 010 7.072m0 0l-2.829-2.829m-4.243 2.829a4.978 4.978 0 01-1.414-2.83m-1.414 5.658a9 9 0 01-2.167-9.238m7.824 2.167a1 1 0 111.414 1.414m-1.414-1.414L3 3m8.293 8.293l1.414 1.414" />
    </svg>
  ),
  warning: <IconAlert className="h-4 w-4" />,
  info: <IconInfo className="h-4 w-4" />,
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
              Dismiss
            </Button>
          )}
        </div>
      </div>
    </div>
  )
}
