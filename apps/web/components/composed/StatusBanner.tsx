'use client'

import { useState } from 'react'
import { cn } from '@sloughgpt/strui'

type BannerVariant = 'info' | 'success' | 'warning' | 'error'

interface StatusBannerProps {
  variant?: BannerVariant
  message: string
  dismissible?: boolean
  onRetry?: () => void
  onDismiss?: () => void
  className?: string
}

const variantClasses: Record<BannerVariant, string> = {
  info: 'bg-primary/10 border-primary/20 text-primary',
  success: 'bg-success/10 border-success/20 text-success',
  warning: 'bg-warning/10 border-warning/20 text-warning',
  error: 'bg-destructive/10 border-destructive/20 text-destructive',
}

export function StatusBanner({
  variant = 'info',
  message,
  dismissible = true,
  onRetry,
  onDismiss,
  className,
}: StatusBannerProps) {
  const [dismissed, setDismissed] = useState(false)

  if (dismissed) return null

  const handleDismiss = () => {
    setDismissed(true)
    onDismiss?.()
  }

  return (
    <div
      className={cn(
        'rounded-lg border px-2.5 py-1.5 text-[11px] flex items-center justify-between',
        variantClasses[variant],
        className,
      )}
      role={variant === 'error' ? 'alert' : 'status'}
    >
      <span>{message}</span>
      <div className="flex items-center gap-1.5 ml-1.5">
        {onRetry && (
          <button
            type="button"
            className="underline text-current opacity-80 hover:opacity-100 text-[10px]"
            onClick={onRetry}
            aria-label="Retry"
          >
            Retry
          </button>
        )}
        {(dismissible || onDismiss) && (
          <button
            type="button"
            className="underline text-current opacity-80 hover:opacity-100 text-[10px]"
            onClick={handleDismiss}
            aria-label="Dismiss"
          >
            Dismiss
          </button>
        )}
      </div>
    </div>
  )
}
