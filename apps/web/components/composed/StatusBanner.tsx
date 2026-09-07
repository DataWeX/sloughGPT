'use client'

import { useState } from 'react'
import { cn } from '@sloughgpt/strui'

type BannerVariant = 'info' | 'success' | 'warning' | 'error'

interface StatusBannerProps {
  variant?: BannerVariant
  message: string
  dismissible?: boolean
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
  className,
}: StatusBannerProps) {
  const [dismissed, setDismissed] = useState(false)

  if (dismissed) return null

  return (
    <div
      className={cn(
        'rounded-md border px-4 py-3 text-sm flex items-center justify-between',
        variantClasses[variant],
        className,
      )}
      role={variant === 'error' ? 'alert' : 'status'}
    >
      <span>{message}</span>
      {dismissible && (
        <button
          type="button"
          className="ml-2 underline text-current opacity-80 hover:opacity-100"
          onClick={() => setDismissed(true)}
          aria-label="Dismiss"
        >
          Dismiss
        </button>
      )}
    </div>
  )
}
