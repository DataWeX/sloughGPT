'use client'

import * as React from 'react'
import { cn } from '../../lib/cn'

export interface ProgressProps {
  value?: number
  max?: number
  /** Variant controls fill color using design tokens */
  variant?: 'default' | 'success' | 'warning' | 'error'
  /** Track height */
  size?: 'xs' | 'sm' | 'default' | 'lg'
  /** Animated indeterminate state (ignores value) */
  indeterminate?: boolean
  label?: string
  showValue?: boolean
  className?: string
}

const variantClasses = {
  default: 'bg-primary',
  success: 'bg-success',
  warning: 'bg-warning',
  error: 'bg-destructive',
}

const sizeClasses = {
  xs: 'h-1',
  sm: 'h-1.5',
  default: 'h-2',
  lg: 'h-3',
}

export function Progress({
  value = 0,
  max = 100,
  variant = 'default',
  size = 'default',
  indeterminate = false,
  label,
  showValue = false,
  className = '',
}: ProgressProps) {
  const pct = indeterminate ? 0 : Math.min(100, Math.max(0, (value / max) * 100))

  return (
    <div className={cn('w-full space-y-1', className)}>
      {(label || showValue) && (
        <div className="flex justify-between text-xs text-muted-foreground">
          {label && <span>{label}</span>}
          {showValue && !indeterminate && (
            <span className="font-medium text-foreground">{Math.round(pct)}%</span>
          )}
        </div>
      )}
      <div
        className={cn('w-full overflow-hidden rounded-full bg-muted', sizeClasses[size])}
        role="progressbar"
        aria-valuenow={indeterminate ? undefined : value}
        aria-valuemin={0}
        aria-valuemax={indeterminate ? undefined : max}
        aria-label={label}
        aria-busy={indeterminate}
      >
        {indeterminate ? (
          <div
            className={cn(
              'h-full w-1/3 rounded-full animate-[progress-indeterminate_1.4s_ease-in-out_infinite]',
              variantClasses[variant],
            )}
          />
        ) : (
          <div
            className={cn('h-full rounded-full transition-all duration-500 ease-smooth', variantClasses[variant])}
            style={{ width: `${pct}%` }}
          />
        )}
      </div>
    </div>
  )
}
