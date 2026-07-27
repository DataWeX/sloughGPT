import type { HTMLAttributes } from 'react'

import { cn } from '../../lib/cn'

export interface ProgressBarProps extends Omit<HTMLAttributes<HTMLDivElement>, 'children'> {
  /** 0–`max` (ignored when `indeterminate`). */
  value?: number
  max?: number
  /** Visually indeterminate (ignores value). */
  indeterminate?: boolean
  /** Color variant for the fill bar. */
  variant?: 'default' | 'success' | 'warning' | 'error'
  /** Optional label shown above the bar. */
  label?: string
  /** Show percentage text above the bar. */
  showValue?: boolean
  /** Track height size. */
  size?: 'xs' | 'sm' | 'default' | 'lg'
}

const variantColors = {
  default: 'bg-primary',
  success: 'bg-success',
  warning: 'bg-warning',
  error: 'bg-destructive',
} as const

const sizeHeights = {
  xs: 'h-1',
  sm: 'h-1.5',
  default: 'h-2',
  lg: 'h-3',
} as const

/** Accessible linear progress for jobs, uploads, and context fill. */
export function ProgressBar({
  value = 0,
  max = 100,
  className,
  indeterminate,
  variant = 'default',
  label,
  showValue = false,
  size = 'default',
  ...props
}: ProgressBarProps) {
  const pct = Math.min(100, Math.max(0, (value / max) * 100))

  return (
    <div className={cn('space-y-1', className)}>
      {(label || showValue) && (
        <div className="flex justify-between text-xs text-muted-foreground">
          {label && <span>{label}</span>}
          {showValue && <span className="font-medium text-foreground">{Math.round(pct)}%</span>}
        </div>
      )}
      <div
        role="progressbar"
        aria-valuemin={0}
        aria-valuemax={max}
        aria-valuenow={indeterminate ? undefined : Math.round(value)}
        aria-valuetext={indeterminate ? undefined : `${Math.round(pct)}%`}
        className={cn('w-full overflow-hidden rounded-full bg-muted', sizeHeights[size])}
        {...props}
      >
        {indeterminate ? (
          <div className={cn('absolute inset-y-0 left-0 animate-pulse rounded-full opacity-60', variantColors[variant])} style={{ width: '100%' }} />
        ) : (
          <div
            className={cn('absolute inset-y-0 left-0 h-full rounded-full transition-[width] duration-300 ease-smooth', variantColors[variant])}
            style={{ width: `${pct}%` }}
          />
        )}
      </div>
    </div>
  )
}
