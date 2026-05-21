import type { HTMLAttributes } from 'react'

import { cn } from '../../lib/cn'

export type StatusDotTone = 'success' | 'warning' | 'destructive' | 'muted' | 'primary'

export interface StatusDotProps extends Omit<HTMLAttributes<HTMLSpanElement>, 'children'> {
  tone?: StatusDotTone
  /** Emphasize “live” / streaming / in-flight state. */
  pulse?: boolean
  /** Optional screen-reader label (also shown when `showLabel` is true). */
  label?: string
  /** Show `label` inline after the dot (default: only when `label` is set). */
  showLabel?: boolean
}

/** Shared class map for dots and timeline nodes. */
export const STATUS_DOT_TONE_CLASSES: Record<StatusDotTone, string> = {
  success: 'bg-success',
  warning: 'bg-warning',
  destructive: 'bg-destructive',
  muted: 'bg-muted-foreground/60',
  primary: 'bg-primary',
}

/** Compact health / connection indicator: colored dot with optional pulse and label. */
export function StatusDot({
  tone = 'muted',
  pulse = false,
  label,
  showLabel,
  className,
  ...props
}: StatusDotProps) {
  const visibleLabel = showLabel ?? Boolean(label)

  return (
    <span
      className={cn('inline-flex items-center gap-2', className)}
      {...props}
    >
      <span className="relative inline-flex h-2.5 w-2.5 shrink-0">
        {pulse ? (
          <span
            className={cn(
              'absolute inline-flex h-full w-full animate-ping rounded-full opacity-40',
              STATUS_DOT_TONE_CLASSES[tone],
            )}
            aria-hidden
          />
        ) : null}
        <span
          className={cn('relative inline-flex h-2.5 w-2.5 rounded-full', STATUS_DOT_TONE_CLASSES[tone])}
          aria-hidden
        />
      </span>
      {visibleLabel && label ? (
        <span className="text-xs font-medium text-muted-foreground">{label}</span>
      ) : null}
    </span>
  )
}
