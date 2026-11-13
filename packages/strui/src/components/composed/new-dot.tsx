import type { HTMLAttributes } from 'react'

import { cn } from '../../lib/cn'
import { isNewFeature } from '../../versions'
import { STATUS_DOT_TONE_CLASSES } from './status-dot'

export interface NewDotProps extends Omit<HTMLAttributes<HTMLSpanElement>, 'children'> {
  /** Feature key in FEATURE_VERSIONS (e.g. "consciousness"). Renders nothing unless fresh. */
  feature: string
  /** Accessible label + tooltip. Defaults to "New". */
  label?: string
}

/**
 * Freshness marker: pulsing accent dot for recently launched features.
 * Reads the component registry (`isNewFeature`) — auto-expires, no flags
 * to flip. Renders null for stale or unknown features.
 */
export function NewDot({ feature, label = 'New', className, ...props }: NewDotProps) {
  if (!isNewFeature(feature)) return null
  return (
    <span
      className={cn('relative inline-flex h-2 w-2 shrink-0', className)}
      role="img"
      aria-label={label}
      title={label}
      {...props}
    >
      <span
        className={cn(
          'absolute inline-flex h-full w-full rounded-full opacity-50 motion-safe:animate-ping',
          STATUS_DOT_TONE_CLASSES.accent,
        )}
        aria-hidden
      />
      <span
        className={cn('relative inline-flex h-2 w-2 rounded-full', STATUS_DOT_TONE_CLASSES.accent)}
        aria-hidden
      />
    </span>
  )
}
