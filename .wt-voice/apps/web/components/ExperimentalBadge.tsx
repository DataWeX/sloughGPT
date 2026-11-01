'use client'

import { Badge } from '@sloughgpt/strui'
import { isTrackedExperimental } from '@/lib/experimental'

export interface ExperimentalBadgeProps {
  /** Registry feature key (e.g. "voice"). Renders nothing unless tracked experimental. */
  feature: string
  className?: string
}

/** Experimental tag driven by the tracking registry — not hardcoded status. */
export function ExperimentalBadge({ feature, className }: ExperimentalBadgeProps) {
  if (!isTrackedExperimental(feature)) return null
  return (
    <Badge variant="warning" size="sm" className={className}>
      Experimental
    </Badge>
  )
}
