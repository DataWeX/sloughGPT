import type { ComponentProps } from 'react'

import { Badge } from '../ui/badge'
import { cn } from '../../lib/cn'

export type JobStatusState = 'idle' | 'queued' | 'running' | 'success' | 'error' | 'cancelled'

export interface JobStatusProps {
  status: JobStatusState
  className?: string
}

type BadgeVariant = NonNullable<ComponentProps<typeof Badge>['variant']>

const copy: Record<JobStatusState, { label: string; variant: BadgeVariant }> = {
  idle: { label: 'idle', variant: 'secondary' },
  queued: { label: 'queued', variant: 'warning' },
  running: { label: 'running', variant: 'default' },
  success: { label: 'success', variant: 'success' },
  error: { label: 'failed', variant: 'destructive' },
  cancelled: { label: 'cancelled', variant: 'outline' },
}

/** Compact training / inference job state for tables and headers. */
export function JobStatus({ status, className }: JobStatusProps) {
  const { label, variant } = copy[status]
  return (
    <Badge 
      variant={variant} 
      className={cn('font-mono text-[0.65rem] uppercase', className)}
      role="status"
      aria-label={`Job status: ${label}`}
    >
      {label}
    </Badge>
  )
}
