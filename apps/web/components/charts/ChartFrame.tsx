'use client'

import type { ReactNode } from 'react'
import { cn, EmptyCard, Skeleton } from '@sloughgpt/strui'
import { StatusBanner } from '@/components/composed/StatusBanner'

export type ChartFrameState = 'loading' | 'error' | 'empty' | 'ready'

export interface ChartFrameProps {
  /** Accessible name of the chart (required — charts are role="img" regions). */
  label: string
  /** Optional header title (chart-label style: text-xs uppercase tracking-wider). */
  title?: ReactNode
  /** Header action slot, e.g. an export button — stays outside the role="img"
   *  region so assistive tech can reach it. */
  action?: ReactNode
  loading?: boolean
  error?: string | null
  onRetry?: () => void
  empty?: boolean
  emptyMessage?: string
  emptyDescription?: string
  /** Fixed pixel height of the chart region (loading + ready). */
  height?: number
  /** sr-only text alternative rendered above the chart region. */
  srSummary?: ReactNode
  className?: string
  children?: ReactNode
}

/** State precedence mirrors PageContainer: loading > error > empty > ready.
 *  Exported for tests and callers that derive the same state. */
export function chartFrameState(
  props: Pick<ChartFrameProps, 'loading' | 'error' | 'empty'>,
): ChartFrameState {
  if (props.loading) return 'loading'
  if (props.error) return 'error'
  if (props.empty) return 'empty'
  return 'ready'
}

/**
 * ChartFrame — the shared chart layer's state/a11y wrapper (card 96d1507c,
 * slice a). Owns ONLY presentation states and accessibility: loading/empty/
 * error/ready, the labeled focusable chart region, header title/action.
 * It never fetches data and never owns journey logic — the caller does.
 */
export function ChartFrame({
  label,
  title,
  action,
  loading,
  error,
  onRetry,
  empty,
  emptyMessage,
  emptyDescription,
  height,
  srSummary,
  className,
  children,
}: ChartFrameProps) {
  const state = chartFrameState({ loading, error, empty })
  const regionStyle = height != null ? { height } : undefined

  return (
    <div className={cn('w-full', className)} data-testid="chart-frame" data-state={state}>
      {(title || action) && (
        <div className="mb-2 flex items-center justify-between gap-2">
          {title && (
            <span className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
              {title}
            </span>
          )}
          {action && <div className="flex items-center gap-1">{action}</div>}
        </div>
      )}

      {srSummary && <p className="sr-only">{srSummary}</p>}

      {state === 'loading' && (
        <div aria-busy="true" aria-live="polite" style={regionStyle}>
          <Skeleton className="h-full w-full" />
        </div>
      )}

      {state === 'error' && (
        <div className="py-2">
          <StatusBanner variant="error" message={error ?? ''} dismissible={false} onRetry={onRetry} />
        </div>
      )}

      {state === 'empty' && (
        <EmptyCard message={emptyMessage} description={emptyDescription} />
      )}

      {state === 'ready' && (
        <div
          role="img"
          aria-label={label}
          tabIndex={0}
          className="w-full rounded focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
          style={regionStyle}
        >
          {children}
        </div>
      )}
    </div>
  )
}
