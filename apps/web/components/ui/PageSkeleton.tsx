'use client'

import { Skeleton } from '@sloughgpt/strui'
import { cn } from '@sloughgpt/strui'

interface PageSkeletonProps {
  /** Number of card skeletons to show. */
  cards?: number
  /** Show header skeleton (title + subtitle). Use false when parent renders real header. */
  header?: boolean
  /** Show grid skeleton instead of card skeleton. */
  grid?: boolean
  className?: string
}

/**
 * Responsive page loading skeleton.
 * No outer page wrapper — parent `PageContainer` handles `sl-page` and max-width.
 * Set header=false when the real AppRouteHeader is already visible.
 */
export function PageSkeleton({ cards = 3, header = false, grid = false, className }: PageSkeletonProps) {
  return (
    <div className={cn('space-y-4', className)} aria-busy="true" aria-live="polite">
      {header && (
        <div className="space-y-2 py-2">
          <Skeleton className="h-8 w-48 md:h-9 md:w-56" />
          <Skeleton className="h-4 w-72 max-w-full" />
        </div>
      )}
      {grid ? (
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 sm:gap-4">
          {Array.from({ length: 4 }).map((_, i) => (
            <Skeleton key={i} className="h-24 rounded-lg" />
          ))}
        </div>
      ) : (
        <div className="space-y-3">
          {Array.from({ length: cards }).map((_, i) => (
            <Skeleton key={i} className="h-32 rounded-lg" />
          ))}
        </div>
      )}
    </div>
  )
}

export function CardSkeleton() {
  return (
    <div className="rounded-lg border border-border/60 bg-card p-3 sm:p-4 space-y-3">
      <Skeleton className="h-4 w-32" />
      <Skeleton className="h-3 w-full" />
      <Skeleton className="h-3 w-3/4" />
    </div>
  )
}

export function ListSkeleton({ items = 5 }: { items?: number }) {
  return (
    <div className="space-y-2">
      {Array.from({ length: items }).map((_, i) => (
        <Skeleton key={i} className="h-16 rounded-lg" />
      ))}
    </div>
  )
}
