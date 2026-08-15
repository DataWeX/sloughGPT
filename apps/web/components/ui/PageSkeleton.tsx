'use client'

import { Skeleton } from '@sloughgpt/strui'

interface PageSkeletonProps {
  /** Number of card skeletons to show. */
  cards?: number
  /** Show header skeleton. */
  header?: boolean
  /** Show grid skeleton instead of card skeleton. */
  grid?: boolean
}

/**
 * Responsive page loading skeleton.
 *
 * Matches PageContainer layout:
 * - sl-page responsive padding
 * - Title skeleton: h-8 on mobile, h-9 on md+ (approximates sl-h1)
 * - Subtitle skeleton: h-4
 * - Cards: full-width on mobile, 2-col on sm, 4-col on sm+ for grid
 */
export function PageSkeleton({ cards = 3, header = true, grid = false }: PageSkeletonProps) {
  return (
    <div className="sl-page mx-auto max-w-4xl space-y-4">
      {header && (
        <div className="space-y-2">
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
