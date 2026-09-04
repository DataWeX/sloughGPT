'use client'

import { type ReactNode } from 'react'
import { cn } from '../../lib/cn'
import { Card, CardContent } from '../ui/card'
import { Skeleton } from '../ui/skeleton'

export interface LoadingCardProps {
  /** Card title */
  title?: string
  /** Number of skeleton lines to show */
  lines?: number
  /** Height of the skeleton content */
  height?: string
  /** Additional CSS classes */
  className?: string
  /** Test ID for testing */
  testId?: string
}

/**
 * Card with loading skeleton.
 *
 * Renders a card with a title and animated skeleton content.
 * Used as a placeholder while data is loading.
 *
 * @example
 * ```tsx
 * <LoadingCard title="Health" lines={3} />
 * <LoadingCard height="h-32" />
 * ```
 */
export function LoadingCard({
  title,
  lines = 2,
  height,
  className,
  testId,
}: LoadingCardProps) {
  return (
    <Card className={className} data-testid={testId}>
      {title && (
        <div className="px-4 pt-4 pb-0">
          <span className="text-sm font-medium">{title}</span>
        </div>
      )}
      <CardContent className={title ? 'pt-2' : undefined}>
        {height ? (
          <Skeleton className={cn('w-full rounded', height)} />
        ) : (
          <div className="space-y-2">
            {Array.from({ length: lines }, (_, i) => (
              <Skeleton
                key={i}
                className={cn('h-3 rounded', i === lines - 1 ? 'w-2/3' : 'w-full')}
              />
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  )
}
