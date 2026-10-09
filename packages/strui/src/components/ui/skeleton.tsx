'use client'

import { cn } from '../../lib/cn'

interface SkeletonProps {
  className?: string
  /** Number of lines to render (text skeleton) */
  lines?: number
  /** Width of last line as percentage (text skeleton) */
  lastLineWidth?: string
}

export function Skeleton({ className, lines, lastLineWidth = '60%' }: SkeletonProps) {
  if (lines && lines > 1) {
    return (
      <div className={cn('space-y-2', className)}>
        {Array.from({ length: lines }).map((_, i) => (
          <div
            key={i}
            className="h-4 animate-pulse rounded bg-muted"
            style={{ width: i === lines - 1 ? lastLineWidth : '100%' }}
          />
        ))}
      </div>
    )
  }
  return <div className={cn('animate-pulse rounded-md bg-muted', className)} />
}
