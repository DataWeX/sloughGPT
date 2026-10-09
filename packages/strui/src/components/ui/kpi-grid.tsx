'use client'

import { type ReactNode } from 'react'
import { cn } from '../../lib/cn'

interface KpiGridProps {
  children: ReactNode
  columns?: 1 | 2 | 3 | 4 | 5
  className?: string
}

export function KpiGrid({ children, columns = 4, className }: KpiGridProps) {
  const cols = {
    1: 'grid-cols-1',
    2: 'grid-cols-1 sm:grid-cols-2',
    3: 'grid-cols-1 sm:grid-cols-3',
    4: 'grid-cols-2 sm:grid-cols-4',
    5: 'grid-cols-2 sm:grid-cols-3 lg:grid-cols-5',
  }
  return (
    <div className={cn('grid gap-3', cols[columns], className)}>
      {children}
    </div>
  )
}
