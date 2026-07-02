import type { ReactNode } from 'react'

import { cn } from '../../lib/cn'

export interface KpiGridProps {
  children: ReactNode
  columns?: 2 | 3 | 4
  className?: string
}

export function KpiGrid({ children, columns = 4, className }: KpiGridProps) {
  return (
    <div className={cn(
      "grid gap-3",
      columns === 2 && "grid-cols-2",
      columns === 3 && "grid-cols-3",
      columns === 4 && "grid-cols-2 sm:grid-cols-4",
      className
    )}>
      {children}
    </div>
  )
}
