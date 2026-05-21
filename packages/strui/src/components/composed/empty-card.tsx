import type { ReactNode } from 'react'

import { cn } from '../../lib/cn'

export interface EmptyCardProps {
  message?: string
  action?: ReactNode
  className?: string
}

export function EmptyCard({ message = 'No items', action, className }: EmptyCardProps) {
  return (
    <div className={cn("flex flex-col items-center justify-center py-12 px-4 text-center border border-dashed rounded-lg", className)}>
      <p className="text-sm text-muted-foreground">{message}</p>
      {action && <div className="mt-3">{action}</div>}
    </div>
  )
}
