'use client'

import { type ReactNode } from 'react'
import { cn } from '../../lib/cn'

interface EmptyCardProps {
  message?: string
  description?: string
  action?: ReactNode
  icon?: ReactNode
  className?: string
}

export function EmptyCard({ message = 'No items', description, action, icon, className }: EmptyCardProps) {
  return (
    <div className={cn('flex flex-col items-center justify-center py-12 px-4 text-center border border-dashed border-border rounded-lg', className)}>
      {icon && (
        <div className="mb-3 flex items-center justify-center w-10 h-10 rounded-xl bg-muted/50 text-muted-foreground">
          {icon}
        </div>
      )}
      <p className="text-sm font-medium text-foreground">{message}</p>
      {description && <p className="text-xs text-muted-foreground mt-1 max-w-xs">{description}</p>}
      {action && <div className="mt-4">{action}</div>}
    </div>
  )
}
