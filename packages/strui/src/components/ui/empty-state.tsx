'use client'

import { type ReactNode } from 'react'
import { cn } from '../../lib/cn'

interface EmptyStateProps {
  icon?: ReactNode
  title: string
  description?: string
  action?: ReactNode
  size?: 'sm' | 'default' | 'lg'
  className?: string
}

export function EmptyState({ icon, title, description, action, size = 'default', className }: EmptyStateProps) {
  const paddings = { sm: 'py-8 px-3', default: 'py-12 px-4', lg: 'py-16 px-6' }

  return (
    <div className={cn('flex flex-col items-center justify-center text-center', paddings[size], className)}>
      {icon && (
        <div className="mb-4 flex items-center justify-center rounded-xl bg-muted/50 p-3 text-muted-foreground">
          {icon}
        </div>
      )}
      <p className="text-sm font-medium text-foreground">{title}</p>
      {description && <p className="text-xs text-muted-foreground mt-1 max-w-xs">{description}</p>}
      {action && <div className="mt-4">{action}</div>}
    </div>
  )
}
