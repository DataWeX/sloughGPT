'use client'

import { type ReactNode } from 'react'
import { cn } from '../../lib/cn'

interface ListSectionProps {
  title?: string
  description?: string
  children: ReactNode
  className?: string
}

export function ListSection({ title, description, children, className }: ListSectionProps) {
  return (
    <div className={cn('space-y-1', className)}>
      {(title || description) && (
        <div className="px-3 pb-1">
          {title && <p className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wider">{title}</p>}
          {description && <p className="text-xs text-muted-foreground mt-0.5">{description}</p>}
        </div>
      )}
      <div className="border border-border rounded-lg overflow-hidden divide-y divide-border">{children}</div>
    </div>
  )
}
