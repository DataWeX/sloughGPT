'use client'

import { type ReactNode } from 'react'
import { cn } from '../../lib/cn'

interface CardDeckProps {
  title?: string
  description?: string
  footer?: ReactNode
  children: ReactNode
  className?: string
}

export function CardDeck({ title, description, footer, children, className }: CardDeckProps) {
  return (
    <div className={cn('border border-border rounded-lg overflow-hidden bg-card', className)}>
      {(title || description) && (
        <div className="px-4 py-3 border-b border-border">
          {title && <p className="font-medium text-sm text-foreground">{title}</p>}
          {description && <p className="text-xs text-muted-foreground mt-0.5">{description}</p>}
        </div>
      )}
      <div className="p-4">{children}</div>
      {footer && <div className="px-4 py-3 border-t border-border bg-muted/30">{footer}</div>}
    </div>
  )
}
