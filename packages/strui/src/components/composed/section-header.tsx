import type { ReactNode } from 'react'

import { cn } from '../../lib/cn'

export interface SectionHeaderProps {
  title: string
  description?: string
  action?: ReactNode
  className?: string
}

/** In-page section title (smaller than **PageHeader**) — forms, cards, AI panels. */
export function SectionHeader({ title, description, action, className }: SectionHeaderProps) {
  return (
    <div
      className={cn(
        'flex flex-col gap-2 border-b border-border pb-3 sm:flex-row sm:items-center sm:justify-between',
        className,
      )}
    >
      <div className="min-w-0">
        <h2 className="sl-h2">{title}</h2>
        {description ? <p className="mt-1 text-sm text-muted-foreground">{description}</p> : null}
      </div>
      {action ? <div className="flex shrink-0 flex-wrap gap-2">{action}</div> : null}
    </div>
  )
}
