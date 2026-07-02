import type { ReactNode } from 'react'

import { cn } from '../../lib/cn'

export interface PageHeaderProps {
  title: string
  description?: string
  /** Buttons or menus (e.g. primary CTA, overflow menu). */
  actions?: ReactNode
  className?: string
}

/** Top-of-page title row with optional description and action cluster (stacks on mobile). */
export function PageHeader({ title, description, actions, className }: PageHeaderProps) {
  return (
    <header
      className={cn(
        'str-safe-x flex flex-col gap-4 border-b border-border bg-card/40 px-4 py-5 backdrop-blur-sm sm:flex-row sm:items-start sm:justify-between sm:py-6',
        className,
      )}
    >
      <div className="min-w-0">
        <h1 className="sl-h1">{title}</h1>
        {description ? <p className="mt-1.5 max-w-2xl text-sm text-muted-foreground">{description}</p> : null}
      </div>
      {actions ? <div className="flex shrink-0 flex-wrap items-center gap-2">{actions}</div> : null}
    </header>
  )
}
