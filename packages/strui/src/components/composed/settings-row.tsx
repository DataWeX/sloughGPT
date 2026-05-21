import type { ReactNode } from 'react'

import { cn } from '../../lib/cn'

export interface SettingsRowProps {
  title: string
  description?: string
  /** Switch, select, or compact control — right-aligned on wide screens. */
  control: ReactNode
  className?: string
}

/** Two-column settings row: title/description + control; stacks on narrow viewports. */
export function SettingsRow({ title, description, control, className }: SettingsRowProps) {
  return (
    <div
      className={cn(
        'flex flex-col gap-3 border-b border-border py-4 last:border-b-0 sm:flex-row sm:items-center sm:justify-between sm:gap-8',
        className,
      )}
    >
      <div className="min-w-0 flex-1 space-y-1">
        <p className="text-sm font-medium text-foreground">{title}</p>
        {description ? <p className="text-sm text-muted-foreground">{description}</p> : null}
      </div>
      <div className="flex shrink-0 items-center justify-start sm:justify-end">{control}</div>
    </div>
  )
}
