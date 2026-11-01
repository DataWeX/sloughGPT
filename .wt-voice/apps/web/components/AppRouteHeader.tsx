import type { ReactNode } from 'react'

import { cn } from '@sloughgpt/strui'

/** Standard title block for `AppRouteHeader` `left` slot (heading + optional subtitle). */
export function AppRouteHeaderLead({
  title,
  subtitle,
  children,
}: {
  title: ReactNode
  subtitle?: ReactNode
  children?: ReactNode
}) {
  return (
    <div className="min-w-0">
      {typeof title === 'string' ? <h1 className="sl-h1">{title}</h1> : title}
      {subtitle != null ? (
        <p className="mt-1 text-sm leading-5 text-muted-foreground">{subtitle}</p>
      ) : null}
      {children}
    </div>
  )
}

export type AppRouteHeaderProps = {
  /** Primary title / subtitle block — stays left, wraps on small screens. */
  left: ReactNode
  /** Secondary cluster (actions, status) — stays right; use justify-end content. */
  right?: ReactNode
  className?: string
  /** Make header sticky at top */
  sticky?: boolean
}

/**
 * Shared page header: one row, justify-between, wraps on mobile.
 * No outer padding — parent `sl-page` handles page gutters.
 * Use inside `PageContainer` or a `mx-auto max-w-*` column.
 */
export function AppRouteHeader({ left, right, className, sticky = false }: AppRouteHeaderProps) {
  return (
    <header
      className={cn(
        'flex w-full min-w-0 flex-wrap items-center justify-between gap-x-4 gap-y-3 py-2',
        sticky && 'sticky top-0 z-10 -mx-4 bg-background/80 backdrop-blur supports-[backdrop-filter]:bg-background/60 px-4',
        className,
      )}
    >
      <div className="flex min-w-0 flex-1 flex-wrap items-center gap-1.5 md:gap-2">{left}</div>
      {right != null ? (
        <div className="flex min-w-0 shrink-0 flex-wrap items-center justify-end gap-2">{right}</div>
      ) : null}
    </header>
  )
}
