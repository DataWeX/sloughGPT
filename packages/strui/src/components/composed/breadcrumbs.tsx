import type { ReactNode } from 'react'

import { cn } from '../../lib/cn'

export interface BreadcrumbItem {
  label: ReactNode
  /** When set, renders as a link (use `href="#"` + `onClick` from the host if needed). */
  href?: string
}

export interface BreadcrumbsProps {
  items: BreadcrumbItem[]
  className?: string
  /** Separator between segments (default: `/`). */
  separator?: ReactNode
}

/** Compact trail for settings sub-pages, dataset paths, and admin shells. */
export function Breadcrumbs({ items, className, separator = '/' }: BreadcrumbsProps) {
  if (items.length === 0) return null

  return (
    <nav aria-label="Breadcrumb" className={cn('text-xs text-muted-foreground', className)}>
      <ol className="flex flex-wrap items-center gap-x-1.5 gap-y-1">
        {items.map((item, i) => (
          <li key={i} className="flex min-w-0 items-center gap-1.5">
            {i > 0 ? (
              <span className="select-none text-border" aria-hidden>
                {separator}
              </span>
            ) : null}
            {item.href != null && item.href !== '' ? (
              <a
                href={item.href}
                className="truncate sl-hover-primary underline-offset-2 hover:underline"
              >
                {item.label}
              </a>
            ) : (
              <span
                className={cn(
                  'min-w-0 truncate',
                  i === items.length - 1 ? 'font-medium text-foreground' : 'text-muted-foreground',
                )}
              >
                {item.label}
              </span>
            )}
          </li>
        ))}
      </ol>
    </nav>
  )
}
