import type { DetailsHTMLAttributes, ReactNode } from 'react'

import { ChevronDownIcon } from '../../lib/icons'
import { cn } from '../../lib/cn'

export interface FoldSectionProps extends Omit<DetailsHTMLAttributes<HTMLDetailsElement>, 'children'> {
  /** Visible heading in the summary row (not the HTML `title` tooltip attribute). */
  heading: ReactNode
  children: ReactNode
}

/**
 * Collapsible block using native `<details>` — settings, advanced fields, log excerpts.
 */
export function FoldSection({ heading, children, className, ...props }: FoldSectionProps) {
  return (
    <details
      className={cn(
        'group rounded-none border border-border bg-card/30 open:bg-card/50',
        className,
      )}
      {...props}
    >
      <summary
        className={cn(
          'flex cursor-pointer list-none items-center justify-between gap-2 px-4 py-3 text-left text-sm font-medium text-foreground',
          'marker:content-none [&::-webkit-details-marker]:hidden',
        )}
      >
        <span className="min-w-0">{heading}</span>
        <ChevronDownIcon className="h-4 w-4 shrink-0 text-muted-foreground transition-transform duration-200 group-open:rotate-180" />
      </summary>
      <div className="border-t border-border px-4 py-3 text-sm text-muted-foreground">{children}</div>
    </details>
  )
}
