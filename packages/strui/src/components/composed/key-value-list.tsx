import type { ReactNode } from 'react'

import { cn } from '../../lib/cn'

export interface KeyValueItem {
  label: string
  value: ReactNode
  /** When set, value is monospace (IDs, paths). */
  mono?: boolean
}

export interface KeyValueListProps {
  items: KeyValueItem[]
  className?: string
  /** Compact rows for sidebars and tool panels. */
  dense?: boolean
}

/** Semantic definition list for metadata, API responses, and run summaries. */
export function KeyValueList({ items, className, dense }: KeyValueListProps) {
  return (
    <dl
      className={cn(
        'divide-y divide-border rounded-none border border-border bg-card/40 text-sm',
        className,
      )}
    >
      {items.map((row, i) => (
        <div
          key={i}
          className={cn(
            'flex flex-col gap-0.5 sm:flex-row sm:items-baseline sm:justify-between sm:gap-4',
            dense ? 'px-3 py-2' : 'px-4 py-3',
          )}
        >
          <dt className="shrink-0 text-xs font-medium uppercase tracking-wider text-muted-foreground">
            {row.label}
          </dt>
          <dd
            className={cn(
              'min-w-0 break-all text-foreground sm:text-right',
              row.mono && 'font-mono text-xs',
            )}
          >
            {row.value}
          </dd>
        </div>
      ))}
    </dl>
  )
}
