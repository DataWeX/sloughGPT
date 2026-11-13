'use client'

import { type ReactNode } from 'react'
import { cn } from '../../lib/cn'

export interface DetailItem {
  label: string
  value: ReactNode
  mono?: boolean
  href?: string
  valueClassName?: string
}

export interface DetailListProps {
  /** Array of detail items */
  items: DetailItem[]
  /** Layout variant */
  layout?: 'rows' | 'grid'
  /** Grid columns (only for grid layout) */
  columns?: 2 | 3 | 4
  /** Additional CSS classes */
  className?: string
  /** Test ID for testing */
  testId?: string
}

/**
 * List of key-value detail items.
 *
 * Renders a list of label/value pairs in either row or grid layout.
 * Useful for settings panels, metadata displays, and detail views.
 *
 * @example
 * ```tsx
 * <DetailList
 *   items={[
 *     { label: 'Model', value: 'GPT-2' },
 *     { label: 'Path', value: '/models/gpt2', mono: true },
 *     { label: 'Size', value: '500 MB' },
 *   ]}
 * />
 * ```
 */
export function DetailList({
  items,
  layout = 'rows',
  columns = 2,
  className,
  testId,
}: DetailListProps) {
  if (layout === 'grid') {
    const gridCols = {
      2: 'grid-cols-2',
      3: 'grid-cols-3',
      4: 'grid-cols-4',
    }

    return (
      <div className={cn('grid gap-3', gridCols[columns], className)} data-testid={testId}>
        {items.map((item, i) => (
          <div key={i}>
            <div className="text-[10px] text-muted-foreground uppercase tracking-wider font-medium">{item.label}</div>
            <div className={cn('text-sm font-semibold mt-0.5', item.mono && 'font-mono text-xs', item.valueClassName)}>
              {item.href ? (
                <a href={item.href} target="_blank" rel="noopener noreferrer" className="text-primary hover:underline">
                  {item.value}
                </a>
              ) : (
                item.value
              )}
            </div>
          </div>
        ))}
      </div>
    )
  }

  return (
    <div className={cn('space-y-1.5 text-[11px]', className)} data-testid={testId}>
      {items.map((item, i) => (
        <div key={i} className="flex items-center justify-between">
          <span className="text-muted-foreground">{item.label}</span>
          <span className={cn('font-numeric', item.mono && 'font-mono text-[10px]', item.valueClassName)}>
            {item.href ? (
              <a href={item.href} target="_blank" rel="noopener noreferrer" className="text-primary hover:underline">
                {item.value}
              </a>
            ) : (
              item.value
            )}
          </span>
        </div>
      ))}
    </div>
  )
}
