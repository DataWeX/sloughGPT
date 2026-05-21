import type { ReactNode } from 'react'

import { cn } from '../../lib/cn'

export interface ReasoningPanelProps {
  title?: string
  children: ReactNode
  className?: string
  /** Open by default (e.g. while streaming “thinking”). */
  defaultOpen?: boolean
}

/**
 * Collapsible chain-of-thought / scratch space — uses native `<details>` (no extra Radix dep).
 */
export function ReasoningPanel({
  title = 'Reasoning',
  children,
  className,
  defaultOpen,
}: ReasoningPanelProps) {
  return (
    <details
      className={cn(
        'rounded-none border border-border bg-muted/25 open:border-primary/25 open:bg-muted/35',
        className,
      )}
      open={defaultOpen}
    >
      <summary className="cursor-pointer list-none px-3 py-2 text-sm font-medium text-foreground marker:content-none [&::-webkit-details-marker]:hidden">
        <span className="select-none">{title}</span>
      </summary>
      <div className="border-t border-border px-3 py-2 text-sm leading-relaxed text-muted-foreground">
        {children}
      </div>
    </details>
  )
}
