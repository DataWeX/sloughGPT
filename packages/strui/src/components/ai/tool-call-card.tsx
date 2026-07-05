import * as React from 'react'

import { Badge } from '../ui/badge'
import { cn } from '../../lib/cn'

export type ToolCallState = 'pending' | 'ok' | 'error'

export interface ToolCallCardProps extends React.HTMLAttributes<HTMLDivElement> {
  name: string
  /** JSON or short text; keep small for mobile. */
  argsPreview?: string
  state?: ToolCallState
}

const stateBadge = (s: ToolCallState) => {
  switch (s) {
    case 'pending':
      return <Badge variant="warning">pending</Badge>
    case 'error':
      return <Badge variant="destructive">error</Badge>
    default:
      return <Badge variant="success">ok</Badge>
  }
}

/** Compact tool / function call surface for agent UIs. */
export function ToolCallCard({
  className,
  name,
  argsPreview,
  state = 'ok',
  ...props
}: ToolCallCardProps) {
  const stateLabel = state === 'pending' ? 'pending' : state === 'error' ? 'error' : 'ok'

  return (
    <article
      className={cn(
        'str-safe-x max-w-full rounded-none border border-border bg-muted/30 p-3 font-mono text-xs shadow-sm sm:text-sm',
        className,
      )}
      aria-label={`Tool call: ${name}, status ${stateLabel}`}
      {...props}
    >
      <div className="mb-2 flex flex-wrap items-center gap-2">
        <span className="font-sans text-sm font-semibold text-foreground">{name}</span>
        {stateBadge(state)}
      </div>
      {argsPreview ? (
        <pre
          className="str-tool-args overflow-x-auto whitespace-pre-wrap break-all text-muted-foreground"
          aria-label={`Arguments for ${name}`}
        >
          {argsPreview}
        </pre>
      ) : null}
    </article>
  )
}
