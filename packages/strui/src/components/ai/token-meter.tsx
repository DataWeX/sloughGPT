import * as React from 'react'

import { cn } from '../../lib/cn'

export interface TokenMeterProps extends React.HTMLAttributes<HTMLDivElement> {
  /** Approximate prompt + completion tokens. */
  total?: number
  contextLimit?: number
}

function formatTok(n: number) {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}k`
  return String(n)
}

/** Inline usage strip for chat headers (mobile-friendly, no chart deps). */
export function TokenMeter({ className, total = 0, contextLimit, ...props }: TokenMeterProps) {
  const pct =
    contextLimit && contextLimit > 0 ? Math.min(100, Math.round((total / contextLimit) * 100)) : null

  const usageId = React.useId()

  return (
    <div
      className={cn(
        'flex flex-wrap items-center gap-2 text-xs text-muted-foreground sm:text-[13px]',
        className,
      )}
      role="meter"
      aria-label={`Token usage: ${formatTok(total)} of ${contextLimit ? formatTok(contextLimit) : 'unlimited'}`}
      aria-valuenow={total}
      aria-valuemin={0}
      aria-valuemax={contextLimit ?? undefined}
      aria-valuetext={`${pct !== null ? pct : 0}% of context limit`}
      {...props}
    >
      <span className="font-mono tabular-nums">{formatTok(total)} tok</span>
      {pct !== null ? (
        <div
          className="h-1.5 min-w-[4rem] max-w-[8rem] flex-1 rounded-none bg-muted"
          role="progressbar"
          aria-labelledby={usageId}
          aria-valuenow={pct}
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuetext={`${pct}% of context`}
        >
          <span
            className="block h-full rounded-none bg-primary/70 transition-all duration-200 ease-smooth"
            style={{ width: `${pct}%` }}
          />
        </div>
      ) : null}
      <span id={usageId} className="sr-only">
        {pct !== null ? `${pct}% of context limit used` : 'No context limit set'}
      </span>
      {contextLimit ? (
        <span className="font-mono tabular-nums text-muted-foreground/80">/ {formatTok(contextLimit)}</span>
      ) : null}
    </div>
  )
}
