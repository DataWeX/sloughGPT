'use client'

import { cn } from '@sloughgpt/strui'

export interface ChartTooltipEntry {
  name?: string
  value?: number | string
  color?: string
  stroke?: string
}

export interface ChartTooltipProps {
  active?: boolean
  payload?: ChartTooltipEntry[]
  label?: string | number
  /** Converts the x-axis label (e.g. step number) to display text. */
  labelFormatter?: (label: string | number) => string
  className?: string
}

/**
 * ChartTooltip — token-styled tooltip content for recharts
 * (`<Tooltip content={<ChartTooltip />} />`). Renders nothing until the
 * tooltip is active. Numeric values use the 4-decimal chart convention;
 * colors come from the series token passed through by recharts.
 */
export function ChartTooltip({ active, payload, label, labelFormatter, className }: ChartTooltipProps) {
  if (!active || !payload?.length) return null

  return (
    <div
      className={cn(
        'rounded-md border border-border bg-popover px-2 py-1.5 text-xs shadow-sm',
        className,
      )}
    >
      {label !== undefined && (
        <div className="mb-1 font-medium text-foreground">
          {labelFormatter ? labelFormatter(label) : String(label)}
        </div>
      )}
      <div className="space-y-0.5">
        {payload.map((entry, i) => (
          <div key={i} className="flex items-center gap-1.5">
            <span
              className="h-2 w-2 shrink-0 rounded-full"
              style={{ backgroundColor: entry.color ?? entry.stroke }}
              aria-hidden="true"
            />
            <span className="text-muted-foreground">{entry.name}</span>
            <span className="ml-auto font-numeric text-foreground">
              {typeof entry.value === 'number' ? entry.value.toFixed(4) : String(entry.value ?? '')}
            </span>
          </div>
        ))}
      </div>
    </div>
  )
}
