'use client'

import { useMemo } from 'react'
import { cn, Card, CardHeader, CardTitle, CardContent } from '@sloughgpt/strui'

interface TrendPoint {
  hour: string
  count: number
}

interface AdminErrorTrendCardProps {
  trends: TrendPoint[]
  total: number
  groupedCount: number
  lastHourCount: number
  topError: string | null
}

export function AdminErrorTrendCard({
  trends,
  total,
  groupedCount,
  lastHourCount,
  topError,
}: AdminErrorTrendCardProps) {
  const maxCount = useMemo(() => Math.max(1, ...trends.map(t => t.count)), [trends])

  return (
    <Card data-testid="admin-error-trend">
      <CardHeader>
        <CardTitle className="text-base">Hourly Trend</CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="grid grid-cols-2 gap-2">
          <div className="rounded-md bg-muted/30 px-3 py-2">
            <p className="text-[9px] text-muted-foreground">Total Errors</p>
            <p className="text-lg font-bold font-mono" data-testid="total-errors">{total}</p>
          </div>
          <div className="rounded-md bg-muted/30 px-3 py-2">
            <p className="text-[9px] text-muted-foreground">Grouped</p>
            <p className="text-lg font-bold font-mono" data-testid="grouped-count">{groupedCount}</p>
          </div>
          <div className="rounded-md bg-muted/30 px-3 py-2">
            <p className="text-[9px] text-muted-foreground">Last Hour</p>
            <p className="text-lg font-bold font-mono" data-testid="last-hour-count">{lastHourCount}</p>
          </div>
          <div className="rounded-md bg-muted/30 px-3 py-2">
            <p className="text-[9px] text-muted-foreground">Top Error</p>
            <p className="text-sm font-medium truncate" data-testid="top-error">{topError ?? '—'}</p>
          </div>
        </div>

        {trends.length > 0 && (
          <div className="flex items-end gap-1 h-24" data-testid="bar-chart">
            {trends.map((t, i) => (
              <div key={i} className="flex-1 flex flex-col items-center gap-0.5">
                <div
                  className={cn(
                    'w-full rounded-t-sm transition-all duration-300',
                    t.count > 0 ? 'bg-destructive/60' : 'bg-muted/40'
                  )}
                  style={{ height: `${Math.max(2, (t.count / maxCount) * 100)}%` }}
                  title={`${t.hour}: ${t.count}`}
                />
                <span className="text-[7px] text-muted-foreground leading-none">{t.hour}</span>
              </div>
            ))}
          </div>
        )}

        {trends.length === 0 && (
          <p className="text-sm text-muted-foreground text-center py-2">No trend data.</p>
        )}
      </CardContent>
    </Card>
  )
}
