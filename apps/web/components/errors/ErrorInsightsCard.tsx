'use client'

import { cn, Card, CardHeader, CardTitle, CardContent } from '@sloughgpt/strui'

interface GroupedError {
  message: string
  count: number
  latest?: string
  lastSeen?: string
}

interface ErrorInsightsCardProps {
  grouped: GroupedError[]
  recent: Array<{ message: string; timestamp: string }>
  total: number
}

const severityEntriesColors: Record<string, string> = {
  Critical: 'bg-destructive/15 text-destructive',
  Warning: 'bg-warning/15 text-warning',
  Security: 'bg-accent/15 text-accent',
  Error: 'bg-muted text-muted-foreground',
}

function severityLevel(message: string): { label: string; color: string } {
  const lower = message.toLowerCase()
  if (lower.includes('crash') || lower.includes('fatal') || lower.includes('oom'))
    return { label: 'Critical', color: 'bg-destructive/15 text-destructive' }
  if (lower.includes('timeout') || lower.includes('slow') || lower.includes('degraded'))
    return { label: 'Warning', color: 'bg-warning/15 text-warning' }
  if (lower.includes('auth') || lower.includes('permission') || lower.includes('forbidden'))
    return { label: 'Security', color: 'bg-accent/15 text-accent' }
  return { label: 'Error', color: 'bg-muted text-muted-foreground' }
}

function computeInsights(grouped: GroupedError[], recent: Array<{ message: string; timestamp: string }>, total: number) {
  if (grouped.length === 0 && recent.length === 0) return null

  const topError = grouped.length > 0 ? grouped.reduce((a, b) => a.count > b.count ? a : b) : null
  const severityCounts: Record<string, number> = {}
  for (const g of grouped) {
    const { label } = severityLevel(g.message)
    severityCounts[label] = (severityCounts[label] || 0) + g.count
  }

  const recentWindow = 3600000
  const now = Date.now()
  const recentErrors = recent.filter(r => {
    try { return now - new Date(r.timestamp).getTime() < recentWindow } catch { return false }
  })

  return { topError, severityCounts, recentCount: recentErrors.length, uniqueErrors: grouped.length }
}

export function ErrorInsightsCard({ grouped, recent, total }: ErrorInsightsCardProps) {
  const insights = computeInsights(grouped, recent, total)
  if (!insights) return null

  const severityEntries = Object.entries(insights.severityCounts).sort((a, b) => b[1] - a[1])

  return (
    <Card data-testid="error-insights">
      <CardHeader>
        <CardTitle className="text-base">Error Insights</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-4">
          <div>
            <div className="text-[10px] text-muted-foreground uppercase tracking-wider font-medium">Total</div>
            <div className="text-sm font-semibold">{total}</div>
          </div>
          <div>
            <div className="text-[10px] text-muted-foreground uppercase tracking-wider font-medium">Unique</div>
            <div className="text-sm font-semibold">{insights.uniqueErrors}</div>
          </div>
          <div>
            <div className="text-[10px] text-muted-foreground uppercase tracking-wider font-medium">Last Hour</div>
            <div className="text-sm font-semibold">{insights.recentCount}</div>
          </div>
          <div>
            <div className="text-[10px] text-muted-foreground uppercase tracking-wider font-medium">Top Error</div>
            <div className="text-sm font-semibold truncate">{insights.topError?.count ?? 0}x</div>
          </div>
        </div>
        {severityEntries.length > 0 && (
          <div className="space-y-1.5">
            <div className="text-[10px] font-medium uppercase tracking-wider text-muted-foreground">By Severity</div>
            {severityEntries.map(([label, count]) => (
              <div key={label} className="flex items-center gap-2">
                <span className={cn('text-[9px] px-1.5 py-0.5 rounded font-medium', severityEntriesColors[label] ?? 'bg-muted text-muted-foreground')}>
                  {label}
                </span>
                <div className="flex-1 h-1.5 rounded-full bg-muted/50 overflow-hidden">
                  <div
                    className="h-full rounded-full bg-primary/60"
                    style={{ width: `${(count / total) * 100}%` }}
                  />
                </div>
                <span className="text-[10px] font-mono text-muted-foreground w-8 text-right">{count}</span>
              </div>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  )
}
