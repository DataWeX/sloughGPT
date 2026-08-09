'use client'

import { Card, CardHeader, CardTitle, CardContent } from '@sloughgpt/strui'
import type { UserAdapterInfo } from '@/lib/user-adapters-controller'

interface AdapterHealthCardProps {
  adapters: UserAdapterInfo[]
}

export function AdapterHealthCard({ adapters }: AdapterHealthCardProps) {
  if (adapters.length === 0) return null

  const totalFeedback = adapters.reduce((s, a) => s + (a.feedback_count ?? 0), 0)
  const avgFeedback = (totalFeedback / adapters.length).toFixed(1)

  const byRank: Record<number, number> = {}
  for (const a of adapters) {
    const rank = a.rank ?? 0
    byRank[rank] = (byRank[rank] ?? 0) + 1
  }
  const sortedRanks = Object.entries(byRank).sort((a, b) => Number(b[0]) - Number(a[0]))

  const withFeedback = adapters.filter(a => (a.feedback_count ?? 0) > 0)
  const topFeedback = [...adapters].sort((a, b) => (b.feedback_count ?? 0) - (a.feedback_count ?? 0)).slice(0, 5)

  const now = Date.now()
  const ages = adapters
    .filter(a => a.created_at)
    .map(a => Math.floor((now - new Date(a.created_at!).getTime()) / 86400000))
  const avgAge = ages.length > 0 ? Math.round(ages.reduce((a, b) => a + b, 0) / ages.length) : null

  return (
    <Card data-testid="adapter-health">
      <CardHeader>
        <CardTitle className="text-base">Adapter Health</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-4 gap-2 mb-3">
          <div className="rounded-md bg-muted/30 p-2 text-center">
            <div className="text-[10px] text-muted-foreground">Adapters</div>
            <div className="text-sm font-mono font-medium">{adapters.length}</div>
          </div>
          <div className="rounded-md bg-muted/30 p-2 text-center">
            <div className="text-[10px] text-muted-foreground">Total Feedback</div>
            <div className="text-sm font-mono font-medium">{totalFeedback}</div>
          </div>
          <div className="rounded-md bg-muted/30 p-2 text-center">
            <div className="text-[10px] text-muted-foreground">Avg Feedback</div>
            <div className="text-sm font-mono font-medium">{avgFeedback}</div>
          </div>
          <div className="rounded-md bg-muted/30 p-2 text-center">
            <div className="text-[10px] text-muted-foreground">Avg Age</div>
            <div className="text-sm font-mono font-medium">{avgAge != null ? `${avgAge}d` : '—'}</div>
          </div>
        </div>

        {sortedRanks.length > 0 && (
          <div className="mb-3">
            <div className="text-[10px] text-muted-foreground mb-1">By Rank</div>
            <div className="flex flex-wrap gap-1.5">
              {sortedRanks.map(([rank, count]) => (
                <span key={rank} className={`text-[9px] px-1.5 py-0.5 rounded font-medium ${
                  Number(rank) > 0 ? 'bg-success/15 text-success' : 'bg-muted text-muted-foreground'
                }`}>
                  Rank {rank} ({count})
                </span>
              ))}
            </div>
          </div>
        )}

        {withFeedback.length > 0 && (
          <div>
            <div className="text-[10px] text-muted-foreground mb-1">Most Active</div>
            <div className="space-y-1">
              {topFeedback.map(a => (
                <div key={a.user_id} className="flex items-center justify-between text-[11px] py-0.5 border-b border-border/30 last:border-0">
                  <span className="font-medium truncate">{a.user_id}</span>
                  <span className="font-mono text-muted-foreground">{a.feedback_count} fb</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
