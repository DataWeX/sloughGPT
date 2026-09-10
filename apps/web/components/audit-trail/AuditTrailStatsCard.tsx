'use client'

import { Card, CardHeader, CardTitle, CardContent, cn } from '@sloughgpt/strui'

interface Activity {
  type: string
  action: string
  status: string
  timestamp: string
}

interface AuditTrailStatsCardProps {
  activities: Activity[]
}

export function AuditTrailStatsCard({ activities }: AuditTrailStatsCardProps) {
  if (activities.length === 0) return null

  const typeCounts: Record<string, number> = {}
  const statusCounts: Record<string, number> = {}
  for (const a of activities) {
    typeCounts[a.type] = (typeCounts[a.type] || 0) + 1
    if (a.status) statusCounts[a.status] = (statusCounts[a.status] || 0) + 1
  }

  const last24h = activities.filter(a => Date.now() - new Date(a.timestamp).getTime() < 86400000).length
  const last7d = activities.filter(a => Date.now() - new Date(a.timestamp).getTime() < 604800000).length

  return (
    <Card data-testid="audit-trail-stats">
      <CardHeader>
        <CardTitle className="text-base">Overview</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-3">
          <div>
            <div className="text-[10px] text-muted-foreground uppercase tracking-wider font-medium">Total</div>
            <div className="text-sm font-semibold mt-0.5">{activities.length}</div>
          </div>
          <div>
            <div className="text-[10px] text-muted-foreground uppercase tracking-wider font-medium">Last 24h</div>
            <div className="text-sm font-semibold mt-0.5">{last24h}</div>
          </div>
          <div>
            <div className="text-[10px] text-muted-foreground uppercase tracking-wider font-medium">Last 7d</div>
            <div className="text-sm font-semibold mt-0.5">{last7d}</div>
          </div>
          <div>
            <div className="text-[10px] text-muted-foreground uppercase tracking-wider font-medium">Types</div>
            <div className="text-sm font-semibold mt-0.5">{Object.keys(typeCounts).length}</div>
          </div>
        </div>

        <div className="flex gap-2 flex-wrap">
          {Object.entries(typeCounts).map(([type, count]) => (
            <div key={type} className="flex items-center gap-1.5">
              <span className={cn(
                'text-[9px] px-1.5 py-0.5 rounded font-medium',
                type === 'training' ? 'bg-primary/15 text-primary' :
                type === 'audit' ? 'bg-accent/15 text-accent' :
                'bg-muted text-muted-foreground'
              )}>
                {type}
              </span>
              <span className="text-xs font-medium">{count}</span>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  )
}
