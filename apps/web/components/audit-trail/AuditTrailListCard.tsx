'use client'

import { useState } from 'react'
import { Card, CardHeader, CardTitle, CardContent, cn } from '@sloughgpt/strui'

interface Activity {
  type: string
  action: string
  detail: string
  status: string
  timestamp: string
  user: string
}

interface AuditTrailListCardProps {
  activities: Activity[]
}

const TYPE_COLORS: Record<string, string> = {
  training: 'bg-primary/15 text-primary',
  audit: 'bg-accent/15 text-accent',
  system: 'bg-muted text-muted-foreground',
}

const STATUS_COLORS: Record<string, string> = {
  completed: 'bg-success/15 text-success',
  success: 'bg-success/15 text-success',
  failed: 'bg-destructive/15 text-destructive',
  failure: 'bg-destructive/15 text-destructive',
  running: 'bg-primary/15 text-primary',
}

export function AuditTrailListCard({ activities }: AuditTrailListCardProps) {
  const [expanded, setExpanded] = useState<number | null>(null)

  return (
    <Card data-testid="audit-trail-list">
      <CardHeader>
        <CardTitle className="text-base">
          Events
          <span className="text-muted-foreground font-normal ml-2">({activities.length})</span>
        </CardTitle>
      </CardHeader>
      <CardContent>
        {activities.length === 0 ? (
          <div className="text-sm text-muted-foreground text-center py-4">No events recorded.</div>
        ) : (
          <div className="space-y-1 max-h-96 overflow-y-auto">
            {activities.map((a, i) => (
              <div key={i} className="rounded border border-border hover:border-primary/30 transition-colors">
                <button
                  className="w-full flex items-center justify-between p-2.5 text-left"
                  onClick={() => setExpanded(expanded === i ? null : i)}
                  data-testid={`audit-event-${i}`}
                >
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center gap-2">
                      <span className={cn('text-[9px] px-1.5 py-0.5 rounded font-medium', TYPE_COLORS[a.type] ?? 'bg-muted text-muted-foreground')}>
                        {a.type}
                      </span>
                      <span className="text-xs font-medium">{a.action}</span>
                    </div>
                    {a.detail && <div className="text-[10px] text-muted-foreground mt-0.5 truncate">{a.detail}</div>}
                  </div>
                  <div className="flex items-center gap-2 shrink-0">
                    {a.user && <span className="text-[10px] text-muted-foreground">{a.user}</span>}
                    {a.status && (
                      <span className={cn('text-[9px] px-1.5 py-0.5 rounded font-medium', STATUS_COLORS[a.status] ?? 'bg-muted text-muted-foreground')}>
                        {a.status}
                      </span>
                    )}
                    <span className="text-[9px] text-muted-foreground whitespace-nowrap">
                      {new Date(a.timestamp).toLocaleString()}
                    </span>
                  </div>
                </button>
                {expanded === i && (
                  <div className="px-2.5 pb-2.5 pt-1 border-t border-border/30">
                    <div className="grid grid-cols-2 gap-2 text-[10px]">
                      <div><span className="text-muted-foreground">Type: </span>{a.type}</div>
                      <div><span className="text-muted-foreground">Action: </span>{a.action}</div>
                      <div><span className="text-muted-foreground">User: </span>{a.user || '—'}</div>
                      <div><span className="text-muted-foreground">Time: </span>{new Date(a.timestamp).toLocaleString()}</div>
                    </div>
                    {a.detail && (
                      <div className="mt-2 p-2 bg-muted/30 rounded text-[10px] font-mono whitespace-pre-wrap">{a.detail}</div>
                    )}
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  )
}
