'use client'

import { Card, CardHeader, CardTitle, CardContent, cn } from '@sloughgpt/strui'

interface AutoTrainStatus {
  enabled: boolean
  threshold: number
  pending_count: number
  last_train: {
    started_at: string
    completed_at: string | null
    pairs_used: number
    checkpoint: string
  } | null
}

interface TrainingStats {
  total: number
  pending: number
  synced: number
  used: number
  by_quality: Record<string, number>
}

interface AutoTrainStatusCardProps {
  status: AutoTrainStatus | null
  stats: TrainingStats | null
}

export function AutoTrainStatusCard({ status, stats }: AutoTrainStatusCardProps) {
  return (
    <Card data-testid="auto-train-status">
      <CardHeader>
        <CardTitle className="text-base">Status</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          <div>
            <div className="text-[10px] text-muted-foreground uppercase tracking-wider font-medium">Mode</div>
            <div className={cn('text-sm font-semibold mt-0.5', status?.enabled ? 'text-success' : 'text-muted-foreground')}>
              {status?.enabled ? 'Enabled' : 'Disabled'}
            </div>
          </div>
          <div>
            <div className="text-[10px] text-muted-foreground uppercase tracking-wider font-medium">Threshold</div>
            <div className="text-sm font-semibold mt-0.5">{status?.threshold ?? 10}</div>
          </div>
          <div>
            <div className="text-[10px] text-muted-foreground uppercase tracking-wider font-medium">Pending</div>
            <div className="text-sm font-semibold mt-0.5">{stats?.pending ?? 0}</div>
          </div>
          <div>
            <div className="text-[10px] text-muted-foreground uppercase tracking-wider font-medium">Total Pairs</div>
            <div className="text-sm font-semibold mt-0.5">{stats?.total ?? 0}</div>
          </div>
        </div>

        {stats && Object.keys(stats.by_quality).length > 0 && (
          <div className="mt-4">
            <div className="text-[10px] text-muted-foreground uppercase tracking-wider font-medium mb-1.5">Quality Breakdown</div>
            <div className="flex gap-2 flex-wrap">
              {Object.entries(stats.by_quality).map(([q, count]) => (
                <div key={q} className="flex items-center gap-1.5">
                  <span className={cn(
                    'text-[9px] px-1.5 py-0.5 rounded font-medium',
                    q === 'good' ? 'bg-success/15 text-success' :
                    q === 'bad' ? 'bg-destructive/15 text-destructive' :
                    'bg-muted text-muted-foreground'
                  )}>
                    {q}
                  </span>
                  <span className="text-xs font-medium">{count}</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
