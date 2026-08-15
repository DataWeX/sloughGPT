'use client'

import { Card, CardContent } from '@sloughgpt/strui'
import { StatCard, KpiGrid } from '@sloughgpt/strui'
import { Button } from '@sloughgpt/strui'
import { systemController, type ExecutorStatus } from '@/lib/system-controller'

interface ExecutorPoolProps {
  status: ExecutorStatus
  onRefresh: () => void
}

export function ExecutorPool({ status, onRefresh }: ExecutorPoolProps) {
  if (!status.initialized) return null

  return (
    <Card className="p-3">
      <div className="flex items-center justify-between mb-2">
        <span className="text-xs font-medium text-muted-foreground uppercase tracking-wider">Pool</span>
        {status.total_tracked > 0 && (
          <Button variant="outline" size="sm" className="text-[10px] h-7" onClick={async () => { await systemController.purgeExecutorJobs(3600); onRefresh() }}>
            Purge
          </Button>
        )}
      </div>
      <CardContent className="p-0">
        <KpiGrid columns={4}>
          <StatCard
            label="Active"
            value={status.active_jobs.toString()} numeric
            icon={<span className={`inline-block w-2 h-2 rounded-full ${status.active_jobs > 0 ? 'bg-warning' : 'bg-success'}`} />}
          />
          <StatCard label="Workers" value={status.max_workers.toString()} numeric />
          <StatCard label="Tracked" value={status.total_tracked.toString()} numeric />
          <StatCard label="Queue" value={status.jobs.filter(j => j.status === 'queued').length.toString()} numeric />
        </KpiGrid>
        {status.jobs.length > 0 && (
          <div className="mt-2 text-[11px] text-muted-foreground space-y-0.5 font-mono">
            {status.jobs.slice(0, 4).map(j => {
              const ds = j.cancel_requested && j.status === 'running' ? 'cancelling' : j.status
              return (
                <div key={j.job_id} className="flex items-center gap-1.5 px-1 py-0.5 rounded hover:bg-muted/30 transition-colors">
                  <span className={`text-[9px] px-1.5 py-0.5 rounded font-medium ${
                    ds === 'running' ? 'bg-warning/15 text-warning' : ds === 'cancelling' ? 'bg-warning/15 text-warning' :
                    ds === 'completed' ? 'bg-success/15 text-success' : ds === 'failed' ? 'bg-destructive/15 text-destructive' : 'bg-muted text-muted-foreground'
                  }`}>{ds}</span>
                  <span className="truncate">{j.job_id}</span>
                  {j.elapsed_s != null && <span className="text-muted-foreground/60">{j.elapsed_s.toFixed(1)}s</span>}
                  {(j.status === 'running' || j.status === 'queued') && !j.cancel_requested && (
                    <Button variant="ghost" size="sm" className="text-[10px] h-7 text-destructive hover:text-destructive ml-auto"
                      onClick={async () => { await systemController.cancelExecutorJob(j.job_id); onRefresh() }}>
                      Cancel
                    </Button>
                  )}
                </div>
              )
            })}
            {status.jobs.length > 4 && <div className="text-muted-foreground/40">+{status.jobs.length - 4} more</div>}
          </div>
        )}
      </CardContent>
    </Card>
  )
}
