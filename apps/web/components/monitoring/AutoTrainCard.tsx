'use client'

import { memo } from 'react'
import { Card, CardContent } from '@sloughgpt/strui'
import { StatCard, KpiGrid } from '@sloughgpt/strui'
import type { AutoTrainStatus } from '@/lib/training-controller'

interface AutoTrainCardProps {
  status: AutoTrainStatus
}

export const AutoTrainCard = memo(function AutoTrainCard({ status }: AutoTrainCardProps) {
  return (
    <Card className="p-3">
      <span className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-2 block">Auto-Trainer</span>
      <CardContent className="p-0">
        <KpiGrid columns={2}>
          <StatCard
            label="Status"
            value={status.enabled ? <span className="font-mono">Running</span> : <span className="font-mono">Off</span>}
            icon={<span className={`inline-block w-2 h-2 rounded-full ${status.enabled ? 'bg-success' : 'bg-muted-foreground/50'}`} />}
          />
          <StatCard label="Queue" value={`${status.pending_conversations ?? 0}/${status.threshold ?? 0}`} numeric />
          <StatCard label="Trains" value={(status.total_trains ?? 0).toString()} numeric />
          <StatCard label="Loss" value={status.last_loss != null ? status.last_loss.toFixed(4) : '...'} numeric />
        </KpiGrid>
        {status.last_train && (
          <p className="text-[11px] text-muted-foreground mt-1.5 font-mono">
            Last: {new Date(status.last_train).toLocaleString()}
            {status.last_checkpoint && <> · {status.last_checkpoint}</>}
          </p>
        )}
        <p className="text-[10px] text-muted-foreground/50 mt-0.5 font-mono">
          {status.session_count ?? 0} sessions · {status.response_log_count ?? 0} logs · {status.interval_s ?? 0}s
        </p>
      </CardContent>
    </Card>
  )
})
