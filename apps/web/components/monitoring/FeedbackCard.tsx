'use client'

import { memo } from 'react'
import { cn, Card, CardContent } from '@sloughgpt/strui'
import { StatCard, KpiGrid, Skeleton } from '@sloughgpt/strui'
import { Button } from '@sloughgpt/strui'
import { logger } from '@/lib/dev-log'
import { apiPost } from '@/lib/http-client'
import { useToastStore } from '@/lib/toast-store'

interface DpoStatus {
  status: string
  last_run: string | null
  accepted_count: number
  rejected_count: number
  result: { perplexity_delta?: number; bleu_delta?: number; verdict?: string; report_path?: string } | null
}

interface VisualStatus {
  visual_loaded: boolean
  training: { status: string }
}

interface FeedbackCardProps {
  dpoStatus: DpoStatus | null
  visualStatus: VisualStatus | null
  dpoRunning: boolean
  onDpoRunningChange: (v: boolean) => void
  onRefresh: () => void
}

export const FeedbackCard = memo(function FeedbackCard({ dpoStatus, visualStatus, dpoRunning, onDpoRunningChange, onRefresh }: FeedbackCardProps) {
  const addToast = useToastStore(s => s.addToast)
  if (!dpoStatus && !visualStatus) return null

  return (
    <Card className="p-3">
      <span className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-2 block">Feedback + Vision</span>
      <CardContent className="p-0">
        <KpiGrid columns={2}>
          <StatCard
            label="Feedback"
            value={dpoStatus ? <span className="font-mono">{dpoStatus.status}</span> : <Skeleton className="h-5 w-16" />}
            icon={<span className={cn('inline-block w-2 h-2 rounded-full', !dpoStatus ? 'bg-warning' :
              dpoStatus.status === 'running' ? 'bg-warning' :
              dpoStatus.status === 'completed' ? 'bg-success' :
              dpoStatus.status === 'error' ? 'bg-destructive' : 'bg-muted-foreground/50')} />}
          />
          <StatCard
            label="Vision"
            value={visualStatus ? <span className="font-mono">{visualStatus.visual_loaded ? 'Yes' : 'No'}</span> : <Skeleton className="h-5 w-12" />}
            icon={<span className={cn('inline-block w-2 h-2 rounded-full', !visualStatus ? 'bg-warning' : visualStatus.visual_loaded ? 'bg-success' : 'bg-muted-foreground/50')} />}
          />
          <StatCard label="Accepted" value={dpoStatus?.accepted_count != null ? dpoStatus.accepted_count.toString() : <Skeleton className="h-5 w-8" />} numeric />
          <StatCard label="Rejected" value={dpoStatus?.rejected_count != null ? dpoStatus.rejected_count.toString() : <Skeleton className="h-5 w-8" />} numeric />
        </KpiGrid>
        <div className="mt-2">
          <Button
            size="sm"
            className="h-6 text-[11px]"
            disabled={dpoRunning || dpoStatus?.status === 'running'}
            onClick={async () => {
              onDpoRunningChange(true)
              try {
                await apiPost('/multimodal/dpo', {})
                onRefresh()
              } catch (err) {
                addToast('DPO training failed', 'error')
                logger.error('Could not dpo training', { exception: String(err) })
              }
              onDpoRunningChange(false)
            }}
          >
            {dpoRunning || dpoStatus?.status === 'running' ? 'Running...' : 'Run feedback'}
          </Button>
        </div>
      </CardContent>
    </Card>
  )
})
