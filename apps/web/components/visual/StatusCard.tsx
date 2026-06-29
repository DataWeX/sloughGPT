'use client'

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { StatCard, KpiGrid } from '@/components/strui'
import { cn } from '@/lib/cn'

interface StatusCardProps {
  loading: boolean
  visualStatus: { loaded?: boolean; vision_encoder?: string; llm?: string } | null
  dpoStatus: { status?: string } | null
}

export default function StatusCard({ loading, visualStatus, dpoStatus }: StatusCardProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Status</CardTitle>
      </CardHeader>
      <CardContent>
        {loading ? (
          <div className="h-20 animate-pulse rounded bg-muted" />
        ) : (
          <KpiGrid columns={4}>
            <StatCard
              label="Visual AI Loaded"
              value={visualStatus?.loaded ? 'Yes' : 'No'}
              icon={<span className={cn("w-2 h-2 rounded-full", visualStatus?.loaded ? 'bg-success' : 'bg-muted-foreground/40')} />}
            />
            <StatCard label="Vision Encoder" value={visualStatus?.vision_encoder || '—'} />
            <StatCard label="LLM" value={visualStatus?.llm || '—'} />
            <StatCard
              label="DPO Status"
              value={dpoStatus?.status || 'idle'}
              icon={<span className={cn("w-2 h-2 rounded-full",
                dpoStatus?.status === 'running' ? 'bg-warning animate-pulse' :
                dpoStatus?.status === 'completed' ? 'bg-success' : 'bg-muted-foreground/40'
              )} />}
            />
          </KpiGrid>
        )}
      </CardContent>
    </Card>
  )
}
