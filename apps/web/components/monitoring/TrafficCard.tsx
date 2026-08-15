'use client'

import { Card, CardContent } from '@sloughgpt/strui'
import { StatCard, KpiGrid } from '@sloughgpt/strui'
import type { LiveHealthSnapshot } from '@/hooks/useLiveStatus'

interface TrafficCardProps {
  liveHealth: LiveHealthSnapshot | null
}

export function formatTokens(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`
  return `${n}`
}

export function TrafficCard({ liveHealth }: TrafficCardProps) {
  if (!liveHealth) return null

  const rpm = liveHealth.requests_per_minute ?? 0
  const totalTokens = liveHealth.total_tokens ?? 0
  const avgPerReq = liveHealth.avg_tokens_per_request ?? 0
  if (rpm <= 0 && totalTokens <= 0 && avgPerReq <= 0) return null

  return (
    <Card className="p-3">
      <span className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-2 block">Traffic</span>
      <CardContent className="p-0">
        <KpiGrid columns={3}>
          <StatCard
            label="Requests/min"
            value={rpm > 0 ? rpm.toFixed(1) : '0'} numeric
            icon={<span className={`inline-block w-2 h-2 rounded-full ${rpm > 0 ? 'bg-success' : 'bg-muted-foreground/50'}`} />}
          />
          <StatCard
            label="Total tokens"
            value={formatTokens(totalTokens)} numeric
            icon={<span className={`inline-block w-2 h-2 rounded-full ${totalTokens > 0 ? 'bg-success' : 'bg-muted-foreground/50'}`} />}
          />
          <StatCard
            label="Avg tokens/req"
            value={avgPerReq > 0 ? avgPerReq.toFixed(0) : '0'} numeric
          />
        </KpiGrid>
      </CardContent>
    </Card>
  )
}
