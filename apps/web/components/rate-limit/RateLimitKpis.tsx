'use client'

import { KpiGrid, StatCard, cn } from '@sloughgpt/strui'

export interface RateLimitStatus {
  enabled: boolean
  requests_per_minute?: number
  burst_size?: number
}

interface RateLimitKpisProps {
  status?: RateLimitStatus | null
}

export function RateLimitKpis({ status }: RateLimitKpisProps) {
  return (
    <KpiGrid columns={3}>
      <StatCard
        label="Status"
        value={status?.enabled ? 'Active' : 'Inactive'}
        icon={
          <span
            className={cn(
              'inline-block w-2 h-2 rounded-full',
              status?.enabled ? 'bg-success' : 'bg-muted-foreground',
            )}
          />
        }
      />
      <StatCard
        label="Requests/min"
        value={status?.requests_per_minute?.toString() ?? '—'}
      />
      <StatCard label="Burst Size" value={status?.burst_size?.toString() ?? '—'} />
    </KpiGrid>
  )
}
