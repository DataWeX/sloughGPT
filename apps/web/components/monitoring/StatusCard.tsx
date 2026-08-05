'use client'

import { Card, CardContent } from '@sloughgpt/strui'
import { StatCard, KpiGrid } from '@sloughgpt/strui'
import { formatUptime } from '@/lib/chat-utils'
import type { LiveHealthSnapshot, ConnectionStatus } from '@/hooks/useLiveStatus'
import type { DetailedHealth } from '@/lib/system-controller'

interface StatusCardProps {
  liveHealth: LiveHealthSnapshot | null
  detailed: DetailedHealth | null
  connectionStatus: ConnectionStatus
  inferenceRate: number
  loaded: boolean
}

export function StatusCard({ liveHealth, detailed, connectionStatus, inferenceRate, loaded }: StatusCardProps) {
  const apiOk = (liveHealth?.health_status ?? detailed?.status) === 'healthy'
  const modelLoaded = liveHealth?.model_loaded ?? detailed?.model_loaded ?? false

  return (
    <Card className="p-3">
      <div className="flex items-center justify-between mb-2">
        <span className="text-xs font-medium text-muted-foreground uppercase tracking-wider">Status</span>
        {connectionStatus === 'connected' && liveHealth && (
          <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded-full bg-success/10 text-success text-[10px] font-medium">
            <span className="inline-block h-1.5 w-1.5 rounded-full bg-success animate-pulse" />
            live
          </span>
        )}
        {connectionStatus === 'connecting' && (
          <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded-full bg-warning/10 text-warning text-[10px] font-medium">
            <span className="inline-block h-1.5 w-1.5 rounded-full bg-warning animate-pulse" />
            reconnecting
          </span>
        )}
        {(connectionStatus === 'offline' || connectionStatus === 'reloading') && (
          <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded-full bg-destructive/10 text-destructive text-[10px] font-medium">
            <span className="inline-block h-1.5 w-1.5 rounded-full bg-destructive" />
            {connectionStatus === 'reloading' ? 'reloading' : 'offline'}
          </span>
        )}
      </div>
      <CardContent className="p-0">
        <KpiGrid columns={2}>
          <StatCard
            label="API"
            value={!loaded ? '...' : <span className="font-mono">{apiOk ? 'Healthy' : 'Error'}</span>}
            icon={<span className={`inline-block w-2 h-2 rounded-full ${!loaded ? 'bg-warning' : apiOk ? 'bg-success' : 'bg-destructive'}`} />}
          />
          <StatCard
            label="Model"
            value={!loaded ? '...' : <span className="font-mono">{modelLoaded ? (liveHealth?.model_type || detailed?.model_type || 'Loaded') : 'Not loaded'}</span>}
            icon={<span className={`inline-block w-2 h-2 rounded-full ${!loaded ? 'bg-warning' : modelLoaded ? 'bg-success' : 'bg-warning'}`} />}
          />
          <StatCard
            label="Uptime"
            value={!loaded ? '...' : <span className="font-mono">{formatUptime(liveHealth?.uptime_seconds ?? detailed?.uptime_seconds ?? 0)}</span>}
          />
          <StatCard
            label="Responses"
            value={!loaded ? '...' : <span className="font-mono">{String(liveHealth?.inference_count ?? detailed?.inference?.inference_count ?? 0)}</span>}
          />
          <StatCard
            label="Rate"
            value={!loaded ? '...' : <span className="font-mono">{inferenceRate.toFixed(1)}/min</span>}
          />
          <StatCard
            label="Tokens/s"
            value={!loaded ? '...' : <span className="font-mono">{liveHealth?.tokens_per_sec != null ? liveHealth.tokens_per_sec.toFixed(1) : '...'}</span>}
            icon={<span className={`inline-block w-2 h-2 rounded-full ${liveHealth && (liveHealth.tokens_per_sec ?? 0) > 0 ? 'bg-success' : 'bg-muted-foreground/50'}`} />}
          />
          <StatCard
            label="Errors"
            value={!loaded ? '...' : <span className="font-mono">{String(liveHealth?.error_count ?? 0)}</span>}
            icon={<span className={`inline-block w-2 h-2 rounded-full ${(liveHealth?.error_count ?? 0) > 0 ? 'bg-destructive' : 'bg-success'}`} />}
          />
        </KpiGrid>
      </CardContent>
    </Card>
  )
}
