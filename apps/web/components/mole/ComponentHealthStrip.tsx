'use client'

import { Card, CardContent, CardHeader, CardTitle, Skeleton, StatusDot } from '@sloughgpt/strui'
import type { ConnectionStatus, LiveHealthSnapshot } from '@/hooks/useLiveStatus'

interface ComponentHealthStripProps {
  health: LiveHealthSnapshot | null
  connectionStatus?: ConnectionStatus
}

/**
 * Live component strip: inference on one side, engines/system on the other,
 * so "inference setup and engines" are visible at a glance.
 *
 * Fields are consumed straight off the `useLiveStatus` snapshot — the typed
 * health-block contract from card 294818b7 has not landed yet; when it does,
 * swap this component's props for that block type.
 */

/** Compact, bounded summary of the free-form quantization payload. */
export function quantSummary(q: unknown): string | null {
  if (q == null) return null
  if (typeof q === 'string') return q || null
  if (typeof q === 'number') return `${q}-bit`
  if (typeof q === 'object') {
    const o = q as Record<string, unknown>
    if (typeof o.bits === 'number') return `${o.bits}-bit`
    if (typeof o.method === 'string') return o.method
    if (typeof o.dtype === 'string') return o.dtype
    return 'on'
  }
  return null
}

function connectionTone(status?: ConnectionStatus): 'success' | 'warning' | 'destructive' | 'muted' {
  if (status === 'connected') return 'success'
  if (status === 'connecting' || status === 'reloading') return 'warning'
  if (status === 'offline' || status === 'error') return 'destructive'
  return 'muted'
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-baseline justify-between gap-2">
      <dt className="text-[10px] uppercase tracking-wider text-muted-foreground">{label}</dt>
      <dd className="truncate text-xs font-medium text-foreground">{value}</dd>
    </div>
  )
}

function Cell({
  label,
  tone,
  status,
  rows,
}: {
  label: string
  tone: 'success' | 'warning' | 'destructive' | 'muted' | 'primary'
  status: string
  rows: Array<{ label: string; value: string }>
}) {
  return (
    <div
      data-testid="component-cell"
      className="rounded-md border border-border/50 bg-card/50 px-3 py-2"
    >
      <div className="flex items-center justify-between gap-2">
        <span className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
          {label}
        </span>
        <StatusDot tone={tone} label={status} showLabel className="text-[10px] text-foreground" />
      </div>
      <dl className="mt-1.5 space-y-0.5">
        {rows.map((r) => (
          <Row key={r.label} {...r} />
        ))}
      </dl>
    </div>
  )
}

export function ComponentHealthStrip({ health, connectionStatus }: ComponentHealthStripProps) {
  const connected = connectionStatus === 'connected'

  if (!health) {
    return (
      <Card data-testid="component-health-strip">
        <CardHeader className="pb-2">
          <CardTitle className="text-base">Live components</CardTitle>
        </CardHeader>
        <CardContent className="grid grid-cols-1 gap-3 sm:grid-cols-2">
          <Skeleton className="h-16 w-full" />
          <Skeleton className="h-16 w-full" />
        </CardContent>
      </Card>
    )
  }

  const quant = quantSummary(health.quantization)
  const score = Math.round(health.health_score ?? 0)

  return (
    <Card data-testid="component-health-strip">
      <CardHeader className="flex flex-row items-center justify-between gap-2 pb-2">
        <CardTitle className="text-base">Live components</CardTitle>
        <StatusDot
          tone={connectionTone(connectionStatus)}
          label={connected ? 'live' : (connectionStatus ?? 'unknown')}
          showLabel
          className="text-[10px] text-muted-foreground"
        />
      </CardHeader>
      <CardContent className="grid grid-cols-1 gap-3 sm:grid-cols-2">
        <Cell
          label="Inference"
          tone={health.model_loaded ? 'success' : 'warning'}
          status={health.model_loaded ? 'ready' : 'no model'}
          rows={[
            {
              label: 'Model',
              value: health.model_loaded
                ? (health.model_type || 'loaded')
                : health.model_loading
                  ? 'loading…'
                  : 'not loaded',
            },
            { label: 'Quantization', value: quant ?? '—' },
            { label: 'Requests', value: String(health.inference_count ?? 0) },
          ]}
        />
        <Cell
          label="Engines / system"
          tone={
            score >= 80 ? 'success' : score >= 50 || score === 0 ? 'warning' : 'destructive'
          }
          status={health.health_status || (score >= 80 ? 'healthy' : 'degraded')}
          rows={[
            { label: 'Health score', value: `${score}/100` },
            {
              label: 'CPU',
              value: health.cpu_percent != null ? `${health.cpu_percent.toFixed(1)}%` : '—',
            },
            {
              label: 'Memory',
              value: health.memory_percent != null ? `${health.memory_percent.toFixed(1)}%` : '—',
            },
          ]}
        />
      </CardContent>
    </Card>
  )
}
