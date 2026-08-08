'use client'

import { useMemo } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@sloughgpt/strui'
import { Badge } from '@sloughgpt/strui'
import type { Checkpoint } from '@/lib/souls-controller'

interface TrainingDashboardProps {
  checkpoints: Checkpoint[]
}

function formatDuration(seconds: number): string {
  if (seconds < 60) return `${Math.round(seconds)}s`
  const m = Math.floor(seconds / 60)
  const s = Math.round(seconds % 60)
  if (m < 60) return `${m}m ${s}s`
  const h = Math.floor(m / 60)
  return `${h}h ${m % 60}m`
}

function formatSize(mb: number): string {
  if (mb < 1) return `${Math.round(mb * 1024)}KB`
  if (mb < 1024) return `${mb.toFixed(1)}MB`
  return `${(mb / 1024).toFixed(1)}GB`
}

export function TrainingDashboard({ checkpoints }: TrainingDashboardProps) {
  const stats = useMemo(() => {
    if (!checkpoints.length) return null

    const withLoss = checkpoints.filter(c => c.loss != null && c.loss > 0)
    const bestLoss = withLoss.length ? Math.min(...withLoss.map(c => c.loss!)) : null
    const avgLoss = withLoss.length ? withLoss.reduce((s, c) => s + c.loss!, 0) / withLoss.length : null
    const totalDuration = checkpoints.reduce((s, c) => s + (c.training_duration_s ?? 0), 0)
    const totalSize = checkpoints.reduce((s, c) => s + (c.size_mb ?? 0), 0)
    const loaded = checkpoints.filter(c => c.is_loaded).length
    const types = new Set(checkpoints.map(c => c.model_type).filter(Boolean))
    const datasets = new Set(checkpoints.map(c => c.training_dataset).filter(Boolean))

    return {
      total: checkpoints.length,
      bestLoss,
      avgLoss,
      totalDuration,
      totalSize,
      loaded,
      types: types.size,
      datasets: datasets.size,
    }
  }, [checkpoints])

  if (!stats) return null

  const kpis = [
    { label: 'Checkpoints', value: String(stats.total), sub: `${stats.loaded} loaded` },
    { label: 'Best Loss', value: stats.bestLoss != null ? stats.bestLoss.toFixed(3) : '—', sub: stats.avgLoss != null ? `avg ${stats.avgLoss.toFixed(3)}` : undefined },
    { label: 'Training Time', value: formatDuration(stats.totalDuration), sub: undefined },
    { label: 'Total Size', value: formatSize(stats.totalSize), sub: undefined },
    { label: 'Model Types', value: String(stats.types), sub: undefined },
    { label: 'Datasets', value: String(stats.datasets), sub: undefined },
  ]

  return (
    <Card data-testid="training-dashboard">
      <CardHeader className="flex flex-row items-center justify-between py-3">
        <CardTitle className="text-base">Dashboard</CardTitle>
        {stats.loaded > 0 && (
          <Badge variant="outline" className="text-[10px] px-1.5 py-0 h-4 bg-primary/10 text-primary border-primary/20">
            {stats.loaded} active
          </Badge>
        )}
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-3 gap-3">
          {kpis.map(kpi => (
            <div key={kpi.label} className="space-y-0.5">
              <span className="text-[10px] text-muted-foreground/50 uppercase tracking-wider">{kpi.label}</span>
              <p className="text-lg font-semibold">{kpi.value}</p>
              {kpi.sub && <p className="text-[10px] text-muted-foreground">{kpi.sub}</p>}
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  )
}
