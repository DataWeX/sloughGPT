/**
 * Focal metrics — the four headline KPIs across the top of the dashboard.
 */

import { KpiGrid, StatCard } from '@sloughgpt/strui'
import { levelName } from './types'

export interface DashboardFocalMetricsProps {
  totalEpisodes: number
  score: number | null
  avgGrowth: string
  consciousnessLevel: number
  isLive: boolean
}

export function DashboardFocalMetrics({
  totalEpisodes,
  score,
  avgGrowth,
  consciousnessLevel,
  isLive,
}: DashboardFocalMetricsProps) {
  return (
    <KpiGrid columns={4}>
      <StatCard label="Total Episodes" value={totalEpisodes} numeric />
      <StatCard
        label="Quality Score"
        value={score !== null ? `${score.toFixed(0)}/100` : '—'}
        numeric={score !== null}
        description="How well responses match personality goals"
      />
      <StatCard label="Avg Growth" value={avgGrowth} numeric />
      <StatCard
        label="Consciousness Level"
        value={levelName(consciousnessLevel)}
        description={`Depth of reflection · ${isLive ? 'live' : 'auto-refresh'}`}
      />
    </KpiGrid>
  )
}
