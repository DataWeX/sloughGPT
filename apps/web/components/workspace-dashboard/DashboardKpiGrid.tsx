'use client'

import { Card, CardContent, StatCard, KpiGrid } from '@sloughgpt/strui'

interface KpiItem {
  label: string
  value: number | string
}

interface DashboardKpiGridProps {
  kpis: KpiItem[]
}

export function DashboardKpiGrid({ kpis }: DashboardKpiGridProps) {
  return (
    <KpiGrid className="mb-6">
      {kpis.map((kpi) => (
        <StatCard key={kpi.label} label={kpi.label} value={kpi.value} />
      ))}
    </KpiGrid>
  )
}
