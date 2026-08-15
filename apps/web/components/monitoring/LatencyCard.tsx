'use client'

import { Card, CardContent } from '@sloughgpt/strui'
import { StatCard, KpiGrid } from '@sloughgpt/strui'

interface LatencyCardProps {
  chartHistory: Array<{ time: string; cpu: number; mem: number; tokens?: number; latency?: number }>
}

export function LatencyCard({ chartHistory }: LatencyCardProps) {
  const latencies = chartHistory.map(h => h.latency ?? 0).filter(l => l > 0)
  if (latencies.length === 0) return null

  const sorted = [...latencies].sort((a, b) => a - b)
  const avg = latencies.reduce((s, l) => s + l, 0) / latencies.length
  const p95 = sorted[Math.floor(sorted.length * 0.95)] ?? sorted[sorted.length - 1]

  return (
    <Card className="p-3">
      <span className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-2 block">Latency</span>
      <CardContent className="p-0">
        <KpiGrid columns={2}>
          <StatCard label="Avg" value={avg.toFixed(0) + "ms"} numeric />
          <StatCard label="P95" value={p95.toFixed(0) + "ms"} numeric />
          <StatCard label="Min" value={sorted[0].toFixed(0) + "ms"} numeric />
          <StatCard label="Max" value={sorted[sorted.length - 1].toFixed(0) + "ms"} numeric />
        </KpiGrid>
      </CardContent>
    </Card>
  )
}
