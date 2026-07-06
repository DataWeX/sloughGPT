'use client'

import { Card, CardContent, CardHeader, CardTitle } from '@sloughgpt/strui'
import { StatCard, KpiGrid } from '@sloughgpt/strui'
import type { BenchmarkResult } from '@/lib/benchmark-controller'

interface SummaryCardProps {
  completedResults: [string, BenchmarkResult][]
  models: { id: string; name: string }[]
}

export default function SummaryCard({ completedResults, models }: SummaryCardProps) {
  if (completedResults.length < 2) return null

  return (
    <Card>
      <CardHeader><CardTitle className="text-base">Summary</CardTitle></CardHeader>
      <CardContent>
        <KpiGrid columns={completedResults.length >= 4 ? 4 : 2}>
          {completedResults.map(([modelId, r]) => {
            const modelName = models.find(m => m.id === modelId)?.name || modelId
            return <StatCard key={modelId} label={modelName} value={`${r.throughput_tokens_per_sec.toFixed(1)} tok/s`} />
          })}
        </KpiGrid>
      </CardContent>
    </Card>
  )
}
