'use client'

import { Card, CardContent } from '@sloughgpt/strui'
import { StatCard, KpiGrid } from '@sloughgpt/strui'

interface BenchQuality {
  status: string
  total_responses: number
  coherence_score: number
  quality_score: number
  repetition_rate: number
  avg_length: number
  empty_rate: number
}

interface BenchStats {
  total: number
  avg_tokens: number
  models: string[]
}

interface QualityCardProps {
  quality: BenchQuality
  stats: BenchStats | null
}

function QualityDot({ score }: { score: number }) {
  return (
    <span className={`inline-block w-2 h-2 rounded-full ${
      score > 0.7 ? 'bg-success' : score > 0.4 ? 'bg-warning' : 'bg-destructive'
    }`} />
  )
}

export function QualityCard({ quality, stats }: QualityCardProps) {
  if (quality.status !== 'ok') return null

  return (
    <Card className="p-3">
      <span className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-2 block">Quality</span>
      <CardContent className="p-0">
        <KpiGrid columns={2}>
          <StatCard label="Coherence" value={quality.coherence_score.toFixed(2)} numeric icon={<QualityDot score={quality.coherence_score} />} />
          <StatCard label="Score" value={quality.quality_score.toFixed(2)} numeric icon={<QualityDot score={quality.quality_score} />} />
          <StatCard label="Responses" value={quality.total_responses.toString()} numeric />
          <StatCard label="Repetition" value={(quality.repetition_rate * 100).toFixed(1) + "%"} numeric />
        </KpiGrid>
        <div className="flex gap-3 mt-1.5 text-[11px] text-muted-foreground font-numeric">
          <span>Avg: {quality.avg_length.toFixed(1)}w</span>
          <span>Empty: {(quality.empty_rate * 100).toFixed(1)}%</span>
          {stats && <span>Tokens: {stats.avg_tokens.toFixed(0)}</span>}
        </div>
      </CardContent>
    </Card>
  )
}
