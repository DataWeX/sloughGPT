'use client'

import { Card, CardContent } from '@sloughgpt/strui'

export interface KbStats {
  total_items: number
  topics: string[]
  avg_importance: number
  source_count: number
}

interface KbStatsCardsProps {
  stats: KbStats | null
}

export function KbStatsCards({ stats }: KbStatsCardsProps) {
  if (!stats) return null

  const cards = [
    { label: 'Total Entries', value: stats.total_items },
    { label: 'Topics', value: stats.topics.length },
    { label: 'Avg Importance', value: stats.avg_importance.toFixed(2) },
    { label: 'Sources', value: stats.source_count },
  ]

  return (
    <div className="grid grid-cols-2 gap-1.5 sm:grid-cols-4">
      {cards.map(s => (
        <Card key={s.label}>
          <CardContent className="p-3 text-center">
            <div className="text-xs text-muted-foreground">{s.label}</div>
            <div className="text-[11px] font-mono font-medium tabular-nums">{s.value}</div>
          </CardContent>
        </Card>
      ))}
    </div>
  )
}
