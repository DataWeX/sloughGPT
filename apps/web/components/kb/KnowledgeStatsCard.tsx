'use client'

import { Card, CardContent, CardHeader, CardTitle } from '@sloughgpt/strui'

interface KnowledgeStatsCardProps {
  totalItems: number
  topics: string[]
  avgImportance: number
  sources: Record<string, unknown>
}

export function KnowledgeStatsCard({ totalItems, topics, avgImportance, sources }: KnowledgeStatsCardProps) {
  const stats = [
    { label: 'Total Entries', value: totalItems },
    { label: 'Topics', value: topics.length },
    { label: 'Avg Importance', value: avgImportance.toFixed(2) },
    { label: 'Sources', value: Object.keys(sources).length },
  ]

  return (
    <Card>
      <CardHeader className="pb-2 pt-2.5 px-2.5">
        <CardTitle className="text-[11px] font-medium">Knowledge Stats</CardTitle>
      </CardHeader>
      <CardContent className="px-2.5 pb-2.5">
        <div className="grid grid-cols-2 gap-1.5 sm:grid-cols-4">
          {stats.map(s => (
            <div key={s.label} className="rounded-md bg-muted/30 p-3 text-center">
              <div className="text-xs text-muted-foreground">{s.label}</div>
              <div className="text-[11px] font-mono font-medium tabular-nums">{s.value}</div>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  )
}
