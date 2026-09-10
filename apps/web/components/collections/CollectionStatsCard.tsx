'use client'

import { Card, CardHeader, CardTitle, CardContent } from '@sloughgpt/strui'

interface CollectionStats {
  pipelines: number
  sources: number
  stores: number
  filters: number
}

interface CollectionStatsCardProps {
  stats: CollectionStats | null
}

export function CollectionStatsCard({ stats }: CollectionStatsCardProps) {
  if (!stats) return null

  const items = [
    { label: 'Pipelines', value: stats.pipelines },
    { label: 'Sources', value: stats.sources },
    { label: 'Stores', value: stats.stores },
    { label: 'Filters', value: stats.filters },
  ]

  return (
    <Card data-testid="collection-stats">
      <CardHeader>
        <CardTitle className="text-base">Overview</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          {items.map(item => (
            <div key={item.label}>
              <div className="text-[10px] text-muted-foreground uppercase tracking-wider font-medium">{item.label}</div>
              <div className="text-sm font-semibold mt-0.5">{item.value}</div>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  )
}
