'use client'

import { Card, CardContent } from '@sloughgpt/strui'

export interface KpiItem {
  label: string
  value: number | string
}

export interface UsageKpiGridProps {
  items: KpiItem[]
}

export function UsageKpiGrid({ items }: UsageKpiGridProps) {
  return (
    <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
      {items.map(item => (
        <Card key={item.label}>
          <CardContent className="p-3 text-center">
            <div className="text-lg font-bold">{item.value}</div>
            <div className="text-[10px] text-muted-foreground">{item.label}</div>
          </CardContent>
        </Card>
      ))}
    </div>
  )
}
