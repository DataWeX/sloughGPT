'use client'

import { Card, CardHeader, CardTitle, CardContent } from '@sloughgpt/strui'

interface DatasetStatsCardProps {
  totalDatasets: number
  totalSize: number
  totalRows: number
  recentUploads: number
}

function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(1)} GB`
}

export function DatasetStatsCard({
  totalDatasets,
  totalSize,
  totalRows,
  recentUploads,
}: DatasetStatsCardProps) {
  const items = [
    { label: 'Total Datasets', value: String(totalDatasets) },
    { label: 'Total Size', value: formatSize(totalSize) },
    { label: 'Total Rows', value: totalRows.toLocaleString() },
    { label: 'Recent Uploads', value: String(recentUploads) },
  ]

  return (
    <Card data-testid="dataset-stats">
      <CardHeader>
        <CardTitle className="text-base">Dataset Overview</CardTitle>
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
