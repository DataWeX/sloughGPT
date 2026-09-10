'use client'

import { Card, CardHeader, CardTitle, CardContent, Input } from '@sloughgpt/strui'
import { Search } from 'lucide-react'

interface DatasetPreview {
  columns: string[]
  rows: any[][]
}

interface DatasetPreviewCardProps {
  datasetName: string
  preview: DatasetPreview | null
  loading: boolean
  search: string
  onSearchChange: (value: string) => void
}

export function DatasetPreviewCard({
  datasetName,
  preview,
  loading,
  search,
  onSearchChange,
}: DatasetPreviewCardProps) {
  return (
    <Card data-testid="dataset-preview">
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="text-base">Preview: {datasetName}</CardTitle>
        </div>
        <div className="relative">
          <Search className="absolute left-2 top-1/2 h-3 w-3 -translate-y-1/2 text-muted-foreground" />
          <Input
            placeholder="Filter rows..."
            value={search}
            onChange={e => onSearchChange(e.target.value)}
            className="pl-7 h-8 text-xs"
            data-testid="preview-search"
          />
        </div>
      </CardHeader>
      <CardContent>
        {loading ? (
          <div className="space-y-2">
            {Array.from({ length: 4 }).map((_, i) => (
              <div key={i} className="h-6 rounded bg-muted animate-pulse" />
            ))}
          </div>
        ) : !preview || preview.columns.length === 0 ? (
          <p className="text-sm text-muted-foreground text-center py-6">No preview data available.</p>
        ) : (
          <div className="max-h-80 overflow-auto">
            <table className="w-full text-xs" data-testid="preview-table">
              <thead>
                <tr>
                  {preview.columns.map(col => (
                    <th key={col} className="text-left font-medium text-muted-foreground px-2 py-1 border-b border-border/40">
                      {col}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {preview.rows.map((row, ri) => (
                  <tr key={ri} className="hover:bg-muted/50">
                    {row.map((cell, ci) => (
                      <td key={ci} className="px-2 py-1 border-b border-border/20 max-w-[200px] truncate">
                        {cell != null ? String(cell) : ''}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
