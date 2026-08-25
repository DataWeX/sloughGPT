'use client'

import { useMemo } from 'react'
import { Card, CardHeader, CardTitle, CardContent, StatCard, KpiGrid } from '@sloughgpt/strui'
import type { FileEntry } from '@/lib/files-controller'

interface FileStatsCardProps {
  files: FileEntry[]
}

function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

function extGroup(filename: string): string {
  const ext = filename.split('.').pop()?.toLowerCase() ?? ''
  if (['txt', 'md'].includes(ext)) return 'Text'
  if (['json', 'jsonl', 'csv'].includes(ext)) return 'Data'
  if (['py', 'js', 'ts', 'tsx', 'jsx'].includes(ext)) return 'Code'
  if (['html', 'css'].includes(ext)) return 'Web'
  if (ext === 'pdf') return 'PDF'
  return 'Other'
}

const GROUP_COLORS: Record<string, string> = {
  Text: 'bg-success/15 text-success',
  Data: 'bg-primary/15 text-primary',
  Code: 'bg-warning/15 text-warning',
  Web: 'bg-accent/15 text-accent',
  PDF: 'bg-destructive/15 text-destructive',
  Other: 'bg-muted text-muted-foreground',
}

export function FileStatsCard({ files }: FileStatsCardProps) {
  const totalSize = useMemo(() => files.reduce((s, f) => s + (f.size ?? 0), 0), [files])
  const indexed = useMemo(() => files.filter(f => f.ingested).length, [files])
  const notIndexed = files.length - indexed

  const sorted = useMemo(() => {
    const groups: Record<string, { count: number; size: number }> = {}
    for (const f of files) {
      const g = extGroup(f.filename)
      if (!groups[g]) groups[g] = { count: 0, size: 0 }
      groups[g].count++
      groups[g].size += f.size ?? 0
    }
    return Object.entries(groups).sort((a, b) => b[1].count - a[1].count)
  }, [files])

  if (files.length === 0) return null

  return (
    <Card data-testid="file-stats">
      <CardHeader>
        <CardTitle className="text-base">File Overview</CardTitle>
      </CardHeader>
      <CardContent>
        <KpiGrid columns={3} className="mb-3">
          <StatCard label="Total" value={files.length} />
          <StatCard label="Size" value={formatSize(totalSize)} />
          <StatCard label="Indexed" value={<>{indexed}<span className="text-muted-foreground">/{files.length}</span></>} />
        </KpiGrid>
        <div className="space-y-1.5">
          {sorted.map(([group, { count, size }]) => (
            <div key={group} className="flex items-center justify-between text-[11px] py-0.5">
              <div className="flex items-center gap-2">
                <span className={`text-[9px] px-1.5 py-0.5 rounded font-medium ${GROUP_COLORS[group] ?? GROUP_COLORS.Other}`}>
                  {group}
                </span>
                <span className="text-muted-foreground">{count} file{count !== 1 ? 's' : ''}</span>
              </div>
              <span className="font-mono text-muted-foreground">{formatSize(size)}</span>
            </div>
          ))}
        </div>
        {notIndexed > 0 && (
          <div className="mt-2 text-[10px] text-warning">
            {notIndexed} file{notIndexed !== 1 ? 's' : ''} not indexed
          </div>
        )}
      </CardContent>
    </Card>
  )
}
