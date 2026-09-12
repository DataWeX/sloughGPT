'use client'

import { useMemo } from 'react'
import { cn } from '@sloughgpt/strui'
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
  Text: 'bg-[#28c840]/10 text-[#28c840]',
  Data: 'bg-[#0a7aff]/10 text-[#0a7aff]',
  Code: 'bg-[#febc2e]/10 text-[#febc2e]',
  Web: 'bg-[#bf5af2]/10 text-[#bf5af2]',
  PDF: 'bg-[#ff5f57]/10 text-[#ff5f57]',
  Other: 'bg-white/[0.06] text-[#8e8e93]',
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
    <div className="rounded-xl border border-white/[0.06] bg-[#0a0a0a] overflow-hidden" data-testid="file-stats">
      <div className="flex items-center h-11 px-4 bg-[#1c1c1e] border-b border-white/[0.06]">
        <span className="text-[11px] font-medium text-[#8e8e93]">File Overview</span>
      </div>
      <div className="px-4 py-3">
        {/* KPI row */}
        <div className="grid grid-cols-3 gap-3 mb-3">
          {[
            { label: 'Total', value: files.length },
            { label: 'Size', value: formatSize(totalSize) },
            { label: 'Indexed', value: `${indexed}/${files.length}` },
          ].map(kpi => (
            <div key={kpi.label} className="rounded-lg border border-white/[0.04] bg-[#111111] px-3 py-2">
              <p className="text-[10px] text-[#636366] uppercase tracking-wider">{kpi.label}</p>
              <p className="text-[13px] font-medium text-[#c7c7cc] font-mono mt-0.5">{kpi.value}</p>
            </div>
          ))}
        </div>
        {/* Group breakdown */}
        <div className="space-y-1">
          {sorted.map(([group, { count, size }]) => (
            <div key={group} className="flex items-center justify-between text-[11px] py-1">
              <div className="flex items-center gap-2">
                <span className={cn('text-[9px] px-1.5 py-0.5 rounded font-medium', GROUP_COLORS[group] ?? GROUP_COLORS.Other)}>
                  {group}
                </span>
                <span className="text-[#636366]">{count} file{count !== 1 ? 's' : ''}</span>
              </div>
              <span className="font-mono text-[#636366]">{formatSize(size)}</span>
            </div>
          ))}
        </div>
        {notIndexed > 0 && (
          <div className="mt-2 text-[10px] text-[#febc2e]">
            {notIndexed} file{notIndexed !== 1 ? 's' : ''} not indexed
          </div>
        )}
      </div>
    </div>
  )
}
