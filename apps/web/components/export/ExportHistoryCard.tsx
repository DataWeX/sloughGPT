'use client'

import { useState, useEffect } from 'react'
import { cn, Card, CardHeader, CardTitle, CardContent } from '@sloughgpt/strui'
import { timeAgo } from '@/lib/time-ago'

const STORAGE_KEY = 'sloughgpt-export-history'

interface ExportRecord {
  format: string
  timestamp: number
  fileCount: number
  label: string
}

function loadHistory(): ExportRecord[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    if (!raw) return []
    return JSON.parse(raw) as ExportRecord[]
  } catch {
    return []
  }
}

function saveHistory(history: ExportRecord[]) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(history.slice(0, 20)))
}

export function recordExport(format: string, fileCount: number) {
  const history = loadHistory()
  history.unshift({
    format,
    timestamp: Date.now(),
    fileCount,
    label: format.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase()),
  })
  saveHistory(history.slice(0, 20))
}

function formatColor(format: string): string {
  if (format.includes('sou')) return 'bg-success/15 text-success'
  if (format.includes('onnx')) return 'bg-primary/15 text-primary'
  if (format.includes('gguf')) return 'bg-warning/15 text-warning'
  if (format.includes('torch')) return 'bg-accent/15 text-accent'
  return 'bg-muted text-muted-foreground'
}

export function ExportHistoryCard() {
  const [history, setHistory] = useState<ExportRecord[]>([])

  useEffect(() => {
    setHistory(loadHistory())
  }, [])

  if (history.length === 0) return null

  const formatCounts: Record<string, number> = {}
  for (const r of history) {
    formatCounts[r.label] = (formatCounts[r.label] || 0) + 1
  }
  const topFormats = Object.entries(formatCounts).sort((a, b) => b[1] - a[1]).slice(0, 4)

  return (
    <Card data-testid="export-history">
      <CardHeader>
        <CardTitle className="text-base">Export History</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-4">
          <div>
            <div className="text-[10px] text-muted-foreground uppercase tracking-wider font-medium">Total Exports</div>
            <div className="text-lg font-semibold">{history.length}</div>
          </div>
          <div>
            <div className="text-[10px] text-muted-foreground uppercase tracking-wider font-medium">Formats Used</div>
            <div className="text-lg font-semibold">{Object.keys(formatCounts).length}</div>
          </div>
          <div>
            <div className="text-[10px] text-muted-foreground uppercase tracking-wider font-medium">Last Export</div>
            <div className="text-lg font-semibold">{timeAgo(history[0].timestamp)}</div>
          </div>
          <div>
            <div className="text-[10px] text-muted-foreground uppercase tracking-wider font-medium">Total Files</div>
            <div className="text-lg font-semibold">{history.reduce((s, r) => s + r.fileCount, 0)}</div>
          </div>
        </div>
        <div className="text-[10px] font-medium uppercase tracking-wider text-muted-foreground mb-1.5">Recent</div>
        <div className="space-y-1">
          {history.slice(0, 5).map((r, idx) => (
            <div key={idx} className="flex items-center justify-between text-[11px]">
              <div className="flex items-center gap-2">
                <span className={cn('text-[9px] px-1.5 py-0.5 rounded font-medium', formatColor(r.format))}>
                  {r.label}
                </span>
                <span className="text-muted-foreground">{r.fileCount} file{r.fileCount !== 1 ? 's' : ''}</span>
              </div>
              <span className="text-muted-foreground">{timeAgo(r.timestamp)}</span>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  )
}
