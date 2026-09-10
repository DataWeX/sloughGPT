'use client'

import { useState, useMemo } from 'react'
import { cn, Card, CardHeader, CardTitle, CardContent, Button } from '@sloughgpt/strui'

interface BenchmarkHistoryEntry {
  timestamp: string
  model: string
  throughput?: number
  latency?: number
  memory?: number
  tokens?: number
  quality?: number
}

interface BenchmarkHistoryCardProps {
  history: BenchmarkHistoryEntry[]
  onClear?: () => void
}

type SortKey = 'timestamp' | 'throughput' | 'latency' | 'memory' | 'model'
type SortDir = 'asc' | 'desc'

function formatTimestamp(ts: string): string {
  try {
    return new Date(ts).toLocaleString()
  } catch {
    return ts
  }
}

export function BenchmarkHistoryCard({ history, onClear }: BenchmarkHistoryCardProps) {
  const [sortKey, setSortKey] = useState<SortKey>('timestamp')
  const [sortDir, setSortDir] = useState<SortDir>('desc')
  const [modelFilter, setModelFilter] = useState<string>('all')

  const models = useMemo(() => {
    const set = new Set(history.map(h => h.model).filter(Boolean))
    return Array.from(set).sort()
  }, [history])

  const filtered = useMemo(() => {
    let data = modelFilter === 'all' ? history : history.filter(h => h.model === modelFilter)
    data = [...data].sort((a, b) => {
      const aVal = sortKey === 'timestamp' ? new Date(a.timestamp).getTime() : (a[sortKey] ?? 0)
      const bVal = sortKey === 'timestamp' ? new Date(b.timestamp).getTime() : (b[sortKey] ?? 0)
      return sortDir === 'asc' ? (aVal as number) - (bVal as number) : (bVal as number) - (aVal as number)
    })
    return data
  }, [history, sortKey, sortDir, modelFilter])

  const toggleSort = (key: SortKey) => {
    if (sortKey === key) {
      setSortDir(d => d === 'asc' ? 'desc' : 'asc')
    } else {
      setSortKey(key)
      setSortDir('desc')
    }
  }

  const stats = useMemo(() => {
    if (filtered.length === 0) return null
    const throughputs = filtered.map(h => h.throughput).filter((v): v is number => v != null && v > 0)
    const latencies = filtered.map(h => h.latency).filter((v): v is number => v != null && v > 0)
    const memories = filtered.map(h => h.memory).filter((v): v is number => v != null && v > 0)
    return {
      count: filtered.length,
      avgThroughput: throughputs.length > 0 ? throughputs.reduce((a, b) => a + b, 0) / throughputs.length : null,
      avgLatency: latencies.length > 0 ? latencies.reduce((a, b) => a + b, 0) / latencies.length : null,
      avgMemory: memories.length > 0 ? memories.reduce((a, b) => a + b, 0) / memories.length : null,
      bestThroughput: throughputs.length > 0 ? Math.max(...throughputs) : null,
      bestLatency: latencies.length > 0 ? Math.min(...latencies) : null,
    }
  }, [filtered])

  if (history.length === 0) {
    return (
      <Card data-testid="benchmark-history">
        <CardHeader><CardTitle className="text-base">Benchmark History</CardTitle></CardHeader>
        <CardContent>
          <p className="text-xs text-muted-foreground text-center py-3">No benchmark history yet.</p>
        </CardContent>
      </Card>
    )
  }

  return (
    <Card data-testid="benchmark-history">
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="text-base">Benchmark History</CardTitle>
          <div className="flex items-center gap-1">
            <select
              className="text-[10px] border border-border/40 rounded px-1.5 py-0.5 bg-background"
              value={modelFilter}
              onChange={e => setModelFilter(e.target.value)}
              aria-label="Filter by model"
            >
              <option value="all">All models</option>
              {models.map(m => <option key={m} value={m}>{m}</option>)}
            </select>
            {onClear && (
              <Button size="sm" variant="ghost" className="h-5 text-[9px] text-destructive" onClick={onClear}>
                Clear
              </Button>
            )}
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-3">
        {stats && (
          <div className="grid grid-cols-2 sm:grid-cols-5 gap-1.5">
            {[
              { label: 'Runs', value: String(stats.count) },
              { label: 'Avg Throughput', value: stats.avgThroughput != null ? `${stats.avgThroughput.toFixed(1)} tok/s` : '—' },
              { label: 'Best Throughput', value: stats.bestThroughput != null ? `${stats.bestThroughput.toFixed(1)} tok/s` : '—' },
              { label: 'Avg Latency', value: stats.avgLatency != null ? `${stats.avgLatency.toFixed(0)} ms` : '—' },
              { label: 'Avg Memory', value: stats.avgMemory != null ? `${stats.avgMemory.toFixed(0)} MB` : '—' },
            ].map(s => (
              <div key={s.label} className="rounded-md bg-muted/20 p-1.5 text-center">
                <p className="text-[9px] text-muted-foreground">{s.label}</p>
                <p className="text-[10px] font-mono font-medium">{s.value}</p>
              </div>
            ))}
          </div>
        )}

        <div className="overflow-x-auto">
          <table className="w-full text-[10px]">
            <thead>
              <tr className="border-b border-border/40">
                {([
                  { key: 'timestamp' as SortKey, label: 'Time' },
                  { key: 'model' as SortKey, label: 'Model' },
                  { key: 'throughput' as SortKey, label: 'Throughput' },
                  { key: 'latency' as SortKey, label: 'Latency' },
                  { key: 'memory' as SortKey, label: 'Memory' },
                ]).map(col => (
                  <th
                    key={col.key}
                    className="text-left py-1.5 px-1 text-muted-foreground font-medium cursor-pointer hover:text-foreground select-none"
                    onClick={() => col.key !== 'model' && toggleSort(col.key)}
                  >
                    {col.label}
                    {sortKey === col.key && <span className="ml-0.5">{sortDir === 'asc' ? '↑' : '↓'}</span>}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {filtered.slice(0, 50).map((h, i) => (
                <tr key={i} className="border-b border-border/20 last:border-0 hover:bg-muted/20">
                  <td className="py-1.5 px-1 text-muted-foreground">{formatTimestamp(h.timestamp)}</td>
                  <td className="py-1.5 px-1 font-medium">{h.model}</td>
                  <td className="py-1.5 px-1 font-mono">{h.throughput != null ? h.throughput.toFixed(1) : '—'}</td>
                  <td className="py-1.5 px-1 font-mono">{h.latency != null ? `${h.latency.toFixed(0)}ms` : '—'}</td>
                  <td className="py-1.5 px-1 font-mono">{h.memory != null ? `${h.memory.toFixed(0)}MB` : '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {filtered.length > 50 && (
          <p className="text-[9px] text-muted-foreground text-center">Showing 50 of {filtered.length} runs</p>
        )}
      </CardContent>
    </Card>
  )
}
