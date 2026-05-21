'use client'

export const dynamic = 'force-dynamic'

import { useEffect, useState } from 'react'

import { AppRouteHeader, AppRouteHeaderLead } from '@/components/AppRouteHeader'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { apiGet } from '@/lib/http-client'
import { useApiHealth } from '@/hooks/useApiHealth'
import { cn } from '@/lib/cn'

interface CompareModel {
  name: string
  family: string
  params: string
  loaded: boolean
  tokens_per_sec: number
  memory_mb: number
  inference_count: number
}

interface CompareResponse {
  models: CompareModel[]
  current_model: string | null
}

const FAMILY_COLORS: Record<string, string> = {
  gpt: 'bg-green-500/10 text-green-600 dark:text-green-400 border-green-500/30',
  llama: 'bg-orange-500/10 text-orange-600 dark:text-orange-400 border-orange-500/30',
  phi: 'bg-pink-500/10 text-pink-600 dark:text-pink-400 border-pink-500/30',
  qwen: 'bg-red-500/10 text-red-600 dark:text-red-400 border-red-500/30',
  other: 'bg-muted/50 text-muted-foreground border-border/50',
}

function formatTps(tps: number): string {
  if (tps === 0) return '—'
  return `${tps.toFixed(1)} tok/s`
}

function formatMemory(mb: number): string {
  if (mb === 0) return '—'
  return mb > 1024 ? `${(mb / 1024).toFixed(2)} GB` : `${mb.toFixed(0)} MB`
}

export default function ComparePage() {
  const [data, setData] = useState<CompareResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [sortKey, setSortKey] = useState<keyof CompareModel>('params')
  const [sortAsc, setSortAsc] = useState(false)
  const { state: health } = useApiHealth()

  const fetchData = async () => {
    setLoading(true)
    try {
      const res = await apiGet<CompareResponse>('/benchmark/compare')
      setData(res)
    } catch {
      setData(null)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { fetchData() }, [])

  const sorted = data
    ? [...data.models].sort((a, b) => {
        if (sortKey === 'params') {
          const extractNum = (p: string) => {
            const n = parseFloat(p)
            return p.includes('B') ? n * 1000 : n
          }
          return sortAsc
            ? extractNum(a.params) - extractNum(b.params)
            : extractNum(b.params) - extractNum(a.params)
        }
        const va = a[sortKey] as number
        const vb = b[sortKey] as number
        return sortAsc ? va - vb : vb - va
      })
    : []

  const toggleSort = (key: keyof CompareModel) => {
    if (sortKey === key) {
      setSortAsc(!sortAsc)
    } else {
      setSortKey(key)
      setSortAsc(false)
    }
  }

  const activeRuntimeId = health !== null && health !== 'offline' && health.model_loaded ? health.model_type : null

  return (
    <div className="sl-page mx-auto max-w-5xl">
      <AppRouteHeader
        className="items-start"
        left={<AppRouteHeaderLead title="Compare" subtitle="Model metrics side by side" />}
        right={
          <Button type="button" variant="secondary" size="sm" onClick={fetchData} disabled={loading}>
            {loading ? 'Loading...' : 'Refresh'}
          </Button>
        }
      />

      <div className="space-y-4">
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Available models</CardTitle>
          </CardHeader>
          <CardContent className="p-0">
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-border/50">
                    <Th onClick={() => toggleSort('name')} active={sortKey === 'name'} asc={sortAsc}>Model</Th>
                    <Th onClick={() => toggleSort('family')} active={sortKey === 'family'} asc={sortAsc}>Family</Th>
                    <Th onClick={() => toggleSort('params')} active={sortKey === 'params'} asc={sortAsc}>Parameters</Th>
                    <Th onClick={() => toggleSort('tokens_per_sec')} active={sortKey === 'tokens_per_sec'} asc={sortAsc}>Throughput</Th>
                    <Th onClick={() => toggleSort('memory_mb')} active={sortKey === 'memory_mb'} asc={sortAsc}>Memory</Th>
                    <Th className="text-right">Status</Th>
                  </tr>
                </thead>
                <tbody>
                  {loading ? (
                    Array.from({ length: 5 }).map((_, i) => (
                      <tr key={i} className="border-b border-border/30">
                        {Array.from({ length: 6 }).map((_, j) => (
                          <td key={j} className="px-4 py-3"><div className="h-4 w-full animate-pulse rounded bg-muted" /></td>
                        ))}
                      </tr>
                    ))
                  ) : sorted.length === 0 ? (
                    <tr>
                      <td colSpan={6} className="px-4 py-12 text-center text-sm text-muted-foreground">No models available</td>
                    </tr>
                  ) : sorted.map((m) => {
                    const isLoaded = m.loaded || (activeRuntimeId && m.name === activeRuntimeId)
                    const color = FAMILY_COLORS[m.family] || FAMILY_COLORS.other
                    return (
                      <tr
                        key={m.name}
                        className={cn(
                          "border-b border-border/30 transition-colors hover:bg-muted/30",
                          isLoaded && "bg-primary/[0.04]",
                        )}
                      >
                        <td className="px-4 py-3">
                          <div className="flex items-center gap-2">
                            <span className={cn("font-medium truncate max-w-[240px]", isLoaded && "text-primary")}>{m.name}</span>
                          </div>
                        </td>
                        <td className="px-4 py-3">
                          <span className={cn("text-xs px-2 py-0.5 rounded-full border font-mono", color)}>{m.family}</span>
                        </td>
                        <td className="px-4 py-3 font-mono text-xs">{m.params}</td>
                        <td className="px-4 py-3 font-mono text-xs">{formatTps(m.tokens_per_sec)}</td>
                        <td className="px-4 py-3 font-mono text-xs">{formatMemory(m.memory_mb)}</td>
                        <td className="px-4 py-3 text-right">
                          {isLoaded ? (
                            <span className="text-xs px-2 py-0.5 rounded-full bg-primary/10 text-primary border border-primary/20 font-medium">Loaded</span>
                          ) : (
                            <span className="text-xs text-muted-foreground">Available</span>
                          )}
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          </CardContent>
        </Card>

        {data && (
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Summary</CardTitle>
            </CardHeader>
            <CardContent className="text-sm text-muted-foreground space-y-1">
              <p>{data.models.length} models available in catalog.</p>
              {data.current_model ? (
                <p><strong className="text-foreground">{data.current_model}</strong> is currently loaded{data.models.find(m => m.name === data.current_model) && <> — <span className="text-primary font-medium">{formatTps(data.models.find(m => m.name === data.current_model)!.tokens_per_sec)}</span> throughput</>}.</p>
              ) : (
                <p>No model currently loaded. Go to <a href="/models" className="text-primary underline underline-offset-2">Models</a> to load one.</p>
              )}
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  )
}

function Th({ children, onClick, active, asc, className }: { children: React.ReactNode; onClick?: () => void; active?: boolean; asc?: boolean; className?: string }) {
  return (
    <th
      className={cn("px-4 py-3 text-left text-xs font-medium text-muted-foreground uppercase tracking-wider cursor-pointer select-none hover:text-foreground transition-colors", className)}
      onClick={onClick}
    >
      <div className="flex items-center gap-1">
        {children}
        {active && (
          <span className="text-[10px]">{asc ? '▲' : '▼'}</span>
        )}
      </div>
    </th>
  )
}
