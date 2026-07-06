'use client'

export const dynamic = 'force-dynamic'

import { useState, useEffect, useCallback, useMemo } from 'react'
import { AppRouteHeader, AppRouteHeaderLead } from '@/components/AppRouteHeader'
import { Card, CardContent, CardHeader, CardTitle } from '@sloughgpt/strui'
import { Button } from '@sloughgpt/strui'
import { Badge, Chip, EmptyCard, KpiGrid, StatCard } from '@sloughgpt/strui'
import { SearchInput } from '@sloughgpt/strui'
import { IconRefresh, IconDownload } from '@sloughgpt/strui'
import { apiGet, apiDelete } from '@/lib/http-client'

interface ErrorEntry {
  id: string
  message: string
  source: string
  stack?: string
  url?: string
  line?: number
  col?: number
  client_host?: string
  timestamp: string
  metadata?: Record<string, unknown>
  fingerprint?: string
}

interface ErrorSummary {
  errors: ErrorEntry[]
  unread_count: number
  total: number
  offset: number
  limit: number
}

interface ErrorGroup {
  fingerprint: string
  message: string
  source: string
  count: number
  latest: string
  sample_id: string
  sample_url: string
  sample_line: number | null
}

interface TrendBucket {
  hour: string
  count: number
}

function classifyError(entry: ErrorEntry): string {
  const msg = (entry.message || '').toLowerCase()
  if (msg.includes('hydrat') || msg.includes('text content does not match')) return 'hydration'
  if (msg.includes('cannot read propert') || msg.includes('is not a function') || msg.includes('is undefined')) return 'null-access'
  if (msg.includes('type ') && msg.includes('not assignable')) return 'type-error'
  if (msg.includes('chunk') || msg.includes('loading chunk')) return 'chunk-load'
  if (msg.includes('fetch') || msg.includes('network') || msg.includes('econnrefused')) return 'network'
  if (msg.includes('401') || msg.includes('unauthorized')) return 'auth'
  if (msg.includes('404') || msg.includes('not found')) return 'not-found'
  if (msg.includes('500') || msg.includes('internal server')) return 'server-error'
  if (msg.includes('cors') || msg.includes('access-control')) return 'cors'
  if (msg.includes('module not found') || msg.includes('cannot find module')) return 'build'
  if (msg.includes('maximum update depth') || msg.includes('too many re-renders')) return 'infinite-loop'
  if (entry.source?.includes('error-boundary') || entry.source?.includes('CustomErrorHandler')) return 'boundary'
  return 'other'
}

const CATEGORY_COLORS: Record<string, string> = {
  hydration: 'bg-blue-500/10 text-blue-600 border-blue-500/30',
  'null-access': 'bg-orange-500/10 text-orange-600 border-orange-500/30',
  'type-error': 'bg-purple-500/10 text-purple-600 border-purple-500/30',
  'chunk-load': 'bg-yellow-500/10 text-yellow-600 border-yellow-500/30',
  network: 'bg-red-500/10 text-red-600 border-red-500/30',
  auth: 'bg-amber-500/10 text-amber-600 border-amber-500/30',
  'not-found': 'bg-gray-500/10 text-gray-600 border-gray-500/30',
  'server-error': 'bg-red-600/10 text-red-700 border-red-600/30',
  cors: 'bg-pink-500/10 text-pink-600 border-pink-500/30',
  build: 'bg-indigo-500/10 text-indigo-600 border-indigo-500/30',
  'infinite-loop': 'bg-red-400/10 text-red-500 border-red-400/30',
  boundary: 'bg-destructive/10 text-destructive border-destructive/30',
  other: 'bg-muted border-border/60 text-muted-foreground',
}

const CATEGORY_LABELS: Record<string, string> = {
  hydration: 'Hydration', 'null-access': 'Null Access', 'type-error': 'Type Error',
  'chunk-load': 'Chunk Load', network: 'Network', auth: 'Auth',
  'not-found': 'Not Found', 'server-error': 'Server Error', cors: 'CORS',
  build: 'Build', 'infinite-loop': 'Infinite Loop', boundary: 'Error Boundary', other: 'Other',
}

type Tab = 'all' | 'grouped' | 'trends'

function Sparkline({ data }: { data: TrendBucket[] }) {
  const max = Math.max(...data.map(d => d.count), 1)
  const w = 200
  const h = 40
  const barW = Math.max(Math.floor(w / data.length) - 1, 2)

  return (
    <svg width={w} height={h} className="inline-block" role="img" aria-label="Error trend chart">
      {data.map((d, i) => {
        const barH = max > 0 ? (d.count / max) * (h - 4) : 0
        return (
          <rect
            key={i}
            x={i * (barW + 1)}
            y={h - barH - 2}
            width={barW}
            height={barH}
            className={d.count > 0 ? 'fill-destructive/60' : 'fill-muted/30'}
            rx={1}
          />
        )
      })}
    </svg>
  )
}

export default function ErrorsPage() {
  const [errors, setErrors] = useState<ErrorEntry[]>([])
  const [total, setTotal] = useState(0)
  const [unreadCount, setUnreadCount] = useState(0)
  const [loading, setLoading] = useState(true)
  const [refreshing, setRefreshing] = useState(false)
  const [search, setSearch] = useState('')
  const [activeCategory, setActiveCategory] = useState('')
  const [selectedError, setSelectedError] = useState<ErrorEntry | null>(null)
  const [clearing, setClearing] = useState(false)
  const [tab, setTab] = useState<Tab>('all')

  // Grouped & trends
  const [groups, setGroups] = useState<ErrorGroup[]>([])
  const [trends, setTrends] = useState<TrendBucket[]>([])

  const fetchErrors = useCallback(async (showRefreshing = false) => {
    if (showRefreshing) setRefreshing(true)
    try {
      const [recentData, groupedData, trendsData] = await Promise.all([
        apiGet<ErrorSummary>('/errors/recent?limit=200'),
        apiGet<{ groups: ErrorGroup[] }>('/errors/grouped'),
        apiGet<{ trends: TrendBucket[] }>('/errors/trends?hours=24'),
      ])
      setErrors(recentData.errors || [])
      setTotal(recentData.total || 0)
      setUnreadCount(recentData.unread_count || 0)
      setGroups(groupedData.groups || [])
      setTrends(trendsData.trends || [])
    } catch {
      setErrors([])
    }
    setLoading(false)
    setRefreshing(false)
  }, [])

  useEffect(() => { fetchErrors() }, [fetchErrors])

  useEffect(() => {
    const id = setInterval(() => {
      if (!document.hidden) fetchErrors()
    }, 10000)
    return () => clearInterval(id)
  }, [fetchErrors])

  const categories = useMemo(() => {
    const map = new Map<string, number>()
    for (const e of errors) map.set(classifyError(e), (map.get(classifyError(e)) || 0) + 1)
    return [...map.entries()].map(([category, count]) => ({ category, count })).sort((a, b) => b.count - a.count)
  }, [errors])

  const filtered = useMemo(() => {
    let list = errors
    if (activeCategory) list = list.filter(e => classifyError(e) === activeCategory)
    if (search) {
      const q = search.toLowerCase()
      list = list.filter(e => e.message?.toLowerCase().includes(q) || e.source?.toLowerCase().includes(q) || e.url?.toLowerCase().includes(q))
    }
    return list
  }, [errors, activeCategory, search])

  const handleClear = async () => {
    setClearing(true)
    try {
      await apiDelete('/errors/clear')
      setErrors([]); setTotal(0); setUnreadCount(0); setSelectedError(null); setGroups([])
    } catch { /* ignore */ }
    setClearing(false)
  }

  const handleExport = () => {
    window.open('/errors/export', '_blank')
  }

  const timeSince = (ts: string) => {
    const diff = Date.now() - new Date(ts).getTime()
    if (diff < 60_000) return 'just now'
    if (diff < 3_600_000) return `${Math.floor(diff / 60_000)}m ago`
    if (diff < 86_400_000) return `${Math.floor(diff / 3_600_000)}h ago`
    return `${Math.floor(diff / 86_400_000)}d ago`
  }

  const totalHourly = useMemo(() => {
    return trends.reduce((s, t) => s + t.count, 0)
  }, [trends])

  return (
    <div className="sl-page mx-auto max-w-4xl">
      <AppRouteHeader
        left={<AppRouteHeaderLead title="Error Monitor" />}
        right={
          <div className="flex items-center gap-2">
            <span className="text-[11px] text-muted-foreground hidden sm:inline">
              {total} total · {unreadCount} new
            </span>
            <Button variant="outline" size="sm" onClick={handleExport} disabled={errors.length === 0}>
              <IconDownload className="w-3.5 h-3.5 mr-1" /> Export
            </Button>
            <Button variant="outline" size="sm" onClick={() => fetchErrors(true)} disabled={refreshing}>
              <IconRefresh className="w-3.5 h-3.5 mr-1" />
              {refreshing ? 'Refreshing...' : 'Refresh'}
            </Button>
          </div>
        }
      />
      <div className="space-y-4">
        {/* Stats */}
        <KpiGrid columns={4}>
          <StatCard label="Total errors" value={total.toString()} icon={<span className="inline-block w-2 h-2 rounded-full bg-destructive" />} />
          <StatCard label="Unread" value={unreadCount.toString()} icon={<span className={`inline-block w-2 h-2 rounded-full ${unreadCount > 0 ? 'bg-warning' : 'bg-success'}`} />} />
          <StatCard label="Unique groups" value={groups.length.toString()} />
          <StatCard label="Last 24h" value={totalHourly.toString()} />
        </KpiGrid>

        {/* Tabs */}
        <div className="flex gap-1 rounded-lg bg-muted/50 p-1">
          {([['all', 'All Errors'], ['grouped', 'Grouped'], ['trends', 'Trends']] as const).map(([key, label]) => (
            <button
              key={key}
              className={`flex-1 rounded-md px-3 py-1.5 text-xs font-medium transition-colors ${tab === key ? 'bg-background text-foreground shadow-sm' : 'text-muted-foreground hover:text-foreground'}`}
              onClick={() => setTab(key)}
            >
              {label}
            </button>
          ))}
        </div>

        {/* Category chips */}
        {categories.length > 0 && (
          <div className="flex flex-wrap gap-2">
            <Chip label={`All (${errors.length})`} selected={!activeCategory} onClick={() => setActiveCategory('')} />
            {categories.map(c => (
              <Chip
                key={c.category}
                label={`${CATEGORY_LABELS[c.category] || c.category} (${c.count})`}
                selected={activeCategory === c.category}
                onClick={() => setActiveCategory(activeCategory === c.category ? '' : c.category)}
              />
            ))}
          </div>
        )}

        {/* Tab: All Errors */}
        {tab === 'all' && (
          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle className="text-base">
                  Errors {activeCategory ? `· ${CATEGORY_LABELS[activeCategory]}` : ''}
                </CardTitle>
                {errors.length > 0 && (
                  <Button variant="destructive" size="sm" onClick={handleClear} disabled={clearing}>
                    {clearing ? 'Clearing...' : 'Clear all'}
                  </Button>
                )}
              </div>
              <div className="mt-2">
                <SearchInput value={search} onChange={setSearch} placeholder="Search errors..." className="w-full" />
              </div>
            </CardHeader>
            <CardContent>
              {loading ? (
                <div className="space-y-2">
                  {[1, 2, 3].map(i => <div key={i} className="h-16 animate-pulse bg-muted rounded-md" />)}
                </div>
              ) : filtered.length === 0 ? (
                <EmptyCard message={errors.length === 0 ? 'No errors logged yet' : 'No errors match your search'} />
              ) : (
                <div className="space-y-2 max-h-[600px] overflow-y-auto">
                  {filtered.map(entry => {
                    const cat = classifyError(entry)
                    const colorClass = CATEGORY_COLORS[cat] || CATEGORY_COLORS.other
                    const isSelected = selectedError?.id === entry.id
                    return (
                      <div key={entry.id}>
                        <button
                          className={`w-full text-left rounded-lg border p-3 transition-all duration-150 hover:bg-muted/40 ${isSelected ? 'ring-1 ring-primary/40' : ''} ${colorClass}`}
                          onClick={() => setSelectedError(isSelected ? null : entry)}
                          aria-expanded={isSelected}
                        >
                          <div className="flex items-start gap-2">
                            <div className="flex-1 min-w-0">
                              <div className="flex items-center gap-2 mb-1">
                                <Badge variant="default" className={`text-[10px] px-1.5 py-0 font-medium ${colorClass}`} label={CATEGORY_LABELS[cat] || cat} />
                                <span className="text-[10px] text-muted-foreground/70 font-mono">{timeSince(entry.timestamp)}</span>
                                {entry.source && <span className="text-[10px] text-muted-foreground/60 truncate">{entry.source}</span>}
                              </div>
                              <p className="text-sm font-medium leading-snug truncate">{entry.message}</p>
                              {entry.url && <p className="text-[11px] text-muted-foreground/60 mt-1 truncate font-mono">{entry.url}{entry.line ? `:${entry.line}` : ''}</p>}
                            </div>
                          </div>
                        </button>
                        {isSelected && (
                          <div className="mt-1 rounded-lg border border-border/60 bg-muted/30 p-3 space-y-2 animate-in fade-in slide-in-from-top-1">
                            {entry.stack && (
                              <div>
                                <p className="text-[10px] font-medium text-muted-foreground mb-1">Stack trace</p>
                                <pre className="text-[11px] font-mono text-muted-foreground whitespace-pre-wrap break-all max-h-40 overflow-y-auto rounded bg-background/50 p-2">{entry.stack}</pre>
                              </div>
                            )}
                            <div className="flex flex-wrap gap-x-4 gap-y-1 text-[11px] text-muted-foreground">
                              <span>ID: <span className="font-mono">{entry.id}</span></span>
                              {entry.line && <span>Line: <span className="font-mono">{entry.line}</span></span>}
                              {entry.col && <span>Col: <span className="font-mono">{entry.col}</span></span>}
                              {entry.client_host && <span>Host: <span className="font-mono">{entry.client_host}</span></span>}
                              <span>Time: <span className="font-mono">{new Date(entry.timestamp).toLocaleString()}</span></span>
                            </div>
                          </div>
                        )}
                      </div>
                    )
                  })}
                </div>
              )}
            </CardContent>
          </Card>
        )}

        {/* Tab: Grouped */}
        {tab === 'grouped' && (
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Grouped by Message ({groups.length} unique)</CardTitle>
            </CardHeader>
            <CardContent>
              {groups.length === 0 ? (
                <EmptyCard message="No errors to group" />
              ) : (
                <div className="space-y-2 max-h-[600px] overflow-y-auto">
                  {groups.map(g => {
                    const cat = classifyError({ message: g.message, source: g.source } as ErrorEntry)
                    const colorClass = CATEGORY_COLORS[cat] || CATEGORY_COLORS.other
                    return (
                      <div key={g.fingerprint} className={`rounded-lg border p-3 ${colorClass}`}>
                        <div className="flex items-start justify-between gap-2">
                          <div className="flex-1 min-w-0">
                            <div className="flex items-center gap-2 mb-1">
                              <Badge variant="default" className={`text-[10px] px-1.5 py-0 font-medium ${colorClass}`} label={CATEGORY_LABELS[cat] || cat} />
                              <span className="text-[10px] font-mono font-bold">{g.count}x</span>
                              <span className="text-[10px] text-muted-foreground/70">{timeSince(g.latest)}</span>
                            </div>
                            <p className="text-sm font-medium leading-snug">{g.message}</p>
                            {g.sample_url && <p className="text-[11px] text-muted-foreground/60 mt-1 font-mono">{g.sample_url}{g.sample_line ? `:${g.sample_line}` : ''}</p>}
                          </div>
                        </div>
                      </div>
                    )
                  })}
                </div>
              )}
            </CardContent>
          </Card>
        )}

        {/* Tab: Trends */}
        {tab === 'trends' && (
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Error Trends (last 24h)</CardTitle>
            </CardHeader>
            <CardContent>
              {trends.every(t => t.count === 0) ? (
                <EmptyCard message="No errors in the last 24 hours" />
              ) : (
                <div className="space-y-3">
                  <Sparkline data={trends} />
                  <div className="grid grid-cols-6 gap-2 text-[11px]">
                    {trends.filter((_, i) => i % 4 === 0).map((t, i) => (
                      <div key={i} className="text-center">
                        <div className="font-mono text-muted-foreground">{t.hour.split('T')[1]?.slice(0, 2) || '?'}:00</div>
                        <div className={`font-bold ${t.count > 0 ? 'text-destructive' : 'text-muted-foreground/50'}`}>{t.count}</div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  )
}
