'use client'

import { useState, useEffect, useRef } from 'react'
import { Card, CardHeader, CardTitle, CardContent, Button, Input } from '@sloughgpt/strui'
import { IconRefresh, IconTrash, IconDownload } from '@sloughgpt/strui'
import { AppRouteHeader, AppRouteHeaderLead } from '@/components/AppRouteHeader'
import { downloadJson } from '@/lib/download-utils'
import { errorsController } from '@/lib/errors-controller'
import { useToastStore } from '@/lib/toast-store'

export default function ErrorsPage() {
  const [grouped, setGrouped] = useState<Awaited<ReturnType<typeof errorsController.getGrouped>>>([])
  const [recent, setRecent] = useState<Awaited<ReturnType<typeof errorsController.getRecent>>['errors']>([])
  const [trends, setTrends] = useState<Awaited<ReturnType<typeof errorsController.getTrends>>>([])
  const [total, setTotal] = useState(0)
  const [loading, setLoading] = useState(true)
  const [clearing, setClearing] = useState(false)
  const [autoRefresh, setAutoRefresh] = useState(false)
  const [search, setSearch] = useState('')
  const intervalRef = useRef<NodeJS.Timeout | null>(null)
  const addToast = useToastStore(s => s.addToast)

  const fetchData = async () => {
    setLoading(true)
    try {
      const [g, r, t] = await Promise.all([
        errorsController.getGrouped(),
        errorsController.getRecent(30),
        errorsController.getTrends(24),
      ])
      setGrouped(g)
      setRecent(r.errors)
      setTotal(r.total)
      setTrends(t)
    } catch {
      addToast('Failed to load error data', 'error')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { fetchData() }, [])

  useEffect(() => {
    if (autoRefresh) {
      intervalRef.current = setInterval(fetchData, 10000)
      const onVis = () => { if (!document.hidden && intervalRef.current) { clearInterval(intervalRef.current); intervalRef.current = setInterval(fetchData, 10000) } }
      document.addEventListener('visibilitychange', onVis)
      return () => { clearInterval(intervalRef.current!); document.removeEventListener('visibilitychange', onVis) }
    } else if (intervalRef.current) {
      clearInterval(intervalRef.current)
    }
  }, [autoRefresh])

  const handleClear = async () => {
    setClearing(true)
    try {
      await errorsController.clear()
      await fetchData()
    } catch {
      addToast('Failed to clear errors', 'error')
    } finally {
      setClearing(false)
    }
  }

  const handleExport = async () => {
    try {
      const data = await errorsController.export()
      downloadJson(data, `errors-${Date.now()}.json`)
    } catch {
      addToast('Failed to export errors', 'error')
    }
  }

  const maxTrend = Math.max(...trends.map(t => t.count), 1)

  if (loading) {
    return (
      <div className="sl-page mx-auto max-w-4xl">
        <AppRouteHeader left={<AppRouteHeaderLead title="Errors" subtitle="Client-side error monitoring" />} />
        <div className="space-y-4">
          <Card><CardContent><div className="h-32 animate-pulse bg-muted/50 rounded" /></CardContent></Card>
        </div>
      </div>
    )
  }

  return (
    <div className="sl-page mx-auto max-w-4xl">
      <AppRouteHeader left={<AppRouteHeaderLead title="Errors" subtitle={`${total} total errors`} />} />
      <div className="space-y-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between">
            <CardTitle className="text-base">Actions</CardTitle>
            <div className="flex items-center gap-2">
              <Button size="sm" variant={autoRefresh ? 'default' : 'ghost'} onClick={() => setAutoRefresh(!autoRefresh)}>
                {autoRefresh ? 'Auto-refresh ON' : 'Auto-refresh'}
              </Button>
              <Button size="sm" variant="ghost" onClick={fetchData}>
                <IconRefresh className="h-3.5 w-3.5" />
              </Button>
            </div>
          </CardHeader>
          <CardContent>
            <div className="flex gap-2">
              <Button size="sm" variant="outline" onClick={handleExport}>
                <span className="inline-flex items-center gap-1.5">
                  <IconDownload className="h-3.5 w-3.5" />
                  Export JSON
                </span>
              </Button>
              <Button size="sm" variant="outline" onClick={handleClear} disabled={clearing} className="text-destructive">
                <span className="inline-flex items-center gap-1.5">
                  <IconTrash className="h-3.5 w-3.5" />
                  {clearing ? 'Clearing...' : 'Clear All'}
                </span>
              </Button>
            </div>
          </CardContent>
        </Card>

        {trends.length > 0 && (
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Hourly Trend (24h)</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex items-end gap-1 h-24">
                {trends.map((t, i) => (
                  <div
                    key={i}
                    className="flex-1 bg-primary/20 rounded-t min-w-[4px]"
                    style={{ height: `${Math.max((t.count / maxTrend) * 100, 2)}%` }}
                    title={`${t.hour.split('T')[1]}: ${t.count}`}
                  />
                ))}
              </div>
              <div className="flex justify-between text-[10px] text-muted-foreground mt-1">
                <span>{trends[0]?.hour.split('T')[1]}</span>
                <span>{trends[trends.length - 1]?.hour.split('T')[1]}</span>
              </div>
            </CardContent>
          </Card>
        )}

        <Card>
          <CardHeader className="flex flex-row items-center justify-between">
            <CardTitle className="text-base">Grouped Errors ({grouped.length})</CardTitle>
            <Input
              value={search}
              onChange={e => setSearch(e.target.value)}
              placeholder="Search errors..."
              className="h-7 w-48 text-xs"
            />
          </CardHeader>
          <CardContent>
            {grouped.length === 0 ? (
              <p className="text-sm text-muted-foreground">No errors logged.</p>
            ) : (
              <div className="space-y-2">
                {grouped
                  .filter(g => !search || g.message.toLowerCase().includes(search.toLowerCase()) || g.source.toLowerCase().includes(search.toLowerCase()))
                  .map(g => (
                  <div key={g.fingerprint} className="rounded-md border border-border/60 px-3 py-2 text-sm hover:bg-muted/50 transition-colors">
                    <div className="flex items-start justify-between gap-2">
                      <div className="min-w-0">
                        <div className="font-medium truncate">{g.message}</div>
                        <div className="text-xs text-muted-foreground">
                          {g.source} · {g.sample_url && <span className="truncate max-w-[200px] inline-block">{g.sample_url}</span>}
                          {g.sample_line != null && `:${g.sample_line}`}
                        </div>
                      </div>
                      <div className="text-right shrink-0">
                        <span className="text-xs font-mono bg-destructive/10 text-destructive px-1.5 py-0.5 rounded">
                          ×{g.count}
                        </span>
                        <div className="text-[10px] text-muted-foreground mt-0.5">
                          {g.latest && new Date(g.latest).toLocaleDateString()}
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-base">Recent Errors</CardTitle>
          </CardHeader>
          <CardContent>
            {recent.length === 0 ? (
              <p className="text-sm text-muted-foreground">No recent errors.</p>
            ) : (
              <div className="space-y-1 max-h-96 overflow-y-auto">
                {recent.map(e => (
                  <div key={e.id} className="flex items-start gap-2 text-xs py-1.5 border-b border-border/30 last:border-0">
                    <span className="font-mono text-muted-foreground shrink-0 w-16">
                      {e.timestamp ? new Date(e.timestamp).toLocaleTimeString() : '—'}
                    </span>
                    <span className="truncate">{e.message}</span>
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
