'use client'

import { useState, useEffect, useCallback } from 'react'
import { Card, CardHeader, CardTitle, CardContent, Button, Input, StatCard, KpiGrid, Skeleton } from '@sloughgpt/strui'
import { IconRefresh } from '@sloughgpt/strui'
import { PageContainer } from '@/components/PageContainer'
import { apiGet } from '@/lib/http-client'
import { SecurityOverviewCard } from '@/components/security/SecurityOverviewCard'
import { useToastStore } from '@/lib/toast-store'
import { logger } from '@/lib/dev-log'

interface AuditLog {
  event_type: string
  timestamp: string
  user?: string
  resource?: string
  detail?: string
  extra?: Record<string, unknown>
}

interface AuditResponse {
  logs?: AuditLog[]
  count?: number
}

function mergeLogs(a: AuditLog[], b: AuditLog[]): AuditLog[] {
  const seen = new Set<string>()
  const out: AuditLog[] = []
  for (const l of [...a, ...b]) {
    const key = `${l.timestamp}|${l.event_type}`
    if (seen.has(key)) continue
    seen.add(key)
    out.push(l)
  }
  return out
}

export default function SecurityPage() {
  const [logs, setLogs] = useState<AuditLog[]>([])
  const [keyInfo, setKeyInfo] = useState<{ count: number; configured: boolean } | null>(null)
  const [loading, setLoading] = useState(true)
  const [loadingMore, setLoadingMore] = useState(false)
  const [historyMode, setHistoryMode] = useState(false)
  const [filter, setFilter] = useState('')
  const addToast = useToastStore(s => s.addToast)

  const eventParam = () => {
    const f = filter.trim()
    return f ? `&event_type=${encodeURIComponent(f)}` : ''
  }

  const fetchData = useCallback(async (useHistory = false) => {
    setLoading(true)
    try {
      const auditUrl = `${useHistory ? '/security/audit?history=true&limit=100' : '/security/audit?limit=100'}${eventParam()}`
      const [logsRes, keysRes] = await Promise.all([
        apiGet<AuditResponse>(auditUrl).catch((e) => { logger.warning('Could not audit log fetch', e); return null }),
        apiGet<{ count: number; configured: boolean }>('/security/keys').catch((e) => { logger.warning('Could not security keys fetch', e); return null }),
      ])
      setLogs(logsRes?.logs ?? [])
      const keysData = keysRes && 'count' in keysRes ? keysRes : null
      setKeyInfo(keysData)
    } catch {
      addToast('Could not load security data', 'error')
    } finally {
      setLoading(false)
    }
  }, [])

  const toggleHistory = () => {
    const next = !historyMode
    setHistoryMode(next)
    fetchData(next)
  }

  const loadOlder = async () => {
    if (logs.length === 0 || loadingMore) return
    setLoadingMore(true)
    try {
      const oldest = logs.reduce<string | null>(
        (min, l) => (l.timestamp && (!min || l.timestamp < min) ? l.timestamp : min),
        null,
      )
      if (!oldest) return
      const before = encodeURIComponent(oldest)
      const res = await apiGet<AuditResponse>(`/security/audit?history=true&limit=100&before=${before}${eventParam()}`)
      const older = res?.logs ?? []
      setLogs(prev => mergeLogs(prev, older))
    } catch {
      addToast('Could not load older audit logs', 'error')
    } finally {
      setLoadingMore(false)
    }
  }

  useEffect(() => { fetchData(false) }, [fetchData])

  const filteredLogs = filter.trim()
    ? logs.filter(l => l.event_type?.toLowerCase().includes(filter.toLowerCase()))
    : logs

  if (loading) {
    return (
      <PageContainer title="Security" subtitle="Audit logs & API keys" loadingCards={4}>
        <KpiGrid>
          <StatCard label="Loading" value={<Skeleton className="h-5 w-12" />} />
          <StatCard label="Loading" value={<Skeleton className="h-5 w-12" />} />
          <StatCard label="Loading" value={<Skeleton className="h-5 w-12" />} />
          <StatCard label="Loading" value={<Skeleton className="h-5 w-12" />} />
        </KpiGrid>
        <Card><CardContent><div className="h-32 animate-pulse bg-muted/50 rounded" /></CardContent></Card>
        <Card><CardContent><div className="h-64 animate-pulse bg-muted/50 rounded" /></CardContent></Card>
      </PageContainer>
    )
  }

  return (
    <PageContainer title="Security" subtitle="Audit logs & API keys">
      <KpiGrid>
        <StatCard label="API Keys" value={keyInfo?.configured ? `${keyInfo.count} configured` : 'None'} />
        <StatCard label="Audit Logs" value={logs.length} />
        <StatCard label="History Mode" value={historyMode ? 'Persisted' : 'Session'} />
        <StatCard label="Filter" value={filter || 'All'} />
      </KpiGrid>

      <SecurityOverviewCard
        logs={logs}
        apiKeyConfigured={keyInfo?.configured ?? false}
        apiKeyCount={keyInfo?.count ?? 0}
      />

      <Card>
        <CardHeader className="flex flex-row items-center justify-between">
          <CardTitle className="text-base">Audit Logs</CardTitle>
          <div className="flex items-center gap-1">
            <Button
              size="sm"
              variant={historyMode ? 'default' : 'ghost'}
              onClick={toggleHistory}
            >
              {historyMode ? 'Persisted' : 'Session'}
            </Button>
            <Button
              size="sm"
              variant="ghost"
              onClick={loadOlder}
              disabled={loadingMore || !historyMode}
            >
              {loadingMore ? (
                <span className="inline-block h-4 w-4 animate-spin rounded-full border-2 border-current border-t-transparent" />
              ) : (
                'Load older'
              )}
            </Button>
            <Button size="sm" variant="ghost" onClick={() => fetchData(historyMode)} aria-label="Refresh audit logs">
              <IconRefresh className="h-4 w-4" />
            </Button>
          </div>
        </CardHeader>
        <CardContent className="space-y-3">
          <Input
            value={filter}
            onChange={e => setFilter(e.target.value)}
            placeholder="Filter by event type..."
          />
          {filteredLogs.length === 0 ? (
            <div className="text-center py-6 text-sm text-muted-foreground space-y-1">
              <div>No audit logs found.</div>
              <div className="text-xs text-muted-foreground/70">Activities are logged automatically as you use the app.</div>
            </div>
          ) : (
            <div className="space-y-2 max-h-96 overflow-y-auto">
              {filteredLogs.map((log, i) => (
                <div key={i} className="rounded-md border border-border/60 px-3 py-2 text-sm">
                  <div className="flex items-center justify-between">
                    <span className="font-medium">{log.event_type}</span>
                    <span className="text-xs text-muted-foreground">
                      {log.timestamp ? new Date(log.timestamp).toLocaleString() : '—'}
                    </span>
                  </div>
                  {log.resource && <div className="text-xs text-muted-foreground mt-0.5">Resource: {log.resource}</div>}
                  {log.user && <div className="text-xs text-muted-foreground mt-0.5">User: {log.user}</div>}
                  {log.detail && <div className="text-xs text-muted-foreground mt-0.5">{log.detail}</div>}
                  {log.extra && Object.keys(log.extra).length > 0 && (
                    <div className="text-xs text-muted-foreground mt-0.5 font-mono">
                      {JSON.stringify(log.extra).slice(0, 120)}
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>
    </PageContainer>
  )
}
