'use client'

import { useState, useEffect, useCallback, useRef } from 'react'
import { Card, CardHeader, CardTitle, CardContent, Button, Input, StatCard, KpiGrid, Skeleton } from '@sloughgpt/strui'
import { IconRefresh } from '@sloughgpt/strui'
import { PageContainer } from '@/components/PageContainer'
import { apiGet, apiPost, apiDelete } from '@/lib/http-client'
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

interface ApiKey {
  id: string
  name: string
  key_hash: string
  scopes: string[]
  created_at: number
  revoked: boolean
  expires_at?: number
}

interface KeysResponse {
  keys: ApiKey[]
  count: number
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
  const [keys, setKeys] = useState<ApiKey[]>([])
  const [loading, setLoading] = useState(true)
  const [loadingMore, setLoadingMore] = useState(false)
  const [historyMode, setHistoryMode] = useState(false)
  const [filter, setFilter] = useState('')
  const [newKeyName, setNewKeyName] = useState('')
  const [creating, setCreating] = useState(false)
  const [newKeyValue, setNewKeyValue] = useState<string | null>(null)
  const filterRef = useRef(filter)
  filterRef.current = filter
  const addToast = useToastStore(s => s.addToast)

  const eventParam = () => {
    const f = filterRef.current.trim()
    return f ? `&event_type=${encodeURIComponent(f)}` : ''
  }

  const fetchKeys = useCallback(async () => {
    try {
      const res = await apiGet<KeysResponse>('/security/keys')
      setKeys(res?.keys ?? [])
    } catch {
      logger.warning('Could not fetch API keys')
    }
  }, [])

  const fetchData = useCallback(async (useHistory = false) => {
    setLoading(true)
    try {
      const auditUrl = `${useHistory ? '/security/audit?history=true&limit=100' : '/security/audit?limit=100'}${eventParam()}`
      const [logsRes] = await Promise.all([
        apiGet<AuditResponse>(auditUrl).catch((e) => { logger.warning('Could not audit log fetch', e); return null }),
        fetchKeys(),
      ])
      setLogs(logsRes?.logs ?? [])
    } catch {
      addToast('Could not load security data', 'error')
    } finally {
      setLoading(false)
    }
  }, [fetchKeys])

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

  const createKey = async () => {
    if (!newKeyName.trim()) return
    setCreating(true)
    try {
      const res = await apiPost<{ key: string; id: string }>('/security/keys', { name: newKeyName, scopes: ['*'] })
      if (res?.key) {
        setNewKeyValue(res.key)
        setNewKeyName('')
        await fetchKeys()
        addToast('API key created — copy it now, it won\'t be shown again', 'success')
      }
    } catch {
      addToast('Could not create API key', 'error')
    } finally {
      setCreating(false)
    }
  }

  const revokeKey = async (id: string) => {
    try {
      await apiDelete(`/security/keys/${id}`)
      await fetchKeys()
      addToast('API key revoked', 'success')
    } catch {
      addToast('Could not revoke API key', 'error')
    }
  }

  const rotateKey = async (id: string) => {
    try {
      const res = await apiPost<{ key: string }>(`/security/keys/${id}/rotate`)
      if (res?.key) {
        setNewKeyValue(res.key)
        await fetchKeys()
        addToast('API key rotated — copy the new key', 'success')
      }
    } catch {
      addToast('Could not rotate API key', 'error')
    }
  }

  useEffect(() => { fetchData(false) }, [fetchData])

  const filteredLogs = filter.trim()
    ? logs.filter(l => l.event_type?.toLowerCase().includes(filter.toLowerCase()))
    : logs

  const activeKeys = keys.filter(k => !k.revoked)
  const revokedKeys = keys.filter(k => k.revoked)

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
        <StatCard label="API Keys" value={activeKeys.length > 0 ? `${activeKeys.length} active` : 'None'} />
        <StatCard label="Audit Logs" value={logs.length} />
        <StatCard label="History Mode" value={historyMode ? 'Persisted' : 'Session'} />
        <StatCard label="Filter" value={filter || 'All'} />
      </KpiGrid>

      <SecurityOverviewCard
        logs={logs}
        apiKeyConfigured={activeKeys.length > 0}
        apiKeyCount={activeKeys.length}
      />

      <Card>
        <CardHeader>
          <CardTitle className="text-base">API Keys</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex gap-2">
            <Input
              value={newKeyName}
              onChange={e => setNewKeyName(e.target.value)}
              placeholder="Key name (e.g. 'ci-pipeline')"
              className="flex-1"
            />
            <Button size="sm" onClick={createKey} disabled={creating || !newKeyName.trim()}>
              {creating ? 'Creating...' : 'Create Key'}
            </Button>
          </div>

          {newKeyValue && (
            <div className="rounded-md bg-success/10 border border-success/30 p-3">
              <div className="text-sm font-medium text-success mb-1">New API Key (copy now)</div>
              <code className="text-xs break-all">{newKeyValue}</code>
              <Button size="sm" variant="ghost" className="mt-2" onClick={() => setNewKeyValue(null)}>Dismiss</Button>
            </div>
          )}

          {activeKeys.length === 0 && revokedKeys.length === 0 ? (
            <div className="text-center py-4 text-sm text-muted-foreground">No API keys yet. Create one above.</div>
          ) : (
            <div className="space-y-2">
              {activeKeys.map(k => (
                <div key={k.id} className="flex items-center justify-between rounded-md border border-border/60 px-3 py-2">
                  <div>
                    <span className="font-medium text-sm">{k.name}</span>
                    <span className="text-xs text-muted-foreground ml-2">{k.key_hash}</span>
                    <span className="text-xs text-muted-foreground ml-2">scopes: {k.scopes.join(', ')}</span>
                  </div>
                  <div className="flex gap-1">
                    <Button size="sm" variant="ghost" onClick={() => rotateKey(k.id)}>Rotate</Button>
                    <Button size="sm" variant="ghost" className="text-destructive" onClick={() => revokeKey(k.id)}>Revoke</Button>
                  </div>
                </div>
              ))}
              {revokedKeys.length > 0 && (
                <div className="text-xs text-muted-foreground mt-2">{revokedKeys.length} revoked key(s) hidden</div>
              )}
            </div>
          )}
        </CardContent>
      </Card>

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
