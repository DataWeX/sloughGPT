'use client'

import { useState, useEffect } from 'react'
import { Card, CardHeader, CardTitle, CardContent, Button, Input } from '@sloughgpt/strui'
import { IconRefresh } from '@sloughgpt/strui'
import { AppRouteHeader, AppRouteHeaderLead } from '@/components/AppRouteHeader'
import { apiGet } from '@/lib/http-client'
import { useToastStore } from '@/lib/toast-store'

interface AuditLog {
  event_type: string
  timestamp: string
  details?: Record<string, unknown>
  ip?: string
  user?: string
}

export default function SecurityPage() {
  const [logs, setLogs] = useState<AuditLog[]>([])
  const [keyInfo, setKeyInfo] = useState<{ count: number; configured: boolean } | null>(null)
  const [loading, setLoading] = useState(true)
  const [filter, setFilter] = useState('')
  const addToast = useToastStore(s => s.addToast)

  const fetchData = async () => {
    try {
      const [logsRes, keysRes] = await Promise.all([
        apiGet<{ data?: { logs?: AuditLog[] } }>('/security/audit?limit=100').catch(() => null),
        apiGet<{ data?: { count: number; configured: boolean } } | { count: number; configured: boolean }>('/security/keys').catch(() => null),
      ])
      setLogs(logsRes?.data?.logs ?? [])
      const keysData = keysRes && 'data' in keysRes && keysRes.data ? keysRes.data : keysRes
      setKeyInfo(keysData as { count: number; configured: boolean } | null)
    } catch {
      addToast('Failed to load security data', 'error')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { fetchData() }, [])

  const filteredLogs = filter.trim()
    ? logs.filter(l => l.event_type?.toLowerCase().includes(filter.toLowerCase()))
    : logs

  if (loading) {
    return (
      <div className="sl-page mx-auto max-w-4xl">
        <AppRouteHeader left={<AppRouteHeaderLead title="Security" subtitle="Audit logs & API keys" />} />
        <div className="space-y-4">
          <Card><CardContent><div className="h-32 animate-pulse bg-muted/50 rounded" /></CardContent></Card>
        </div>
      </div>
    )
  }

  return (
    <div className="sl-page mx-auto max-w-4xl">
      <AppRouteHeader left={<AppRouteHeaderLead title="Security" subtitle="Audit logs & API keys" />} />
      <div className="space-y-4">
        {keyInfo && (
          <div className="grid grid-cols-2 gap-3">
            <div className="rounded-md bg-muted/30 p-3 text-center">
              <div className="text-xs text-muted-foreground">API Keys</div>
              <div className={`text-lg font-mono font-medium ${keyInfo.configured ? 'text-success' : 'text-warning'}`}>
                {keyInfo.configured ? `${keyInfo.count} configured` : 'None configured'}
              </div>
            </div>
            <div className="rounded-md bg-muted/30 p-3 text-center">
              <div className="text-xs text-muted-foreground">Audit Logs</div>
              <div className="text-lg font-mono font-medium">{logs.length}</div>
            </div>
          </div>
        )}

        <Card>
          <CardHeader className="flex flex-row items-center justify-between">
            <CardTitle className="text-base">Audit Logs</CardTitle>
            <Button size="sm" variant="ghost" onClick={fetchData}>
              <IconRefresh className="h-3.5 w-3.5" />
            </Button>
          </CardHeader>
          <CardContent className="space-y-3">
            <Input
              value={filter}
              onChange={e => setFilter(e.target.value)}
              placeholder="Filter by event type..."
            />
            {filteredLogs.length === 0 ? (
              <p className="text-sm text-muted-foreground">No audit logs found.</p>
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
                    {log.ip && <div className="text-xs text-muted-foreground mt-0.5">IP: {log.ip}</div>}
                    {log.user && <div className="text-xs text-muted-foreground mt-0.5">User: {log.user}</div>}
                    {log.details && Object.keys(log.details).length > 0 && (
                      <div className="text-xs text-muted-foreground mt-0.5 font-mono">
                        {JSON.stringify(log.details).slice(0, 120)}
                      </div>
                    )}
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
