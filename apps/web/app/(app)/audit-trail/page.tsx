'use client'

import { useState, useEffect, useCallback, useMemo } from 'react'
import { Card, CardHeader, CardTitle, CardContent, Button, Input, StatCard, KpiGrid, Skeleton } from '@sloughgpt/strui'
import { IconRefresh, IconDownload } from '@/components/icons/NavIcons'
import { PageContainer } from '@/components/PageContainer'
import { AppRouteHeader, AppRouteHeaderLead } from '@/components/AppRouteHeader'
import { apiGet } from '@/lib/http-client'
import { useAuthStore } from '@/lib/auth'
import { logger } from '@/lib/dev-log'

interface Activity {
  type: string
  action: string
  detail: string
  status: string
  timestamp: string
  user: string
}

export default function AuditTrailPage() {
  const [activities, setActivities] = useState<Activity[]>([])
  const [loading, setLoading] = useState(true)
  const [filter, setFilter] = useState('')
  const [dateFrom, setDateFrom] = useState('')
  const [dateTo, setDateTo] = useState('')
  const [typeFilter, setTypeFilter] = useState('all')
  const { currentWorkspace } = useAuthStore()

  const fetchActivities = useCallback(async () => {
    if (!currentWorkspace?.id) {
      setLoading(false)
      return
    }
    try {
      const params = new URLSearchParams()
      if (dateFrom) params.set('from_date', dateFrom)
      if (dateTo) params.set('to_date', dateTo)
      if (typeFilter !== 'all') params.set('type', typeFilter)
      const qs = params.toString()
      const res = await apiGet<{ data: { activities: Activity[] } }>(
        `/workspaces/${currentWorkspace.id}/activity${qs ? `?${qs}` : ''}`
      )
      setActivities(res?.data?.activities ?? [])
    } catch {
      logger.warning('Could not fetch workspace activity')
    } finally {
      setLoading(false)
    }
  }, [currentWorkspace?.id, dateFrom, dateTo, typeFilter])

  useEffect(() => { fetchActivities() }, [fetchActivities])

  const filteredActivities = useMemo(() => {
    return activities.filter(a => {
      // Text filter (client-side)
      if (filter) {
        const q = filter.toLowerCase()
        return (
          a.action.toLowerCase().includes(q) ||
          a.detail.toLowerCase().includes(q) ||
          a.type.toLowerCase().includes(q) ||
          a.user.toLowerCase().includes(q)
        )
      }
      return true
    })
  }, [activities, filter])

  const exportCsv = () => {
    const headers = ['timestamp', 'type', 'action', 'detail', 'status', 'user']
    const rows = filteredActivities.map(a => [
      a.timestamp,
      a.type,
      a.action,
      `"${(a.detail || '').replace(/"/g, '""')}"`,
      a.status,
      a.user,
    ])
    const csv = [headers.join(','), ...rows.map(r => r.join(','))].join('\n')
    const blob = new Blob([csv], { type: 'text/csv' })
    const url = URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = `audit-trail-${currentWorkspace?.id || 'all'}-${new Date().toISOString().slice(0, 10)}.csv`
    link.click()
    URL.revokeObjectURL(url)
  }

  const formatTime = (ts: string) => {
    if (!ts) return ''
    try { return new Date(ts).toLocaleString() } catch { return ts }
  }

  const typeCounts = useMemo(() => {
    const counts: Record<string, number> = {}
    for (const a of activities) {
      counts[a.type] = (counts[a.type] || 0) + 1
    }
    return counts
  }, [activities])

  const uniqueTypes = useMemo(() => {
    const types = new Set(activities.map(a => a.type))
    return Array.from(types).sort()
  }, [activities])

  if (loading) {
    return (
      <PageContainer title="Audit Trail">
        <Skeleton className="h-8 w-64 mb-4" />
        <Skeleton className="h-64 w-full" />
      </PageContainer>
    )
  }

  return (
    <PageContainer title="Audit Trail">
      <AppRouteHeader left={<AppRouteHeaderLead title="Audit Trail" />} />

      <KpiGrid className="mb-6">
        <StatCard label="Total Events" value={activities.length} />
        <StatCard label="Filtered" value={filteredActivities.length} />
        <StatCard label="Training" value={typeCounts['training'] ?? 0} />
        <StatCard label="Audit" value={typeCounts['audit'] ?? 0} />
      </KpiGrid>

      {/* Filters */}
      <Card className="mb-4">
        <CardContent className="py-3 space-y-2">
          <div className="flex items-center gap-2">
            <Input
              value={filter}
              onChange={e => setFilter(e.target.value)}
              placeholder="Filter by action, type, user..."
              className="flex-1 h-6 text-[10px]"
            />
            <select
              value={typeFilter}
              onChange={e => setTypeFilter(e.target.value)}
              className="h-6 text-[10px] rounded-md border border-border bg-background px-2"
            >
              <option value="all">All types</option>
              {uniqueTypes.map(t => (
                <option key={t} value={t}>{t}</option>
              ))}
            </select>
          </div>
          <div className="flex items-center gap-2">
            <div className="flex items-center gap-1">
              <label className="text-[10px] text-muted-foreground">From:</label>
              <Input
                type="date"
                value={dateFrom}
                onChange={e => setDateFrom(e.target.value)}
                className="h-6 text-[10px] w-32"
              />
            </div>
            <div className="flex items-center gap-1">
              <label className="text-[10px] text-muted-foreground">To:</label>
              <Input
                type="date"
                value={dateTo}
                onChange={e => setDateTo(e.target.value)}
                className="h-6 text-[10px] w-32"
              />
            </div>
            {(dateFrom || dateTo || typeFilter !== 'all') && (
              <Button
                size="sm"
                variant="ghost"
                className="h-6 text-[10px]"
                onClick={() => { setDateFrom(''); setDateTo(''); setTypeFilter('all') }}
              >
                Clear filters
              </Button>
            )}
            <div className="flex-1" />
            <Button size="sm" variant="outline" className="h-6 text-[10px]" onClick={fetchActivities}>
              <IconRefresh className="h-3 w-3 mr-1" />
              Refresh
            </Button>
            <Button size="sm" variant="outline" className="h-6 text-[10px]" onClick={exportCsv} disabled={filteredActivities.length === 0}>
              <IconDownload className="h-3 w-3 mr-1" />
              Export CSV
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Activity feed */}
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-xs">Events ({filteredActivities.length})</CardTitle>
        </CardHeader>
        <CardContent className="space-y-1">
          {filteredActivities.length === 0 ? (
            <p className="text-xs text-muted-foreground py-4 text-center">
              {filter || dateFrom || dateTo || typeFilter !== 'all' ? 'No events match filters' : 'No events recorded'}
            </p>
          ) : (
            filteredActivities.map((a, i) => (
              <div key={i} className="flex items-center justify-between px-3 py-2 rounded-md text-[10px] hover:bg-muted/50 border border-transparent hover:border-border/50">
                <div className="min-w-0 flex-1">
                  <div className="flex items-center gap-2">
                    <span className={`px-1.5 py-0.5 rounded text-[9px] font-medium ${
                      a.type === 'training' ? 'bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400' :
                      a.type === 'audit' ? 'bg-purple-100 text-purple-700 dark:bg-purple-900/30 dark:text-purple-400' :
                      'bg-muted text-muted-foreground'
                    }`}>
                      {a.type}
                    </span>
                    <span className="font-medium">{a.action}</span>
                  </div>
                  {a.detail && <div className="text-muted-foreground mt-0.5">{a.detail}</div>}
                </div>
                <div className="flex items-center gap-3 shrink-0">
                  {a.user && <span className="text-muted-foreground">{a.user}</span>}
                  {a.status && (
                    <span className={`px-1.5 py-0.5 rounded-full text-[9px] ${
                      a.status === 'completed' || a.status === 'success' ? 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400' :
                      a.status === 'failed' || a.status === 'failure' ? 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400' :
                      a.status === 'running' ? 'bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400' :
                      'bg-muted text-muted-foreground'
                    }`}>
                      {a.status}
                    </span>
                  )}
                  <span className="text-muted-foreground whitespace-nowrap">{formatTime(a.timestamp)}</span>
                </div>
              </div>
            ))
          )}
        </CardContent>
      </Card>
    </PageContainer>
  )
}
