'use client'

import { useState, useEffect, useCallback } from 'react'
import { Card, CardHeader, CardTitle, CardContent, Button, Input, StatCard, KpiGrid, Skeleton } from '@sloughgpt/strui'
import { IconRefresh } from '@/components/icons/NavIcons'
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
  const { currentWorkspace } = useAuthStore()

  const fetchActivities = useCallback(async () => {
    if (!currentWorkspace?.id) {
      setLoading(false)
      return
    }
    try {
      const res = await apiGet<{ data: { activities: Activity[] } }>(
        `/workspaces/${currentWorkspace.id}/activity`
      )
      setActivities(res?.data?.activities ?? [])
    } catch {
      logger.warning('Could not fetch workspace activity')
    } finally {
      setLoading(false)
    }
  }, [currentWorkspace?.id])

  useEffect(() => { fetchActivities() }, [fetchActivities])

  const filteredActivities = activities.filter(a => {
    if (!filter) return true
    const q = filter.toLowerCase()
    return (
      a.action.toLowerCase().includes(q) ||
      a.detail.toLowerCase().includes(q) ||
      a.type.toLowerCase().includes(q) ||
      a.user.toLowerCase().includes(q)
    )
  })

  const formatTime = (ts: string) => {
    if (!ts) return ''
    try {
      const d = new Date(ts)
      return d.toLocaleString()
    } catch {
      return ts
    }
  }

  const typeCounts = activities.reduce((acc, a) => {
    acc[a.type] = (acc[a.type] || 0) + 1
    return acc
  }, {} as Record<string, number>)

  if (loading) {
    return (
      <PageContainer title="Audit Trail" subtitle="Workspace change history" loadingCards={2}>
        <KpiGrid>
          <StatCard label="Loading" value={<Skeleton className="h-3.5 w-10" />} />
          <StatCard label="Loading" value={<Skeleton className="h-3.5 w-10" />} />
        </KpiGrid>
        <Card><CardContent><div className="h-48 animate-pulse bg-muted/50 rounded-lg" /></CardContent></Card>
      </PageContainer>
    )
  }

  return (
    <div className="sl-page mx-auto max-w-4xl">
      <AppRouteHeader
        left={<AppRouteHeaderLead title="Audit Trail" subtitle="Workspace change history" />}
      />

      <div className="space-y-4">
        <KpiGrid>
          <StatCard label="Total Events" value={activities.length} />
          <StatCard label="Training" value={typeCounts['training'] ?? 0} />
          <StatCard label="Audit" value={typeCounts['audit'] ?? 0} />
        </KpiGrid>

        {/* Filter */}
        <Card>
          <CardContent className="py-2">
            <div className="flex items-center gap-2">
              <Input
                value={filter}
                onChange={e => setFilter(e.target.value)}
                placeholder="Filter by action, type, user..."
                className="flex-1 h-6 text-[10px]"
              />
              <Button size="sm" variant="outline" className="h-6 text-[10px]" onClick={fetchActivities}>
                <IconRefresh className="h-3 w-3 mr-1" />
                Refresh
              </Button>
            </div>
          </CardContent>
        </Card>

        {/* Activity feed */}
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-xs">Events</CardTitle>
          </CardHeader>
          <CardContent className="space-y-1">
            {filteredActivities.length === 0 ? (
              <p className="text-xs text-muted-foreground py-4 text-center">
                {filter ? 'No events match filter' : 'No events recorded'}
              </p>
            ) : (
              filteredActivities.map((a, i) => (
                <div key={i} className="flex items-center justify-between px-3 py-2 rounded-md text-[10px] hover:bg-muted/50 border border-transparent hover:border-border/50">
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center gap-2">
                      <span className={`px-1.5 py-0.5 rounded text-[9px] font-medium ${
                        a.type === 'training' ? 'bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400' :
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
      </div>
    </div>
  )
}
