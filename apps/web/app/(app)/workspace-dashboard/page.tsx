'use client'

import { useState, useEffect, useCallback } from 'react'
import { Card, CardHeader, CardTitle, CardContent, StatCard, KpiGrid, Skeleton } from '@sloughgpt/strui'
import { PageContainer } from '@/components/PageContainer'
import { AppRouteHeader, AppRouteHeaderLead } from '@/components/AppRouteHeader'
import { apiGet } from '@/lib/http-client'
import { useAuthStore } from '@/lib/auth'
import { logger } from '@/lib/dev-log'

interface WorkspaceStats {
  workspace_id: string
  name: string
  member_count: number
  dataset_count: number
  training_jobs: number
  active_training_jobs: number
  knowledge_items: number
}

interface Activity {
  type: string
  action: string
  detail: string
  status: string
  timestamp: string
  user: string
}

export default function WorkspaceDashboardPage() {
  const [stats, setStats] = useState<WorkspaceStats | null>(null)
  const [activities, setActivities] = useState<Activity[]>([])
  const [loading, setLoading] = useState(true)
  const { currentWorkspace } = useAuthStore()

  const fetchData = useCallback(async () => {
    if (!currentWorkspace?.id) {
      setLoading(false)
      return
    }
    try {
      const [statsRes, activityRes] = await Promise.all([
        apiGet<{ data: WorkspaceStats }>(`/workspaces/${currentWorkspace.id}/stats`),
        apiGet<{ data: { activities: Activity[] } }>(`/workspaces/${currentWorkspace.id}/activity`),
      ])
      setStats(statsRes?.data ?? null)
      setActivities(activityRes?.data?.activities ?? [])
    } catch {
      logger.warning('Could not fetch workspace dashboard data')
    } finally {
      setLoading(false)
    }
  }, [currentWorkspace?.id])

  useEffect(() => { fetchData() }, [fetchData])

  const formatTime = (ts: string) => {
    if (!ts) return ''
    try {
      const d = new Date(ts)
      return d.toLocaleString()
    } catch {
      return ts
    }
  }

  if (loading) {
    return (
      <PageContainer title="Dashboard" subtitle="Workspace overview" loadingCards={3}>
        <KpiGrid>
          <StatCard label="Loading" value={<Skeleton className="h-3.5 w-10" />} />
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
        left={<AppRouteHeaderLead title="Dashboard" subtitle={`Overview for ${stats?.name ?? currentWorkspace?.name ?? 'workspace'}`} />}
      />

      <div className="space-y-4">
        <KpiGrid>
          <StatCard label="Members" value={stats?.member_count ?? 0} />
          <StatCard label="Training Jobs" value={stats?.training_jobs ?? 0} />
          <StatCard label="Active Jobs" value={stats?.active_training_jobs ?? 0} />
          <StatCard label="Datasets" value={stats?.dataset_count ?? 0} />
          <StatCard label="Knowledge Items" value={stats?.knowledge_items ?? 0} />
        </KpiGrid>

        {/* Active training indicator */}
        {(stats?.active_training_jobs ?? 0) > 0 && (
          <Card>
            <CardContent className="py-3">
              <div className="flex items-center gap-2 text-xs">
                <span className="relative flex h-2 w-2">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75" />
                  <span className="relative inline-flex rounded-full h-2 w-2 bg-green-500" />
                </span>
                <span className="text-green-600 dark:text-green-400 font-medium">
                  {stats?.active_training_jobs} training job{stats?.active_training_jobs !== 1 ? 's' : ''} running
                </span>
              </div>
            </CardContent>
          </Card>
        )}

        {/* Activity feed */}
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-xs">Recent Activity</CardTitle>
          </CardHeader>
          <CardContent className="space-y-1">
            {activities.length === 0 ? (
              <p className="text-xs text-muted-foreground py-4 text-center">No recent activity</p>
            ) : (
              activities.slice(0, 15).map((a, i) => (
                <div key={i} className="flex items-center justify-between px-2 py-1.5 rounded text-[10px] hover:bg-muted/50">
                  <div className="min-w-0 flex-1">
                    <span className="font-medium">{a.action}</span>
                    {a.detail && <span className="text-muted-foreground ml-1.5 truncate">{a.detail}</span>}
                  </div>
                  <div className="flex items-center gap-2 shrink-0">
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
