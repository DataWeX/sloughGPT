'use client'

import { useState, useEffect, useCallback, useMemo } from 'react'
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

interface UsageData {
  members: { total: number; admins: number; users: number; viewers: number }
  training: { total: number; running: number; completed: number; failed: number; total_minutes: number }
  datasets: { total: number }
  knowledge: { total: number }
  api_keys: { total: number }
}

export default function WorkspaceDashboardPage() {
  const [stats, setStats] = useState<WorkspaceStats | null>(null)
  const [activities, setActivities] = useState<Activity[]>([])
  const [usage, setUsage] = useState<UsageData | null>(null)
  const [loading, setLoading] = useState(true)
  const [healthStatus, setHealthStatus] = useState<string | null>(null)
  const [checkingHealth, setCheckingHealth] = useState(false)
  const { currentWorkspace } = useAuthStore()

  const runHealthCheck = useCallback(async () => {
    if (!currentWorkspace?.id) return
    setCheckingHealth(true)
    try {
      const res = await apiGet<{ data: { status: string } }>(`/workspaces/${currentWorkspace.id}/health`)
      setHealthStatus(res?.data?.status ?? null)
    } catch {
      setHealthStatus('error')
    } finally {
      setCheckingHealth(false)
    }
  }, [currentWorkspace?.id])

  const fetchData = useCallback(async () => {
    if (!currentWorkspace?.id) {
      setLoading(false)
      return
    }
    try {
      const [statsRes, activityRes, usageRes] = await Promise.all([
        apiGet<{ data: WorkspaceStats }>(`/workspaces/${currentWorkspace.id}/stats`),
        apiGet<{ data: { activities: Activity[] } }>(`/workspaces/${currentWorkspace.id}/activity`),
        apiGet<{ data: UsageData }>(`/workspaces/${currentWorkspace.id}/usage`),
      ])
      setStats(statsRes?.data ?? null)
      setActivities(activityRes?.data?.activities ?? [])
      setUsage(usageRes?.data ?? null)
      runHealthCheck()
    } catch {
      logger.warning('Could not fetch workspace dashboard data')
    } finally {
      setLoading(false)
    }
  }, [currentWorkspace?.id, runHealthCheck])

  useEffect(() => { fetchData() }, [fetchData])

  const activityByType = useMemo(() => {
    const map: Record<string, number> = {}
    for (const a of activities) {
      map[a.type] = (map[a.type] || 0) + 1
    }
    return map
  }, [activities])

  const trainingByStatus = useMemo(() => {
    if (!usage?.training) return { completed: 0, failed: 0, running: 0 }
    return {
      completed: usage.training.completed,
      failed: usage.training.failed,
      running: usage.training.running,
    }
  }, [usage])

  const formatTime = (ts: string) => {
    if (!ts) return ''
    try { return new Date(ts).toLocaleString() } catch { return ts }
  }

  const maxActivityType = useMemo(() => {
    let max = 0
    let maxType = ''
    for (const [type, count] of Object.entries(activityByType)) {
      if (count > max) { max = count; maxType = type }
    }
    return { type: maxType, count: max }
  }, [activityByType])

  if (loading) {
    return (
      <PageContainer title="Workspace Dashboard">
        <Skeleton className="h-8 w-64 mb-4" />
        <Skeleton className="h-64 w-full" />
      </PageContainer>
    )
  }

  return (
    <PageContainer title="Workspace Dashboard">
      <AppRouteHeader
        left={<AppRouteHeaderLead title={`${stats?.name ?? currentWorkspace?.name ?? 'Workspace'} — Dashboard`} />}
      />

      {/* KPIs */}
      <KpiGrid className="mb-6">
        <StatCard label="Members" value={stats?.member_count ?? 0} />
        <StatCard label="Training Jobs" value={stats?.training_jobs ?? 0} />
        <StatCard label="Active Jobs" value={stats?.active_training_jobs ?? 0} />
        <StatCard label="Datasets" value={stats?.dataset_count ?? 0} />
        <StatCard label="Knowledge" value={stats?.knowledge_items ?? 0} />
        <StatCard label="API Keys" value={usage?.api_keys?.total ?? 0} />
      </KpiGrid>

      {/* Active training indicator */}
      {(stats?.active_training_jobs ?? 0) > 0 && (
        <Card className="mb-4">
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

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 mb-4">
        {/* Training Status */}
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-xs">Training Overview</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              <div className="flex items-center justify-between text-[10px]">
                <span className="text-muted-foreground">Completed</span>
                <span className="font-medium text-green-600">{trainingByStatus.completed}</span>
              </div>
              <div className="w-full bg-muted rounded-full h-1.5">
                <div
                  className="bg-green-500 h-1.5 rounded-full"
                  style={{ width: `${usage?.training?.total ? (trainingByStatus.completed / usage.training.total) * 100 : 0}%` }}
                />
              </div>

              <div className="flex items-center justify-between text-[10px]">
                <span className="text-muted-foreground">Running</span>
                <span className="font-medium text-blue-600">{trainingByStatus.running}</span>
              </div>
              <div className="w-full bg-muted rounded-full h-1.5">
                <div
                  className="bg-blue-500 h-1.5 rounded-full"
                  style={{ width: `${usage?.training?.total ? (trainingByStatus.running / usage.training.total) * 100 : 0}%` }}
                />
              </div>

              <div className="flex items-center justify-between text-[10px]">
                <span className="text-muted-foreground">Failed</span>
                <span className="font-medium text-red-600">{trainingByStatus.failed}</span>
              </div>
              <div className="w-full bg-muted rounded-full h-1.5">
                <div
                  className="bg-red-500 h-1.5 rounded-full"
                  style={{ width: `${usage?.training?.total ? (trainingByStatus.failed / usage.training.total) * 100 : 0}%` }}
                />
              </div>

              <div className="pt-2 border-t text-[10px] text-muted-foreground">
                Total training time: {usage?.training?.total_minutes ?? 0} min
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Activity Breakdown */}
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-xs">Activity Breakdown</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              {Object.entries(activityByType)
                .sort(([, a], [, b]) => b - a)
                .map(([type, count]) => (
                  <div key={type} className="flex items-center justify-between text-[10px]">
                    <div className="flex items-center gap-2">
                      <span className={`px-1.5 py-0.5 rounded text-[9px] font-medium ${
                        type === 'training' ? 'bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400' :
                        type === 'audit' ? 'bg-purple-100 text-purple-700 dark:bg-purple-900/30 dark:text-purple-400' :
                        'bg-muted text-muted-foreground'
                      }`}>
                        {type}
                      </span>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="w-24 bg-muted rounded-full h-1.5">
                        <div
                          className={`h-1.5 rounded-full ${
                            type === 'training' ? 'bg-blue-500' :
                            type === 'audit' ? 'bg-purple-500' : 'bg-gray-500'
                          }`}
                          style={{ width: `${maxActivityType.count ? (count / maxActivityType.count) * 100 : 0}%` }}
                        />
                      </div>
                      <span className="font-medium w-6 text-right">{count}</span>
                    </div>
                  </div>
                ))}
              {Object.keys(activityByType).length === 0 && (
                <p className="text-[10px] text-muted-foreground text-center py-4">No activity data</p>
              )}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Quick health status */}
      {healthStatus && (
        <Card className="mb-4">
          <CardContent className="py-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2 text-xs">
                <span className={`h-2 w-2 rounded-full ${
                  healthStatus === 'healthy' ? 'bg-green-500' :
                  healthStatus === 'warning' ? 'bg-yellow-500' : 'bg-red-500'
                }`} />
                <span className="font-medium capitalize">Workspace {healthStatus}</span>
              </div>
              <button
                onClick={runHealthCheck}
                disabled={checkingHealth}
                className="text-[10px] text-muted-foreground hover:text-foreground"
              >
                {checkingHealth ? 'Checking...' : 'Re-check'}
              </button>
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
