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

interface AuditLog {
  event_type: string
  timestamp: string
  user?: string
  resource?: string
  detail?: string
}

export default function WorkspaceDashboardPage() {
  const [stats, setStats] = useState<WorkspaceStats | null>(null)
  const [recentAudit, setRecentAudit] = useState<AuditLog[]>([])
  const [loading, setLoading] = useState(true)
  const { currentWorkspace } = useAuthStore()

  const fetchData = useCallback(async () => {
    if (!currentWorkspace?.id) {
      setLoading(false)
      return
    }
    try {
      const [statsRes, auditRes] = await Promise.all([
        apiGet<{ data: WorkspaceStats }>(`/workspaces/${currentWorkspace.id}/stats`),
        apiGet<{ data: { logs: AuditLog[] } }>('/security/audit?limit=20'),
      ])
      setStats(statsRes?.data ?? null)
      setRecentAudit(auditRes?.data?.logs ?? [])
    } catch {
      logger.warning('Could not fetch workspace dashboard data')
    } finally {
      setLoading(false)
    }
  }, [currentWorkspace?.id])

  useEffect(() => { fetchData() }, [fetchData])

  if (!currentWorkspace) {
    return (
      <PageContainer title="Workspace Dashboard" subtitle="Select a workspace first">
        <Card>
          <CardContent className="py-8 text-center text-sm text-muted-foreground">
            No workspace selected. Use the workspace switcher in the header to select one.
          </CardContent>
        </Card>
      </PageContainer>
    )
  }

  if (loading) {
    return (
      <PageContainer title="Workspace Dashboard" subtitle={currentWorkspace.name} loadingCards={4}>
        <KpiGrid>
          <StatCard label="Loading" value={<Skeleton className="h-3.5 w-10" />} />
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
        left={<AppRouteHeaderLead title="Workspace Dashboard" subtitle={currentWorkspace.name} />}
      />

      <div className="space-y-4">
        <KpiGrid>
          <StatCard label="Members" value={stats?.member_count ?? 0} />
          <StatCard label="Datasets" value={stats?.dataset_count ?? 0} />
          <StatCard label="Training Jobs" value={stats?.training_jobs ?? 0} />
          <StatCard label="Knowledge Items" value={stats?.knowledge_items ?? 0} />
        </KpiGrid>

        {stats && stats.active_training_jobs > 0 && (
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-xs">Active Training</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex items-center gap-2 text-xs">
                <span className="inline-block h-2 w-2 rounded-full bg-green-500 animate-pulse" />
                <span>{stats.active_training_jobs} job{stats.active_training_jobs !== 1 ? 's' : ''} running</span>
              </div>
            </CardContent>
          </Card>
        )}

        {/* Recent activity */}
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-xs">Recent Activity</CardTitle>
          </CardHeader>
          <CardContent>
            {recentAudit.length === 0 ? (
              <p className="text-[10px] text-muted-foreground py-4 text-center">No recent activity</p>
            ) : (
              <div className="space-y-0.5">
                {recentAudit.slice(0, 10).map((log, i) => (
                  <div key={i} className="flex items-center justify-between px-2 py-1.5 rounded text-[10px] hover:bg-muted/50">
                    <div className="min-w-0 flex-1">
                      <span className="font-medium">{log.event_type}</span>
                      {log.resource && <span className="text-muted-foreground ml-1.5">· {log.resource}</span>}
                    </div>
                    <div className="shrink-0 text-muted-foreground">
                      {log.timestamp ? new Date(log.timestamp).toLocaleTimeString() : ''}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>

        {/* Quick links */}
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-xs">Quick Links</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 gap-2 text-[10px]">
              <a href="/workspaces" className="px-3 py-2 rounded-md hover:bg-muted/50 transition-colors">Manage Members</a>
              <a href="/datasets" className="px-3 py-2 rounded-md hover:bg-muted/50 transition-colors">Datasets</a>
              <a href="/training" className="px-3 py-2 rounded-md hover:bg-muted/50 transition-colors">Training Jobs</a>
              <a href="/kb" className="px-3 py-2 rounded-md hover:bg-muted/50 transition-colors">Knowledge Base</a>
              <a href="/security" className="px-3 py-2 rounded-md hover:bg-muted/50 transition-colors">API Keys</a>
              <a href="/profile" className="px-3 py-2 rounded-md hover:bg-muted/50 transition-colors">Profile</a>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
