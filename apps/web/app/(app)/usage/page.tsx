'use client'

import { useState, useEffect, useCallback } from 'react'
import { Card, CardHeader, CardTitle, CardContent, StatCard, KpiGrid, Skeleton } from '@sloughgpt/strui'
import { PageContainer } from '@/components/PageContainer'
import { AppRouteHeader, AppRouteHeaderLead } from '@/components/AppRouteHeader'
import { apiGet } from '@/lib/http-client'
import { useAuthStore } from '@/lib/auth'
import { logger } from '@/lib/dev-log'

interface WorkspaceUsage {
  workspace_id: string
  name: string
  members: {
    total: number
    by_role: Record<string, number>
  }
  training: {
    by_status: Record<string, number>
    total_minutes: number
  }
  datasets: number
  knowledge_items: number
  api_keys: number
}

export default function UsagePage() {
  const [usage, setUsage] = useState<WorkspaceUsage | null>(null)
  const [loading, setLoading] = useState(true)
  const { currentWorkspace } = useAuthStore()

  const fetchUsage = useCallback(async () => {
    if (!currentWorkspace?.id) {
      setLoading(false)
      return
    }
    try {
      const res = await apiGet<{ data: WorkspaceUsage }>(`/workspaces/${currentWorkspace.id}/usage`)
      setUsage(res?.data ?? null)
    } catch {
      logger.warning('Could not fetch workspace usage')
    } finally {
      setLoading(false)
    }
  }, [currentWorkspace?.id])

  useEffect(() => { fetchUsage() }, [fetchUsage])

  if (loading) {
    return (
      <PageContainer title="Usage" subtitle="Workspace usage metrics" loadingCards={4}>
        <KpiGrid>
          <StatCard label="Loading" value={<Skeleton className="h-3.5 w-10" />} />
          <StatCard label="Loading" value={<Skeleton className="h-3.5 w-10" />} />
          <StatCard label="Loading" value={<Skeleton className="h-3.5 w-10" />} />
          <StatCard label="Loading" value={<Skeleton className="h-3.5 w-10" />} />
        </KpiGrid>
      </PageContainer>
    )
  }

  if (!currentWorkspace) {
    return (
      <PageContainer title="Usage" subtitle="No workspace selected">
        <Card><CardContent><p className="text-xs text-muted-foreground text-center py-8">Select a workspace to view usage</p></CardContent></Card>
      </PageContainer>
    )
  }

  return (
    <div className="sl-page mx-auto max-w-4xl">
      <AppRouteHeader
        left={<AppRouteHeaderLead title="Usage" subtitle={`Metrics for ${usage?.name ?? currentWorkspace.name}`} />}
      />

      <div className="space-y-4">
        <KpiGrid>
          <StatCard label="Members" value={usage?.members.total ?? 0} />
          <StatCard label="Datasets" value={usage?.datasets ?? 0} />
          <StatCard label="Knowledge Items" value={usage?.knowledge_items ?? 0} />
          <StatCard label="API Keys" value={usage?.api_keys ?? 0} />
        </KpiGrid>

        {/* Training overview */}
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-xs">Training Jobs</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              <div className="text-center p-2 rounded bg-muted/30">
                <div className="text-lg font-bold text-green-500">{usage?.training.by_status.completed ?? 0}</div>
                <div className="text-[10px] text-muted-foreground">Completed</div>
              </div>
              <div className="text-center p-2 rounded bg-muted/30">
                <div className="text-lg font-bold text-blue-500">{usage?.training.by_status.running ?? 0}</div>
                <div className="text-[10px] text-muted-foreground">Running</div>
              </div>
              <div className="text-center p-2 rounded bg-muted/30">
                <div className="text-lg font-bold text-yellow-500">{usage?.training.by_status.queued ?? 0}</div>
                <div className="text-[10px] text-muted-foreground">Queued</div>
              </div>
              <div className="text-center p-2 rounded bg-muted/30">
                <div className="text-lg font-bold text-red-500">{usage?.training.by_status.failed ?? 0}</div>
                <div className="text-[10px] text-muted-foreground">Failed</div>
              </div>
            </div>
            <div className="mt-3 text-[10px] text-muted-foreground text-center">
              Total training time: {usage?.training.total_minutes ?? 0} minutes
            </div>
          </CardContent>
        </Card>

        {/* Member roles */}
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-xs">Member Roles</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-1">
              {Object.entries(usage?.members.by_role ?? {}).map(([role, count]) => (
                <div key={role} className="flex items-center justify-between px-2 py-1.5 rounded text-[10px] hover:bg-muted/50">
                  <span className="font-medium capitalize">{role}</span>
                  <span className="text-muted-foreground">{count}</span>
                </div>
              ))}
              {Object.keys(usage?.members.by_role ?? {}).length === 0 && (
                <p className="text-[10px] text-muted-foreground text-center py-2">No members</p>
              )}
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
