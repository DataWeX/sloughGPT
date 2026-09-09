'use client'

import { useState, useEffect, useCallback } from 'react'
import { Card, CardHeader, CardTitle, CardContent, Button, Input, StatCard, KpiGrid, Skeleton } from '@sloughgpt/strui'
import { IconRefresh } from '@/components/icons/NavIcons'
import { PageContainer } from '@/components/PageContainer'
import { AppRouteHeader, AppRouteHeaderLead } from '@/components/AppRouteHeader'
import { apiGet } from '@/lib/http-client'
import { useAuthStore } from '@/lib/auth'
import { useToastStore } from '@/lib/toast-store'

interface Notification {
  type: string
  title: string
  detail: string
  status: string
  timestamp: string
}

interface NotificationsResponse {
  data: { notifications: Notification[] }
}

export default function NotificationsPage() {
  const { currentWorkspace } = useAuthStore()
  const addToast = useToastStore(s => s.addToast)
  const [notifications, setNotifications] = useState<Notification[]>([])
  const [loading, setLoading] = useState(true)
  const [filter, setFilter] = useState('')

  const fetchNotifications = useCallback(async () => {
    if (!currentWorkspace?.id) {
      setLoading(false)
      return
    }
    try {
      const res = await apiGet<NotificationsResponse>(
        `/workspaces/${currentWorkspace.id}/notifications`
      )
      setNotifications(res?.data?.notifications ?? [])
    } catch {
      addToast('Could not load notifications', 'error')
    } finally {
      setLoading(false)
    }
  }, [currentWorkspace?.id, addToast])

  useEffect(() => { fetchNotifications() }, [fetchNotifications])

  const filtered = notifications.filter(n => {
    if (!filter) return true
    const q = filter.toLowerCase()
    return (
      n.title.toLowerCase().includes(q) ||
      n.detail.toLowerCase().includes(q) ||
      n.type.toLowerCase().includes(q)
    )
  })

  const typeCounts = notifications.reduce((acc, n) => {
    acc[n.type] = (acc[n.type] || 0) + 1
    return acc
  }, {} as Record<string, number>)

  const formatTime = (ts: string) => {
    if (!ts) return ''
    try { return new Date(ts).toLocaleString() } catch { return ts }
  }

  if (loading) {
    return (
      <PageContainer>
        <Skeleton className="h-8 w-64 mb-4" />
        <Skeleton className="h-64 w-full" />
      </PageContainer>
    )
  }

  return (
    <PageContainer>
      <AppRouteHeader>
        <AppRouteHeaderLead>Notifications</AppRouteHeaderLead>
      </AppRouteHeader>

      <KpiGrid className="mb-6">
        <StatCard label="Total" value={notifications.length} />
        <StatCard label="Training" value={typeCounts['training'] ?? 0} />
        <StatCard label="Members" value={typeCounts['member'] ?? 0} />
      </KpiGrid>

      <Card className="mb-4">
        <CardContent className="py-2">
          <div className="flex gap-2">
            <Input
              value={filter}
              onChange={e => setFilter(e.target.value)}
              placeholder="Filter notifications..."
              className="flex-1 h-6 text-[10px]"
            />
            <Button size="sm" variant="outline" className="h-6 text-[10px]" onClick={fetchNotifications}>
              <IconRefresh className="h-3 w-3 mr-1" />
              Refresh
            </Button>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-xs">Recent Events</CardTitle>
        </CardHeader>
        <CardContent className="space-y-1">
          {filtered.length === 0 ? (
            <p className="text-xs text-muted-foreground py-4 text-center">
              {filter ? 'No notifications match filter' : 'No notifications'}
            </p>
          ) : (
            filtered.map((n, i) => (
              <div key={i} className="flex items-center justify-between px-3 py-2 rounded-md text-[10px] hover:bg-muted/50 border border-transparent hover:border-border/50">
                <div className="min-w-0 flex-1">
                  <div className="flex items-center gap-2">
                    <span className={`px-1.5 py-0.5 rounded text-[9px] font-medium ${
                      n.type === 'training' ? 'bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400' :
                      n.type === 'member' ? 'bg-purple-100 text-purple-700 dark:bg-purple-900/30 dark:text-purple-400' :
                      'bg-muted text-muted-foreground'
                    }`}>
                      {n.type}
                    </span>
                    <span className="font-medium">{n.title}</span>
                  </div>
                  {n.detail && <div className="text-muted-foreground mt-0.5">{n.detail}</div>}
                </div>
                <div className="flex items-center gap-3 shrink-0">
                  {n.status && (
                    <span className={`px-1.5 py-0.5 rounded-full text-[9px] ${
                      n.status === 'completed' ? 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400' :
                      n.status === 'failed' ? 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400' :
                      'bg-muted text-muted-foreground'
                    }`}>
                      {n.status}
                    </span>
                  )}
                  <span className="text-muted-foreground whitespace-nowrap">{formatTime(n.timestamp)}</span>
                </div>
              </div>
            ))
          )}
        </CardContent>
      </Card>
    </PageContainer>
  )
}
