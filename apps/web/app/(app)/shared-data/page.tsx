'use client'

import { useState, useEffect, useCallback } from 'react'
import { Card, CardHeader, CardTitle, CardContent, Button, Input, Skeleton, EmptyCard } from '@sloughgpt/strui'
import { PageContainer } from '@/components/PageContainer'
import { AppRouteHeader, AppRouteHeaderLead } from '@/components/AppRouteHeader'
import { apiGet, apiPost, apiDelete } from '@/lib/http-client'
import { useAuthStore } from '@/lib/auth'
import { useToastStore } from '@/lib/toast-store'

interface Share {
  id: string
  resource_type: string
  resource_id: string
  source_workspace_id: string
  target_workspace_id: string
  permission: string
  shared_by: string
  shared_at: string
}

interface SharedItem {
  share: Share
  dataset?: { id: string; name: string; size: number }
}

export default function SharedDataPage() {
  const { currentWorkspace } = useAuthStore()
  const addToast = useToastStore(s => s.addToast)
  const [shares, setShares] = useState<Share[]>([])
  const [loading, setLoading] = useState(true)
  const [showShareDialog, setShowShareDialog] = useState(false)
  const [shareType, setShareType] = useState('dataset')
  const [shareResourceId, setShareResourceId] = useState('')
  const [shareTargetWs, setShareTargetWs] = useState('')
  const [sharePermission, setSharePermission] = useState('read')
  const [workspaces, setWorkspaces] = useState<Array<{ id: string; name: string }>>([])

  const fetchShares = useCallback(async () => {
    if (!currentWorkspace?.id) { setLoading(false); return }
    try {
      const res = await apiGet<{ data: { shares: Share[] } }>(`/workspaces/${currentWorkspace.id}/shared`)
      setShares(res?.data?.shares ?? [])
    } catch {
      addToast('Could not load shared data', 'error')
    } finally {
      setLoading(false)
    }
  }, [currentWorkspace?.id, addToast])

  const fetchWorkspaces = useCallback(async () => {
    try {
      const res = await apiGet<{ data: Array<{ id: string; name: string }> }>('/workspaces')
      setWorkspaces((res?.data ?? []).filter(w => w.id !== currentWorkspace?.id))
    } catch { /* silent */ }
  }, [currentWorkspace?.id])

  useEffect(() => { fetchShares(); fetchWorkspaces() }, [fetchShares, fetchWorkspaces])

  const handleShare = async () => {
    if (!shareResourceId.trim() || !shareTargetWs) {
      addToast('Fill in all fields', 'error')
      return
    }
    try {
      await apiPost(`/workspaces/${currentWorkspace?.id}/share`, {
        resource_type: shareType,
        resource_id: shareResourceId.trim(),
        target_workspace_id: shareTargetWs,
        permission: sharePermission,
      })
      addToast('Data shared successfully', 'success')
      setShowShareDialog(false)
      setShareResourceId('')
      setShareTargetWs('')
      fetchShares()
    } catch {
      addToast('Could not share data', 'error')
    }
  }

  const handleRevoke = async (shareId: string) => {
    try {
      await apiDelete(`/workspaces/${currentWorkspace?.id}/share/${shareId}`)
      addToast('Share revoked', 'success')
      fetchShares()
    } catch {
      addToast('Could not revoke share', 'error')
    }
  }

  const incoming = shares.filter(s => s.target_workspace_id === currentWorkspace?.id)
  const outgoing = shares.filter(s => s.source_workspace_id === currentWorkspace?.id)

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
      <AppRouteHeader
        left={<AppRouteHeaderLead>Shared Data</AppRouteHeaderLead>}
        right={
          <Button size="sm" className="h-6 text-[10px]" onClick={() => setShowShareDialog(true)}>
            Share Data
          </Button>
        }
      />

      {/* Share dialog */}
      {showShareDialog && (
        <Card className="mb-4 border-primary/30">
          <CardHeader className="pb-2">
            <CardTitle className="text-xs">Share Data with Another Workspace</CardTitle>
          </CardHeader>
          <CardContent className="space-y-2">
            <div className="flex gap-2">
              <select
                value={shareType}
                onChange={e => setShareType(e.target.value)}
                className="h-6 text-[10px] rounded-md border border-border bg-background px-2"
              >
                <option value="dataset">Dataset</option>
                <option value="knowledge">Knowledge</option>
                <option value="api_key">API Key</option>
              </select>
              <Input
                value={shareResourceId}
                onChange={e => setShareResourceId(e.target.value)}
                placeholder={`${shareType} ID or name`}
                className="flex-1 h-6 text-[10px]"
              />
            </div>
            <div className="flex gap-2">
              <select
                value={shareTargetWs}
                onChange={e => setShareTargetWs(e.target.value)}
                className="flex-1 h-6 text-[10px] rounded-md border border-border bg-background px-2"
              >
                <option value="">Select workspace...</option>
                {workspaces.map(w => (
                  <option key={w.id} value={w.id}>{w.name}</option>
                ))}
              </select>
              <select
                value={sharePermission}
                onChange={e => setSharePermission(e.target.value)}
                className="h-6 text-[10px] rounded-md border border-border bg-background px-2"
              >
                <option value="read">Read only</option>
                <option value="admin">Admin</option>
              </select>
            </div>
            <div className="flex gap-2">
              <Button size="sm" className="h-6 text-[10px]" onClick={handleShare}>Share</Button>
              <Button size="sm" variant="ghost" className="h-6 text-[10px]" onClick={() => setShowShareDialog(false)}>Cancel</Button>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Incoming shares */}
      <Card className="mb-4">
        <CardHeader className="pb-2">
          <CardTitle className="text-xs">Shared With Me ({incoming.length})</CardTitle>
        </CardHeader>
        <CardContent className="space-y-1">
          {incoming.length === 0 ? (
            <p className="text-[10px] text-muted-foreground text-center py-4">No data shared with this workspace</p>
          ) : (
            incoming.map(s => (
              <div key={s.id} className="flex items-center justify-between px-3 py-2 rounded-md text-[10px] hover:bg-muted/50 border border-transparent hover:border-border/50">
                <div className="min-w-0 flex-1">
                  <div className="flex items-center gap-2">
                    <span className="px-1.5 py-0.5 rounded text-[9px] font-medium bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400">
                      {s.resource_type}
                    </span>
                    <span className="font-medium">{s.resource_id}</span>
                  </div>
                  <div className="text-muted-foreground mt-0.5">
                    From workspace {s.source_workspace_id.slice(0, 8)}... · {s.permission}
                  </div>
                </div>
                <span className="text-muted-foreground">{new Date(s.shared_at).toLocaleDateString()}</span>
              </div>
            ))
          )}
        </CardContent>
      </Card>

      {/* Outgoing shares */}
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-xs">Shared by Me ({outgoing.length})</CardTitle>
        </CardHeader>
        <CardContent className="space-y-1">
          {outgoing.length === 0 ? (
            <p className="text-[10px] text-muted-foreground text-center py-4">No data shared from this workspace</p>
          ) : (
            outgoing.map(s => (
              <div key={s.id} className="flex items-center justify-between px-3 py-2 rounded-md text-[10px] hover:bg-muted/50 border border-transparent hover:border-border/50">
                <div className="min-w-0 flex-1">
                  <div className="flex items-center gap-2">
                    <span className="px-1.5 py-0.5 rounded text-[9px] font-medium bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400">
                      {s.resource_type}
                    </span>
                    <span className="font-medium">{s.resource_id}</span>
                  </div>
                  <div className="text-muted-foreground mt-0.5">
                    To workspace {s.target_workspace_id.slice(0, 8)}... · {s.permission}
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-muted-foreground">{new Date(s.shared_at).toLocaleDateString()}</span>
                  <Button size="sm" variant="ghost" className="h-5 text-[9px] text-destructive" onClick={() => handleRevoke(s.id)}>
                    Revoke
                  </Button>
                </div>
              </div>
            ))
          )}
        </CardContent>
      </Card>
    </PageContainer>
  )
}
