'use client'

import { useState, useEffect, useCallback } from 'react'
import { Card, CardHeader, CardTitle, CardContent, Button, Input, Skeleton } from '@sloughgpt/strui'
import { PageContainer } from '@/components/PageContainer'
import { AppRouteHeader, AppRouteHeaderLead } from '@/components/AppRouteHeader'
import { apiGet, apiPut } from '@/lib/http-client'
import { useAuthStore } from '@/lib/auth'
import { useToastStore } from '@/lib/toast-store'
import { logger } from '@/lib/dev-log'

interface WorkspaceDetail {
  id: string
  name: string
  description: string
  tenant_id: string
  created_at: string
  updated_at: string
  member_count: number
}

export default function WorkspaceSettingsPage() {
  const [workspace, setWorkspace] = useState<WorkspaceDetail | null>(null)
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [name, setName] = useState('')
  const [description, setDescription] = useState('')
  const [dirty, setDirty] = useState(false)
  const addToast = useToastStore(s => s.addToast)
  const { currentWorkspace } = useAuthStore()

  const fetchWorkspace = useCallback(async () => {
    if (!currentWorkspace?.id) {
      setLoading(false)
      return
    }
    try {
      const res = await apiGet<{ data: WorkspaceDetail }>(`/workspaces/${currentWorkspace.id}`)
      const data = res?.data
      if (data) {
        setWorkspace(data)
        setName(data.name)
        setDescription(data.description)
      }
    } catch {
      logger.warning('Could not fetch workspace settings')
    } finally {
      setLoading(false)
    }
  }, [currentWorkspace?.id])

  useEffect(() => { fetchWorkspace() }, [fetchWorkspace])

  useEffect(() => {
    if (!workspace) return
    setDirty(name !== workspace.name || description !== workspace.description)
  }, [name, description, workspace])

  const handleSave = async () => {
    if (!workspace || !dirty) return
    setSaving(true)
    try {
      await apiPut(`/workspaces/${workspace.id}`, { name, description })
      setWorkspace(prev => prev ? { ...prev, name, description, updated_at: new Date().toISOString() } : prev)
      setDirty(false)
      addToast('Workspace settings saved', 'success')
    } catch {
      addToast('Could not save workspace settings', 'error')
    } finally {
      setSaving(false)
    }
  }

  const handleReset = () => {
    if (!workspace) return
    setName(workspace.name)
    setDescription(workspace.description)
    setDirty(false)
  }

  const formatDate = (iso: string) => {
    if (!iso) return '—'
    try {
      return new Date(iso).toLocaleString()
    } catch {
      return iso
    }
  }

  return (
    <PageContainer>
      <AppRouteHeader>
        <AppRouteHeaderLead>Workspace Settings</AppRouteHeaderLead>
      </AppRouteHeader>

      {loading ? (
        <div className="space-y-4">
          <Skeleton className="h-40 w-full" />
          <Skeleton className="h-60 w-full" />
        </div>
      ) : !workspace ? (
        <Card>
          <CardContent className="py-12 text-center text-muted-foreground">
            No workspace selected. Use the workspace switcher to select one.
          </CardContent>
        </Card>
      ) : (
        <div className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>General</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-1.5">
                <label className="text-xs font-medium text-muted-foreground">Name</label>
                <Input
                  value={name}
                  onChange={e => setName(e.target.value)}
                  placeholder="Workspace name"
                />
              </div>
              <div className="space-y-1.5">
                <label className="text-xs font-medium text-muted-foreground">Description</label>
                <textarea
                  value={description}
                  onChange={e => setDescription(e.target.value)}
                  placeholder="Optional description"
                  rows={3}
                  className="flex w-full rounded-md border border-border bg-transparent px-3 py-2 text-sm placeholder:text-muted-foreground/50 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring disabled:cursor-not-allowed disabled:opacity-50 resize-none"
                />
              </div>
              <div className="flex gap-2 pt-1">
                <Button
                  onClick={handleSave}
                  disabled={!dirty || saving}
                  size="sm"
                >
                  {saving ? 'Saving...' : 'Save Changes'}
                </Button>
                {dirty && (
                  <Button
                    onClick={handleReset}
                    variant="outline"
                    size="sm"
                  >
                    Reset
                  </Button>
                )}
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Details</CardTitle>
            </CardHeader>
            <CardContent>
              <dl className="grid grid-cols-2 gap-x-4 gap-y-3 text-sm">
                <dt className="text-muted-foreground">Workspace ID</dt>
                <dd className="font-mono text-xs">{workspace.id}</dd>

                <dt className="text-muted-foreground">Tenant ID</dt>
                <dd className="font-mono text-xs">{workspace.tenant_id || '—'}</dd>

                <dt className="text-muted-foreground">Members</dt>
                <dd>{workspace.member_count}</dd>

                <dt className="text-muted-foreground">Created</dt>
                <dd>{formatDate(workspace.created_at)}</dd>

                <dt className="text-muted-foreground">Last Updated</dt>
                <dd>{formatDate(workspace.updated_at)}</dd>
              </dl>
            </CardContent>
          </Card>
        </div>
      )}
    </PageContainer>
  )
}
