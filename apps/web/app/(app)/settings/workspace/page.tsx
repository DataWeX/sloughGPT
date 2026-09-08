'use client'

import { useState, useEffect, useCallback } from 'react'
import { PageContainer } from '@/components/PageContainer'
import { AppRouteHeader, AppRouteHeaderLead } from '@/components/AppRouteHeader'
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
  Button,
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
  Input,
  Skeleton,
  StatCard,
  KpiGrid,
  Textarea,
} from '@sloughgpt/strui'
import { IconTrash } from '@/components/icons/NavIcons'
import { apiGet, apiPut, apiDelete } from '@/lib/http-client'
import { useAuthStore } from '@/lib/auth'
import { useToastStore } from '@/lib/toast-store'
import { useLocale } from '@/hooks/useLocale'

interface Workspace {
  id: string
  name: string
  description: string
  tenant_id: string
  member_count: number
  created_at: string
}

interface WorkspaceResponse {
  data: Workspace
}

interface UsageData {
  members: { total: number; admins: number; users: number; viewers: number }
  training: { total: number; running: number; completed: number; failed: number; total_minutes: number }
  datasets: { total: number }
  knowledge: { total: number }
  api_keys: { total: number }
}

interface UsageResponse {
  data: UsageData
}

interface HealthData {
  status: string
  checks: Record<string, string>
}

interface HealthResponse {
  data: HealthData
}

export default function WorkspaceSettingsPage() {
  const { currentWorkspace } = useAuthStore()
  const addToast = useToastStore(s => s.addToast)
  const { t } = useLocale()

  const [workspace, setWorkspace] = useState<Workspace | null>(null)
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)

  const [name, setName] = useState('')
  const [description, setDescription] = useState('')

  const [usage, setUsage] = useState<UsageData | null>(null)
  const [health, setHealth] = useState<HealthData | null>(null)

  const [deleteConfirm, setDeleteConfirm] = useState('')
  const [deleting, setDeleting] = useState(false)

  const wsId = currentWorkspace?.id

  const fetchWorkspace = useCallback(async () => {
    if (!wsId) return
    try {
      const res = await apiGet<WorkspaceResponse>(`/workspaces/${wsId}`)
      const ws = res?.data
      if (ws) {
        setWorkspace(ws)
        setName(ws.name)
        setDescription(ws.description)
      }
    } catch {
      addToast('Could not load workspace', 'error')
    } finally {
      setLoading(false)
    }
  }, [wsId, addToast])

  const fetchUsage = useCallback(async () => {
    if (!wsId) return
    try {
      const res = await apiGet<UsageResponse>(`/workspaces/${wsId}/usage`)
      setUsage(res?.data ?? null)
    } catch { /* optional */ }
  }, [wsId])

  const fetchHealth = useCallback(async () => {
    if (!wsId) return
    try {
      const res = await apiGet<HealthResponse>(`/workspaces/${wsId}/health`)
      setHealth(res?.data ?? null)
    } catch { /* optional */ }
  }, [wsId])

  useEffect(() => {
    fetchWorkspace()
    fetchUsage()
    fetchHealth()
  }, [fetchWorkspace, fetchUsage, fetchHealth])

  const handleSave = async () => {
    if (!wsId || !name.trim()) return
    setSaving(true)
    try {
      await apiPut(`/workspaces/${wsId}`, { name: name.trim(), description: description.trim() })
      addToast('Workspace updated', 'success')
      await fetchWorkspace()
    } catch {
      addToast('Could not update workspace', 'error')
    } finally {
      setSaving(false)
    }
  }

  const handleDelete = async () => {
    if (!wsId || deleteConfirm !== workspace?.name) return
    setDeleting(true)
    try {
      await apiDelete(`/workspaces/${wsId}`)
      addToast('Workspace deleted', 'success')
      window.location.href = '/workspaces'
    } catch {
      addToast('Could not delete workspace', 'error')
    } finally {
      setDeleting(false)
      setDeleteConfirm('')
    }
  }

  const hasChanges = workspace && (name !== workspace.name || description !== workspace.description)

  if (loading) {
    return (
      <PageContainer>
        <Skeleton className="h-8 w-64 mb-4" />
        <Skeleton className="h-64 w-full" />
      </PageContainer>
    )
  }

  if (!workspace) {
    return (
      <PageContainer>
        <p className="text-muted-foreground">No workspace selected.</p>
      </PageContainer>
    )
  }

  return (
    <PageContainer>
      <AppRouteHeader>
        <AppRouteHeaderLead>{workspace.name} — Settings</AppRouteHeaderLead>
      </AppRouteHeader>

      {/* KPIs */}
      <KpiGrid className="mb-6">
        <StatCard label="Members" value={String(usage?.members?.total ?? workspace.member_count)} />
        <StatCard label="Training Jobs" value={String(usage?.training?.total ?? 0)} />
        <StatCard label="Datasets" value={String(usage?.datasets?.total ?? 0)} />
        <StatCard label="Knowledge" value={String(usage?.knowledge?.total ?? 0)} />
      </KpiGrid>

      {/* General Settings */}
      <Card className="mb-6">
        <CardHeader>
          <CardTitle>General</CardTitle>
          <CardDescription>Workspace name and description</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div>
            <label className="text-sm font-medium mb-1 block">Name</label>
            <Input
              value={name}
              onChange={e => setName(e.target.value)}
              placeholder="Workspace name"
              maxLength={200}
            />
          </div>
          <div>
            <label className="text-sm font-medium mb-1 block">Description</label>
            <Textarea
              value={description}
              onChange={e => setDescription(e.target.value)}
              placeholder="Workspace description"
              maxLength={1000}
              rows={3}
            />
          </div>
        </CardContent>
        <CardFooter className="flex justify-end">
          <Button
            onClick={handleSave}
            disabled={!hasChanges || saving}
          >
            {saving ? 'Saving...' : 'Save Changes'}
          </Button>
        </CardFooter>
      </Card>

      {/* Health */}
      {health && (
        <Card className="mb-6">
          <CardHeader>
            <CardTitle>Health</CardTitle>
            <CardDescription>Workspace health status</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="flex items-center gap-2 mb-2">
              <span className={`inline-block w-2 h-2 rounded-full ${
                health.status === 'healthy' ? 'bg-green-500' :
                health.status === 'warning' ? 'bg-yellow-500' : 'bg-red-500'
              }`} />
              <span className="font-medium capitalize">{health.status}</span>
            </div>
            {health.checks && Object.keys(health.checks).length > 0 && (
              <div className="mt-2 space-y-1 text-sm text-muted-foreground">
                {Object.entries(health.checks).map(([key, val]) => (
                  <div key={key} className="flex justify-between">
                    <span>{key}</span>
                    <span>{val}</span>
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {/* Danger Zone */}
      <Card className="border-destructive">
        <CardHeader>
          <CardTitle className="text-destructive">Danger Zone</CardTitle>
          <CardDescription>
            Permanently delete this workspace and all its data. This action cannot be undone.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <AlertDialog>
            <AlertDialogTrigger asChild>
              <Button variant="destructive">
                <IconTrash className="mr-2 h-4 w-4" />
                Delete Workspace
              </Button>
            </AlertDialogTrigger>
            <AlertDialogContent>
              <AlertDialogHeader>
                <AlertDialogTitle>Delete workspace &ldquo;{workspace.name}&rdquo;?</AlertDialogTitle>
                <AlertDialogDescription>
                  This will permanently delete the workspace, all members, training jobs, datasets,
                  and knowledge. Type the workspace name to confirm.
                </AlertDialogDescription>
              </AlertDialogHeader>
              <Input
                value={deleteConfirm}
                onChange={e => setDeleteConfirm(e.target.value)}
                placeholder={workspace.name}
                className="mt-2"
              />
              <AlertDialogFooter>
                <AlertDialogCancel onClick={() => setDeleteConfirm('')}>Cancel</AlertDialogCancel>
                <AlertDialogAction
                  onClick={handleDelete}
                  disabled={deleteConfirm !== workspace.name || deleting}
                  className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
                >
                  {deleting ? 'Deleting...' : 'Delete'}
                </AlertDialogAction>
              </AlertDialogFooter>
            </AlertDialogContent>
          </AlertDialog>
        </CardContent>
      </Card>
    </PageContainer>
  )
}
