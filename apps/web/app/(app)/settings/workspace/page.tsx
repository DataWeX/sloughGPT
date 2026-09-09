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
  Slider,
  StatCard,
  KpiGrid,
  Switch,
  Textarea,
} from '@sloughgpt/strui'
import { IconTrash } from '@/components/icons/NavIcons'
import { apiGet, apiPut, apiDelete, apiPost } from '@/lib/http-client'
import { useAuthStore } from '@/lib/auth'
import { useToastStore } from '@/lib/toast-store'
import { useLocale } from '@/hooks/useLocale'

interface WorkspaceSettings {
  workspace_id: string
  name: string
  description: string
  default_model: string
  data_retention_days: number
  max_members: number
  allow_sharing: boolean
  created_at: string
  updated_at: string
  member_count?: number
}

interface SettingsResponse {
  data: WorkspaceSettings
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

  const [settings, setSettings] = useState<WorkspaceSettings | null>(null)
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)

  const [name, setName] = useState('')
  const [description, setDescription] = useState('')
  const [defaultModel, setDefaultModel] = useState('')
  const [retentionDays, setRetentionDays] = useState(90)
  const [maxMembers, setMaxMembers] = useState(50)
  const [allowSharing, setAllowSharing] = useState(true)

  const [usage, setUsage] = useState<UsageData | null>(null)
  const [health, setHealth] = useState<HealthData | null>(null)

  const [deleteConfirm, setDeleteConfirm] = useState('')
  const [deleting, setDeleting] = useState(false)
  const [cleaning, setCleaning] = useState(false)
  const [cloning, setCloning] = useState(false)

  const wsId = currentWorkspace?.id

  const fetchSettings = useCallback(async () => {
    if (!wsId) return
    try {
      const res = await apiGet<SettingsResponse>(`/workspaces/${wsId}/settings`)
      const s = res?.data
      if (s) {
        setSettings(s)
        setName(s.name)
        setDescription(s.description)
        setDefaultModel(s.default_model || '')
        setRetentionDays(s.data_retention_days || 90)
        setMaxMembers(s.max_members || 50)
        setAllowSharing(s.allow_sharing)
      }
    } catch {
      addToast('Could not load workspace settings', 'error')
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
    fetchSettings()
    fetchUsage()
    fetchHealth()
  }, [fetchSettings, fetchUsage, fetchHealth])

  const handleSave = async () => {
    if (!wsId || !name.trim()) return
    setSaving(true)
    try {
      await apiPut(`/workspaces/${wsId}/settings`, {
        name: name.trim(),
        description: description.trim(),
        default_model: defaultModel.trim() || null,
        data_retention_days: retentionDays,
        max_members: maxMembers,
        allow_sharing: allowSharing,
      })
      addToast('Settings saved', 'success')
      await fetchSettings()
    } catch {
      addToast('Could not save settings', 'error')
    } finally {
      setSaving(false)
    }
  }

  const handleDelete = async () => {
    if (!wsId || deleteConfirm !== settings?.name) return
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

  const handleCleanup = async () => {
    if (!wsId) return
    setCleaning(true)
    try {
      const res = await apiPost<{ data: { retention_days: number; cleaned: Record<string, number> } }>(
        `/workspaces/${wsId}/cleanup`,
        {}
      )
      const cleaned = res?.data?.cleaned ?? {}
      const total = Object.values(cleaned).reduce((a, b) => a + b, 0)
      addToast(`Cleanup complete: ${total} items removed`, 'success')
    } catch {
      addToast('Could not run cleanup', 'error')
    } finally {
      setCleaning(false)
    }
  }

  const handleClone = async () => {
    if (!wsId) return
    setCloning(true)
    try {
      const res = await apiPost<{ data: { id: string; name: string; members_cloned: number } }>(
        `/workspaces/${wsId}/clone`,
        {}
      )
      const data = res?.data
      if (data) {
        addToast(`Cloned to "${data.name}" with ${data.members_cloned} members`, 'success')
      }
    } catch {
      addToast('Could not clone workspace', 'error')
    } finally {
      setCloning(false)
    }
  }

  const hasChanges = settings && (
    name !== settings.name ||
    description !== settings.description ||
    defaultModel !== (settings.default_model || '') ||
    retentionDays !== settings.data_retention_days ||
    maxMembers !== settings.max_members ||
    allowSharing !== settings.allow_sharing
  )

  if (loading) {
    return (
      <PageContainer>
        <Skeleton className="h-8 w-64 mb-4" />
        <Skeleton className="h-64 w-full" />
      </PageContainer>
    )
  }

  if (!settings) {
    return (
      <PageContainer>
        <p className="text-muted-foreground">No workspace selected.</p>
      </PageContainer>
    )
  }

  return (
    <PageContainer>
      <AppRouteHeader>
        <AppRouteHeaderLead>{settings.name} — Settings</AppRouteHeaderLead>
      </AppRouteHeader>

      <KpiGrid className="mb-6">
        <StatCard label="Members" value={String(usage?.members?.total ?? settings.member_count ?? 0)} />
        <StatCard label="Training Jobs" value={String(usage?.training?.total ?? 0)} />
        <StatCard label="Datasets" value={String(usage?.datasets?.total ?? 0)} />
        <StatCard label="Knowledge" value={String(usage?.knowledge?.total ?? 0)} />
      </KpiGrid>

      {/* General */}
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
      </Card>

      {/* Defaults */}
      <Card className="mb-6">
        <CardHeader>
          <CardTitle>Defaults</CardTitle>
          <CardDescription>Default model and policies for this workspace</CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          <div>
            <label className="text-sm font-medium mb-1 block">Default Model</label>
            <Input
              value={defaultModel}
              onChange={e => setDefaultModel(e.target.value)}
              placeholder="e.g. llama-3.1-8b"
              maxLength={200}
            />
            <p className="text-xs text-muted-foreground mt-1">
              Model used when no specific model is selected
            </p>
          </div>

          <div>
            <label className="text-sm font-medium mb-1 block">
              Data Retention: {retentionDays} days
            </label>
            <Slider
              value={[retentionDays]}
              onValueChange={([v]) => setRetentionDays(v)}
              min={7}
              max={365}
              step={1}
            />
            <p className="text-xs text-muted-foreground mt-1">
              Training jobs and data older than this are automatically cleaned up
            </p>
          </div>

          <div>
            <label className="text-sm font-medium mb-1 block">
              Max Members: {maxMembers}
            </label>
            <Slider
              value={[maxMembers]}
              onValueChange={([v]) => setMaxMembers(v)}
              min={2}
              max={500}
              step={1}
            />
            <p className="text-xs text-muted-foreground mt-1">
              Maximum number of members allowed in this workspace
            </p>
          </div>

          <div className="flex items-center justify-between">
            <div>
              <label className="text-sm font-medium block">Allow Sharing</label>
              <p className="text-xs text-muted-foreground">
                Allow members to share datasets and knowledge outside the workspace
              </p>
            </div>
            <Switch
              checked={allowSharing}
              onCheckedChange={setAllowSharing}
            />
          </div>
        </CardContent>
        <CardFooter className="flex justify-end">
          <Button onClick={handleSave} disabled={!hasChanges || saving}>
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
                {health.checks.map((check: any, i: number) => (
                  <div key={i} className="flex justify-between">
                    <span>{check.name}</span>
                    <span>{check.detail}</span>
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {/* Data Retention Cleanup */}
      <Card className="mb-6">
        <CardHeader>
          <CardTitle>Data Retention</CardTitle>
          <CardDescription>
            Clean up training jobs and audit logs older than {retentionDays} days
          </CardDescription>
        </CardHeader>
        <CardContent className="flex gap-2">
          <Button variant="outline" onClick={handleCleanup} disabled={cleaning}>
            {cleaning ? 'Cleaning...' : 'Run Cleanup Now'}
          </Button>
          <Button variant="outline" onClick={handleClone} disabled={cloning}>
            {cloning ? 'Cloning...' : 'Clone Workspace'}
          </Button>
        </CardContent>
      </Card>

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
                <AlertDialogTitle>Delete workspace &ldquo;{settings.name}&rdquo;?</AlertDialogTitle>
                <AlertDialogDescription>
                  This will permanently delete the workspace, all members, training jobs, datasets,
                  and knowledge. Type the workspace name to confirm.
                </AlertDialogDescription>
              </AlertDialogHeader>
              <Input
                value={deleteConfirm}
                onChange={e => setDeleteConfirm(e.target.value)}
                placeholder={settings.name}
                className="mt-2"
              />
              <AlertDialogFooter>
                <AlertDialogCancel onClick={() => setDeleteConfirm('')}>Cancel</AlertDialogCancel>
                <AlertDialogAction
                  onClick={handleDelete}
                  disabled={deleteConfirm !== settings.name || deleting}
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
