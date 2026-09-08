'use client'

import { useState, useEffect, useCallback } from 'react'
import { Card, CardHeader, CardTitle, CardContent, Button, Input, Skeleton } from '@sloughgpt/strui'
import { PageContainer } from '@/components/PageContainer'
import { AppRouteHeader, AppRouteHeaderLead } from '@/components/AppRouteHeader'
import { apiGet, apiPut } from '@/lib/http-client'
import { useAuthStore } from '@/lib/auth'
import { useToastStore } from '@/lib/toast-store'
import { logger } from '@/lib/dev-log'

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
}

export default function WorkspaceSettingsPage() {
  const [settings, setSettings] = useState<WorkspaceSettings | null>(null)
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [form, setForm] = useState({
    name: '',
    description: '',
    default_model: '',
    data_retention_days: 90,
    max_members: 50,
    allow_sharing: true,
  })
  const { currentWorkspace } = useAuthStore()
  const addToast = useToastStore(s => s.addToast)

  const fetchSettings = useCallback(async () => {
    if (!currentWorkspace?.id) {
      setLoading(false)
      return
    }
    try {
      const res = await apiGet<{ data: WorkspaceSettings }>(`/workspaces/${currentWorkspace.id}/settings`)
      const data = res?.data
      if (data) {
        setSettings(data)
        setForm({
          name: data.name,
          description: data.description,
          default_model: data.default_model,
          data_retention_days: data.data_retention_days,
          max_members: data.max_members,
          allow_sharing: data.allow_sharing,
        })
      }
    } catch {
      logger.warning('Could not fetch workspace settings')
    } finally {
      setLoading(false)
    }
  }, [currentWorkspace?.id])

  useEffect(() => { fetchSettings() }, [fetchSettings])

  const handleSave = async () => {
    if (!currentWorkspace?.id) return
    setSaving(true)
    try {
      await apiPut(`/workspaces/${currentWorkspace.id}/settings`, form)
      addToast('Settings saved', 'success')
      await fetchSettings()
    } catch {
      addToast('Could not save settings', 'error')
    } finally {
      setSaving(false)
    }
  }

  const hasChanges = settings && (
    form.name !== settings.name ||
    form.description !== settings.description ||
    form.default_model !== settings.default_model ||
    form.data_retention_days !== settings.data_retention_days ||
    form.max_members !== settings.max_members ||
    form.allow_sharing !== settings.allow_sharing
  )

  if (loading) {
    return (
      <PageContainer>
        <AppRouteHeader>
          <AppRouteHeaderLead>Workspace Settings</AppRouteHeaderLead>
        </AppRouteHeader>
        <div className="space-y-4">
          <Skeleton className="h-40" />
          <Skeleton className="h-60" />
        </div>
      </PageContainer>
    )
  }

  if (!settings) {
    return (
      <PageContainer>
        <AppRouteHeader>
          <AppRouteHeaderLead>Workspace Settings</AppRouteHeaderLead>
        </AppRouteHeader>
        <Card>
          <CardContent className="py-12 text-center text-muted-foreground">
            No workspace selected. Use the workspace switcher to select one.
          </CardContent>
        </Card>
      </PageContainer>
    )
  }

  return (
    <PageContainer>
      <AppRouteHeader>
        <AppRouteHeaderLead>Workspace Settings</AppRouteHeaderLead>
      </AppRouteHeader>

      <div className="space-y-6">
        {/* General */}
        <Card>
          <CardHeader>
            <CardTitle>General</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <label className="text-sm font-medium text-foreground">Name</label>
              <Input
                value={form.name}
                onChange={e => setForm(f => ({ ...f, name: e.target.value }))}
                placeholder="Workspace name"
              />
            </div>
            <div className="space-y-2">
              <label className="text-sm font-medium text-foreground">Description</label>
              <textarea
                value={form.description}
                onChange={e => setForm(f => ({ ...f, description: e.target.value }))}
                placeholder="Describe this workspace"
                rows={3}
                className="w-full rounded-lg border border-border bg-background px-3 py-2 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring"
              />
            </div>
          </CardContent>
        </Card>

        {/* Defaults */}
        <Card>
          <CardHeader>
            <CardTitle>Defaults</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <label className="text-sm font-medium text-foreground">Default Model</label>
              <Input
                value={form.default_model}
                onChange={e => setForm(f => ({ ...f, default_model: e.target.value }))}
                placeholder="e.g. llama-3.2-3b"
              />
              <p className="text-xs text-muted-foreground">
                Model used by default for new conversations in this workspace
              </p>
            </div>
          </CardContent>
        </Card>

        {/* Limits */}
        <Card>
          <CardHeader>
            <CardTitle>Limits</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
              <div className="space-y-2">
                <label className="text-sm font-medium text-foreground">Data Retention (days)</label>
                <Input
                  type="number"
                  min={7}
                  max={365}
                  value={form.data_retention_days}
                  onChange={e => setForm(f => ({ ...f, data_retention_days: parseInt(e.target.value) || 90 }))}
                />
                <p className="text-xs text-muted-foreground">
                  Training jobs and datasets older than this are auto-archived
                </p>
              </div>
              <div className="space-y-2">
                <label className="text-sm font-medium text-foreground">Max Members</label>
                <Input
                  type="number"
                  min={2}
                  max={500}
                  value={form.max_members}
                  onChange={e => setForm(f => ({ ...f, max_members: parseInt(e.target.value) || 50 }))}
                />
                <p className="text-xs text-muted-foreground">
                  Maximum number of members in this workspace
                </p>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Sharing */}
        <Card>
          <CardHeader>
            <CardTitle>Sharing</CardTitle>
          </CardHeader>
          <CardContent>
            <label className="flex items-center gap-3 cursor-pointer">
              <input
                type="checkbox"
                checked={form.allow_sharing}
                onChange={e => setForm(f => ({ ...f, allow_sharing: e.target.checked }))}
                className="h-4 w-4 rounded border-border"
              />
              <div>
                <span className="text-sm font-medium text-foreground">Allow data sharing</span>
                <p className="text-xs text-muted-foreground">
                  Allow members to share datasets and knowledge with other workspaces
                </p>
              </div>
            </label>
          </CardContent>
        </Card>

        {/* Save */}
        <div className="flex items-center justify-between rounded-lg border border-border bg-card p-4">
          <div className="text-sm text-muted-foreground">
            {hasChanges ? 'You have unsaved changes' : 'No changes'}
          </div>
          <Button
            onClick={handleSave}
            disabled={!hasChanges || saving}
          >
            {saving ? 'Saving...' : 'Save Settings'}
          </Button>
        </div>
      </div>
    </PageContainer>
  )
}
