'use client'

import { useState, useEffect, useCallback } from 'react'
import { Card, CardHeader, CardTitle, CardContent, Button, Input, StatCard, KpiGrid, Skeleton } from '@sloughgpt/strui'
import { IconPlus, IconTrash } from '@/components/icons/NavIcons'
import { PageContainer } from '@/components/PageContainer'
import { AppRouteHeader, AppRouteHeaderLead } from '@/components/AppRouteHeader'
import { apiGet, apiPost, apiPut, apiDelete } from '@/lib/http-client'
import { useAuthStore } from '@/lib/auth'
import { useToastStore } from '@/lib/toast-store'
import { logger } from '@/lib/dev-log'

interface Workspace {
  id: string
  name: string
  description: string
  tenant_id: string
  role: string
  member_count: number
  created_at?: string
}

interface WorkspacesResponse {
  data: Workspace[]
  meta: { total: number }
}

interface Member {
  user_id: string
  username: string
  email: string
  role: string
}

interface MembersResponse {
  data: Member[]
}

export default function WorkspacesPage() {
  const [workspaces, setWorkspaces] = useState<Workspace[]>([])
  const [loading, setLoading] = useState(true)
  const [newName, setNewName] = useState('')
  const [newDesc, setNewDesc] = useState('')
  const [creating, setCreating] = useState(false)
  const [selectedWs, setSelectedWs] = useState<string | null>(null)
  const [members, setMembers] = useState<Member[]>([])
  const [loadingMembers, setLoadingMembers] = useState(false)
  const [addMemberId, setAddMemberId] = useState('')
  const [addMemberRole, setAddMemberRole] = useState('member')
  const [editingWs, setEditingWs] = useState<string | null>(null)
  const [editName, setEditName] = useState('')
  const [editDesc, setEditDesc] = useState('')
  const addToast = useToastStore(s => s.addToast)
  const { currentWorkspace, switchWorkspace } = useAuthStore()

  const fetchWorkspaces = useCallback(async () => {
    try {
      const res = await apiGet<WorkspacesResponse>('/workspaces')
      setWorkspaces(res?.data ?? [])
    } catch {
      logger.warning('Could not fetch workspaces')
    } finally {
      setLoading(false)
    }
  }, [])

  const createWorkspace = async () => {
    if (!newName.trim()) return
    setCreating(true)
    try {
      await apiPost('/workspaces', { name: newName, description: newDesc })
      setNewName('')
      setNewDesc('')
      await fetchWorkspaces()
      addToast('Workspace created', 'success')
    } catch {
      addToast('Could not create workspace', 'error')
    } finally {
      setCreating(false)
    }
  }

  const deleteWorkspace = async (id: string) => {
    if (!confirm('Delete this workspace? This cannot be undone.')) return
    try {
      await apiDelete(`/workspaces/${id}`)
      if (selectedWs === id) setSelectedWs(null)
      await fetchWorkspaces()
      addToast('Workspace deleted', 'success')
    } catch {
      addToast('Could not delete workspace', 'error')
    }
  }

  const exportWorkspace = async (id: string, name: string) => {
    try {
      const res = await apiGet<{ data: Record<string, unknown> }>(`/workspaces/${id}/export`)
      const data = res?.data
      if (data) {
        const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' })
        const url = URL.createObjectURL(blob)
        const a = document.createElement('a')
        a.href = url
        a.download = `${name.replace(/\s+/g, '_')}_export.json`
        a.click()
        URL.revokeObjectURL(url)
        addToast('Workspace exported', 'success')
      }
    } catch {
      addToast('Could not export workspace', 'error')
    }
  }

  const startEdit = (ws: Workspace) => {
    setEditingWs(ws.id)
    setEditName(ws.name)
    setEditDesc(ws.description)
  }

  const saveEdit = async () => {
    if (!editingWs || !editName.trim()) return
    try {
      await apiPut(`/workspaces/${editingWs}`, { name: editName, description: editDesc })
      setEditingWs(null)
      await fetchWorkspaces()
      addToast('Workspace updated', 'success')
    } catch {
      addToast('Could not update workspace', 'error')
    }
  }

  const importWorkspace = async () => {
    const input = document.createElement('input')
    input.type = 'file'
    input.accept = '.json'
    input.onchange = async (e) => {
      const file = (e.target as HTMLInputElement).files?.[0]
      if (!file) return
      try {
        const text = await file.text()
        const data = JSON.parse(text)
        await apiPost('/workspaces/import', { workspace_data: data })
        await fetchWorkspaces()
        addToast('Workspace imported', 'success')
      } catch {
        addToast('Could not import workspace', 'error')
      }
    }
    input.click()
  }

  const fetchMembers = useCallback(async (wsId: string) => {
    setLoadingMembers(true)
    try {
      const res = await apiGet<MembersResponse>(`/workspaces/${wsId}/members`)
      setMembers(res?.data ?? [])
    } catch {
      logger.warning('Could not fetch members')
    } finally {
      setLoadingMembers(false)
    }
  }, [])

  const addMember = async () => {
    if (!selectedWs || !addMemberId.trim()) return
    try {
      await apiPost(`/workspaces/${selectedWs}/members`, { user_id: addMemberId, role: addMemberRole })
      setAddMemberId('')
      await fetchMembers(selectedWs)
      addToast('Member added', 'success')
    } catch {
      addToast('Could not add member', 'error')
    }
  }

  const removeMember = async (userId: string) => {
    if (!selectedWs) return
    try {
      await apiDelete(`/workspaces/${selectedWs}/members/${userId}`)
      await fetchMembers(selectedWs)
      addToast('Member removed', 'success')
    } catch {
      addToast('Could not remove member', 'error')
    }
  }

  useEffect(() => { fetchWorkspaces() }, [fetchWorkspaces])

  useEffect(() => {
    if (selectedWs) fetchMembers(selectedWs)
  }, [selectedWs, fetchMembers])

  const activeCount = workspaces.length

  if (loading) {
    return (
      <PageContainer title="Workspaces" subtitle="Manage workspaces and members" loadingCards={3}>
        <KpiGrid>
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
        left={<AppRouteHeaderLead title="Workspaces" subtitle="Manage workspaces and members" />}
      />

      <div className="space-y-4">
        <KpiGrid>
          <StatCard label="Workspaces" value={activeCount} />
          <StatCard label="Current" value={currentWorkspace?.name ?? 'None'} />
        </KpiGrid>

        {/* Create workspace */}
        <Card>
          <CardHeader className="pb-2">
            <div className="flex items-center justify-between">
              <CardTitle className="text-xs">Create Workspace</CardTitle>
              <Button size="sm" variant="outline" className="h-6 text-[10px]" onClick={importWorkspace}>
                Import from JSON
              </Button>
            </div>
          </CardHeader>
          <CardContent className="space-y-2">
            <div className="flex gap-1.5">
              <Input
                value={newName}
                onChange={e => setNewName(e.target.value)}
                placeholder="Workspace name"
                className="flex-1 h-6 text-[10px]"
              />
              <Input
                value={newDesc}
                onChange={e => setNewDesc(e.target.value)}
                placeholder="Description (optional)"
                className="flex-1 h-6 text-[10px]"
              />
              <Button size="sm" className="h-6 text-[10px]" onClick={createWorkspace} disabled={creating || !newName.trim()}>
                {creating ? 'Creating...' : 'Create'}
              </Button>
            </div>
          </CardContent>
        </Card>

        {/* Workspace list */}
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-xs">Workspaces</CardTitle>
          </CardHeader>
          <CardContent className="space-y-1">
            {workspaces.length === 0 ? (
              <p className="text-xs text-muted-foreground py-4 text-center">No workspaces yet</p>
            ) : (
              workspaces.map(ws => (
                <div
                  key={ws.id}
                  className={`px-3 py-2 rounded-md text-xs transition-colors ${
                    selectedWs === ws.id ? 'bg-primary/[0.08] border border-primary/40' : 'hover:bg-muted/50 border border-transparent'
                  }`}
                >
                  {editingWs === ws.id ? (
                    /* Edit mode */
                    <div className="space-y-2">
                      <div className="flex gap-1.5">
                        <Input
                          value={editName}
                          onChange={e => setEditName(e.target.value)}
                          placeholder="Name"
                          className="flex-1 h-6 text-[10px]"
                        />
                        <Input
                          value={editDesc}
                          onChange={e => setEditDesc(e.target.value)}
                          placeholder="Description"
                          className="flex-1 h-6 text-[10px]"
                        />
                      </div>
                      <div className="flex gap-1.5">
                        <Button size="sm" className="h-5 text-[10px]" onClick={saveEdit}>Save</Button>
                        <Button size="sm" variant="ghost" className="h-5 text-[10px]" onClick={() => setEditingWs(null)}>Cancel</Button>
                      </div>
                    </div>
                  ) : (
                    /* View mode */
                    <div className="flex items-center justify-between cursor-pointer" onClick={() => setSelectedWs(selectedWs === ws.id ? null : ws.id)}>
                      <div className="min-w-0 flex-1">
                        <div className="font-medium truncate">{ws.name}</div>
                        {ws.description && <div className="text-[10px] text-muted-foreground truncate">{ws.description}</div>}
                      </div>
                      <div className="flex items-center gap-2 shrink-0">
                        <span className="text-[10px] text-muted-foreground">{ws.member_count ?? 0} members</span>
                        {ws.id === currentWorkspace?.id && (
                          <span className="text-[10px] px-1.5 py-0.5 rounded-full bg-primary/10 text-primary">Active</span>
                        )}
                        <Button
                          size="sm"
                          variant="ghost"
                          className="h-5 text-[10px]"
                          onClick={e => { e.stopPropagation(); startEdit(ws) }}
                        >
                          Edit
                        </Button>
                        <Button
                          size="sm"
                          variant="ghost"
                          className="h-5 text-[10px]"
                          onClick={e => { e.stopPropagation(); exportWorkspace(ws.id, ws.name) }}
                        >
                          Export
                        </Button>
                        <Button
                          size="sm"
                          variant="ghost"
                          className="h-5 w-5 p-0 text-muted-foreground hover:text-destructive"
                          onClick={e => { e.stopPropagation(); deleteWorkspace(ws.id) }}
                        >
                          <IconTrash className="h-3 w-3" />
                        </Button>
                      </div>
                    </div>
                  )}
                </div>
              ))
            )}
          </CardContent>
        </Card>

        {/* Members panel */}
        {selectedWs && (
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-xs">
                Members — {workspaces.find(w => w.id === selectedWs)?.name}
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-2">
              <div className="flex gap-1.5">
                <Input
                  value={addMemberId}
                  onChange={e => setAddMemberId(e.target.value)}
                  placeholder="User ID"
                  className="flex-1 h-6 text-[10px]"
                />
                <select
                  value={addMemberRole}
                  onChange={e => setAddMemberRole(e.target.value)}
                  className="h-6 text-[10px] rounded-md border border-border bg-background px-2"
                >
                  <option value="viewer">Viewer</option>
                  <option value="member">Member</option>
                  <option value="admin">Admin</option>
                </select>
                <Button size="sm" className="h-6 text-[10px]" onClick={addMember} disabled={!addMemberId.trim()}>
                  Add
                </Button>
              </div>

              {loadingMembers ? (
                <div className="h-16 animate-pulse bg-muted/50 rounded" />
              ) : members.length === 0 ? (
                <p className="text-[10px] text-muted-foreground py-2 text-center">No members</p>
              ) : (
                <div className="space-y-0.5">
                  {members.map(m => (
                    <div key={m.user_id} className="flex items-center justify-between px-2 py-1.5 rounded text-[10px] hover:bg-muted/50">
                      <div className="min-w-0 flex-1">
                        <span className="font-medium">{m.username}</span>
                        <span className="text-muted-foreground ml-1.5">{m.email}</span>
                      </div>
                      <div className="flex items-center gap-2 shrink-0">
                        <span className="text-muted-foreground">{m.role}</span>
                        <Button
                          size="sm"
                          variant="ghost"
                          className="h-4 w-4 p-0 text-muted-foreground hover:text-destructive"
                          onClick={() => removeMember(m.user_id)}
                        >
                          <IconTrash className="h-2.5 w-2.5" />
                        </Button>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  )
}
