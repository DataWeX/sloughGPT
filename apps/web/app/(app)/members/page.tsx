'use client'

import { useState, useEffect, useCallback } from 'react'
import { Card, CardHeader, CardTitle, CardContent, Button, Input, StatCard, KpiGrid, Skeleton } from '@sloughgpt/strui'
import { IconPlus, IconTrash } from '@/components/icons/NavIcons'
import { PageContainer } from '@/components/PageContainer'
import { AppRouteHeader, AppRouteHeaderLead } from '@/components/AppRouteHeader'
import { apiGet, apiPost, apiDelete } from '@/lib/http-client'
import { useAuthStore } from '@/lib/auth'
import { useToastStore } from '@/lib/toast-store'
import { logger } from '@/lib/dev-log'

interface Workspace {
  id: string
  name: string
  member_count: number
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
  joined_at?: string
}

interface MembersResponse {
  data: Member[]
}

export default function MembersPage() {
  const [workspaces, setWorkspaces] = useState<Workspace[]>([])
  const [selectedWs, setSelectedWs] = useState<string | null>(null)
  const [members, setMembers] = useState<Member[]>([])
  const [loading, setLoading] = useState(true)
  const [loadingMembers, setLoadingMembers] = useState(false)
  const [addMemberId, setAddMemberId] = useState('')
  const [addMemberRole, setAddMemberRole] = useState('member')
  const [searchQuery, setSearchQuery] = useState('')
  const addToast = useToastStore(s => s.addToast)
  const { currentWorkspace, switchWorkspace } = useAuthStore()

  const fetchWorkspaces = useCallback(async () => {
    try {
      const res = await apiGet<WorkspacesResponse>('/workspaces')
      setWorkspaces(res?.data ?? [])
      // Auto-select current workspace
      if (currentWorkspace?.id) {
        setSelectedWs(currentWorkspace.id)
      } else if ((res?.data ?? []).length > 0) {
        setSelectedWs(res!.data[0].id)
      }
    } catch {
      logger.warning('Could not fetch workspaces')
    } finally {
      setLoading(false)
    }
  }, [currentWorkspace?.id])

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
    if (!confirm('Remove this member from the workspace?')) return
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

  const filteredMembers = members.filter(m =>
    m.username.toLowerCase().includes(searchQuery.toLowerCase()) ||
    m.email.toLowerCase().includes(searchQuery.toLowerCase()) ||
    m.role.toLowerCase().includes(searchQuery.toLowerCase())
  )

  const roleCounts = members.reduce((acc, m) => {
    acc[m.role] = (acc[m.role] || 0) + 1
    return acc
  }, {} as Record<string, number>)

  if (loading) {
    return (
      <PageContainer title="Members" subtitle="Manage workspace members" loadingCards={3}>
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
        left={<AppRouteHeaderLead title="Members" subtitle="Manage workspace members" />}
      />

      <div className="space-y-4">
        {/* Workspace selector */}
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-xs">Select Workspace</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex gap-2">
              {workspaces.map(ws => (
                <button
                  key={ws.id}
                  onClick={() => setSelectedWs(ws.id)}
                  className={`px-3 py-1.5 rounded-md text-xs transition-colors ${
                    selectedWs === ws.id
                      ? 'bg-primary text-primary-foreground'
                      : 'bg-muted hover:bg-muted/80 text-muted-foreground'
                  }`}
                >
                  {ws.name}
                  <span className="ml-1.5 text-[10px] opacity-70">({ws.member_count})</span>
                </button>
              ))}
            </div>
          </CardContent>
        </Card>

        {selectedWs && (
          <>
            {/* KPI grid */}
            <KpiGrid>
              <StatCard label="Total Members" value={members.length} />
              <StatCard label="Admins" value={roleCounts['admin'] ?? 0} />
              <StatCard label="Members" value={roleCounts['member'] ?? 0} />
              <StatCard label="Viewers" value={roleCounts['viewer'] ?? 0} />
            </KpiGrid>

            {/* Add member form */}
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-xs">Add Member</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="flex gap-2">
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
                    <IconPlus className="h-3 w-3 mr-1" />
                    Add
                  </Button>
                </div>
              </CardContent>
            </Card>

            {/* Members list */}
            <Card>
              <CardHeader className="pb-2">
                <div className="flex items-center justify-between">
                  <CardTitle className="text-xs">Members</CardTitle>
                  <Input
                    value={searchQuery}
                    onChange={e => setSearchQuery(e.target.value)}
                    placeholder="Search..."
                    className="w-48 h-6 text-[10px]"
                  />
                </div>
              </CardHeader>
              <CardContent className="space-y-1">
                {loadingMembers ? (
                  <div className="h-32 animate-pulse bg-muted/50 rounded" />
                ) : filteredMembers.length === 0 ? (
                  <p className="text-[10px] text-muted-foreground py-4 text-center">
                    {searchQuery ? 'No members match search' : 'No members in this workspace'}
                  </p>
                ) : (
                  filteredMembers.map(m => (
                    <div
                      key={m.user_id}
                      className="flex items-center justify-between px-3 py-2 rounded-md text-[10px] hover:bg-muted/50 border border-transparent hover:border-border/50"
                    >
                      <div className="min-w-0 flex-1">
                        <div className="font-medium">{m.username}</div>
                        <div className="text-muted-foreground">{m.email}</div>
                      </div>
                      <div className="flex items-center gap-3 shrink-0">
                        <span className={`px-2 py-0.5 rounded-full text-[9px] font-medium ${
                          m.role === 'admin' ? 'bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400' :
                          m.role === 'member' ? 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400' :
                          'bg-muted text-muted-foreground'
                        }`}>
                          {m.role}
                        </span>
                        {m.joined_at && (
                          <span className="text-muted-foreground whitespace-nowrap">
                            Joined {new Date(m.joined_at).toLocaleDateString()}
                          </span>
                        )}
                        <Button
                          size="sm"
                          variant="ghost"
                          className="h-5 w-5 p-0 text-muted-foreground hover:text-destructive"
                          onClick={() => removeMember(m.user_id)}
                        >
                          <IconTrash className="h-3 w-3" />
                        </Button>
                      </div>
                    </div>
                  ))
                )}
              </CardContent>
            </Card>
          </>
        )}
      </div>
    </div>
  )
}
