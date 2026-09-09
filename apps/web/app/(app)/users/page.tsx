'use client'

import { useState, useEffect, useCallback } from 'react'
import { Card, CardHeader, CardTitle, CardContent, Button, Input, StatCard, KpiGrid, Skeleton } from '@sloughgpt/strui'
import { IconPlus, IconTrash, IconRefresh } from '@/components/icons/NavIcons'
import { PageContainer } from '@/components/PageContainer'
import { AppRouteHeader, AppRouteHeaderLead } from '@/components/AppRouteHeader'
import { apiGet, apiPost, apiPut, apiDelete } from '@/lib/http-client'
import { useToastStore } from '@/lib/toast-store'
import { logger } from '@/lib/dev-log'
import { useRefreshShortcut } from '@/hooks/useRefreshShortcut'

interface User {
  id: string
  username: string
  email: string
  role: string
  status: string
  display_name: string
  tenant_id: string
  created_at: string
  last_login_at: string
}

interface UsersResponse {
  data: User[]
  meta: { total: number }
}

export default function UsersAdminPage() {
  const [users, setUsers] = useState<User[]>([])
  const [loading, setLoading] = useState(true)
  const [creating, setCreating] = useState(false)
  const [newUsername, setNewUsername] = useState('')
  const [newEmail, setNewEmail] = useState('')
  const [newPassword, setNewPassword] = useState('')
  const [newRole, setNewRole] = useState('user')
  const [editingUser, setEditingUser] = useState<string | null>(null)
  const [editEmail, setEditEmail] = useState('')
  const [editRole, setEditRole] = useState('')
  const [editStatus, setEditStatus] = useState('')
  const addToast = useToastStore(s => s.addToast)

  const fetchUsers = useCallback(async () => {
    try {
      const res = await apiGet<UsersResponse>('/users')
      setUsers(res?.data ?? [])
    } catch {
      logger.warning('Could not fetch users')
    } finally {
      setLoading(false)
    }
  }, [])

  useRefreshShortcut(fetchUsers)

  const createUser = async () => {
    if (!newUsername.trim() || !newEmail.trim() || !newPassword.trim()) return
    setCreating(true)
    try {
      await apiPost('/users', {
        username: newUsername,
        email: newEmail,
        password: newPassword,
        role: newRole,
      })
      setNewUsername('')
      setNewEmail('')
      setNewPassword('')
      setNewRole('user')
      await fetchUsers()
      addToast('User created', 'success')
    } catch {
      addToast('Could not create user', 'error')
    } finally {
      setCreating(false)
    }
  }

  const deleteUser = async (id: string, username: string) => {
    if (!confirm(`Delete user "${username}"? This cannot be undone.`)) return
    try {
      await apiDelete(`/users/${id}`)
      await fetchUsers()
      addToast('User deleted', 'success')
    } catch {
      addToast('Could not delete user', 'error')
    }
  }

  const startEdit = (user: User) => {
    setEditingUser(user.id)
    setEditEmail(user.email)
    setEditRole(user.role)
    setEditStatus(user.status)
  }

  const saveEdit = async () => {
    if (!editingUser) return
    try {
      await apiPut(`/users/${editingUser}`, {
        email: editEmail,
        role: editRole,
        status: editStatus,
      })
      setEditingUser(null)
      await fetchUsers()
      addToast('User updated', 'success')
    } catch {
      addToast('Could not update user', 'error')
    }
  }

  useEffect(() => { fetchUsers() }, [fetchUsers])

  const activeUsers = users.filter(u => u.status === 'active')
  const adminUsers = users.filter(u => u.role === 'admin' || u.role === 'owner')

  if (loading) {
    return (
      <PageContainer title="Users" subtitle="Manage user accounts" loadingCards={3}>
        <KpiGrid>
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
        left={<AppRouteHeaderLead title="Users" subtitle="Manage user accounts" />}
        right={
          <Button size="sm" variant="ghost" className="h-6 text-[10px]" onClick={() => fetchUsers()}>
            <IconRefresh className="h-3 w-3 mr-1" /> Refresh
          </Button>
        }
      />

      <div className="space-y-4">
        <KpiGrid>
          <StatCard label="Total Users" value={users.length} />
          <StatCard label="Active" value={activeUsers.length} />
          <StatCard label="Admins" value={adminUsers.length} />
        </KpiGrid>

        {/* Create user */}
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-xs">Create User</CardTitle>
          </CardHeader>
          <CardContent className="space-y-2">
            <div className="grid grid-cols-2 gap-1.5">
              <Input
                value={newUsername}
                onChange={e => setNewUsername(e.target.value)}
                placeholder="Username"
                className="h-6 text-[10px]"
              />
              <Input
                value={newEmail}
                onChange={e => setNewEmail(e.target.value)}
                placeholder="Email"
                type="email"
                className="h-6 text-[10px]"
              />
              <Input
                value={newPassword}
                onChange={e => setNewPassword(e.target.value)}
                placeholder="Password (min 8 chars)"
                type="password"
                className="h-6 text-[10px]"
              />
              <select
                value={newRole}
                onChange={e => setNewRole(e.target.value)}
                className="h-6 text-[10px] rounded-md border border-border bg-background px-2"
              >
                <option value="viewer">Viewer</option>
                <option value="user">User</option>
                <option value="admin">Admin</option>
              </select>
            </div>
            <Button size="sm" className="h-6 text-[10px]" onClick={createUser} disabled={creating || !newUsername.trim() || !newEmail.trim() || !newPassword.trim()}>
              {creating ? 'Creating...' : 'Create User'}
            </Button>
          </CardContent>
        </Card>

        {/* User list */}
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-xs">Users</CardTitle>
          </CardHeader>
          <CardContent className="space-y-0.5">
            {users.length === 0 ? (
              <p className="text-xs text-muted-foreground py-4 text-center">No users yet</p>
            ) : (
              users.map(u => (
                <div key={u.id} className="px-3 py-2 rounded-md hover:bg-muted/50 transition-colors">
                  {editingUser === u.id ? (
                    /* Edit mode */
                    <div className="space-y-2">
                      <div className="grid grid-cols-3 gap-1.5">
                        <Input
                          value={editEmail}
                          onChange={e => setEditEmail(e.target.value)}
                          placeholder="Email"
                          className="h-6 text-[10px]"
                        />
                        <select
                          value={editRole}
                          onChange={e => setEditRole(e.target.value)}
                          className="h-6 text-[10px] rounded-md border border-border bg-background px-2"
                        >
                          <option value="viewer">Viewer</option>
                          <option value="user">User</option>
                          <option value="admin">Admin</option>
                          <option value="owner">Owner</option>
                        </select>
                        <select
                          value={editStatus}
                          onChange={e => setEditStatus(e.target.value)}
                          className="h-6 text-[10px] rounded-md border border-border bg-background px-2"
                        >
                          <option value="active">Active</option>
                          <option value="inactive">Inactive</option>
                          <option value="suspended">Suspended</option>
                        </select>
                      </div>
                      <div className="flex gap-1.5">
                        <Button size="sm" className="h-5 text-[10px]" onClick={saveEdit}>Save</Button>
                        <Button size="sm" variant="ghost" className="h-5 text-[10px]" onClick={() => setEditingUser(null)}>Cancel</Button>
                      </div>
                    </div>
                  ) : (
                    /* View mode */
                    <div className="flex items-center justify-between">
                      <div className="min-w-0 flex-1">
                        <div className="flex items-center gap-2">
                          <span className="font-medium text-xs">{u.username}</span>
                          <span className={`text-[10px] px-1.5 py-0.5 rounded-full ${
                            u.role === 'owner' ? 'bg-purple-100 text-purple-700 dark:bg-purple-900/30 dark:text-purple-400' :
                            u.role === 'admin' ? 'bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400' :
                            'bg-muted text-muted-foreground'
                          }`}>{u.role}</span>
                          <span className={`text-[10px] px-1.5 py-0.5 rounded-full ${
                            u.status === 'active' ? 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400' :
                            'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400'
                          }`}>{u.status}</span>
                        </div>
                        <div className="text-[10px] text-muted-foreground mt-0.5">
                          {u.email} · {u.display_name || '—'} · Last login: {u.last_login_at ? new Date(u.last_login_at).toLocaleDateString() : 'Never'}
                        </div>
                      </div>
                      <div className="flex items-center gap-1 shrink-0">
                        <Button size="sm" variant="ghost" className="h-5 text-[10px]" onClick={() => startEdit(u)}>Edit</Button>
                        <Button
                          size="sm"
                          variant="ghost"
                          className="h-5 w-5 p-0 text-muted-foreground hover:text-destructive"
                          onClick={() => deleteUser(u.id, u.username)}
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
      </div>
    </div>
  )
}
