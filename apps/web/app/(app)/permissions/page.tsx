'use client'

import { useState, useEffect, useCallback } from 'react'
import { Card, CardHeader, CardTitle, CardContent, Skeleton } from '@sloughgpt/strui'
import { PageContainer } from '@/components/PageContainer'
import { AppRouteHeader, AppRouteHeaderLead } from '@/components/AppRouteHeader'
import { apiGet } from '@/lib/http-client'
import { useAuthStore } from '@/lib/auth'
import { useToastStore } from '@/lib/toast-store'
import { useRefreshShortcut } from '@/hooks/useRefreshShortcut'

interface PermissionsResponse {
  data: {
    roles: Record<string, { name: string; permissions: string[] }>
    all_permissions: Record<string, string[]>
    member_permissions: { user_id: string; username: string; role: string; permissions: string[] }[]
  }
}

const ROLE_COLORS: Record<string, string> = {
  owner: 'bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-400',
  admin: 'bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400',
  user: 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400',
  viewer: 'bg-muted text-muted-foreground',
}

const CATEGORY_LABELS: Record<string, string> = {
  model: 'Models',
  train: 'Training',
  chat: 'Chat',
  dataset: 'Datasets',
  knowledge: 'Knowledge',
  user: 'Users',
  tenant: 'Tenants',
  workspace: 'Workspaces',
  system: 'System',
}

export default function PermissionsPage() {
  const { currentWorkspace } = useAuthStore()
  const addToast = useToastStore(s => s.addToast)
  const [roles, setRoles] = useState<Record<string, { name: string; permissions: string[] }>>({})
  const [allPermissions, setAllPermissions] = useState<Record<string, string[]>>({})
  const [memberPerms, setMemberPerms] = useState<{ user_id: string; username: string; role: string; permissions: string[] }[]>([])
  const [loading, setLoading] = useState(true)
  const [selectedRole, setSelectedRole] = useState<string | null>(null)

  const fetchPermissions = useCallback(async () => {
    if (!currentWorkspace?.id) {
      setLoading(false)
      return
    }
    try {
      const res = await apiGet<PermissionsResponse>(`/workspaces/${currentWorkspace.id}/permissions`)
      if (res?.data) {
        setRoles(res.data.roles)
        setAllPermissions(res.data.all_permissions)
        setMemberPerms(res.data.member_permissions)
      }
    } catch {
      addToast('Could not load permissions', 'error')
    } finally {
      setLoading(false)
    }
  }, [currentWorkspace?.id, addToast])

  useRefreshShortcut(fetchPermissions)

  useEffect(() => { fetchPermissions() }, [fetchPermissions])

  if (loading) {
    return (
      <PageContainer title="Permissions">
        <Skeleton className="h-8 w-64 mb-4" />
        <Skeleton className="h-64 w-full" />
      </PageContainer>
    )
  }

  return (
    <PageContainer title="Permissions">
      <AppRouteHeader left={<AppRouteHeaderLead title="Permissions" />} />

      {/* Role-Permission Matrix */}
      <Card className="mb-6">
        <CardHeader className="pb-2">
          <CardTitle className="text-xs">Role Permissions</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="overflow-x-auto">
            <table className="w-full text-[10px]">
              <thead>
                <tr className="border-b">
                  <th className="text-left py-2 pr-4 font-medium">Permission</th>
                  {Object.keys(roles).map(role => (
                    <th key={role} className="text-center py-2 px-2 font-medium">
                      <span className={`px-1.5 py-0.5 rounded text-[9px] font-medium ${ROLE_COLORS[role] || 'bg-muted'}`}>
                        {role}
                      </span>
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {Object.entries(allPermissions).map(([category, perms]) => (
                  <>
                    <tr key={`cat-${category}`} className="border-b bg-muted/30">
                      <td colSpan={Object.keys(roles).length + 1} className="py-1.5 font-medium text-muted-foreground">
                        {CATEGORY_LABELS[category] || category}
                      </td>
                    </tr>
                    {perms.map(perm => (
                      <tr key={perm} className="border-b border-border/50">
                        <td className="py-1.5 pr-4 text-muted-foreground">{perm}</td>
                        {Object.entries(roles).map(([role, { permissions }]) => (
                          <td key={role} className="text-center py-1.5 px-2">
                            {permissions.includes(perm) ? (
                              <span className="text-green-600">✓</span>
                            ) : (
                              <span className="text-muted-foreground/40">—</span>
                            )}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </>
                ))}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>

      {/* Member Permissions */}
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-xs">Member Permissions ({memberPerms.length})</CardTitle>
        </CardHeader>
        <CardContent className="space-y-1">
          {memberPerms.length === 0 ? (
            <p className="text-[10px] text-muted-foreground text-center py-4">No members</p>
          ) : (
            memberPerms.map(m => (
              <div
                key={m.user_id}
                className="flex items-center justify-between px-3 py-2 rounded-md text-[10px] hover:bg-muted/50 border border-transparent hover:border-border/50 cursor-pointer"
                onClick={() => setSelectedRole(selectedRole === m.user_id ? null : m.user_id)}
              >
                <div className="min-w-0 flex-1">
                  <div className="font-medium">{m.username || m.user_id}</div>
                </div>
                <span className={`px-1.5 py-0.5 rounded text-[9px] font-medium ${ROLE_COLORS[m.role] || 'bg-muted'}`}>
                  {m.role}
                </span>
              </div>
            ))
          )}

          {/* Expanded permission details */}
          {selectedRole && memberPerms.find(m => m.user_id === selectedRole) && (
            <div className="mt-2 p-3 rounded-md bg-muted/30 border">
              <p className="text-[10px] font-medium mb-2">
                {memberPerms.find(m => m.user_id === selectedRole)?.username} permissions:
              </p>
              <div className="flex flex-wrap gap-1">
                {memberPerms.find(m => m.user_id === selectedRole)?.permissions.map(p => (
                  <span key={p} className="px-1.5 py-0.5 rounded bg-background text-[9px] border">
                    {p}
                  </span>
                ))}
              </div>
            </div>
          )}
        </CardContent>
      </Card>
    </PageContainer>
  )
}
