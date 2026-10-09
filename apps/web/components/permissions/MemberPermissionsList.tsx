import { useState } from 'react'
import { Card, CardHeader, CardTitle, CardContent } from '@sloughgpt/strui'

const ROLE_COLORS: Record<string, string> = {
  owner: 'bg-warning/15 text-warning dark:bg-warning/10 dark:text-warning',
  admin: 'bg-info/15 text-info dark:bg-info/10 dark:text-info',
  user: 'bg-success/15 text-success dark:bg-success/10 dark:text-success',
  viewer: 'bg-muted text-muted-foreground',
}

export interface MemberPermission {
  user_id: string
  username: string
  role: string
  permissions: string[]
}

export interface MemberPermissionsListProps {
  members: MemberPermission[]
}

export function MemberPermissionsList({ members }: MemberPermissionsListProps) {
  const [selectedId, setSelectedId] = useState<string | null>(null)

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-xs">Member Permissions ({members.length})</CardTitle>
      </CardHeader>
      <CardContent className="space-y-1">
        {members.length === 0 ? (
          <p className="text-[10px] text-muted-foreground text-center py-4">No members</p>
        ) : (
          members.map(m => (
            <div
              key={m.user_id}
              className="flex items-center justify-between px-3 py-2 rounded-md text-[10px] hover:bg-muted/50 border border-transparent hover:border-border/50 cursor-pointer"
              onClick={() => setSelectedId(selectedId === m.user_id ? null : m.user_id)}
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

        {selectedId && members.find(m => m.user_id === selectedId) && (
          <div className="mt-2 p-3 rounded-md bg-muted/30 border">
            <p className="text-[10px] font-medium mb-2">
              {members.find(m => m.user_id === selectedId)?.username} permissions:
            </p>
            <div className="flex flex-wrap gap-1">
              {members.find(m => m.user_id === selectedId)?.permissions.map(p => (
                <span key={p} className="px-1.5 py-0.5 rounded bg-background text-[9px] border">
                  {p}
                </span>
              ))}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
