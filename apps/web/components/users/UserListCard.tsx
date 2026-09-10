'use client'

import { Card, CardHeader, CardTitle, CardContent, Button } from '@sloughgpt/strui'

export interface User {
  id: string
  username: string
  email: string
  role: string
  status: string
  display_name: string
  last_login_at: string
}

export interface UserListCardProps {
  users: User[]
  onEdit: (user: User) => void
  onDelete: (id: string, username: string) => void
}

export function UserListCard({ users, onEdit, onDelete }: UserListCardProps) {
  return (
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
              <div className="flex items-center justify-between">
                <div className="min-w-0 flex-1">
                  <div className="flex items-center gap-2">
                    <span className="font-medium text-xs">{u.username}</span>
                    <span className="text-[10px] px-1.5 py-0.5 rounded-full bg-muted text-muted-foreground">{u.role}</span>
                    <span className={`text-[10px] px-1.5 py-0.5 rounded-full ${
                      u.status === 'active' ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700'
                    }`}>{u.status}</span>
                  </div>
                  <div className="text-[10px] text-muted-foreground mt-0.5">
                    {u.email} · Last login: {u.last_login_at ? new Date(u.last_login_at).toLocaleDateString() : 'Never'}
                  </div>
                </div>
                <div className="flex items-center gap-1 shrink-0">
                  <Button size="sm" variant="ghost" className="h-5 text-[10px]" onClick={() => onEdit(u)}>Edit</Button>
                  <Button
                    size="sm"
                    variant="ghost"
                    className="h-5 w-5 p-0 text-muted-foreground hover:text-destructive"
                    onClick={() => onDelete(u.id, u.username)}
                  >
                    ×
                  </Button>
                </div>
              </div>
            </div>
          ))
        )}
      </CardContent>
    </Card>
  )
}
