'use client'

import { useState } from 'react'
import { Card, CardHeader, CardTitle, CardContent, Button, Input } from '@sloughgpt/strui'

interface Member {
  user_id: string
  username: string
  email: string
  role: string
}

interface WorkspaceMembersCardProps {
  workspaceName: string
  members: Member[]
  loading?: boolean
  onAddMember: (userId: string, role: string) => void
  onRemoveMember: (userId: string) => void
}

export function WorkspaceMembersCard({
  workspaceName,
  members,
  loading,
  onAddMember,
  onRemoveMember,
}: WorkspaceMembersCardProps) {
  const [userId, setUserId] = useState('')
  const [role, setRole] = useState('member')

  const handleAdd = () => {
    if (!userId.trim()) return
    onAddMember(userId.trim(), role)
    setUserId('')
  }

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-xs">Members — {workspaceName}</CardTitle>
      </CardHeader>
      <CardContent className="space-y-2">
        <div className="flex gap-1.5">
          <Input
            value={userId}
            onChange={(e) => setUserId(e.target.value)}
            placeholder="User ID"
            className="flex-1 h-6 text-[10px]"
          />
          <select
            value={role}
            onChange={(e) => setRole(e.target.value)}
            className="h-6 text-[10px] rounded-md border border-border bg-background px-2"
          >
            <option value="viewer">Viewer</option>
            <option value="member">Member</option>
            <option value="admin">Admin</option>
          </select>
          <Button size="sm" className="h-6 text-[10px]" onClick={handleAdd} disabled={!userId.trim()}>
            Add
          </Button>
        </div>

        {loading ? (
          <div className="h-16 animate-pulse bg-muted/50 rounded" />
        ) : members.length === 0 ? (
          <p className="text-[10px] text-muted-foreground py-2 text-center">No members</p>
        ) : (
          <div className="space-y-0.5">
            {members.map((m) => (
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
                    onClick={() => onRemoveMember(m.user_id)}
                  >
                    ×
                  </Button>
                </div>
              </div>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  )
}
