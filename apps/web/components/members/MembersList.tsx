'use client'

import { useState } from 'react'
import { Card, CardHeader, CardTitle, CardContent, Button, Input } from '@sloughgpt/strui'

interface Member {
  user_id: string
  username: string
  email: string
  role: string
  joined_at?: string
}

interface MembersListProps {
  members: Member[]
  loading?: boolean
  onRemove: (userId: string) => void
}

export function MembersList({ members, loading, onRemove }: MembersListProps) {
  const [searchQuery, setSearchQuery] = useState('')

  const filtered = members.filter(m =>
    m.username.toLowerCase().includes(searchQuery.toLowerCase()) ||
    m.email.toLowerCase().includes(searchQuery.toLowerCase()) ||
    m.role.toLowerCase().includes(searchQuery.toLowerCase())
  )

  return (
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
        {loading ? (
          <div className="h-32 animate-pulse bg-muted/50 rounded" />
        ) : filtered.length === 0 ? (
          <p className="text-[10px] text-muted-foreground py-4 text-center">
            {searchQuery ? 'No members match search' : 'No members in this workspace'}
          </p>
        ) : (
          filtered.map(m => (
            <div
              key={m.user_id}
              className="flex items-center justify-between px-3 py-2 rounded-md text-[10px] hover:bg-muted/50"
            >
              <div className="min-w-0 flex-1">
                <div className="font-medium">{m.username}</div>
                <div className="text-muted-foreground">{m.email}</div>
              </div>
              <div className="flex items-center gap-3 shrink-0">
                <span className={`px-2 py-0.5 rounded-full text-[9px] font-medium ${
                  m.role === 'admin' ? 'bg-blue-100 text-blue-700' :
                  m.role === 'member' ? 'bg-green-100 text-green-700' :
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
                  onClick={() => onRemove(m.user_id)}
                >
                  ×
                </Button>
              </div>
            </div>
          ))
        )}
      </CardContent>
    </Card>
  )
}
