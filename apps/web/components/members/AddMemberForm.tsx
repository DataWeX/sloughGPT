'use client'

import { useState } from 'react'
import { Card, CardHeader, CardTitle, CardContent, Button, Input } from '@sloughgpt/strui'

interface AddMemberFormProps {
  onAdd: (userId: string, role: string) => void
  disabled?: boolean
}

export function AddMemberForm({ onAdd, disabled }: AddMemberFormProps) {
  const [userId, setUserId] = useState('')
  const [role, setRole] = useState('member')

  const handleSubmit = () => {
    if (!userId.trim()) return
    onAdd(userId, role)
    setUserId('')
  }

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-xs">Add Member by ID</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="flex gap-2">
          <Input
            value={userId}
            onChange={e => setUserId(e.target.value)}
            placeholder="User ID"
            className="flex-1 h-6 text-[10px]"
          />
          <select
            value={role}
            onChange={e => setRole(e.target.value)}
            className="h-6 text-[10px] rounded-md border border-border bg-background px-2"
          >
            <option value="viewer">Viewer</option>
            <option value="member">Member</option>
            <option value="admin">Admin</option>
          </select>
          <Button size="sm" className="h-6 text-[10px]" onClick={handleSubmit} disabled={!userId.trim() || disabled}>
            Add
          </Button>
        </div>
      </CardContent>
    </Card>
  )
}
