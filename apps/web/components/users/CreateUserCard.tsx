'use client'

import { Card, CardHeader, CardTitle, CardContent, Button, Input } from '@sloughgpt/strui'

export interface CreateUserCardProps {
  username: string
  email: string
  password: string
  role: string
  loading: boolean
  onUsernameChange: (v: string) => void
  onEmailChange: (v: string) => void
  onPasswordChange: (v: string) => void
  onRoleChange: (v: string) => void
  onCreate: () => void
}

export function CreateUserCard({
  username,
  email,
  password,
  role,
  loading,
  onUsernameChange,
  onEmailChange,
  onPasswordChange,
  onRoleChange,
  onCreate,
}: CreateUserCardProps) {
  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-xs">Create User</CardTitle>
      </CardHeader>
      <CardContent className="space-y-2">
        <div className="grid grid-cols-2 gap-1.5">
          <Input
            value={username}
            onChange={e => onUsernameChange(e.target.value)}
            placeholder="Username"
            className="h-6 text-[10px]"
          />
          <Input
            value={email}
            onChange={e => onEmailChange(e.target.value)}
            placeholder="Email"
            type="email"
            className="h-6 text-[10px]"
          />
          <Input
            value={password}
            onChange={e => onPasswordChange(e.target.value)}
            placeholder="Password (min 8 chars)"
            type="password"
            className="h-6 text-[10px]"
          />
          <select
            value={role}
            onChange={e => onRoleChange(e.target.value)}
            className="h-6 text-[10px] rounded-md border border-border bg-background px-2"
          >
            <option value="viewer">Viewer</option>
            <option value="user">User</option>
            <option value="admin">Admin</option>
          </select>
        </div>
        <Button
          size="sm"
          className="h-6 text-[10px]"
          onClick={onCreate}
          disabled={loading || !username.trim() || !email.trim() || !password.trim()}
        >
          {loading ? 'Creating...' : 'Create User'}
        </Button>
      </CardContent>
    </Card>
  )
}
