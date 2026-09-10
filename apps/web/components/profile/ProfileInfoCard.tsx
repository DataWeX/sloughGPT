'use client'

import { Card, CardHeader, CardTitle, CardContent, Button, Input } from '@sloughgpt/strui'

interface ProfileInfoCardProps {
  username?: string
  displayName?: string
  email?: string
  onDisplayNameChange?: (value: string) => void
  onEmailChange?: (value: string) => void
  onSave?: () => void
  saving?: boolean
}

export function ProfileInfoCard({
  username = '',
  displayName = '',
  email = '',
  onDisplayNameChange,
  onEmailChange,
  onSave,
  saving = false,
}: ProfileInfoCardProps) {
  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-xs">Profile Information</CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="space-y-1">
          <label className="text-[10px] text-muted-foreground">Username</label>
          <Input value={username} disabled className="h-6 text-[10px] opacity-60" />
        </div>
        <div className="space-y-1">
          <label className="text-[10px] text-muted-foreground">Display Name</label>
          <Input
            value={displayName}
            onChange={(e) => onDisplayNameChange?.(e.target.value)}
            placeholder="Your display name"
            className="h-6 text-[10px]"
          />
        </div>
        <div className="space-y-1">
          <label className="text-[10px] text-muted-foreground">Email</label>
          <Input
            value={email}
            onChange={(e) => onEmailChange?.(e.target.value)}
            type="email"
            placeholder="your@email.com"
            className="h-6 text-[10px]"
          />
        </div>
        <Button size="sm" className="h-6 text-[10px]" onClick={onSave} disabled={saving}>
          {saving ? 'Saving...' : 'Save Changes'}
        </Button>
      </CardContent>
    </Card>
  )
}
