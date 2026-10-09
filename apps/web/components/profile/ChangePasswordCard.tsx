'use client'

import { useState } from 'react'
import { Card, CardHeader, CardTitle, CardContent, Button, Input } from '@sloughgpt/strui'

interface ChangePasswordCardProps {
  onChangePassword?: (currentPassword: string, newPassword: string) => Promise<void>
  changingPassword?: boolean
}

export function ChangePasswordCard({
  onChangePassword,
  changingPassword = false,
}: ChangePasswordCardProps) {
  const [currentPassword, setCurrentPassword] = useState('')
  const [newPassword, setNewPassword] = useState('')
  const [confirmPassword, setConfirmPassword] = useState('')

  const handleChange = async () => {
    if (!onChangePassword) return
    await onChangePassword(currentPassword, newPassword)
    setCurrentPassword('')
    setNewPassword('')
    setConfirmPassword('')
  }

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-xs">Change Password</CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="space-y-1">
          <label className="text-[10px] text-muted-foreground">Current Password</label>
          <Input
            value={currentPassword}
            onChange={(e) => setCurrentPassword(e.target.value)}
            type="password"
            placeholder="Current password"
            className="h-6 text-[10px]"
          />
        </div>
        <div className="space-y-1">
          <label className="text-[10px] text-muted-foreground">New Password</label>
          <Input
            value={newPassword}
            onChange={(e) => setNewPassword(e.target.value)}
            type="password"
            placeholder="New password (min 8 characters)"
            className="h-6 text-[10px]"
          />
        </div>
        <div className="space-y-1">
          <label className="text-[10px] text-muted-foreground">Confirm New Password</label>
          <Input
            value={confirmPassword}
            onChange={(e) => setConfirmPassword(e.target.value)}
            type="password"
            placeholder="Confirm new password"
            className="h-6 text-[10px]"
          />
        </div>
        <Button
          size="sm"
          className="h-6 text-[10px]"
          onClick={handleChange}
          disabled={changingPassword || !currentPassword || !newPassword}
        >
          {changingPassword ? 'Changing...' : 'Change Password'}
        </Button>
      </CardContent>
    </Card>
  )
}
