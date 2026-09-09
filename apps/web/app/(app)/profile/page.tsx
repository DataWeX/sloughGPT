'use client'

import { useState, useEffect, useCallback } from 'react'
import { Card, CardHeader, CardTitle, CardContent, Button, Input, StatCard, KpiGrid, Skeleton } from '@sloughgpt/strui'
import { PageContainer } from '@/components/PageContainer'
import { AppRouteHeader, AppRouteHeaderLead } from '@/components/AppRouteHeader'
import { apiGet, apiPut, apiPost } from '@/lib/http-client'
import { useAuthStore } from '@/lib/auth'
import { useToastStore } from '@/lib/toast-store'
import { logger } from '@/lib/dev-log'
import { useRefreshShortcut } from '@/hooks/useRefreshShortcut'

interface UserProfile {
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

export default function ProfilePage() {
  const [profile, setProfile] = useState<UserProfile | null>(null)
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [email, setEmail] = useState('')
  const [displayName, setDisplayName] = useState('')
  const [currentPassword, setCurrentPassword] = useState('')
  const [newPassword, setNewPassword] = useState('')
  const [confirmPassword, setConfirmPassword] = useState('')
  const [changingPassword, setChangingPassword] = useState(false)
  const addToast = useToastStore(s => s.addToast)
  const { user, setUser } = useAuthStore()

  const fetchProfile = useCallback(async () => {
    try {
      const res = await apiGet<{ data: UserProfile }>('/users/me/profile')
      const p = res?.data
      if (p) {
        setProfile(p)
        setEmail(p.email)
        setDisplayName(p.display_name || '')
      }
    } catch {
      logger.warning('Could not fetch profile')
    } finally {
      setLoading(false)
    }
  }, [])

  useRefreshShortcut(fetchProfile)

  const updateProfile = async () => {
    setSaving(true)
    try {
      const res = await apiPut<{ data: UserProfile }>('/users/me/profile', {
        email,
        display_name: displayName,
      })
      const p = res?.data
      if (p) {
        setProfile(p)
        if (user) {
          setUser({ ...user, email: p.email })
        }
        addToast('Profile updated', 'success')
      }
    } catch {
      addToast('Could not update profile', 'error')
    } finally {
      setSaving(false)
    }
  }

  const changePassword = async () => {
    if (newPassword !== confirmPassword) {
      addToast('Passwords do not match', 'error')
      return
    }
    if (newPassword.length < 8) {
      addToast('Password must be at least 8 characters', 'error')
      return
    }
    setChangingPassword(true)
    try {
      await apiPost('/users/me/password', {
        current_password: currentPassword,
        new_password: newPassword,
      })
      setCurrentPassword('')
      setNewPassword('')
      setConfirmPassword('')
      addToast('Password changed', 'success')
    } catch {
      addToast('Could not change password — check current password', 'error')
    } finally {
      setChangingPassword(false)
    }
  }

  useEffect(() => { fetchProfile() }, [fetchProfile])

  if (loading) {
    return (
      <PageContainer title="Profile" subtitle="Your account settings" loadingCards={3}>
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
        left={<AppRouteHeaderLead title="Profile" subtitle="Your account settings" />}
      />

      <div className="space-y-4">
        <KpiGrid>
          <StatCard label="Username" value={profile?.username ?? '—'} />
          <StatCard label="Role" value={profile?.role ?? '—'} />
          <StatCard label="Status" value={profile?.status ?? '—'} />
          <StatCard label="Member Since" value={profile?.created_at ? new Date(profile.created_at).toLocaleDateString() : '—'} />
        </KpiGrid>

        {/* Profile info */}
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-xs">Profile Information</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="space-y-1">
              <label className="text-[10px] text-muted-foreground">Username</label>
              <Input value={profile?.username ?? ''} disabled className="h-6 text-[10px] opacity-60" />
            </div>
            <div className="space-y-1">
              <label className="text-[10px] text-muted-foreground">Display Name</label>
              <Input
                value={displayName}
                onChange={e => setDisplayName(e.target.value)}
                placeholder="Your display name"
                className="h-6 text-[10px]"
              />
            </div>
            <div className="space-y-1">
              <label className="text-[10px] text-muted-foreground">Email</label>
              <Input
                value={email}
                onChange={e => setEmail(e.target.value)}
                type="email"
                placeholder="your@email.com"
                className="h-6 text-[10px]"
              />
            </div>
            <Button size="sm" className="h-6 text-[10px]" onClick={updateProfile} disabled={saving}>
              {saving ? 'Saving...' : 'Save Changes'}
            </Button>
          </CardContent>
        </Card>

        {/* Change password */}
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-xs">Change Password</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="space-y-1">
              <label className="text-[10px] text-muted-foreground">Current Password</label>
              <Input
                value={currentPassword}
                onChange={e => setCurrentPassword(e.target.value)}
                type="password"
                placeholder="Current password"
                className="h-6 text-[10px]"
              />
            </div>
            <div className="space-y-1">
              <label className="text-[10px] text-muted-foreground">New Password</label>
              <Input
                value={newPassword}
                onChange={e => setNewPassword(e.target.value)}
                type="password"
                placeholder="New password (min 8 characters)"
                className="h-6 text-[10px]"
              />
            </div>
            <div className="space-y-1">
              <label className="text-[10px] text-muted-foreground">Confirm New Password</label>
              <Input
                value={confirmPassword}
                onChange={e => setConfirmPassword(e.target.value)}
                type="password"
                placeholder="Confirm new password"
                className="h-6 text-[10px]"
              />
            </div>
            <Button
              size="sm"
              className="h-6 text-[10px]"
              onClick={changePassword}
              disabled={changingPassword || !currentPassword || !newPassword}
            >
              {changingPassword ? 'Changing...' : 'Change Password'}
            </Button>
          </CardContent>
        </Card>

        {/* Account details */}
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-xs">Account Details</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 gap-2 text-[10px]">
              <div>
                <span className="text-muted-foreground">User ID: </span>
                <span className="font-mono">{profile?.id}</span>
              </div>
              <div>
                <span className="text-muted-foreground">Tenant ID: </span>
                <span className="font-mono">{profile?.tenant_id || '—'}</span>
              </div>
              <div>
                <span className="text-muted-foreground">Last Login: </span>
                <span>{profile?.last_login_at ? new Date(profile.last_login_at).toLocaleString() : '—'}</span>
              </div>
              <div>
                <span className="text-muted-foreground">Created: </span>
                <span>{profile?.created_at ? new Date(profile.created_at).toLocaleString() : '—'}</span>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
