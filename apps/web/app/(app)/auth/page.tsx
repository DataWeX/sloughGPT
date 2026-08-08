'use client'

import { useState, useEffect } from 'react'
import { Card, CardHeader, CardTitle, CardContent, Button, Input } from '@sloughgpt/strui'
import { IconRefresh } from '@sloughgpt/strui'
import { AppRouteHeader, AppRouteHeaderLead } from '@/components/AppRouteHeader'
import { PUBLIC_API_URL } from '@/lib/config'

type Mode = 'login' | 'register'

interface UserInfo {
  id: string
  username: string
  email: string
}

export default function AuthPage() {
  const [mode, setMode] = useState<Mode>('login')
  const [username, setUsername] = useState('')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [currentUser, setCurrentUser] = useState<UserInfo | null>(null)
  const [checking, setChecking] = useState(true)
  const [token, setToken] = useState<string | null>(null)

  useEffect(() => {
    const saved = localStorage.getItem('auth_token')
    if (saved) {
      setToken(saved)
      fetch(`${PUBLIC_API_URL}/auth/me`, {
        headers: { Authorization: `Bearer ${saved}` },
      })
        .then(r => r.ok ? r.json() : Promise.reject())
        .then(d => setCurrentUser(d))
        .catch(() => { localStorage.removeItem('auth_token'); setToken(null) })
        .finally(() => setChecking(false))
    } else {
      setChecking(false)
    }
  }, [])

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setLoading(true)
    setError(null)
    try {
      const endpoint = mode === 'login' ? '/auth/login' : '/auth/register'
      const body = mode === 'login'
        ? { username, password }
        : { username, email, password }
      const res = await fetch(`${PUBLIC_API_URL}${endpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })
      const data = await res.json()
      if (!res.ok) {
        setError(data.detail ?? 'Authentication failed')
        return
      }
      setToken(data.token)
      setCurrentUser(data.user)
      localStorage.setItem('auth_token', data.token)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Connection failed')
    } finally {
      setLoading(false)
    }
  }

  const handleLogout = () => {
    setToken(null)
    setCurrentUser(null)
    localStorage.removeItem('auth_token')
  }

  if (checking) {
    return (
      <div className="sl-page mx-auto max-w-4xl">
        <AppRouteHeader left={<AppRouteHeaderLead title="Auth" subtitle="Authentication" />} />
        <div className="space-y-4">
          <Card><CardContent><div className="h-32 animate-pulse bg-muted/50 rounded" /></CardContent></Card>
        </div>
      </div>
    )
  }

  return (
    <div className="sl-page mx-auto max-w-4xl">
      <AppRouteHeader left={<AppRouteHeaderLead title="Auth" subtitle={currentUser ? `Logged in as ${currentUser.username}` : 'Authentication'} />} />
      <div className="space-y-4">
        {currentUser ? (
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Current User</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="grid grid-cols-3 gap-3">
                <div className="rounded-md bg-muted/30 p-3 text-center">
                  <div className="text-xs text-muted-foreground">Username</div>
                  <div className="text-sm font-mono font-medium">{currentUser.username}</div>
                </div>
                <div className="rounded-md bg-muted/30 p-3 text-center">
                  <div className="text-xs text-muted-foreground">Email</div>
                  <div className="text-sm font-mono font-medium">{currentUser.email}</div>
                </div>
                <div className="rounded-md bg-muted/30 p-3 text-center">
                  <div className="text-xs text-muted-foreground">User ID</div>
                  <div className="text-sm font-mono font-medium truncate">{currentUser.id}</div>
                </div>
              </div>
              <Button size="sm" variant="outline" onClick={handleLogout}>Logout</Button>
            </CardContent>
          </Card>
        ) : (
          <Card>
            <CardHeader>
              <CardTitle className="text-base">{mode === 'login' ? 'Login' : 'Register'}</CardTitle>
            </CardHeader>
            <CardContent>
              <form onSubmit={handleSubmit} className="space-y-3">
                <Input
                  value={username}
                  onChange={e => setUsername(e.target.value)}
                  placeholder="Username"
                  required
                />
                {mode === 'register' && (
                  <Input
                    type="email"
                    value={email}
                    onChange={e => setEmail(e.target.value)}
                    placeholder="Email"
                    required
                  />
                )}
                <Input
                  type="password"
                  value={password}
                  onChange={e => setPassword(e.target.value)}
                  placeholder="Password"
                  required
                />
                {error && <div className="text-xs text-destructive">{error}</div>}
                <div className="flex items-center gap-3">
                  <Button size="sm" type="submit" disabled={loading}>
                    {loading ? 'Processing...' : mode === 'login' ? 'Login' : 'Register'}
                  </Button>
                  <button
                    type="button"
                    className="text-xs text-primary hover:text-primary/80"
                    onClick={() => { setMode(mode === 'login' ? 'register' : 'login'); setError(null) }}
                  >
                    {mode === 'login' ? 'Create account' : 'Already have an account?'}
                  </button>
                </div>
              </form>
            </CardContent>
          </Card>
        )}

        <Card>
          <CardHeader>
            <CardTitle className="text-base">Token Info</CardTitle>
          </CardHeader>
          <CardContent>
            {token ? (
              <div className="space-y-2">
                <div className="rounded-md bg-muted/30 p-3">
                  <div className="text-xs text-muted-foreground mb-1">JWT Token</div>
                  <div className="text-[10px] font-mono break-all text-muted-foreground">{token.slice(0, 60)}...</div>
                </div>
                <Button
                  size="sm"
                  variant="ghost"
                  onClick={async () => {
                    try {
                      const res = await fetch(`${PUBLIC_API_URL}/auth/verify`, {
                        method: 'POST',
                        headers: { Authorization: `Bearer ${token}` },
                      })
                      const data = await res.json()
                      alert(data?.data?.valid ? 'Token valid' : 'Token invalid')
                    } catch { alert('Verification failed') }
                  }}
                >
                  Verify Token
                </Button>
              </div>
            ) : (
              <p className="text-sm text-muted-foreground">No token. Login or register to get one.</p>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
