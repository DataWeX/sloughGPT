'use client'

import { useState, useEffect } from 'react'
import { Card, CardHeader, CardTitle, CardContent, Button, Input, StatCard, KpiGrid, Skeleton } from '@sloughgpt/strui'
import { IconRefresh } from '@sloughgpt/strui'
import { PageContainer } from '@/components/PageContainer'
import { AuthSessionInfoCard } from '@/components/auth/AuthSessionInfoCard'
import { authController, type UserInfo } from '@/lib/auth-controller'

type Mode = 'login' | 'register'

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
      authController.getMe(saved)
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
      const data = mode === 'login'
        ? await authController.login(username, password)
        : await authController.register(username, email, password)
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
      <PageContainer title="Auth" subtitle="Authentication" loadingCards={3}>
        <KpiGrid>
          <StatCard label="Loading" value={<Skeleton className="h-5 w-12" />} />
          <StatCard label="Loading" value={<Skeleton className="h-5 w-12" />} />
          <StatCard label="Loading" value={<Skeleton className="h-5 w-12" />} />
        </KpiGrid>
        <Card><CardContent><div className="h-32 animate-pulse bg-muted/50 rounded" /></CardContent></Card>
        <Card><CardContent><div className="h-24 animate-pulse bg-muted/50 rounded" /></CardContent></Card>
      </PageContainer>
    )
  }

  return (
    <PageContainer title="Auth" subtitle={currentUser ? `Logged in as ${currentUser.username}` : 'Authentication'}>
      <KpiGrid>
        <StatCard label="Status" value={currentUser ? 'Logged In' : 'Guest'} />
        <StatCard label="User" value={currentUser?.username ?? '—'} />
        <StatCard label="Token" value={token ? 'Active' : 'None'} />
      </KpiGrid>

      {currentUser ? (
        <>
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Current User</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
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
          <AuthSessionInfoCard token={token} user={currentUser} onLogout={handleLogout} />
        </>
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
                    const data = await authController.verify(token!)
                    alert(data?.valid ? 'Token valid' : 'Token invalid')
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
    </PageContainer>
  )
}
