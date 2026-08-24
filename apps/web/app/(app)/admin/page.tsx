'use client'

import { useState, useEffect, useRef } from 'react'
import {
  Card, CardHeader, CardTitle, CardContent, Button, Input, StatCard, KpiGrid, Skeleton,
  Tabs, TabsList, TabsTrigger, TabsContent,
} from '@sloughgpt/strui'
import { IconRefresh, IconTrash, IconDownload } from '@sloughgpt/strui'
import { PageContainer } from '@/components/PageContainer'
import { apiGet } from '@/lib/http-client'
import { SecurityOverviewCard } from '@/components/security/SecurityOverviewCard'
import { AuthSessionInfoCard } from '@/components/auth/AuthSessionInfoCard'
import { ErrorInsightsCard } from '@/components/errors/ErrorInsightsCard'
import { authController, type UserInfo } from '@/lib/auth-controller'
import { errorsController } from '@/lib/errors-controller'
import { downloadJson } from '@/lib/download-utils'
import { useToastStore } from '@/lib/toast-store'
import { logger } from '@/lib/dev-log'

type Tab = 'security' | 'auth' | 'errors'

// ── Security types ────────────────────────────────────────────────────────────
interface AuditLog {
  event_type: string
  timestamp: string
  user?: string
  resource?: string
  detail?: string
  extra?: Record<string, unknown>
}

interface AuditResponse {
  logs?: AuditLog[]
  count?: number
}

function mergeLogs(a: AuditLog[], b: AuditLog[]): AuditLog[] {
  const seen = new Set<string>()
  const out: AuditLog[] = []
  for (const l of [...a, ...b]) {
    const key = `${l.timestamp}|${l.event_type}`
    if (seen.has(key)) continue
    seen.add(key)
    out.push(l)
  }
  return out
}

// ── Auth types ────────────────────────────────────────────────────────────────
type AuthMode = 'login' | 'register'

export default function AdminPage() {
  const [tab, setTab] = useState<Tab>('security')
  const addToast = useToastStore(s => s.addToast)

  // ── Shared header state ─────────────────────────────────────────────────────
  const [globalLoading, setGlobalLoading] = useState(true)

  // ── Security state ──────────────────────────────────────────────────────────
  const [logs, setLogs] = useState<AuditLog[]>([])
  const [keyInfo, setKeyInfo] = useState<{ count: number; configured: boolean } | null>(null)
  const [secLoading, setSecLoading] = useState(true)
  const [loadingMore, setLoadingMore] = useState(false)
  const [historyMode, setHistoryMode] = useState(false)
  const [filter, setFilter] = useState('')

  // ── Auth state ──────────────────────────────────────────────────────────────
  const [authMode, setAuthMode] = useState<AuthMode>('login')
  const [username, setUsername] = useState('')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [authLoading, setAuthLoading] = useState(false)
  const [authError, setAuthError] = useState<string | null>(null)
  const [currentUser, setCurrentUser] = useState<UserInfo | null>(null)
  const [checking, setChecking] = useState(true)
  const [token, setToken] = useState<string | null>(null)

  // ── Errors state ────────────────────────────────────────────────────────────
  const [grouped, setGrouped] = useState<Awaited<ReturnType<typeof errorsController.getGrouped>>>([])
  const [recent, setRecent] = useState<Awaited<ReturnType<typeof errorsController.getRecent>>['errors']>([])
  const [trends, setTrends] = useState<Awaited<ReturnType<typeof errorsController.getTrends>>>([])
  const [total, setTotal] = useState(0)
  const [errLoading, setErrLoading] = useState(true)
  const [clearing, setClearing] = useState(false)
  const [autoRefresh, setAutoRefresh] = useState(false)
  const [search, setSearch] = useState('')
  const intervalRef = useRef<NodeJS.Timeout | null>(null)

  // ── Security helpers ────────────────────────────────────────────────────────
  const eventParam = () => {
    const f = filter.trim()
    return f ? `&event_type=${encodeURIComponent(f)}` : ''
  }

  const fetchSecurity = async (useHistory = false) => {
    setSecLoading(true)
    try {
      const auditUrl = `${useHistory ? '/security/audit?history=true&limit=100' : '/security/audit?limit=100'}${eventParam()}`
      const [logsRes, keysRes] = await Promise.all([
        apiGet<AuditResponse>(auditUrl).catch((e) => { logger.warning('audit log fetch failed', e); return null }),
        apiGet<{ count: number; configured: boolean }>('/security/keys').catch((e) => { logger.warning('security keys fetch failed', e); return null }),
      ])
      setLogs(logsRes?.logs ?? [])
      const keysData = keysRes && 'count' in keysRes ? keysRes : null
      setKeyInfo(keysData)
    } catch {
      addToast('Could not load security data', 'error')
    } finally {
      setSecLoading(false)
    }
  }

  const toggleHistory = () => {
    const next = !historyMode
    setHistoryMode(next)
    fetchSecurity(next)
  }

  const loadOlder = async () => {
    if (logs.length === 0 || loadingMore) return
    setLoadingMore(true)
    try {
      const oldest = logs.reduce<string | null>(
        (min, l) => (l.timestamp && (!min || l.timestamp < min) ? l.timestamp : min),
        null,
      )
      if (!oldest) return
      const before = encodeURIComponent(oldest)
      const res = await apiGet<AuditResponse>(`/security/audit?history=true&limit=100&before=${before}${eventParam()}`)
      const older = res?.logs ?? []
      setLogs(prev => mergeLogs(prev, older))
    } catch {
      addToast('Could not load older audit logs', 'error')
    } finally {
      setLoadingMore(false)
    }
  }

  // ── Auth helpers ────────────────────────────────────────────────────────────
  const handleAuthSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setAuthLoading(true)
    setAuthError(null)
    try {
      const data = authMode === 'login'
        ? await authController.login(username, password)
        : await authController.register(username, email, password)
      setToken(data.token)
      setCurrentUser(data.user)
      localStorage.setItem('auth_token', data.token)
    } catch (err) {
      setAuthError(err instanceof Error ? err.message : 'Connection failed')
    } finally {
      setAuthLoading(false)
    }
  }

  const handleLogout = () => {
    setToken(null)
    setCurrentUser(null)
    localStorage.removeItem('auth_token')
  }

  // ── Errors helpers ──────────────────────────────────────────────────────────
  const fetchErrors = async () => {
    setErrLoading(true)
    try {
      const [g, r, t] = await Promise.all([
        errorsController.getGrouped(),
        errorsController.getRecent(30),
        errorsController.getTrends(24),
      ])
      setGrouped(g)
      setRecent(r.errors)
      setTotal(r.total)
      setTrends(t)
    } catch {
      addToast('Could not load error data', 'error')
    } finally {
      setErrLoading(false)
    }
  }

  const handleClear = async () => {
    setClearing(true)
    try {
      await errorsController.clear()
      await fetchErrors()
    } catch {
      addToast('Could not clear errors', 'error')
    } finally {
      setClearing(false)
    }
  }

  const handleExport = async () => {
    try {
      const data = await errorsController.export()
      downloadJson(data, `errors-${Date.now()}.json`)
    } catch {
      addToast('Could not export errors', 'error')
    }
  }

  const handleExportFiltered = () => {
    const filtered = grouped.filter(g =>
      !search || g.message.toLowerCase().includes(search.toLowerCase()) || g.source.toLowerCase().includes(search.toLowerCase())
    )
    const data = filtered.map(g => ({
      message: g.message,
      source: g.source,
      count: g.count,
      fingerprint: g.fingerprint,
      sample_url: g.sample_url,
      sample_line: g.sample_line,
      latest: g.latest,
    }))
    downloadJson(data, `errors-filtered-${Date.now()}.json`)
    addToast(`Exported ${data.length} error groups`, 'success')
  }

  // ── Init ────────────────────────────────────────────────────────────────────
  useEffect(() => {
    const init = async () => {
      // Auth init
      const saved = localStorage.getItem('auth_token')
      if (saved) {
        setToken(saved)
        try {
          const d = await authController.getMe(saved)
          setCurrentUser(d)
        } catch {
          localStorage.removeItem('auth_token')
          setToken(null)
        }
      }
      setChecking(false)

      // Fetch all tabs in parallel
      await Promise.all([
        fetchSecurity(false),
        fetchErrors(),
      ])
      setGlobalLoading(false)
    }
    init()
  }, [])

  // ── Auto-refresh for errors ─────────────────────────────────────────────────
  useEffect(() => {
    if (tab !== 'errors') return
    if (autoRefresh) {
      intervalRef.current = setInterval(fetchErrors, 10000)
      const onVis = () => { if (!document.hidden && intervalRef.current) { clearInterval(intervalRef.current); intervalRef.current = setInterval(fetchErrors, 10000) } }
      document.addEventListener('visibilitychange', onVis)
      return () => { clearInterval(intervalRef.current!); document.removeEventListener('visibilitychange', onVis) }
    } else if (intervalRef.current) {
      clearInterval(intervalRef.current)
    }
  }, [autoRefresh, tab])

  // ── Refresh handler (header button) ─────────────────────────────────────────
  const refreshAll = async () => {
    if (tab === 'security') {
      await fetchSecurity(historyMode)
    } else if (tab === 'auth') {
      // Auth doesn't need explicit refresh; re-check token
      const saved = localStorage.getItem('auth_token')
      if (saved) {
        try {
          const d = await authController.getMe(saved)
          setCurrentUser(d)
        } catch {
          localStorage.removeItem('auth_token')
          setToken(null)
          setCurrentUser(null)
        }
      }
    } else {
      await fetchErrors()
    }
  }

  // ── Derived state ───────────────────────────────────────────────────────────
  const filteredLogs = filter.trim()
    ? logs.filter(l => l.event_type?.toLowerCase().includes(filter.toLowerCase()))
    : logs

  const lastHourCount = recent.filter(e => {
    const ts = new Date(e.timestamp).getTime()
    return Date.now() - ts < 3600000
  }).length
  const topError = grouped.length > 0 ? grouped[0].message.slice(0, 40) : 'None'
  const maxTrend = Math.max(...trends.map(t => t.count), 1)

  const isInitialLoading = globalLoading || checking

  if (isInitialLoading) {
    return (
      <PageContainer
        title="Admin"
        subtitle="Security, authentication & error monitoring"
        loadingCards={4}
      >
        <KpiGrid>
          <StatCard label="Loading" value={<Skeleton className="h-5 w-12" />} />
          <StatCard label="Loading" value={<Skeleton className="h-5 w-12" />} />
          <StatCard label="Loading" value={<Skeleton className="h-5 w-12" />} />
          <StatCard label="Loading" value={<Skeleton className="h-5 w-12" />} />
        </KpiGrid>
        <Card><CardContent><div className="h-32 animate-pulse bg-muted/50 rounded" /></CardContent></Card>
        <Card><CardContent><div className="h-64 animate-pulse bg-muted/50 rounded" /></CardContent></Card>
      </PageContainer>
    )
  }

  return (
    <PageContainer
      title="Admin"
      subtitle="Security, authentication & error monitoring"
      headerRight={
        <Button size="sm" variant="ghost" onClick={refreshAll} aria-label="Refresh">
          <IconRefresh className="h-4 w-4" />
        </Button>
      }
    >
      <Tabs value={tab} onValueChange={(v) => setTab(v as Tab)}>
        <TabsList>
          <TabsTrigger value="security">Security</TabsTrigger>
          <TabsTrigger value="auth">Auth</TabsTrigger>
          <TabsTrigger value="errors">Errors</TabsTrigger>
        </TabsList>

        {/* ── Security Tab ──────────────────────────────────────────────── */}
        <TabsContent value="security" className="space-y-4">
          <KpiGrid>
            <StatCard label="API Keys" value={keyInfo?.configured ? `${keyInfo.count} configured` : 'None'} />
            <StatCard label="Audit Logs" value={logs.length} />
            <StatCard label="History Mode" value={historyMode ? 'Persisted' : 'Session'} />
            <StatCard label="Filter" value={filter || 'All'} />
          </KpiGrid>

          <SecurityOverviewCard
            logs={logs}
            apiKeyConfigured={keyInfo?.configured ?? false}
            apiKeyCount={keyInfo?.count ?? 0}
          />

          <Card>
            <CardHeader className="flex flex-row items-center justify-between">
              <CardTitle className="text-base">Audit Logs</CardTitle>
              <div className="flex items-center gap-1">
                <Button
                  size="sm"
                  variant={historyMode ? 'default' : 'ghost'}
                  onClick={toggleHistory}
                >
                  {historyMode ? 'Persisted' : 'Session'}
                </Button>
                <Button
                  size="sm"
                  variant="ghost"
                  onClick={loadOlder}
                  disabled={loadingMore || !historyMode}
                >
                  {loadingMore ? (
                    <span className="inline-block h-4 w-4 animate-spin rounded-full border-2 border-current border-t-transparent" />
                  ) : (
                    'Load older'
                  )}
                </Button>
                <Button size="sm" variant="ghost" onClick={() => fetchSecurity(historyMode)} aria-label="Refresh audit logs">
                  <IconRefresh className="h-4 w-4" />
                </Button>
              </div>
            </CardHeader>
            <CardContent className="space-y-3">
              <Input
                value={filter}
                onChange={e => setFilter(e.target.value)}
                placeholder="Filter by event type..."
              />
              {filteredLogs.length === 0 ? (
                <div className="text-center py-6 text-sm text-muted-foreground space-y-1">
                  <div>No audit logs found.</div>
                  <div className="text-xs text-muted-foreground/70">Activities are logged automatically as you use the app.</div>
                </div>
              ) : (
                <div className="space-y-2 max-h-96 overflow-y-auto">
                  {filteredLogs.map((log, i) => (
                    <div key={i} className="rounded-md border border-border/60 px-3 py-2 text-sm">
                      <div className="flex items-center justify-between">
                        <span className="font-medium">{log.event_type}</span>
                        <span className="text-xs text-muted-foreground">
                          {log.timestamp ? new Date(log.timestamp).toLocaleString() : '—'}
                        </span>
                      </div>
                      {log.resource && <div className="text-xs text-muted-foreground mt-0.5">Resource: {log.resource}</div>}
                      {log.user && <div className="text-xs text-muted-foreground mt-0.5">User: {log.user}</div>}
                      {log.detail && <div className="text-xs text-muted-foreground mt-0.5">{log.detail}</div>}
                      {log.extra && Object.keys(log.extra).length > 0 && (
                        <div className="text-xs text-muted-foreground mt-0.5 font-mono">
                          {JSON.stringify(log.extra).slice(0, 120)}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* ── Auth Tab ──────────────────────────────────────────────────── */}
        <TabsContent value="auth" className="space-y-4">
          <KpiGrid>
            <StatCard label="Status" value={currentUser ? 'Logged In' : 'Guest'} />
            <StatCard label="User" value={currentUser?.username ?? '—'} />
            <StatCard label="Token" value={token ? 'Authenticated' : 'Not signed in'} />
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
                <CardTitle className="text-base">{authMode === 'login' ? 'Login' : 'Register'}</CardTitle>
              </CardHeader>
              <CardContent>
                <form onSubmit={handleAuthSubmit} className="space-y-3">
                  <Input
                    value={username}
                    onChange={e => setUsername(e.target.value)}
                    placeholder="Username"
                    required
                  />
                  {authMode === 'register' && (
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
                  {authError && <div className="text-xs text-destructive">{authError}</div>}
                  <div className="flex items-center gap-3">
                    <Button size="sm" type="submit" disabled={authLoading}>
                      {authLoading ? 'Processing...' : authMode === 'login' ? 'Login' : 'Register'}
                    </Button>
                    <button
                      type="button"
                      className="text-xs text-primary hover:text-primary/80"
                      onClick={() => { setAuthMode(authMode === 'login' ? 'register' : 'login'); setAuthError(null) }}
                    >
                      {authMode === 'login' ? 'Create account' : 'Already have an account?'}
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
                        addToast(data?.valid ? 'Token is valid' : 'Token is invalid', data?.valid ? 'success' : 'error')
                      } catch { addToast('Verification failed', 'error') }
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
        </TabsContent>

        {/* ── Errors Tab ────────────────────────────────────────────────── */}
        <TabsContent value="errors" className="space-y-4">
          <KpiGrid>
            <StatCard label="Total Errors" value={String(total)} />
            <StatCard label="Error Groups" value={String(grouped.length)} />
            <StatCard label="Last Hour" value={String(lastHourCount)} />
            <StatCard label="Top Error" value={topError} />
          </KpiGrid>

          <ErrorInsightsCard grouped={grouped} recent={recent} total={total} />

          <Card>
            <CardHeader className="flex flex-row items-center justify-between">
              <CardTitle className="text-base">Actions</CardTitle>
              <div className="flex items-center gap-2">
                <Button size="sm" variant={autoRefresh ? 'default' : 'ghost'} onClick={() => setAutoRefresh(!autoRefresh)}>
                  {autoRefresh ? 'Auto-refresh ON' : 'Auto-refresh'}
                </Button>
                <Button size="sm" variant="ghost" onClick={fetchErrors}>
                  <IconRefresh className="h-4 w-4" />
                </Button>
              </div>
            </CardHeader>
            <CardContent>
              <div className="flex gap-2">
                <Button size="sm" variant="outline" onClick={handleExport}>
                  <span className="inline-flex items-center gap-1.5">
                    <IconDownload className="h-4 w-4" />
                    Export All
                  </span>
                </Button>
                {search && (
                  <Button size="sm" variant="outline" onClick={handleExportFiltered}>
                    <span className="inline-flex items-center gap-1.5">
                      <IconDownload className="h-4 w-4" />
                      Export Filtered
                    </span>
                  </Button>
                )}
                <Button size="sm" variant="outline" onClick={handleClear} disabled={clearing} className="text-destructive">
                  <span className="inline-flex items-center gap-1.5">
                    <IconTrash className="h-4 w-4" />
                    {clearing ? 'Clearing...' : 'Clear All'}
                  </span>
                </Button>
              </div>
            </CardContent>
          </Card>

          {trends.length > 0 && (
            <Card>
              <CardHeader>
                <CardTitle className="text-base">Hourly Trend (24h)</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="flex items-end gap-1 h-24">
                  {trends.map((t, i) => (
                    <div
                      key={i}
                      className="flex-1 bg-primary/20 rounded-t min-w-[4px]"
                      style={{ height: `${Math.max((t.count / maxTrend) * 100, 2)}%` }}
                      title={`${t.hour.split('T')[1]}: ${t.count}`}
                    />
                  ))}
                </div>
                <div className="flex justify-between text-xs text-muted-foreground mt-1">
                  <span>{trends[0]?.hour.split('T')[1]}</span>
                  <span>{trends[trends.length - 1]?.hour.split('T')[1]}</span>
                </div>
              </CardContent>
            </Card>
          )}

          <Card>
            <CardHeader className="flex flex-row flex-wrap items-center justify-between gap-2">
              <CardTitle className="text-base">Grouped Errors ({grouped.length})</CardTitle>
              <Input
                value={search}
                onChange={e => setSearch(e.target.value)}
                placeholder="Search errors..."
                className="h-9 w-full sm:w-48 text-sm"
              />
            </CardHeader>
            <CardContent>
              {grouped.length === 0 ? (
                <p className="text-sm text-muted-foreground">No errors logged.</p>
              ) : (
                <div className="space-y-2">
                  {grouped
                    .filter(g => !search || g.message.toLowerCase().includes(search.toLowerCase()) || g.source.toLowerCase().includes(search.toLowerCase()))
                    .map(g => (
                    <div key={g.fingerprint} className="rounded-md border border-border/60 px-3 py-2 text-sm hover:bg-muted/50 transition-colors">
                      <div className="flex items-start justify-between gap-2">
                        <div className="min-w-0">
                          <div className="font-medium truncate">{g.message}</div>
                          <div className="text-xs text-muted-foreground">
                            {g.source} · {g.sample_url && <span className="truncate max-w-[200px] inline-block">{g.sample_url}</span>}
                            {g.sample_line != null && `:${g.sample_line}`}
                          </div>
                        </div>
                        <div className="text-right shrink-0">
                          <span className="text-xs font-mono bg-destructive/10 text-destructive px-1.5 py-0.5 rounded">
                            ×{g.count}
                          </span>
                          <div className="text-[10px] text-muted-foreground mt-0.5">
                            {g.latest && new Date(g.latest).toLocaleDateString()}
                          </div>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="text-base">Recent Errors</CardTitle>
            </CardHeader>
            <CardContent>
              {recent.length === 0 ? (
                <p className="text-sm text-muted-foreground">No recent errors.</p>
              ) : (
                <div className="space-y-1 max-h-96 overflow-y-auto">
                  {recent.map(e => (
                    <div key={e.id} className="flex items-start gap-2 text-xs py-1.5 border-b border-border/30 last:border-0">
                      <span className="font-mono text-muted-foreground shrink-0 w-16">
                        {e.timestamp ? new Date(e.timestamp).toLocaleTimeString() : '—'}
                      </span>
                      <span className="truncate">{e.message}</span>
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </PageContainer>
  )
}
