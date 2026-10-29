'use client'

import { useState, useEffect, useRef } from 'react'
import {
  Card, CardHeader, CardTitle, CardContent, Button, Input, StatCard, KpiGrid, Skeleton, Spinner,
  Tabs, TabsList, TabsTrigger, TabsContent,
} from '@sloughgpt/strui'
import { IconRefresh, IconTrash, IconDownload } from '@sloughgpt/strui'
import { PageContainer } from '@/components/PageContainer'
import { StatusBanner } from '@/components/composed/StatusBanner'
import { apiGet } from '@/lib/http-client'
import { SecurityOverviewCard } from '@/components/security/SecurityOverviewCard'
import { AuthSessionInfoCard } from '@/components/auth/AuthSessionInfoCard'
import { ErrorInsightsCard } from '@/components/errors/ErrorInsightsCard'
import { authController, type UserInfo } from '@/lib/auth-controller'
import { useRefreshShortcut } from '@/hooks/useRefreshShortcut'
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
        apiGet<AuditResponse>(auditUrl).catch((e) => { logger.warning('Could not audit log fetch', e); return null }),
        apiGet<{ count: number; configured: boolean }>('/security/keys').catch((e) => { logger.warning('Could not security keys fetch', e); return null }),
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
      setAuthError(err instanceof Error ? err.message : 'Could not connection')
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

  useRefreshShortcut(refreshAll)

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
        <Button size="sm" variant="ghost" className="h-6 text-[10px]" onClick={refreshAll} aria-label="Refresh">
          <IconRefresh className="h-3 w-3" />
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
            <CardHeader className="flex flex-row items-center justify-between pb-2 pt-2.5 px-2.5">
              <CardTitle className="text-[11px] font-medium">Audit Logs</CardTitle>
              <div className="flex items-center gap-1">
                <Button
                  size="sm"
                  variant={historyMode ? 'default' : 'ghost'}
                  className="h-6 text-[10px]"
                  onClick={toggleHistory}
                >
                  {historyMode ? 'Persisted' : 'Session'}
                </Button>
                <Button
                  size="sm"
                  variant="ghost"
                  className="h-6 text-[10px]"
                  onClick={loadOlder}
                  disabled={loadingMore || !historyMode}
                >
                  {loadingMore ? (
                    <Spinner size="sm" />
                  ) : (
                    'Load older'
                  )}
                </Button>
                <Button size="sm" variant="ghost" className="h-6 text-[10px]" onClick={() => fetchSecurity(historyMode)} aria-label="Refresh audit logs">
                  <IconRefresh className="h-3 w-3" />
                </Button>
              </div>
            </CardHeader>
            <CardContent className="space-y-2 px-2.5 pb-2.5">
              <Input
                value={filter}
                onChange={e => setFilter(e.target.value)}
                placeholder="Filter by event type..."
                className="h-7 text-[11px]"
              />
              {filteredLogs.length === 0 ? (
                <div className="text-center py-4 text-[10px] text-muted-foreground/60 space-y-0.5">
                  <div>No audit logs found.</div>
                  <div>Activities are logged automatically as you use the app.</div>
                </div>
              ) : (
                <div className="space-y-1 max-h-96 overflow-y-auto">
                  {filteredLogs.map((log, i) => (
                    <div key={i} className="rounded-lg border border-border/40 px-2.5 py-2 text-[11px]">
                      <div className="flex items-center justify-between">
                        <span className="font-medium">{log.event_type}</span>
                        <span className="text-[10px] text-muted-foreground/60 font-mono tabular-nums">
                          {log.timestamp ? new Date(log.timestamp).toLocaleString() : '—'}
                        </span>
                      </div>
                      {log.resource && <div className="text-[10px] text-muted-foreground/60 mt-0.5">Resource: {log.resource}</div>}
                      {log.user && <div className="text-[10px] text-muted-foreground/60 mt-0.5">User: {log.user}</div>}
                      {log.detail && <div className="text-[10px] text-muted-foreground/60 mt-0.5">{log.detail}</div>}
                      {log.extra && Object.keys(log.extra).length > 0 && (
                        <div className="text-[10px] text-muted-foreground/60 mt-0.5 font-mono">
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
                <CardHeader className="pb-2 pt-2.5 px-2.5">
                  <CardTitle className="text-[11px] font-medium">Current User</CardTitle>
                </CardHeader>
                <CardContent className="space-y-2 px-2.5 pb-2.5">
                  <div className="grid grid-cols-1 sm:grid-cols-3 gap-1.5">
                    <div className="rounded-lg bg-muted/20 p-2 text-center">
                      <div className="text-[9px] text-muted-foreground/60 uppercase tracking-wider">Username</div>
                      <div className="text-[11px] font-mono font-medium tabular-nums">{currentUser.username}</div>
                    </div>
                    <div className="rounded-lg bg-muted/20 p-2 text-center">
                      <div className="text-[9px] text-muted-foreground/60 uppercase tracking-wider">Email</div>
                      <div className="text-[11px] font-mono font-medium tabular-nums">{currentUser.email}</div>
                    </div>
                    <div className="rounded-lg bg-muted/20 p-2 text-center">
                      <div className="text-[9px] text-muted-foreground/60 uppercase tracking-wider">User ID</div>
                      <div className="text-[11px] font-mono font-medium tabular-nums truncate">{currentUser.id}</div>
                    </div>
                  </div>
                  <Button size="sm" variant="outline" className="h-7 text-[11px]" onClick={handleLogout}>Logout</Button>
                </CardContent>
              </Card>
              <AuthSessionInfoCard token={token} user={currentUser} onLogout={handleLogout} />
            </>
          ) : (
            <Card>
              <CardHeader className="pb-2 pt-2.5 px-2.5">
                <CardTitle className="text-[11px] font-medium">{authMode === 'login' ? 'Login' : 'Register'}</CardTitle>
              </CardHeader>
              <CardContent className="px-2.5 pb-2.5">
                <form onSubmit={handleAuthSubmit} className="space-y-2">
                  <Input
                    value={username}
                    onChange={e => setUsername(e.target.value)}
                    placeholder="Username"
                    aria-label="Username"
                    className="h-7 text-[11px]"
                    required
                  />
                  {authMode === 'register' && (
                    <Input
                      type="email"
                      value={email}
                      onChange={e => setEmail(e.target.value)}
                      placeholder="Email"
                      aria-label="Email"
                      className="h-7 text-[11px]"
                      required
                    />
                  )}
                  <Input
                    type="password"
                    value={password}
                    onChange={e => setPassword(e.target.value)}
                    placeholder="Password"
                    aria-label="Password"
                    className="h-7 text-[11px]"
                    required
                  />
                  {authError && <StatusBanner variant="error" message={authError} dismissible={false} />}
                  <div className="flex items-center gap-2">
                    <Button size="sm" type="submit" className="h-7 text-[11px]" disabled={authLoading}>
                      {authLoading ? 'Processing...' : authMode === 'login' ? 'Login' : 'Register'}
                    </Button>
                    <button
                      type="button"
                      className="text-[10px] text-primary hover:text-primary/80"
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
            <CardHeader className="pb-2 pt-2.5 px-2.5">
              <CardTitle className="text-[11px] font-medium">Token Info</CardTitle>
            </CardHeader>
            <CardContent className="px-2.5 pb-2.5">
              {token ? (
                <div className="space-y-1.5">
                  <div className="rounded-lg bg-muted/20 p-2">
                    <div className="text-[9px] text-muted-foreground/60 uppercase tracking-wider mb-0.5">JWT Token</div>
                    <div className="text-[10px] font-mono break-all text-muted-foreground/60">{token.slice(0, 60)}...</div>
                  </div>
                  <Button
                    size="sm"
                    variant="ghost"
                    className="h-6 text-[10px]"
                    onClick={async () => {
                      try {
                        const data = await authController.verify(token!)
                        addToast(data?.valid ? 'Token is valid' : 'Token is invalid', data?.valid ? 'success' : 'error')
                      } catch { addToast('Could not verify token', 'error') }
                    }}
                  >
                    Verify Token
                  </Button>
                </div>
              ) : (
                <p className="text-[10px] text-muted-foreground/60">No token. Login or register to get one.</p>
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
            <CardHeader className="flex flex-row items-center justify-between pb-2 pt-2.5 px-2.5">
              <CardTitle className="text-[11px] font-medium">Actions</CardTitle>
              <div className="flex items-center gap-1">
                <Button size="sm" variant={autoRefresh ? 'default' : 'ghost'} className="h-6 text-[10px]" onClick={() => setAutoRefresh(!autoRefresh)}>
                  {autoRefresh ? 'Auto-refresh ON' : 'Auto-refresh'}
                </Button>
                <Button size="sm" variant="ghost" className="h-6 text-[10px]" onClick={fetchErrors} aria-label="Refresh">
                  <IconRefresh className="h-3 w-3" />
                </Button>
              </div>
            </CardHeader>
            <CardContent className="px-2.5 pb-2.5">
              <div className="flex gap-1">
                <Button size="sm" variant="outline" className="h-7 text-[11px]" onClick={handleExport}>
                  <span className="inline-flex items-center gap-1">
                    <IconDownload className="h-3 w-3" />
                    Export All
                  </span>
                </Button>
                {search && (
                  <Button size="sm" variant="outline" className="h-7 text-[11px]" onClick={handleExportFiltered}>
                    <span className="inline-flex items-center gap-1">
                      <IconDownload className="h-3 w-3" />
                      Export Filtered
                    </span>
                  </Button>
                )}
                <Button size="sm" variant="outline" className="h-7 text-[11px] text-destructive" onClick={handleClear} disabled={clearing}>
                  <span className="inline-flex items-center gap-1">
                    <IconTrash className="h-3 w-3" />
                    {clearing ? 'Clearing...' : 'Clear All'}
                  </span>
                </Button>
              </div>
            </CardContent>
          </Card>

          {trends.length > 0 && (
            <Card>
              <CardHeader className="pb-2 pt-2.5 px-2.5">
                <CardTitle className="text-[11px] font-medium">Hourly Trend (24h)</CardTitle>
              </CardHeader>
              <CardContent className="px-2.5 pb-2.5">
                <div className="flex items-end gap-0.5 h-20">
                  {trends.map((t, i) => (
                    <div
                      key={i}
                      className="flex-1 bg-primary/20 rounded-t min-w-[3px]"
                      style={{ height: `${Math.max((t.count / maxTrend) * 100, 2)}%` }}
                      title={`${t.hour.split('T')[1]}: ${t.count}`}
                    />
                  ))}
                </div>
                <div className="flex justify-between text-[9px] text-muted-foreground/60 font-mono mt-0.5">
                  <span>{trends[0]?.hour.split('T')[1]}</span>
                  <span>{trends[trends.length - 1]?.hour.split('T')[1]}</span>
                </div>
              </CardContent>
            </Card>
          )}

          <Card>
            <CardHeader className="flex flex-row flex-wrap items-center justify-between gap-2 pb-2 pt-2.5 px-2.5">
              <CardTitle className="text-[11px] font-medium">Grouped Errors ({grouped.length})</CardTitle>
              <Input
                value={search}
                onChange={e => setSearch(e.target.value)}
                placeholder="Search errors..."
                className="h-7 w-full sm:w-40 text-[11px]"
              />
            </CardHeader>
            <CardContent className="px-2.5 pb-2.5">
              {grouped.length === 0 ? (
                <p className="text-[10px] text-muted-foreground/60">No errors logged.</p>
              ) : (
                <div className="space-y-1">
                  {grouped
                    .filter(g => !search || g.message.toLowerCase().includes(search.toLowerCase()) || g.source.toLowerCase().includes(search.toLowerCase()))
                    .map(g => (
                    <div key={g.fingerprint} className="rounded-lg border border-border/40 px-2.5 py-2 text-[11px] hover:bg-muted/20 transition-colors">
                      <div className="flex items-start justify-between gap-2">
                        <div className="min-w-0">
                          <div className="font-medium truncate">{g.message}</div>
                          <div className="text-[10px] text-muted-foreground/60">
                            {g.source} · {g.sample_url && <span className="truncate max-w-[200px] inline-block">{g.sample_url}</span>}
                            {g.sample_line != null && `:${g.sample_line}`}
                          </div>
                        </div>
                        <div className="text-right shrink-0">
                          <span className="text-[10px] font-mono bg-destructive/10 text-destructive px-1.5 py-0.5 rounded-full">
                            ×{g.count}
                          </span>
                          <div className="text-[9px] text-muted-foreground/60 mt-0.5 font-mono">
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
            <CardHeader className="pb-2 pt-2.5 px-2.5">
              <CardTitle className="text-[11px] font-medium">Recent Errors</CardTitle>
            </CardHeader>
            <CardContent className="px-2.5 pb-2.5">
              {recent.length === 0 ? (
                <p className="text-[10px] text-muted-foreground/60">No recent errors.</p>
              ) : (
                <div className="space-y-0 max-h-96 overflow-y-auto">
                  {recent.map(e => (
                    <div key={e.id} className="flex items-start gap-2 text-[10px] py-1.5 border-b border-border/20 last:border-0">
                      <span className="font-mono text-muted-foreground/60 shrink-0 w-14 tabular-nums">
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
