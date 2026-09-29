'use client'
export const dynamic = 'force-dynamic'

import { useState, useEffect, useCallback } from 'react'
import { PageContainer } from '@/components/PageContainer'
import { Skeleton, cn, Tabs, TabsList, TabsTrigger, TabsContent, StatusDot } from '@sloughgpt/strui'
import { IconRefresh } from '@sloughgpt/strui'
import { TerminalPanel } from '@/components/shell/TerminalPanel'
import { V86TerminalPanel } from '@/components/shell/V86TerminalPanel'
import { FileStatsCard } from '@/components/files/FileStatsCard'
import { filesController, type FileEntry } from '@/lib/files-controller'
import { voiceController, type VoiceStatus } from '@/lib/voice-controller'
import { authFetch } from '@/lib/http-client'
import { useRefreshShortcut } from '@/hooks/useRefreshShortcut'
import { useLiveStatus } from '@/hooks/useLiveStatus'
import { useFileList } from '@/lib/cache'
import { useModels } from '@/lib/cache/api-hooks'
import ModelCatalogCard from '@/components/models/ModelCatalogCard'
import QuantizationCard from '@/components/models/QuantizationCard'
import DownloadsCard from '@/components/models/DownloadsCard'
import EngineStatusCard from '@/components/models/EngineStatusCard'
import { StartupTimeline } from '@/components/startup/StartupTimeline'
import { StartupHistoryChart } from '@/components/startup/StartupHistoryChart'

const QUICK_ACTIONS = [
  {
    label: 'Restart Backend',
    description: 'Restart the FastAPI server',
    endpoint: '/system/restart',
    method: 'POST',
  },
  {
    label: 'Clear Cache',
    description: 'Clear model cache',
    endpoint: '/cache/clear',
    method: 'POST',
  },
  {
    label: 'Reset Metrics',
    description: 'Reset all metrics counters',
    endpoint: '/registry/stats/reset',
    method: 'POST',
  },
  {
    label: 'Reload Models',
    description: 'Reload model registry',
    endpoint: '/models/reload',
    method: 'POST',
  },
  { label: 'Health Check', description: 'Check system health', endpoint: '/health', method: 'GET' },
  {
    label: 'List Endpoints',
    description: 'List all API routes',
    endpoint: '/routes',
    method: 'GET',
  },
]

export default function DeveloperPage() {
  const [tab, setTab] = useState<string>('shell')
  useRefreshShortcut(() => {
    window.location.reload()
  })
  const { startupStage, startupElapsed } = useLiveStatus()

  return (
    <PageContainer
      title="Developer"
      subtitle="Shell, files, voice & API tools"
      headerRight={
        <div className="flex items-center gap-3">
          {startupStage && startupStage !== 'ready' && startupStage !== 'background' && (
            <div className="flex items-center gap-2 rounded-lg border border-warning/20 bg-warning/[0.06] px-3 py-1.5">
              <StatusDot tone="warning" pulse />
              <span className="text-[10px] font-medium text-warning uppercase tracking-wider">
                {startupStage} {startupElapsed > 0 && `· ${startupElapsed.toFixed(0)}s`}
              </span>
            </div>
          )}
          {startupStage === 'ready' && (
            <div className="flex items-center gap-2 rounded-lg border border-success/20 bg-success/[0.06] px-3 py-1.5">
              <StatusDot tone="success" />
              <span className="text-[10px] font-medium text-success uppercase tracking-wider">
                Ready
              </span>
            </div>
          )}
        </div>
      }
    >
      <Tabs value={tab} onValueChange={setTab}>
        <TabsList aria-label="Developer tools">
          <TabsTrigger value="shell">Terminal</TabsTrigger>
          <TabsTrigger value="files">Files</TabsTrigger>
          <TabsTrigger value="voice">Voice</TabsTrigger>
          <TabsTrigger value="models">Models</TabsTrigger>
          <TabsTrigger value="api">API</TabsTrigger>
          <TabsTrigger value="quick">Quick Actions</TabsTrigger>
          <TabsTrigger value="startup">Startup</TabsTrigger>
        </TabsList>

        <TabsContent value="shell">
          <ShellTab />
        </TabsContent>
        <TabsContent value="files">
          <FilesTab />
        </TabsContent>
        <TabsContent value="voice">
          <VoiceTab />
        </TabsContent>
        <TabsContent value="models">
          <ModelsTab />
        </TabsContent>
        <TabsContent value="api">
          <ApiTab />
        </TabsContent>
        <TabsContent value="quick">
          <QuickActionsTab />
        </TabsContent>
        <TabsContent value="startup">
          <StartupTab />
        </TabsContent>
      </Tabs>
    </PageContainer>
  )
}

function ShellTab() {
  const [shellMode, setShellMode] = useState<'backend' | 'v86'>('backend')

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-1 rounded-xl border border-border/20 bg-muted/20 p-1 w-fit">
        <button
          type="button"
          onClick={() => setShellMode('backend')}
          className={cn(
            'rounded-lg px-4 py-1.5 text-[11px] font-medium transition-all duration-200',
            shellMode === 'backend'
              ? 'bg-muted/30 text-muted-foreground shadow-sm shadow-black/20'
              : 'text-muted-foreground/60 hover:text-muted-foreground/80',
          )}
        >
          Backend
        </button>
        <button
          type="button"
          onClick={() => setShellMode('v86')}
          className={cn(
            'rounded-lg px-4 py-1.5 text-[11px] font-medium transition-all duration-200',
            shellMode === 'v86'
              ? 'bg-muted/30 text-muted-foreground shadow-sm shadow-black/20'
              : 'text-muted-foreground/60 hover:text-muted-foreground/80',
          )}
        >
          Browser VM
        </button>
      </div>
      {shellMode === 'backend' ? (
        <TerminalPanel className="h-[calc(100vh-12rem)]" />
      ) : (
        <V86TerminalPanel className="h-[calc(100vh-12rem)]" />
      )}
    </div>
  )
}

function ModelsTab() {
  const { data: modelsData, isLoading: modelsLoading, refetch: refetchModels } = useModels()
  const { healthLegacy: health } = useLiveStatus()
  const models = modelsData ?? []
  const isOnline = health !== null && health !== 'offline'
  const activeRuntimeId =
    health !== null && health !== 'offline' && health.model_loaded ? health.model_type : null

  const handleModelLoaded = useCallback(async () => {
    await refetchModels()
  }, [refetchModels])

  return (
    <div className="space-y-3">
      <ModelCatalogCard
        models={models}
        modelsLoading={modelsLoading}
        activeRuntimeId={activeRuntimeId}
        onModelLoaded={handleModelLoaded}
      />
      <QuantizationCard isOnline={isOnline} />
      <DownloadsCard />
      <EngineStatusCard />
    </div>
  )
}

function FilesTab() {
  const { data: files = [], isLoading: loading, refetch } = useFileList()
  const [search, setSearch] = useState('')

  const filtered = files.filter((f) => f.filename.toLowerCase().includes(search.toLowerCase()))

  return (
    <div className="space-y-4">
      <FileStatsCard files={files} />
      <div className="rounded-xl border border-border/20 bg-card overflow-hidden">
        <div className="flex items-center justify-between h-11 px-4 bg-muted/30 border-b border-border/20">
          <span className="text-[11px] font-medium text-muted-foreground/80">Documents</span>
          <div className="flex items-center gap-2">
            <input
              type="text"
              placeholder="Search..."
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              className="w-40 h-7 rounded-md border border-border/20 bg-muted/20 px-2.5 text-[11px] text-muted-foreground placeholder:text-muted-foreground/40 outline-none focus:border-white/[0.12] transition-colors font-mono"
              aria-label="Search files"
            />
            <button
              type="button"
              onClick={() => {
                refetch()
              }}
              className="h-7 w-7 flex items-center justify-center rounded-md border border-border/20 bg-muted/20 text-muted-foreground/80 hover:text-muted-foreground hover:border-white/[0.12] transition-colors"
              aria-label="Refresh files"
            >
              <IconRefresh className="w-3 h-3" />
            </button>
          </div>
        </div>
        <div className="px-4 py-2">
          {loading ? (
            <div className="space-y-2 py-2">
              {Array.from({ length: 3 }).map((_, i) => (
                <Skeleton key={i} className="h-7 w-full bg-muted/30" />
              ))}
            </div>
          ) : filtered.length === 0 ? (
            <p className="text-[11px] text-muted-foreground/60 py-6 text-center">
              {search ? 'No files match your search.' : 'No files uploaded yet.'}
            </p>
          ) : (
            <div className="max-h-72 overflow-y-auto py-1">
              {filtered.map((f, i) => (
                <div
                  key={f.id}
                  className={cn(
                    'flex items-center justify-between px-2.5 py-2 rounded-lg text-[11px] transition-colors hover:bg-white/[0.04]',
                    i > 0 && 'border-t border-white/[0.04]',
                  )}
                >
                  <span className="font-mono text-muted-foreground truncate">{f.filename}</span>
                  <span className="text-muted-foreground/60 shrink-0 ml-3 font-mono">
                    {f.size ? `${(f.size / 1024).toFixed(1)} KB` : '—'}
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

function VoiceTab() {
  const [status, setStatus] = useState<VoiceStatus | null>(null)
  const [loading, setLoading] = useState(true)
  const [ttsText, setTtsText] = useState('')
  const [generating, setGenerating] = useState(false)
  const [lastResult, setLastResult] = useState<{ duration_ms: number; backend: string } | null>(
    null,
  )
  const [ttsError, setTtsError] = useState<string | null>(null)

  useEffect(() => {
    voiceController
      .getStatus()
      .then((d) => setStatus(d))
      .catch(() => {})
      .finally(() => setLoading(false))
  }, [])

  const handleGenerate = async () => {
    if (!ttsText.trim()) return
    setGenerating(true)
    setTtsError(null)
    setLastResult(null)
    try {
      const data = await voiceController.tts(ttsText)
      if (data.detail) {
        setTtsError(data.detail)
        return
      }
      setLastResult({ duration_ms: data.duration_ms, backend: data.backend })
      if (data.audio) {
        const audio = new Audio(`data:audio/wav;base64,${data.audio}`)
        audio.play().catch(() => {})
      }
    } catch {
      setTtsError('TTS request failed')
    } finally {
      setGenerating(false)
    }
  }

  return (
    <div className="grid grid-cols-2 gap-4">
      {/* Status card */}
      <div className="rounded-xl border border-border/20 bg-card overflow-hidden">
        <div className="flex items-center h-9 px-4 bg-muted/30 border-b border-border/20">
          <span className="text-[11px] font-medium text-muted-foreground/80">Voice Status</span>
        </div>
        <div className="px-4 py-3">
          {loading ? (
            <Skeleton className="h-14 w-full bg-muted/30" />
          ) : status ? (
            <div className="space-y-2.5">
              <div className="flex items-center gap-2.5">
                <StatusDot tone={status.server_tts ? 'success' : 'destructive'} />
                <span className="text-[12px] text-muted-foreground">
                  {status.server_tts ? 'TTS Available' : 'TTS Unavailable'}
                </span>
              </div>
              <p className="text-[11px] text-muted-foreground/60 font-mono">
                {status.model ?? 'no model'}
              </p>
              {status.error && (
                <div className="flex items-start gap-2 text-destructive bg-destructive/[0.08] rounded-lg px-3 py-2 text-[11px]">
                  <span className="shrink-0 text-[10px] font-bold">!</span>
                  {status.error}
                </div>
              )}
            </div>
          ) : (
            <p className="text-[11px] text-muted-foreground/60">Could not load status</p>
          )}
        </div>
      </div>

      {/* Quick Test card */}
      <div className="rounded-xl border border-border/20 bg-card overflow-hidden">
        <div className="flex items-center h-9 px-4 bg-muted/30 border-b border-border/20">
          <span className="text-[11px] font-medium text-muted-foreground/80">Quick Test</span>
        </div>
        <div className="px-4 py-3 space-y-2.5">
          <textarea
            placeholder="Type text to speak..."
            value={ttsText}
            onChange={(e) => setTtsText(e.target.value)}
            className="w-full h-16 rounded-lg border border-border/20 bg-muted/20 px-3 py-2 text-[12px] text-muted-foreground placeholder:text-muted-foreground/40 resize-none outline-none focus:border-white/[0.12] transition-colors font-mono"
            aria-label="Text to speech input"
          />
          <button
            type="button"
            onClick={handleGenerate}
            disabled={generating || !ttsText.trim()}
            className={cn(
              'w-full h-8 rounded-lg text-[11px] font-medium transition-all duration-200',
              generating || !ttsText.trim()
                ? 'bg-success/20 text-success/40 cursor-not-allowed'
                : 'bg-success/10 text-success hover:bg-success/20',
            )}
          >
            {generating ? 'Generating...' : 'Speak'}
          </button>
          {ttsError && (
            <div className="flex items-start gap-2 text-destructive bg-destructive/[0.08] rounded-lg px-3 py-2 text-[11px]">
              <span className="shrink-0 text-[10px] font-bold">!</span>
              {ttsError}
            </div>
          )}
          {lastResult && (
            <p className="text-[10px] text-muted-foreground/60 font-mono">
              {lastResult.duration_ms}ms · {lastResult.backend}
            </p>
          )}
        </div>
      </div>
    </div>
  )
}

interface HistoryEntry {
  id: string
  method: string
  path: string
  body: string
  status: number
  timeMs: number
  timestamp: number
}

const HISTORY_KEY = 'dev-api-playground-history'

function loadHistory(): HistoryEntry[] {
  try {
    return JSON.parse(localStorage.getItem(HISTORY_KEY) || '[]')
  } catch {
    return []
  }
}

function saveHistory(entries: HistoryEntry[]) {
  localStorage.setItem(HISTORY_KEY, JSON.stringify(entries.slice(0, 50)))
}

function ApiTab() {
  const [method, setMethod] = useState('GET')
  const [path, setPath] = useState('/health')
  const [body, setBody] = useState('')
  const [authHeader, setAuthHeader] = useState('')
  const [loading, setLoading] = useState(false)
  const [response, setResponse] = useState<string | null>(null)
  const [responseStatus, setResponseStatus] = useState<number | null>(null)
  const [responseTime, setResponseTime] = useState<number | null>(null)
  const [responseHeaders, setResponseHeaders] = useState<Record<string, string> | null>(null)
  const [history, setHistory] = useState<HistoryEntry[]>(() => loadHistory())

  const METHODS = ['GET', 'POST', 'PUT', 'PATCH', 'DELETE'] as const

  const handleSend = async () => {
    setLoading(true)
    setResponse(null)
    setResponseStatus(null)
    setResponseTime(null)
    setResponseHeaders(null)

    const start = Date.now()
    try {
      const baseUrl = window.location.origin.replace(':3001', ':8000')
      const headers: Record<string, string> = {}
      if (body && method !== 'GET') {
        headers['Content-Type'] = 'application/json'
      }
      if (authHeader.trim()) {
        headers['Authorization'] = authHeader.trim()
      }
      const res = await authFetch(`${baseUrl}${path}`, {
        method,
        headers,
        body: body && method !== 'GET' ? body : undefined,
        noAuth: true,
      })
      const elapsed = Date.now() - start
      setResponseStatus(res.status)
      setResponseTime(elapsed)

      const hdrs: Record<string, string> = {}
      res.headers.forEach((v, k) => {
        hdrs[k] = v
      })
      setResponseHeaders(hdrs)

      const text = await res.text()
      try {
        setResponse(JSON.stringify(JSON.parse(text), null, 2))
      } catch {
        setResponse(text)
      }

      const entry: HistoryEntry = {
        id: Date.now().toString(36),
        method,
        path,
        body,
        status: res.status,
        timeMs: elapsed,
        timestamp: Date.now(),
      }
      const updated = [entry, ...history].slice(0, 50)
      setHistory(updated)
      saveHistory(updated)
    } catch (err) {
      setResponse(err instanceof Error ? err.message : 'Request failed')
      setResponseStatus(0)
    } finally {
      setLoading(false)
    }
  }

  const loadFromHistory = (entry: HistoryEntry) => {
    setMethod(entry.method)
    setPath(entry.path)
    setBody(entry.body)
  }

  const clearHistory = () => {
    setHistory([])
    localStorage.removeItem(HISTORY_KEY)
  }

  const METHOD_COLORS: Record<string, string> = {
    GET: 'bg-success/10 text-success border-success/20',
    POST: 'bg-warning/10 text-warning border-warning/20',
    PUT: 'bg-info/60/10 text-info border-info/60/20',
    PATCH: 'bg-primary/10 text-primary border-primary/20',
    DELETE: 'bg-destructive/10 text-destructive border-destructive/20',
  }

  return (
    <div className="space-y-4">
      {/* API Playground */}
      <div className="rounded-xl border border-border/20 bg-card overflow-hidden">
        <div className="flex items-center h-11 px-4 bg-muted/30 border-b border-border/20">
          <span className="text-[11px] font-medium text-muted-foreground/80">API Playground</span>
        </div>
        <div className="px-4 py-3 space-y-3">
          {/* URL bar */}
          <div className="flex gap-2">
            <select
              value={method}
              onChange={(e) => setMethod(e.target.value)}
              className={cn(
                'h-8 rounded-lg border px-2.5 text-[11px] font-mono font-medium bg-muted/20 outline-none',
                METHOD_COLORS[method] ?? 'border-border/20 text-muted-foreground',
              )}
              aria-label="HTTP method"
            >
              {METHODS.map((m) => (
                <option key={m} value={m}>
                  {m}
                </option>
              ))}
            </select>
            <input
              value={path}
              onChange={(e) => setPath(e.target.value)}
              placeholder="/endpoint"
              className="flex-1 h-8 rounded-lg border border-border/20 bg-muted/20 px-3 text-[12px] font-mono text-muted-foreground placeholder:text-muted-foreground/40 outline-none focus:border-white/[0.12] transition-colors"
              aria-label="Request path"
              onKeyDown={(e) => {
                if (e.key === 'Enter') handleSend()
              }}
            />
            <button
              type="button"
              onClick={handleSend}
              disabled={loading || !path.trim()}
              className={cn(
                'h-8 px-4 rounded-lg text-[11px] font-medium transition-all duration-200',
                loading || !path.trim()
                  ? 'bg-info/20 text-info/40 cursor-not-allowed'
                  : 'bg-info/10 text-info hover:bg-info/20',
              )}
            >
              {loading ? 'Sending...' : 'Send'}
            </button>
          </div>

          {/* Auth header */}
          <input
            value={authHeader}
            onChange={(e) => setAuthHeader(e.target.value)}
            placeholder="Authorization: Bearer <token>"
            className="w-full h-8 rounded-lg border border-border/20 bg-muted/20 px-3 text-[11px] font-mono text-muted-foreground placeholder:text-muted-foreground/40 outline-none focus:border-white/[0.12] transition-colors"
            aria-label="Authorization header"
          />

          {/* Body */}
          {method !== 'GET' && (
            <textarea
              value={body}
              onChange={(e) => setBody(e.target.value)}
              placeholder='{"key": "value"}'
              className="w-full h-24 rounded-lg border border-border/20 bg-muted/20 px-3 py-2 text-[11px] font-mono text-muted-foreground placeholder:text-muted-foreground/40 resize-none outline-none focus:border-white/[0.12] transition-colors"
              aria-label="Request body"
            />
          )}

          {/* Response */}
          {response !== null && (
            <div className="space-y-2">
              <div className="flex items-center gap-3 text-[11px]">
                <span
                  className={cn(
                    'font-mono font-medium',
                    responseStatus && responseStatus >= 200 && responseStatus < 300
                      ? 'text-success'
                      : responseStatus && responseStatus >= 400
                        ? 'text-destructive'
                        : 'text-muted-foreground/60',
                  )}
                >
                  {responseStatus}
                </span>
                {responseTime != null && (
                  <span className="text-muted-foreground/60 font-mono">{responseTime}ms</span>
                )}
              </div>

              {responseHeaders && Object.keys(responseHeaders).length > 0 && (
                <details className="text-[11px]">
                  <summary className="text-muted-foreground/60 cursor-pointer hover:text-muted-foreground/80 transition-colors">
                    Response Headers
                  </summary>
                  <pre className="mt-1.5 rounded-lg border border-white/[0.04] bg-muted/20 p-2.5 font-mono text-[10px] text-muted-foreground overflow-auto max-h-32 leading-relaxed">
                    {Object.entries(responseHeaders)
                      .map(([k, v]) => `${k}: ${v}`)
                      .join('\n')}
                  </pre>
                </details>
              )}

              <pre className="rounded-xl border border-white/[0.04] bg-muted/20 p-3.5 text-[11px] font-mono text-muted-foreground overflow-auto max-h-96 whitespace-pre-wrap leading-relaxed">
                {response}
              </pre>
            </div>
          )}
        </div>
      </div>

      {/* History */}
      {history.length > 0 && (
        <div className="rounded-xl border border-border/20 bg-card overflow-hidden">
          <div className="flex items-center justify-between h-11 px-4 bg-muted/30 border-b border-border/20">
            <span className="text-[11px] font-medium text-muted-foreground/80">
              Request History
            </span>
            <button
              type="button"
              onClick={clearHistory}
              className="text-[10px] text-destructive/60 hover:text-destructive transition-colors"
            >
              Clear
            </button>
          </div>
          <div className="px-2 py-1 max-h-56 overflow-y-auto">
            {history.map((entry, i) => (
              <button
                key={entry.id}
                type="button"
                onClick={() => loadFromHistory(entry)}
                className={cn(
                  'w-full flex items-center gap-2.5 px-3 py-2 rounded-lg text-[11px] hover:bg-white/[0.04] transition-colors text-left',
                  i > 0 && 'border-t border-white/[0.03]',
                )}
              >
                <span
                  className={cn(
                    'font-mono font-medium w-10 shrink-0',
                    entry.status >= 200 && entry.status < 300
                      ? 'text-success'
                      : entry.status >= 400
                        ? 'text-destructive'
                        : 'text-muted-foreground/60',
                  )}
                >
                  {entry.status || 'ERR'}
                </span>
                <span className="font-mono text-muted-foreground/40 w-12 shrink-0">
                  {entry.method}
                </span>
                <span className="font-mono text-muted-foreground truncate flex-1">
                  {entry.path}
                </span>
                <span className="text-muted-foreground/60 shrink-0 font-mono">
                  {entry.timeMs}ms
                </span>
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

function QuickActionsTab() {
  const [results, setResults] = useState<
    Record<string, { status: number; body: string; timeMs: number } | null>
  >({})
  const [running, setRunning] = useState<string | null>(null)

  const executeAction = async (action: (typeof QUICK_ACTIONS)[0]) => {
    setRunning(action.label)
    setResults((prev) => ({ ...prev, [action.label]: null }))
    const start = Date.now()

    try {
      const res = await authFetch(action.endpoint, { method: action.method, noAuth: true })
      const elapsed = Date.now() - start
      const text = await res.text()
      setResults((prev) => ({
        ...prev,
        [action.label]: { status: res.status, body: text, timeMs: elapsed },
      }))
    } catch (err) {
      setResults((prev) => ({
        ...prev,
        [action.label]: {
          status: 0,
          body: err instanceof Error ? err.message : 'Failed',
          timeMs: Date.now() - start,
        },
      }))
    } finally {
      setRunning(null)
    }
  }

  const METHOD_COLORS: Record<string, string> = {
    GET: 'bg-success/10 text-success',
    POST: 'bg-warning/10 text-warning',
    PUT: 'bg-info/10 text-info',
    PATCH: 'bg-primary/10 text-primary',
    DELETE: 'bg-destructive/10 text-destructive',
  }

  return (
    <div className="space-y-4">
      <div className="grid gap-2.5 sm:grid-cols-2 lg:grid-cols-3">
        {QUICK_ACTIONS.map((action) => (
          <button
            key={action.label}
            type="button"
            onClick={() => executeAction(action)}
            disabled={running !== null}
            className={cn(
              'group flex flex-col items-start p-4 rounded-xl border text-left transition-all duration-200',
              'hover:-translate-y-0.5 hover:shadow-lg hover:shadow-black/10',
              running === action.label
                ? 'border-primary/40 bg-primary/[0.04]'
                : results[action.label]
                  ? results[action.label]!.status >= 200 && results[action.label]!.status < 300
                    ? 'border-success/20 bg-success/[0.03]'
                    : 'border-destructive/20 bg-destructive/[0.03]'
                  : 'border-border/20 bg-white/[0.02] hover:border-white/[0.12] hover:bg-white/[0.04]',
            )}
          >
            <div className="flex items-center gap-2 w-full mb-1">
              <span
                className={cn(
                  'text-[10px] font-mono font-medium px-1.5 py-0.5 rounded',
                  METHOD_COLORS[action.method] ?? 'bg-muted text-muted-foreground',
                )}
              >
                {action.method}
              </span>
              <span className="text-sm font-medium text-border">{action.label}</span>
              {running === action.label && (
                <span className="ml-auto flex items-center gap-1.5 text-[10px] text-warning">
                  <StatusDot tone="warning" pulse />
                  running
                </span>
              )}
              {results[action.label] && running !== action.label && (
                <span
                  className={cn(
                    'ml-auto text-[10px] font-mono',
                    results[action.label]!.status >= 200 && results[action.label]!.status < 300
                      ? 'text-success'
                      : results[action.label]!.status >= 400
                        ? 'text-warning'
                        : 'text-destructive',
                  )}
                >
                  {results[action.label]!.status || 'ERR'}
                </span>
              )}
            </div>
            <span className="text-[11px] text-muted-foreground/80">{action.description}</span>
            <span className="text-[10px] text-muted-foreground/40 font-mono mt-1.5">
              {action.endpoint}
            </span>
          </button>
        ))}
      </div>

      {Object.entries(results).map(
        ([label, result]) =>
          result && (
            <div key={label} className="rounded-xl border border-border/20 bg-card overflow-hidden">
              <div className="flex items-center justify-between px-4 py-2.5 bg-muted/30 border-b border-[#0.06]">
                <span className="text-[11px] font-medium text-muted-foreground">{label}</span>
                <div className="flex items-center gap-2">
                  <span
                    className={cn(
                      'text-[10px] font-mono',
                      result.status >= 200 && result.status < 300
                        ? 'text-success'
                        : result.status >= 400
                          ? 'text-warning'
                          : 'text-destructive',
                    )}
                  >
                    {result.status || 'ERR'}
                  </span>
                  <span className="text-[10px] text-muted-foreground/60 font-mono">
                    {result.timeMs}ms
                  </span>
                </div>
              </div>
              <pre className="px-4 py-3 text-[11px] font-mono text-muted-foreground overflow-auto max-h-48 whitespace-pre-wrap leading-relaxed">
                {result.body || 'No response'}
              </pre>
            </div>
          ),
      )}
    </div>
  )
}

function StartupTab() {
  const {
    health,
    startupStage,
    startupElapsed,
    startupModelProgress,
    startupModelProgressMessage,
  } = useLiveStatus()
  const [startupData, setStartupData] = useState<Record<string, unknown> | null>(null)
  const [historyData, setHistoryData] = useState<Record<string, unknown> | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const fetchStartup = async () => {
      try {
        const [progressRes, historyRes] = await Promise.all([
          fetch('/health/startup-progress'),
          fetch('/health/startup-history'),
        ])
        const progressJson = await progressRes.json()
        const historyJson = await historyRes.json()
        setStartupData(progressJson.data)
        setHistoryData(historyJson.data)
      } catch {
        // ignore
      } finally {
        setLoading(false)
      }
    }
    fetchStartup()
    const interval = setInterval(fetchStartup, 2000)
    return () => clearInterval(interval)
  }, [])

  const STAGE_COLORS: Record<string, string> = {
    init: 'text-muted-foreground/60',
    critical: 'text-warning',
    ready: 'text-info',
    background: 'text-success',
  }

  const HOOK_STATUS_COLORS: Record<string, string> = {
    pending: 'text-muted-foreground/60',
    running: 'text-warning',
    ok: 'text-success',
    timeout: 'text-destructive',
    error: 'text-destructive',
  }

  const hooks = startupData?.hooks as
    | Record<
        string,
        {
          name: string
          stage: string
          status: string
          duration_seconds: number
          error: string | null
        }
      >
    | undefined
  const stages = startupData?.stages as
    Record<string, { hooks: string[]; time: number | null }> | undefined
  const stats = historyData?.stats as
    | {
        count: number
        avg_duration: number
        min_duration: number
        max_duration: number
        p50_duration: number
        p95_duration: number
        success_rate: number
      }
    | undefined
  const stageStats = historyData?.stage_stats as
    | Record<string, { avg: number; min: number; max: number; p50: number; count: number }>
    | undefined
  const alerts = historyData?.alerts as
    Array<{ type: string; severity: string; message: string; timestamp: number }> | undefined
  const suggestions = historyData?.suggestions as
    | Array<{ type: string; severity: string; message: string; stage?: string; hook?: string }>
    | undefined

  return (
    <div className="space-y-4">
      {/* Alerts */}
      {alerts && alerts.length > 0 && (
        <div className="space-y-2">
          {alerts.map((alert, i) => (
            <div
              key={`${alert.type}-${i}`}
              className={cn(
                'rounded-xl border px-4 py-3 flex items-center gap-3',
                alert.severity === 'error'
                  ? 'border-destructive/20 bg-destructive/[0.06]'
                  : 'border-warning/20 bg-warning/[0.06]',
              )}
            >
              <StatusDot tone={alert.severity === 'error' ? 'destructive' : 'warning'} />
              <div className="flex-1 min-w-0">
                <span
                  className={cn(
                    'text-[11px] font-medium',
                    alert.severity === 'error' ? 'text-destructive' : 'text-warning',
                  )}
                >
                  {alert.type.replace(/_/g, ' ')}
                </span>
                <span className="text-[10px] text-muted-foreground/60 ml-2">{alert.message}</span>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Suggestions */}
      {suggestions && suggestions.length > 0 && (
        <div className="rounded-xl border border-border/20 bg-card overflow-hidden">
          <div className="h-9 px-4 bg-muted/30 border-b border-border/20 flex items-center">
            <span className="text-[11px] font-medium text-muted-foreground/80">
              Optimization Suggestions
            </span>
            <span className="text-[9px] text-muted-foreground/60 ml-2">
              {suggestions.length} suggestions
            </span>
          </div>
          <div className="divide-y divide-white/[0.04]">
            {suggestions.map((suggestion, i) => (
              <div key={`${suggestion.type}-${i}`} className="px-4 py-3 flex items-start gap-3">
                <StatusDot
                  tone={
                    suggestion.severity === 'error'
                      ? 'destructive'
                      : suggestion.severity === 'warning'
                        ? 'warning'
                        : 'primary'
                  }
                />
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2">
                    <span
                      className={cn(
                        'text-[11px] font-medium',
                        suggestion.severity === 'error'
                          ? 'text-destructive'
                          : suggestion.severity === 'warning'
                            ? 'text-warning'
                            : 'text-info',
                      )}
                    >
                      {suggestion.type.replace(/_/g, ' ')}
                    </span>
                    {suggestion.stage && (
                      <span className="text-[9px] text-muted-foreground/60 bg-muted/30 px-1.5 py-0.5 rounded">
                        {suggestion.stage}
                      </span>
                    )}
                    {suggestion.hook && (
                      <span className="text-[9px] text-muted-foreground/60 bg-muted/30 px-1.5 py-0.5 rounded">
                        {suggestion.hook}
                      </span>
                    )}
                  </div>
                  <span className="text-[10px] text-muted-foreground/80">{suggestion.message}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* History Chart */}
      <StartupHistoryChart />

      {/* Overview */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
        <div className="rounded-xl border border-border/20 bg-card p-4">
          <div className="text-[10px] text-muted-foreground/60 uppercase tracking-wider mb-1">
            Stage
          </div>
          <div
            className={cn(
              'text-[14px] font-medium font-mono',
              STAGE_COLORS[startupStage] ?? 'text-muted-foreground',
            )}
          >
            {startupStage}
          </div>
        </div>
        <div className="rounded-xl border border-border/20 bg-card p-4">
          <div className="text-[10px] text-muted-foreground/60 uppercase tracking-wider mb-1">
            Elapsed
          </div>
          <div className="text-[14px] font-medium font-mono text-muted-foreground">
            {startupElapsed.toFixed(1)}s
          </div>
        </div>
        <div className="rounded-xl border border-border/20 bg-card p-4">
          <div className="text-[10px] text-muted-foreground/60 uppercase tracking-wider mb-1">
            Model
          </div>
          <div className="text-[14px] font-medium font-mono text-muted-foreground">
            {Math.round(startupModelProgress * 100)}%
          </div>
        </div>
        <div className="rounded-xl border border-border/20 bg-card p-4">
          <div className="text-[10px] text-muted-foreground/60 uppercase tracking-wider mb-1">
            Status
          </div>
          <div
            className={cn(
              'text-[14px] font-medium',
              startupStage === 'background' ? 'text-success' : 'text-warning',
            )}
          >
            {startupStage === 'background' ? 'Ready' : 'Starting'}
          </div>
        </div>
      </div>

      {/* Progress bar */}
      <div className="rounded-xl border border-border/20 bg-card p-4">
        <div className="flex items-center justify-between mb-2">
          <span className="text-[11px] text-muted-foreground/80">Startup Progress</span>
          <span className="text-[11px] text-muted-foreground/60 font-mono">
            {startupModelProgressMessage}
          </span>
        </div>
        <div className="h-2 rounded-full bg-muted/30 overflow-hidden">
          <div
            className="h-full rounded-full bg-gradient-to-r from-info to-success transition-all duration-300"
            style={{
              width: `${Math.min(100, (startupModelProgress || (startupStage === 'background' ? 1 : startupStage === 'ready' ? 0.6 : startupStage === 'critical' ? 0.3 : 0.1)) * 100)}%`,
            }}
          />
        </div>
        <div className="flex justify-between mt-1">
          {['init', 'critical', 'ready', 'background'].map((stage, i) => (
            <span
              key={stage}
              className={cn(
                'text-[9px] font-mono',
                i <= ['init', 'critical', 'ready', 'background'].indexOf(startupStage)
                  ? 'text-success'
                  : 'text-foreground',
              )}
            >
              {stage}
            </span>
          ))}
        </div>
      </div>

      {/* Performance comparison */}
      {stats && stats.count > 0 && (
        <div className="rounded-xl border border-border/20 bg-card overflow-hidden">
          <div className="h-9 px-4 bg-muted/30 border-b border-border/20 flex items-center">
            <span className="text-[11px] font-medium text-muted-foreground/80">
              Performance History
            </span>
            <span className="text-[9px] text-muted-foreground/60 ml-2">
              {stats.count} startups recorded
            </span>
          </div>
          <div className="p-4">
            <div className="grid grid-cols-2 lg:grid-cols-5 gap-3">
              <div className="text-center">
                <div className="text-[10px] text-muted-foreground/60 uppercase tracking-wider mb-1">
                  Avg
                </div>
                <div className="text-[14px] font-mono text-muted-foreground">
                  {stats.avg_duration}s
                </div>
              </div>
              <div className="text-center">
                <div className="text-[10px] text-muted-foreground/60 uppercase tracking-wider mb-1">
                  P50
                </div>
                <div className="text-[14px] font-mono text-muted-foreground">
                  {stats.p50_duration}s
                </div>
              </div>
              <div className="text-center">
                <div className="text-[10px] text-muted-foreground/60 uppercase tracking-wider mb-1">
                  P95
                </div>
                <div className="text-[14px] font-mono text-muted-foreground">
                  {stats.p95_duration}s
                </div>
              </div>
              <div className="text-center">
                <div className="text-[10px] text-muted-foreground/60 uppercase tracking-wider mb-1">
                  Min
                </div>
                <div className="text-[14px] font-mono text-success">{stats.min_duration}s</div>
              </div>
              <div className="text-center">
                <div className="text-[10px] text-muted-foreground/60 uppercase tracking-wider mb-1">
                  Max
                </div>
                <div className="text-[14px] font-mono text-destructive">{stats.max_duration}s</div>
              </div>
            </div>
            {startupElapsed > 0 && stats.avg_duration > 0 && (
              <div className="mt-3 pt-3 border-t border-white/[0.04] flex items-center justify-center gap-2">
                <span className="text-[10px] text-muted-foreground/60">Current:</span>
                <span
                  className={cn(
                    'text-[11px] font-mono font-medium',
                    startupElapsed < stats.avg_duration ? 'text-success' : 'text-warning',
                  )}
                >
                  {startupElapsed.toFixed(1)}s
                </span>
                <span className="text-[10px] text-muted-foreground/60">
                  vs avg {stats.avg_duration}s
                </span>
                {startupElapsed < stats.avg_duration && (
                  <span className="text-[9px] text-success">
                    ({((1 - startupElapsed / stats.avg_duration) * 100).toFixed(0)}% faster)
                  </span>
                )}
                {startupElapsed > stats.avg_duration && (
                  <span className="text-[9px] text-warning">
                    (+{((startupElapsed / stats.avg_duration - 1) * 100).toFixed(0)}% slower)
                  </span>
                )}
              </div>
            )}
          </div>
        </div>
      )}

      {/* Stages */}
      {stages && (
        <div className="rounded-xl border border-border/20 bg-card overflow-hidden">
          <div className="h-9 px-4 bg-muted/30 border-b border-border/20 flex items-center">
            <span className="text-[11px] font-medium text-muted-foreground/80">Stages</span>
          </div>
          <div className="divide-y divide-white/[0.04]">
            {Object.entries(stages).map(([stage, data]) => (
              <div key={stage} className="px-4 py-3 flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <StatusDot
                    tone={
                      stage === startupStage ? 'primary' : data.time !== null ? 'success' : 'muted'
                    }
                    pulse={stage === startupStage}
                  />
                  <span className="text-[12px] font-medium text-muted-foreground capitalize">
                    {stage}
                  </span>
                  <span className="text-[10px] text-muted-foreground/60 font-mono">
                    {data.hooks.length} hooks
                  </span>
                </div>
                <span className="text-[11px] font-mono text-muted-foreground/60">
                  {data.time !== null ? `${data.time}s` : '—'}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Stage Stats */}
      {stageStats && Object.keys(stageStats).length > 0 && (
        <div className="rounded-xl border border-border/20 bg-card overflow-hidden">
          <div className="h-9 px-4 bg-muted/30 border-b border-border/20 flex items-center">
            <span className="text-[11px] font-medium text-muted-foreground/80">
              Stage Performance
            </span>
          </div>
          <div className="p-4">
            <div className="grid grid-cols-2 lg:grid-cols-3 gap-3">
              {Object.entries(stageStats).map(([stage, data]) => (
                <div key={stage} className="rounded-lg border border-white/[0.04] bg-muted/20 p-3">
                  <div className="flex items-center gap-2 mb-2">
                    <StatusDot tone={stage === startupStage ? 'primary' : 'success'} />
                    <span className="text-[11px] font-medium text-muted-foreground capitalize">
                      {stage}
                    </span>
                  </div>
                  <div className="grid grid-cols-2 gap-2 text-[10px]">
                    <div>
                      <span className="text-muted-foreground/60">Avg</span>
                      <span className="ml-1 font-mono text-muted-foreground">{data.avg}s</span>
                    </div>
                    <div>
                      <span className="text-muted-foreground/60">P50</span>
                      <span className="ml-1 font-mono text-muted-foreground">{data.p50}s</span>
                    </div>
                    <div>
                      <span className="text-muted-foreground/60">Min</span>
                      <span className="ml-1 font-mono text-success">{data.min}s</span>
                    </div>
                    <div>
                      <span className="text-muted-foreground/60">Max</span>
                      <span className="ml-1 font-mono text-destructive">{data.max}s</span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Timeline */}
      <StartupTimeline />

      {/* Hooks */}
      {hooks && Object.keys(hooks).length > 0 && (
        <div className="rounded-xl border border-border/20 bg-card overflow-hidden">
          <div className="h-9 px-4 bg-muted/30 border-b border-border/20 flex items-center">
            <span className="text-[11px] font-medium text-muted-foreground/80">Hooks</span>
          </div>
          <div className="divide-y divide-white/[0.04]">
            {Object.values(hooks).map((hook) => (
              <div key={hook.name} className="px-4 py-2.5 flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <StatusDot
                    tone={
                      hook.status === 'ok'
                        ? 'success'
                        : hook.status === 'running'
                          ? 'warning'
                          : hook.status === 'error' || hook.status === 'timeout'
                            ? 'destructive'
                            : 'muted'
                    }
                    pulse={hook.status === 'running'}
                  />
                  <span className="text-[11px] font-mono text-muted-foreground">{hook.name}</span>
                  <span className="text-[9px] text-muted-foreground/60 uppercase">
                    {hook.stage}
                  </span>
                </div>
                <div className="flex items-center gap-3">
                  {hook.error && (
                    <span className="text-[9px] text-destructive max-w-32 truncate">
                      {hook.error}
                    </span>
                  )}
                  <span className="text-[10px] font-mono text-muted-foreground/60">
                    {hook.duration_seconds > 0 ? `${hook.duration_seconds}s` : '—'}
                  </span>
                  <span
                    className={cn(
                      'text-[9px] font-mono uppercase',
                      HOOK_STATUS_COLORS[hook.status],
                    )}
                  >
                    {hook.status}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
