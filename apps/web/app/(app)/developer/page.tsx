'use client'
export const dynamic = 'force-dynamic'

import { useState, useEffect, useCallback } from 'react'
import { PageContainer } from '@/components/PageContainer'
import {
  Card, CardContent, CardHeader, CardTitle, Button, Input,
  Textarea, Skeleton, cn,
  Tabs, TabsList, TabsTrigger, TabsContent,
} from '@sloughgpt/strui'
import { IconRefresh } from '@sloughgpt/strui'
import { StatusBanner } from '@/components/composed/StatusBanner'
import { TerminalPanel } from '@/components/shell/TerminalPanel'
import { V86TerminalPanel } from '@/components/shell/V86TerminalPanel'
import { FileStatsCard } from '@/components/files/FileStatsCard'
import { filesController, type FileEntry } from '@/lib/files-controller'
import { voiceController, type VoiceStatus } from '@/lib/voice-controller'
import { authFetch } from '@/lib/http-client'
import { useRefreshShortcut } from '@/hooks/useRefreshShortcut'


const QUICK_ACTIONS = [
  { label: 'Restart Backend', description: 'Restart the FastAPI server', endpoint: '/system/restart', method: 'POST' },
  { label: 'Clear Cache', description: 'Clear model cache', endpoint: '/cache/clear', method: 'POST' },
  { label: 'Reset Metrics', description: 'Reset all metrics counters', endpoint: '/registry/stats/reset', method: 'POST' },
  { label: 'Reload Models', description: 'Reload model registry', endpoint: '/models/reload', method: 'POST' },
  { label: 'Health Check', description: 'Check system health', endpoint: '/health', method: 'GET' },
  { label: 'List Endpoints', description: 'List all API routes', endpoint: '/routes', method: 'GET' },
]

export default function DeveloperPage() {
  const [tab, setTab] = useState<string>('shell')
  useRefreshShortcut(() => { window.location.reload() })

  return (
    <PageContainer title="Developer" subtitle="Terminal, files & quick actions">
      <Tabs value={tab} onValueChange={setTab}>
        <TabsList aria-label="Developer tools">
          <TabsTrigger value="shell">Terminal</TabsTrigger>
          <TabsTrigger value="files">Files</TabsTrigger>
          <TabsTrigger value="voice">Voice</TabsTrigger>
          <TabsTrigger value="api">API</TabsTrigger>
          <TabsTrigger value="quick">Quick Actions</TabsTrigger>
        </TabsList>

        <TabsContent value="shell"><ShellTab /></TabsContent>
        <TabsContent value="files"><FilesTab /></TabsContent>
        <TabsContent value="voice"><VoiceTab /></TabsContent>
        <TabsContent value="api"><ApiTab /></TabsContent>
        <TabsContent value="quick"><QuickActionsTab /></TabsContent>
      </Tabs>
    </PageContainer>
  )
}

function ShellTab() {
  const [shellMode, setShellMode] = useState<'backend' | 'v86'>('backend')

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-1 rounded-lg border border-border/60 bg-muted/30 p-1 w-fit">
        <button
          type="button"
          onClick={() => setShellMode('backend')}
          className={cn(
            'rounded-md px-3 py-1 text-xs font-medium transition-all',
            shellMode === 'backend'
              ? 'bg-background text-foreground shadow-sm'
              : 'text-muted-foreground hover:text-foreground',
          )}
        >
          Backend
        </button>
        <button
          type="button"
          onClick={() => setShellMode('v86')}
          className={cn(
            'rounded-md px-3 py-1 text-xs font-medium transition-all',
            shellMode === 'v86'
              ? 'bg-background text-foreground shadow-sm'
              : 'text-muted-foreground hover:text-foreground',
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

function FilesTab() {
  const [files, setFiles] = useState<FileEntry[]>([])
  const [loading, setLoading] = useState(true)
  const [search, setSearch] = useState('')

  const fetchFiles = useCallback(async () => {
    try {
      setFiles(await filesController.list())
    } catch {
      // silent
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => { fetchFiles() }, [fetchFiles])

  const filtered = files.filter(f =>
    f.filename.toLowerCase().includes(search.toLowerCase()),
  )

  return (
    <div className="space-y-4">
      <FileStatsCard files={files} />
      <Card>
        <CardHeader className="flex flex-row items-center justify-between">
          <CardTitle className="text-base">Documents</CardTitle>
          <div className="flex gap-2">
            <Input
              placeholder="Search files..."
              value={search}
              onChange={e => setSearch(e.target.value)}
              className="w-48 h-8 text-xs"
              aria-label="Search files"
            />
            <Button variant="ghost" size="sm" onClick={() => { setLoading(true); fetchFiles() }} aria-label="Refresh files">
              <IconRefresh className="w-3.5 h-3.5" />
            </Button>
          </div>
        </CardHeader>
        <CardContent>
          {loading ? (
            <div className="space-y-2">
              {Array.from({ length: 3 }).map((_, i) => (
                <Skeleton key={i} className="h-8 w-full" />
              ))}
            </div>
          ) : filtered.length === 0 ? (
            <p className="text-xs text-muted-foreground py-4 text-center">
              {search ? 'No files match your search.' : 'No files uploaded yet.'}
            </p>
          ) : (
            <div className="space-y-1 max-h-64 overflow-y-auto">
              {filtered.map(f => (
                <div
                  key={f.id}
                  className="flex items-center justify-between px-2 py-1.5 rounded text-xs hover:bg-muted/50 transition-colors"
                >
                  <span className="font-mono truncate">{f.filename}</span>
                  <span className="text-muted-foreground shrink-0 ml-2">
                    {f.size ? `${(f.size / 1024).toFixed(1)} KB` : '—'}
                  </span>
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  )
}

function VoiceTab() {
  const [status, setStatus] = useState<VoiceStatus | null>(null)
  const [loading, setLoading] = useState(true)
  const [ttsText, setTtsText] = useState('')
  const [generating, setGenerating] = useState(false)
  const [lastResult, setLastResult] = useState<{ duration_ms: number; backend: string } | null>(null)
  const [ttsError, setTtsError] = useState<string | null>(null)

  useEffect(() => {
    voiceController.getStatus()
      .then(d => setStatus(d))
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
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-4">
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Voice Status</CardTitle>
          </CardHeader>
          <CardContent>
            {loading ? (
              <Skeleton className="h-16 w-full" />
            ) : status ? (
              <div className="space-y-2">
                <div className="flex items-center gap-2">
                  <span className={cn(
                    'w-2 h-2 rounded-full',
                    status.server_tts ? 'bg-emerald-500' : 'bg-red-500',
                  )} />
                  <span className="text-xs">{status.server_tts ? 'TTS Available' : 'TTS Unavailable'}</span>
                </div>
                <p className="text-xs text-muted-foreground">
                  Model: {status.model ?? 'none'}
                </p>
                {status.error && (
                  <StatusBanner variant="error" message={status.error} dismissible={false} />
                )}
              </div>
            ) : (
              <p className="text-xs text-muted-foreground">Could not load status</p>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-base">Quick Test</CardTitle>
          </CardHeader>
          <CardContent className="space-y-2">
            <Textarea
              placeholder="Type text to speak..."
              value={ttsText}
              onChange={e => setTtsText(e.target.value)}
              className="h-16 text-xs resize-none"
              aria-label="Text to speech input"
            />
            <Button
              size="sm"
              onClick={handleGenerate}
              disabled={generating || !ttsText.trim()}
              className="w-full"
            >
              {generating ? 'Generating...' : 'Speak'}
            </Button>
            {ttsError && (
              <StatusBanner variant="error" message={ttsError} dismissible={false} />
            )}
            {lastResult && (
              <p className="text-xs text-muted-foreground">
                {lastResult.duration_ms}ms · {lastResult.backend}
              </p>
            )}
          </CardContent>
        </Card>
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
      const res = await authFetch(`${baseUrl}${path}`, { method, headers, body: body && method !== 'GET' ? body : undefined, noAuth: true })
      const elapsed = Date.now() - start
      setResponseStatus(res.status)
      setResponseTime(elapsed)

      const hdrs: Record<string, string> = {}
      res.headers.forEach((v, k) => { hdrs[k] = v })
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

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <CardTitle className="text-base">API Playground</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="flex gap-2">
            <select
              value={method}
              onChange={e => setMethod(e.target.value)}
              className="h-8 rounded-md border border-border bg-muted/50 px-2 text-xs font-mono"
              aria-label="HTTP method"
            >
              {METHODS.map(m => <option key={m} value={m}>{m}</option>)}
            </select>
            <Input
              value={path}
              onChange={e => setPath(e.target.value)}
              placeholder="/endpoint"
              className="h-8 text-xs font-mono flex-1"
              aria-label="Request path"
              onKeyDown={e => { if (e.key === 'Enter') handleSend() }}
            />
            <Button size="sm" onClick={handleSend} disabled={loading || !path.trim()}>
              {loading ? 'Sending...' : 'Send'}
            </Button>
          </div>

          <Input
            value={authHeader}
            onChange={e => setAuthHeader(e.target.value)}
            placeholder="Authorization: Bearer <token>"
            className="h-8 text-xs font-mono"
            aria-label="Authorization header"
          />

          {method !== 'GET' && (
            <Textarea
              value={body}
              onChange={e => setBody(e.target.value)}
              placeholder='{"key": "value"}'
              className="h-24 text-xs font-mono resize-none"
              aria-label="Request body"
            />
          )}

          {response !== null && (
            <div className="space-y-2">
              <div className="flex items-center gap-3 text-xs">
                <span className={cn(
                  'font-mono font-medium',
                  responseStatus && responseStatus >= 200 && responseStatus < 300 ? 'text-emerald-500' :
                  responseStatus && responseStatus >= 400 ? 'text-red-500' : 'text-muted-foreground',
                )}>
                  {responseStatus}
                </span>
                {responseTime != null && (
                  <span className="text-muted-foreground">{responseTime}ms</span>
                )}
              </div>

              {responseHeaders && Object.keys(responseHeaders).length > 0 && (
                <details className="text-xs">
                  <summary className="text-muted-foreground cursor-pointer hover:text-foreground">Response Headers</summary>
                  <pre className="mt-1 rounded border border-border/30 bg-muted/20 p-2 font-mono text-[10px] overflow-auto max-h-32">
                    {Object.entries(responseHeaders).map(([k, v]) => `${k}: ${v}`).join('\n')}
                  </pre>
                </details>
              )}

              <pre className="rounded-lg border border-border/50 bg-muted/20 p-3 text-xs font-mono overflow-auto max-h-96 whitespace-pre-wrap">
                {response}
              </pre>
            </div>
          )}
        </CardContent>
      </Card>

      {history.length > 0 && (
        <Card>
          <CardHeader className="flex flex-row items-center justify-between">
            <CardTitle className="text-base">Request History</CardTitle>
            <Button variant="ghost" size="sm" onClick={clearHistory} className="text-destructive text-xs">
              Clear
            </Button>
          </CardHeader>
          <CardContent>
            <div className="space-y-1 max-h-48 overflow-y-auto">
              {history.map(entry => (
                <button
                  key={entry.id}
                  type="button"
                  onClick={() => loadFromHistory(entry)}
                  className="w-full flex items-center gap-2 px-2 py-1.5 rounded text-xs hover:bg-muted/50 transition-colors text-left"
                >
                  <span className={cn(
                    'font-mono font-medium w-12 shrink-0',
                    entry.status >= 200 && entry.status < 300 ? 'text-emerald-500' :
                    entry.status >= 400 ? 'text-red-500' : 'text-muted-foreground',
                  )}>
                    {entry.status || 'ERR'}
                  </span>
                  <span className="font-mono text-muted-foreground w-12 shrink-0">{entry.method}</span>
                  <span className="font-mono truncate flex-1">{entry.path}</span>
                  <span className="text-muted-foreground shrink-0">{entry.timeMs}ms</span>
                </button>
              ))}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  )
}

function QuickActionsTab() {
  const [results, setResults] = useState<Record<string, { status: number; body: string; timeMs: number } | null>>({})
  const [running, setRunning] = useState<string | null>(null)

  const executeAction = async (action: typeof QUICK_ACTIONS[0]) => {
    setRunning(action.label)
    setResults(prev => ({ ...prev, [action.label]: null }))
    const start = Date.now()

    try {
      const res = await authFetch(action.endpoint, { method: action.method, noAuth: true })
      const elapsed = Date.now() - start
      const text = await res.text()
      setResults(prev => ({ ...prev, [action.label]: { status: res.status, body: text, timeMs: elapsed } }))
    } catch (err) {
      setResults(prev => ({ ...prev, [action.label]: { status: 0, body: err instanceof Error ? err.message : 'Failed', timeMs: Date.now() - start } }))
    } finally {
      setRunning(null)
    }
  }

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <CardTitle className="text-base">Quick Actions</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
            {QUICK_ACTIONS.map(action => (
              <button
                key={action.label}
                type="button"
                onClick={() => executeAction(action)}
                disabled={running !== null}
                className={cn(
                  'flex flex-col items-start p-3 rounded-lg border text-left transition-all hover:-translate-y-0.5 hover:shadow-sm',
                  running === action.label ? 'border-primary/50 bg-primary/5' : 'border-border/60 hover:border-primary/30',
                  results[action.label]?.status === 200 ? 'border-emerald-500/30 bg-emerald-500/5' :
                  results[action.label]?.status === 404 ? 'border-amber-500/30 bg-amber-500/5' :
                  results[action.label]?.status === 0 ? 'border-red-500/30 bg-red-500/5' : ''
                )}
              >
                <div className="flex items-center gap-2 w-full">
                  <span className="text-sm font-medium">{action.label}</span>
                  {running === action.label && (
                    <span className="ml-auto text-[10px] text-amber-500 animate-pulse">Running...</span>
                  )}
                  {results[action.label] && running !== action.label && (
                    <span className={cn(
                      'ml-auto text-[10px] font-mono',
                      results[action.label]!.status >= 200 && results[action.label]!.status < 300 ? 'text-emerald-500' :
                      results[action.label]!.status >= 400 ? 'text-amber-500' : 'text-red-500'
                    )}>
                      {results[action.label]!.status || 'ERR'} · {results[action.label]!.timeMs}ms
                    </span>
                  )}
                </div>
                <span className="text-xs text-muted-foreground">{action.description}</span>
                <span className="text-[10px] text-muted-foreground/60 font-mono mt-1">{action.method} {action.endpoint}</span>
              </button>
            ))}
          </div>
        </CardContent>
      </Card>

      {Object.entries(results).map(([label, result]) => result && (
        <Card key={label}>
          <CardHeader className="flex flex-row items-center justify-between pb-2">
            <CardTitle className="text-sm">{label}</CardTitle>
            <span className="text-[10px] text-muted-foreground font-mono">
              {result.status || 'ERR'} · {result.timeMs}ms
            </span>
          </CardHeader>
          <CardContent>
            <pre className="text-xs font-mono bg-muted/30 rounded p-3 overflow-auto max-h-48 whitespace-pre-wrap">
              {result.body || 'No response'}
            </pre>
          </CardContent>
        </Card>
      ))}
    </div>
  )
}
