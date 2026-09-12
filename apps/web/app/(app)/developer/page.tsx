'use client'
export const dynamic = 'force-dynamic'

import { useState, useEffect, useCallback } from 'react'
import { PageContainer } from '@/components/PageContainer'
import { Skeleton, cn, Tabs, TabsList, TabsTrigger, TabsContent } from '@sloughgpt/strui'
import { IconRefresh } from '@sloughgpt/strui'
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
      <div className="rounded-xl border border-white/[0.06] bg-[#0a0a0a] overflow-hidden">
        <div className="flex items-center justify-between h-11 px-4 bg-[#1c1c1e] border-b border-white/[0.06]">
          <span className="text-[11px] font-medium text-[#8e8e93]">Documents</span>
          <div className="flex items-center gap-2">
            <input
              type="text"
              placeholder="Search..."
              value={search}
              onChange={e => setSearch(e.target.value)}
              className="w-40 h-7 rounded-md border border-white/[0.06] bg-[#111111] px-2.5 text-[11px] text-[#c7c7cc] placeholder:text-[#48484a] outline-none focus:border-white/[0.12] transition-colors font-mono"
              aria-label="Search files"
            />
            <button
              type="button"
              onClick={() => { setLoading(true); fetchFiles() }}
              className="h-7 w-7 flex items-center justify-center rounded-md border border-white/[0.06] bg-[#111111] text-[#8e8e93] hover:text-[#c7c7cc] hover:border-white/[0.12] transition-colors"
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
                <Skeleton key={i} className="h-7 w-full bg-[#1c1c1e]" />
              ))}
            </div>
          ) : filtered.length === 0 ? (
            <p className="text-[11px] text-[#636366] py-6 text-center">
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
                  <span className="font-mono text-[#c7c7cc] truncate">{f.filename}</span>
                  <span className="text-[#636366] shrink-0 ml-3 font-mono">
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
    <div className="grid grid-cols-2 gap-4">
      {/* Status card */}
      <div className="rounded-xl border border-white/[0.06] bg-[#0a0a0a] overflow-hidden">
        <div className="flex items-center h-9 px-4 bg-[#1c1c1e] border-b border-white/[0.06]">
          <span className="text-[11px] font-medium text-[#8e8e93]">Voice Status</span>
        </div>
        <div className="px-4 py-3">
          {loading ? (
            <Skeleton className="h-14 w-full bg-[#1c1c1e]" />
          ) : status ? (
            <div className="space-y-2.5">
              <div className="flex items-center gap-2.5">
                <span className={cn(
                  'w-2 h-2 rounded-full',
                  status.server_tts ? 'bg-[#28c840]' : 'bg-[#ff5f57]',
                )} />
                <span className="text-[12px] text-[#c7c7cc]">{status.server_tts ? 'TTS Available' : 'TTS Unavailable'}</span>
              </div>
              <p className="text-[11px] text-[#636366] font-mono">
                {status.model ?? 'no model'}
              </p>
              {status.error && (
                <div className="flex items-start gap-2 text-[#ff5f57] bg-[#ff5f57]/[0.08] rounded-lg px-3 py-2 text-[11px]">
                  <span className="shrink-0 text-[10px] font-bold">!</span>
                  {status.error}
                </div>
              )}
            </div>
          ) : (
            <p className="text-[11px] text-[#636366]">Could not load status</p>
          )}
        </div>
      </div>

      {/* Quick Test card */}
      <div className="rounded-xl border border-white/[0.06] bg-[#0a0a0a] overflow-hidden">
        <div className="flex items-center h-9 px-4 bg-[#1c1c1e] border-b border-white/[0.06]">
          <span className="text-[11px] font-medium text-[#8e8e93]">Quick Test</span>
        </div>
        <div className="px-4 py-3 space-y-2.5">
          <textarea
            placeholder="Type text to speak..."
            value={ttsText}
            onChange={e => setTtsText(e.target.value)}
            className="w-full h-16 rounded-lg border border-white/[0.06] bg-[#111111] px-3 py-2 text-[12px] text-[#c7c7cc] placeholder:text-[#48484a] resize-none outline-none focus:border-white/[0.12] transition-colors font-mono"
            aria-label="Text to speech input"
          />
          <button
            type="button"
            onClick={handleGenerate}
            disabled={generating || !ttsText.trim()}
            className={cn(
              'w-full h-8 rounded-lg text-[11px] font-medium transition-all duration-200',
              generating || !ttsText.trim()
                ? 'bg-[#28c840]/20 text-[#28c840]/40 cursor-not-allowed'
                : 'bg-[#28c840]/10 text-[#28c840] hover:bg-[#28c840]/20',
            )}
          >
            {generating ? 'Generating...' : 'Speak'}
          </button>
          {ttsError && (
            <div className="flex items-start gap-2 text-[#ff5f57] bg-[#ff5f57]/[0.08] rounded-lg px-3 py-2 text-[11px]">
              <span className="shrink-0 text-[10px] font-bold">!</span>
              {ttsError}
            </div>
          )}
          {lastResult && (
            <p className="text-[10px] text-[#636366] font-mono">
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

  const METHOD_COLORS: Record<string, string> = {
    GET: 'bg-[#28c840]/10 text-[#28c840] border-[#28c840]/20',
    POST: 'bg-[#febc2e]/10 text-[#febc2e] border-[#febc2e]/20',
    PUT: 'bg-[#5ac8fa]/10 text-[#5ac8fa] border-[#5ac8fa]/20',
    PATCH: 'bg-[#bf5af2]/10 text-[#bf5af2] border-[#bf5af2]/20',
    DELETE: 'bg-[#ff5f57]/10 text-[#ff5f57] border-[#ff5f57]/20',
  }

  return (
    <div className="space-y-4">
      {/* API Playground */}
      <div className="rounded-xl border border-white/[0.06] bg-[#0a0a0a] overflow-hidden">
        <div className="flex items-center h-11 px-4 bg-[#1c1c1e] border-b border-white/[0.06]">
          <span className="text-[11px] font-medium text-[#8e8e93]">API Playground</span>
        </div>
        <div className="px-4 py-3 space-y-3">
          {/* URL bar */}
          <div className="flex gap-2">
            <select
              value={method}
              onChange={e => setMethod(e.target.value)}
              className={cn(
                'h-8 rounded-lg border px-2.5 text-[11px] font-mono font-medium bg-[#111111] outline-none',
                METHOD_COLORS[method] ?? 'border-white/[0.06] text-[#c7c7cc]',
              )}
              aria-label="HTTP method"
            >
              {METHODS.map(m => <option key={m} value={m}>{m}</option>)}
            </select>
            <input
              value={path}
              onChange={e => setPath(e.target.value)}
              placeholder="/endpoint"
              className="flex-1 h-8 rounded-lg border border-white/[0.06] bg-[#111111] px-3 text-[12px] font-mono text-[#c7c7cc] placeholder:text-[#48484a] outline-none focus:border-white/[0.12] transition-colors"
              aria-label="Request path"
              onKeyDown={e => { if (e.key === 'Enter') handleSend() }}
            />
            <button
              type="button"
              onClick={handleSend}
              disabled={loading || !path.trim()}
              className={cn(
                'h-8 px-4 rounded-lg text-[11px] font-medium transition-all duration-200',
                loading || !path.trim()
                  ? 'bg-[#0a7aff]/20 text-[#0a7aff]/40 cursor-not-allowed'
                  : 'bg-[#0a7aff]/10 text-[#0a7aff] hover:bg-[#0a7aff]/20',
              )}
            >
              {loading ? 'Sending...' : 'Send'}
            </button>
          </div>

          {/* Auth header */}
          <input
            value={authHeader}
            onChange={e => setAuthHeader(e.target.value)}
            placeholder="Authorization: Bearer <token>"
            className="w-full h-8 rounded-lg border border-white/[0.06] bg-[#111111] px-3 text-[11px] font-mono text-[#c7c7cc] placeholder:text-[#48484a] outline-none focus:border-white/[0.12] transition-colors"
            aria-label="Authorization header"
          />

          {/* Body */}
          {method !== 'GET' && (
            <textarea
              value={body}
              onChange={e => setBody(e.target.value)}
              placeholder='{"key": "value"}'
              className="w-full h-24 rounded-lg border border-white/[0.06] bg-[#111111] px-3 py-2 text-[11px] font-mono text-[#c7c7cc] placeholder:text-[#48484a] resize-none outline-none focus:border-white/[0.12] transition-colors"
              aria-label="Request body"
            />
          )}

          {/* Response */}
          {response !== null && (
            <div className="space-y-2">
              <div className="flex items-center gap-3 text-[11px]">
                <span className={cn(
                  'font-mono font-medium',
                  responseStatus && responseStatus >= 200 && responseStatus < 300 ? 'text-[#28c840]' :
                  responseStatus && responseStatus >= 400 ? 'text-[#ff5f57]' : 'text-[#636366]',
                )}>
                  {responseStatus}
                </span>
                {responseTime != null && (
                  <span className="text-[#636366] font-mono">{responseTime}ms</span>
                )}
              </div>

              {responseHeaders && Object.keys(responseHeaders).length > 0 && (
                <details className="text-[11px]">
                  <summary className="text-[#636366] cursor-pointer hover:text-[#8e8e93] transition-colors">Response Headers</summary>
                  <pre className="mt-1.5 rounded-lg border border-white/[0.04] bg-[#111111] p-2.5 font-mono text-[10px] text-[#c7c7cc] overflow-auto max-h-32 leading-relaxed">
                    {Object.entries(responseHeaders).map(([k, v]) => `${k}: ${v}`).join('\n')}
                  </pre>
                </details>
              )}

              <pre className="rounded-xl border border-white/[0.04] bg-[#111111] p-3.5 text-[11px] font-mono text-[#c7c7cc] overflow-auto max-h-96 whitespace-pre-wrap leading-relaxed">
                {response}
              </pre>
            </div>
          )}
        </div>
      </div>

      {/* History */}
      {history.length > 0 && (
        <div className="rounded-xl border border-white/[0.06] bg-[#0a0a0a] overflow-hidden">
          <div className="flex items-center justify-between h-11 px-4 bg-[#1c1c1e] border-b border-white/[0.06]">
            <span className="text-[11px] font-medium text-[#8e8e93]">Request History</span>
            <button
              type="button"
              onClick={clearHistory}
              className="text-[10px] text-[#ff5f57]/60 hover:text-[#ff5f57] transition-colors"
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
                <span className={cn(
                  'font-mono font-medium w-10 shrink-0',
                  entry.status >= 200 && entry.status < 300 ? 'text-[#28c840]' :
                  entry.status >= 400 ? 'text-[#ff5f57]' : 'text-[#636366]',
                )}>
                  {entry.status || 'ERR'}
                </span>
                <span className="font-mono text-[#48484a] w-12 shrink-0">{entry.method}</span>
                <span className="font-mono text-[#c7c7cc] truncate flex-1">{entry.path}</span>
                <span className="text-[#636366] shrink-0 font-mono">{entry.timeMs}ms</span>
              </button>
            ))}
          </div>
        </div>
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

  const METHOD_COLORS: Record<string, string> = {
    GET: 'bg-emerald-500/10 text-emerald-500',
    POST: 'bg-amber-500/10 text-amber-500',
    PUT: 'bg-blue-500/10 text-blue-400',
    PATCH: 'bg-purple-500/10 text-purple-400',
    DELETE: 'bg-red-500/10 text-red-500',
  }

  return (
    <div className="space-y-4">
      <div className="grid gap-2.5 sm:grid-cols-2 lg:grid-cols-3">
        {QUICK_ACTIONS.map(action => (
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
                    ? 'border-emerald-500/20 bg-emerald-500/[0.03]'
                    : 'border-red-500/20 bg-red-500/[0.03]'
                  : 'border-white/[0.06] bg-white/[0.02] hover:border-white/[0.12] hover:bg-white/[0.04]',
            )}
          >
            <div className="flex items-center gap-2 w-full mb-1">
              <span className={cn(
                'text-[10px] font-mono font-medium px-1.5 py-0.5 rounded',
                METHOD_COLORS[action.method] ?? 'bg-muted text-muted-foreground',
              )}>
                {action.method}
              </span>
              <span className="text-sm font-medium text-[#e5e5ea]">{action.label}</span>
              {running === action.label && (
                <span className="ml-auto flex items-center gap-1.5 text-[10px] text-[#febc2e]">
                  <span className="w-1.5 h-1.5 rounded-full bg-[#febc2e] animate-pulse" />
                  running
                </span>
              )}
              {results[action.label] && running !== action.label && (
                <span className={cn(
                  'ml-auto text-[10px] font-mono',
                  results[action.label]!.status >= 200 && results[action.label]!.status < 300 ? 'text-[#28c840]' :
                  results[action.label]!.status >= 400 ? 'text-[#febc2e]' : 'text-[#ff5f57]'
                )}>
                  {results[action.label]!.status || 'ERR'}
                </span>
              )}
            </div>
            <span className="text-[11px] text-[#8e8e93]">{action.description}</span>
            <span className="text-[10px] text-[#48484a] font-mono mt-1.5">{action.endpoint}</span>
          </button>
        ))}
      </div>

      {Object.entries(results).map(([label, result]) => result && (
        <div key={label} className="rounded-xl border border-white/[0.06] bg-[#0a0a0a] overflow-hidden">
          <div className="flex items-center justify-between px-4 py-2.5 bg-[#1c1c1e] border-b border-white/[0.06]">
            <span className="text-[11px] font-medium text-[#c7c7cc]">{label}</span>
            <div className="flex items-center gap-2">
              <span className={cn(
                'text-[10px] font-mono',
                result.status >= 200 && result.status < 300 ? 'text-[#28c840]' :
                result.status >= 400 ? 'text-[#febc2e]' : 'text-[#ff5f57]'
              )}>
                {result.status || 'ERR'}
              </span>
              <span className="text-[10px] text-[#636366] font-mono">{result.timeMs}ms</span>
            </div>
          </div>
          <pre className="px-4 py-3 text-[11px] font-mono text-[#c7c7cc] overflow-auto max-h-48 whitespace-pre-wrap leading-relaxed">
            {result.body || 'No response'}
          </pre>
        </div>
      ))}
    </div>
  )
}
