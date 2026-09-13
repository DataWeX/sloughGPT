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
import { useLiveStatus } from '@/hooks/useLiveStatus'
import { StartupTimeline } from '@/components/startup/StartupTimeline'
import { StartupHistoryChart } from '@/components/startup/StartupHistoryChart'


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
  const { startupStage, startupElapsed } = useLiveStatus()

  return (
    <PageContainer
      title="Developer"
      subtitle="Shell, files, voice & API tools"
      headerRight={
        <div className="flex items-center gap-3">
          {startupStage && startupStage !== 'ready' && startupStage !== 'background' && (
            <div className="flex items-center gap-2 rounded-lg border border-[#febc2e]/20 bg-[#febc2e]/[0.06] px-3 py-1.5">
              <span className="w-1.5 h-1.5 rounded-full bg-[#febc2e] animate-pulse" />
              <span className="text-[10px] font-medium text-[#febc2e] uppercase tracking-wider">
                {startupStage} {startupElapsed > 0 && `· ${startupElapsed.toFixed(0)}s`}
              </span>
            </div>
          )}
          {startupStage === 'ready' && (
            <div className="flex items-center gap-2 rounded-lg border border-[#28c840]/20 bg-[#28c840]/[0.06] px-3 py-1.5">
              <span className="w-1.5 h-1.5 rounded-full bg-[#28c840]" />
              <span className="text-[10px] font-medium text-[#28c840] uppercase tracking-wider">Ready</span>
            </div>
          )}
        </div>
      }
    >
      <Tabs value={tab} onValueChange={setTab}>
        <TabsList
          aria-label="Developer tools"
          className="rounded-xl border border-white/[0.06] bg-[#111111] p-1 h-auto"
        >
          <TabsTrigger value="shell" className="rounded-lg px-3.5 py-1.5 text-[11px] font-medium data-[state=active]:bg-[#1c1c1e] data-[state=active]:text-[#c7c7cc] data-[state=active]:shadow-sm data-[state=active]:shadow-black/20 data-[state=inactive]:text-[#636366] data-[state=inactive]:hover:text-[#8e8e93]">Terminal</TabsTrigger>
          <TabsTrigger value="files" className="rounded-lg px-3.5 py-1.5 text-[11px] font-medium data-[state=active]:bg-[#1c1c1e] data-[state=active]:text-[#c7c7cc] data-[state=active]:shadow-sm data-[state=active]:shadow-black/20 data-[state=inactive]:text-[#636366] data-[state=inactive]:hover:text-[#8e8e93]">Files</TabsTrigger>
          <TabsTrigger value="voice" className="rounded-lg px-3.5 py-1.5 text-[11px] font-medium data-[state=active]:bg-[#1c1c1e] data-[state=active]:text-[#c7c7cc] data-[state=active]:shadow-sm data-[state=active]:shadow-black/20 data-[state=inactive]:text-[#636366] data-[state=inactive]:hover:text-[#8e8e93]">Voice</TabsTrigger>
          <TabsTrigger value="api" className="rounded-lg px-3.5 py-1.5 text-[11px] font-medium data-[state=active]:bg-[#1c1c1e] data-[state=active]:text-[#c7c7cc] data-[state=active]:shadow-sm data-[state=active]:shadow-black/20 data-[state=inactive]:text-[#636366] data-[state=inactive]:hover:text-[#8e8e93]">API</TabsTrigger>
          <TabsTrigger value="quick" className="rounded-lg px-3.5 py-1.5 text-[11px] font-medium data-[state=active]:bg-[#1c1c1e] data-[state=active]:text-[#c7c7cc] data-[state=active]:shadow-sm data-[state=active]:shadow-black/20 data-[state=inactive]:text-[#636366] data-[state=inactive]:hover:text-[#8e8e93]">Quick Actions</TabsTrigger>
          <TabsTrigger value="startup" className="rounded-lg px-3.5 py-1.5 text-[11px] font-medium data-[state=active]:bg-[#1c1c1e] data-[state=active]:text-[#c7c7cc] data-[state=active]:shadow-sm data-[state=active]:shadow-black/20 data-[state=inactive]:text-[#636366] data-[state=inactive]:hover:text-[#8e8e93]">Startup</TabsTrigger>
        </TabsList>

        <TabsContent value="shell"><ShellTab /></TabsContent>
        <TabsContent value="files"><FilesTab /></TabsContent>
        <TabsContent value="voice"><VoiceTab /></TabsContent>
        <TabsContent value="api"><ApiTab /></TabsContent>
        <TabsContent value="quick"><QuickActionsTab /></TabsContent>
        <TabsContent value="startup"><StartupTab /></TabsContent>
      </Tabs>
    </PageContainer>
  )
}

function ShellTab() {
  const [shellMode, setShellMode] = useState<'backend' | 'v86'>('backend')

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-1 rounded-xl border border-white/[0.06] bg-[#111111] p-1 w-fit">
        <button
          type="button"
          onClick={() => setShellMode('backend')}
          className={cn(
            'rounded-lg px-4 py-1.5 text-[11px] font-medium transition-all duration-200',
            shellMode === 'backend'
              ? 'bg-[#1c1c1e] text-[#c7c7cc] shadow-sm shadow-black/20'
              : 'text-[#636366] hover:text-[#8e8e93]',
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
              ? 'bg-[#1c1c1e] text-[#c7c7cc] shadow-sm shadow-black/20'
              : 'text-[#636366] hover:text-[#8e8e93]',
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
          <div className="flex items-center justify-between px-4 py-2.5 bg-[#1c1c1e] border-b border-[#0.06]">
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

function StartupTab() {
  const { health, startupStage, startupElapsed, startupModelProgress, startupModelProgressMessage } = useLiveStatus()
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
    init: 'text-[#636366]',
    critical: 'text-[#febc2e]',
    ready: 'text-[#0a7aff]',
    background: 'text-[#28c840]',
  }

  const HOOK_STATUS_COLORS: Record<string, string> = {
    pending: 'text-[#636366]',
    running: 'text-[#febc2e]',
    ok: 'text-[#28c840]',
    timeout: 'text-[#ff5f57]',
    error: 'text-[#ff5f57]',
  }

  const hooks = startupData?.hooks as Record<string, { name: string; stage: string; status: string; duration_seconds: number; error: string | null }> | undefined
  const stages = startupData?.stages as Record<string, { hooks: string[]; time: number | null }> | undefined
  const stats = historyData?.stats as { count: number; avg_duration: number; min_duration: number; max_duration: number; p50_duration: number; p95_duration: number; success_rate: number } | undefined
  const stageStats = historyData?.stage_stats as Record<string, { avg: number; min: number; max: number; p50: number; count: number }> | undefined
  const alerts = historyData?.alerts as Array<{ type: string; severity: string; message: string; timestamp: number }> | undefined
  const suggestions = historyData?.suggestions as Array<{ type: string; severity: string; message: string; stage?: string; hook?: string }> | undefined

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
                  ? 'border-[#ff5f57]/20 bg-[#ff5f57]/[0.06]'
                  : 'border-[#febc2e]/20 bg-[#febc2e]/[0.06]',
              )}
            >
              <span className={cn(
                'w-2 h-2 rounded-full shrink-0',
                alert.severity === 'error' ? 'bg-[#ff5f57]' : 'bg-[#febc2e]',
              )} />
              <div className="flex-1 min-w-0">
                <span className={cn(
                  'text-[11px] font-medium',
                  alert.severity === 'error' ? 'text-[#ff5f57]' : 'text-[#febc2e]',
                )}>
                  {alert.type.replace(/_/g, ' ')}
                </span>
                <span className="text-[10px] text-[#636366] ml-2">{alert.message}</span>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Suggestions */}
      {suggestions && suggestions.length > 0 && (
        <div className="rounded-xl border border-white/[0.06] bg-[#0a0a0a] overflow-hidden">
          <div className="h-9 px-4 bg-[#1c1c1e] border-b border-white/[0.06] flex items-center">
            <span className="text-[11px] font-medium text-[#8e8e93]">Optimization Suggestions</span>
            <span className="text-[9px] text-[#636366] ml-2">{suggestions.length} suggestions</span>
          </div>
          <div className="divide-y divide-white/[0.04]">
            {suggestions.map((suggestion, i) => (
              <div
                key={`${suggestion.type}-${i}`}
                className="px-4 py-3 flex items-start gap-3"
              >
                <span className={cn(
                  'w-2 h-2 rounded-full shrink-0 mt-1',
                  suggestion.severity === 'error' ? 'bg-[#ff5f57]' :
                  suggestion.severity === 'warning' ? 'bg-[#febc2e]' : 'bg-[#0a7aff]',
                )} />
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2">
                    <span className={cn(
                      'text-[11px] font-medium',
                      suggestion.severity === 'error' ? 'text-[#ff5f57]' :
                      suggestion.severity === 'warning' ? 'text-[#febc2e]' : 'text-[#0a7aff]',
                    )}>
                      {suggestion.type.replace(/_/g, ' ')}
                    </span>
                    {suggestion.stage && (
                      <span className="text-[9px] text-[#636366] bg-[#1c1c1e] px-1.5 py-0.5 rounded">{suggestion.stage}</span>
                    )}
                    {suggestion.hook && (
                      <span className="text-[9px] text-[#636366] bg-[#1c1c1e] px-1.5 py-0.5 rounded">{suggestion.hook}</span>
                    )}
                  </div>
                  <span className="text-[10px] text-[#8e8e93]">{suggestion.message}</span>
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
        <div className="rounded-xl border border-white/[0.06] bg-[#0a0a0a] p-4">
          <div className="text-[10px] text-[#636366] uppercase tracking-wider mb-1">Stage</div>
          <div className={cn('text-[14px] font-medium font-mono', STAGE_COLORS[startupStage] ?? 'text-[#c7c7cc]')}>
            {startupStage}
          </div>
        </div>
        <div className="rounded-xl border border-white/[0.06] bg-[#0a0a0a] p-4">
          <div className="text-[10px] text-[#636366] uppercase tracking-wider mb-1">Elapsed</div>
          <div className="text-[14px] font-medium font-mono text-[#c7c7cc]">
            {startupElapsed.toFixed(1)}s
          </div>
        </div>
        <div className="rounded-xl border border-white/[0.06] bg-[#0a0a0a] p-4">
          <div className="text-[10px] text-[#636366] uppercase tracking-wider mb-1">Model</div>
          <div className="text-[14px] font-medium font-mono text-[#c7c7cc]">
            {Math.round(startupModelProgress * 100)}%
          </div>
        </div>
        <div className="rounded-xl border border-white/[0.06] bg-[#0a0a0a] p-4">
          <div className="text-[10px] text-[#636366] uppercase tracking-wider mb-1">Status</div>
          <div className={cn('text-[14px] font-medium', startupStage === 'background' ? 'text-[#28c840]' : 'text-[#febc2e]')}>
            {startupStage === 'background' ? 'Ready' : 'Starting'}
          </div>
        </div>
      </div>

      {/* Progress bar */}
      <div className="rounded-xl border border-white/[0.06] bg-[#0a0a0a] p-4">
        <div className="flex items-center justify-between mb-2">
          <span className="text-[11px] text-[#8e8e93]">Startup Progress</span>
          <span className="text-[11px] text-[#636366] font-mono">{startupModelProgressMessage}</span>
        </div>
        <div className="h-2 rounded-full bg-[#1c1c1e] overflow-hidden">
          <div
            className="h-full rounded-full bg-gradient-to-r from-[#0a7aff] to-[#28c840] transition-all duration-300"
            style={{ width: `${Math.min(100, (startupModelProgress || (startupStage === 'background' ? 1 : startupStage === 'ready' ? 0.6 : startupStage === 'critical' ? 0.3 : 0.1)) * 100)}%` }}
          />
        </div>
        <div className="flex justify-between mt-1">
          {['init', 'critical', 'ready', 'background'].map((stage, i) => (
            <span key={stage} className={cn(
              'text-[9px] font-mono',
              i <= ['init', 'critical', 'ready', 'background'].indexOf(startupStage)
                ? 'text-[#28c840]'
                : 'text-[#2c2c2e]',
            )}>
              {stage}
            </span>
          ))}
        </div>
      </div>

      {/* Performance comparison */}
      {stats && stats.count > 0 && (
        <div className="rounded-xl border border-white/[0.06] bg-[#0a0a0a] overflow-hidden">
          <div className="h-9 px-4 bg-[#1c1c1e] border-b border-white/[0.06] flex items-center">
            <span className="text-[11px] font-medium text-[#8e8e93]">Performance History</span>
            <span className="text-[9px] text-[#636366] ml-2">{stats.count} startups recorded</span>
          </div>
          <div className="p-4">
            <div className="grid grid-cols-2 lg:grid-cols-5 gap-3">
              <div className="text-center">
                <div className="text-[10px] text-[#636366] uppercase tracking-wider mb-1">Avg</div>
                <div className="text-[14px] font-mono text-[#c7c7cc]">{stats.avg_duration}s</div>
              </div>
              <div className="text-center">
                <div className="text-[10px] text-[#636366] uppercase tracking-wider mb-1">P50</div>
                <div className="text-[14px] font-mono text-[#c7c7cc]">{stats.p50_duration}s</div>
              </div>
              <div className="text-center">
                <div className="text-[10px] text-[#636366] uppercase tracking-wider mb-1">P95</div>
                <div className="text-[14px] font-mono text-[#c7c7cc]">{stats.p95_duration}s</div>
              </div>
              <div className="text-center">
                <div className="text-[10px] text-[#636366] uppercase tracking-wider mb-1">Min</div>
                <div className="text-[14px] font-mono text-[#28c840]">{stats.min_duration}s</div>
              </div>
              <div className="text-center">
                <div className="text-[10px] text-[#636366] uppercase tracking-wider mb-1">Max</div>
                <div className="text-[14px] font-mono text-[#ff5f57]">{stats.max_duration}s</div>
              </div>
            </div>
            {startupElapsed > 0 && stats.avg_duration > 0 && (
              <div className="mt-3 pt-3 border-t border-white/[0.04] flex items-center justify-center gap-2">
                <span className="text-[10px] text-[#636366]">Current:</span>
                <span className={cn(
                  'text-[11px] font-mono font-medium',
                  startupElapsed < stats.avg_duration ? 'text-[#28c840]' : 'text-[#febc2e]',
                )}>
                  {startupElapsed.toFixed(1)}s
                </span>
                <span className="text-[10px] text-[#636366]">vs avg {stats.avg_duration}s</span>
                {startupElapsed < stats.avg_duration && (
                  <span className="text-[9px] text-[#28c840]">({((1 - startupElapsed / stats.avg_duration) * 100).toFixed(0)}% faster)</span>
                )}
                {startupElapsed > stats.avg_duration && (
                  <span className="text-[9px] text-[#febc2e]">(+{((startupElapsed / stats.avg_duration - 1) * 100).toFixed(0)}% slower)</span>
                )}
              </div>
            )}
          </div>
        </div>
      )}

      {/* Stages */}
      {stages && (
        <div className="rounded-xl border border-white/[0.06] bg-[#0a0a0a] overflow-hidden">
          <div className="h-9 px-4 bg-[#1c1c1e] border-b border-white/[0.06] flex items-center">
            <span className="text-[11px] font-medium text-[#8e8e93]">Stages</span>
          </div>
          <div className="divide-y divide-white/[0.04]">
            {Object.entries(stages).map(([stage, data]) => (
              <div key={stage} className="px-4 py-3 flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <span className={cn('w-2 h-2 rounded-full', stage === startupStage ? 'bg-[#0a7aff] animate-pulse' : data.time !== null ? 'bg-[#28c840]' : 'bg-[#2c2c2e]')} />
                  <span className="text-[12px] font-medium text-[#c7c7cc] capitalize">{stage}</span>
                  <span className="text-[10px] text-[#636366] font-mono">{data.hooks.length} hooks</span>
                </div>
                <span className="text-[11px] font-mono text-[#636366]">
                  {data.time !== null ? `${data.time}s` : '—'}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Stage Stats */}
      {stageStats && Object.keys(stageStats).length > 0 && (
        <div className="rounded-xl border border-white/[0.06] bg-[#0a0a0a] overflow-hidden">
          <div className="h-9 px-4 bg-[#1c1c1e] border-b border-white/[0.06] flex items-center">
            <span className="text-[11px] font-medium text-[#8e8e93]">Stage Performance</span>
          </div>
          <div className="p-4">
            <div className="grid grid-cols-2 lg:grid-cols-3 gap-3">
              {Object.entries(stageStats).map(([stage, data]) => (
                <div key={stage} className="rounded-lg border border-white/[0.04] bg-[#111111] p-3">
                  <div className="flex items-center gap-2 mb-2">
                    <div className={cn(
                      'w-2 h-2 rounded-full',
                      stage === startupStage ? 'bg-[#0a7aff]' : 'bg-[#28c840]',
                    )} />
                    <span className="text-[11px] font-medium text-[#c7c7cc] capitalize">{stage}</span>
                  </div>
                  <div className="grid grid-cols-2 gap-2 text-[10px]">
                    <div>
                      <span className="text-[#636366]">Avg</span>
                      <span className="ml-1 font-mono text-[#c7c7cc]">{data.avg}s</span>
                    </div>
                    <div>
                      <span className="text-[#636366]">P50</span>
                      <span className="ml-1 font-mono text-[#c7c7cc]">{data.p50}s</span>
                    </div>
                    <div>
                      <span className="text-[#636366]">Min</span>
                      <span className="ml-1 font-mono text-[#28c840]">{data.min}s</span>
                    </div>
                    <div>
                      <span className="text-[#636366]">Max</span>
                      <span className="ml-1 font-mono text-[#ff5f57]">{data.max}s</span>
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
        <div className="rounded-xl border border-white/[0.06] bg-[#0a0a0a] overflow-hidden">
          <div className="h-9 px-4 bg-[#1c1c1e] border-b border-white/[0.06] flex items-center">
            <span className="text-[11px] font-medium text-[#8e8e93]">Hooks</span>
          </div>
          <div className="divide-y divide-white/[0.04]">
            {Object.values(hooks).map((hook) => (
              <div key={hook.name} className="px-4 py-2.5 flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <span className={cn('w-1.5 h-1.5 rounded-full', HOOK_STATUS_COLORS[hook.status] ?? 'text-[#636366]')} />
                  <span className="text-[11px] font-mono text-[#c7c7cc]">{hook.name}</span>
                  <span className="text-[9px] text-[#636366] uppercase">{hook.stage}</span>
                </div>
                <div className="flex items-center gap-3">
                  {hook.error && (
                    <span className="text-[9px] text-[#ff5f57] max-w-32 truncate">{hook.error}</span>
                  )}
                  <span className="text-[10px] font-mono text-[#636366]">
                    {hook.duration_seconds > 0 ? `${hook.duration_seconds}s` : '—'}
                  </span>
                  <span className={cn('text-[9px] font-mono uppercase', HOOK_STATUS_COLORS[hook.status])}>
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
