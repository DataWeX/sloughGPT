'use client'

/** API playground tab — request builder with persisted history. */
import { authFetch } from '@/lib/http-client'
import { cn } from '@sloughgpt/strui'
import { useState } from 'react'

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

export function ApiTab() {
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
