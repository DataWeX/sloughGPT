'use client'

import { useState } from 'react'
import { cn } from '@sloughgpt/strui'
import { authFetch } from '@/lib/http-client'

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

const METHODS = ['GET', 'POST', 'PUT', 'PATCH', 'DELETE'] as const

const METHOD_COLORS: Record<string, string> = {
  GET: 'bg-[#28c840]/10 text-[#28c840] border-[#28c840]/20',
  POST: 'bg-[#febc2e]/10 text-[#febc2e] border-[#febc2e]/20',
  PUT: 'bg-[#5ac8fa]/10 text-[#5ac8fa] border-[#5ac8fa]/20',
  PATCH: 'bg-[#bf5af2]/10 text-[#bf5af2] border-[#bf5af2]/20',
  DELETE: 'bg-[#ff5f57]/10 text-[#ff5f57] border-[#ff5f57]/20',
}

export function DevApiPlaygroundCard() {
  const [method, setMethod] = useState('GET')
  const [path, setPath] = useState('/health')
  const [body, setBody] = useState('')
  const [authHeader, setAuthHeader] = useState('')
  const [loading, setLoading] = useState(false)
  const [response, setResponse] = useState<string | null>(null)
  const [responseStatus, setResponseStatus] = useState<number | null>(null)
  const [responseTime, setResponseTime] = useState<number | null>(null)
  const [history, setHistory] = useState<HistoryEntry[]>(() => loadHistory())

  const handleSend = async () => {
    setLoading(true)
    setResponse(null)
    setResponseStatus(null)
    setResponseTime(null)

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
      <div className="rounded-xl border border-white/[0.06] bg-[#0a0a0a] overflow-hidden">
        <div className="flex items-center h-11 px-4 bg-[#1c1c1e] border-b border-white/[0.06]">
          <span className="text-[11px] font-medium text-[#8e8e93]">API Playground</span>
        </div>
        <div className="px-4 py-3 space-y-3">
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

          <input
            value={authHeader}
            onChange={e => setAuthHeader(e.target.value)}
            placeholder="Authorization: Bearer <token>"
            className="w-full h-8 rounded-lg border border-white/[0.06] bg-[#111111] px-3 text-[11px] font-mono text-[#c7c7cc] placeholder:text-[#48484a] outline-none focus:border-white/[0.12] transition-colors"
            aria-label="Authorization header"
          />

          {method !== 'GET' && (
            <textarea
              value={body}
              onChange={e => setBody(e.target.value)}
              placeholder='{"key": "value"}'
              className="w-full h-24 rounded-lg border border-white/[0.06] bg-[#111111] px-3 py-2 text-[11px] font-mono text-[#c7c7cc] placeholder:text-[#48484a] resize-none outline-none focus:border-white/[0.12] transition-colors"
              aria-label="Request body"
            />
          )}

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
              <pre className="rounded-xl border border-white/[0.04] bg-[#111111] p-3.5 text-[11px] font-mono text-[#c7c7cc] overflow-auto max-h-96 whitespace-pre-wrap leading-relaxed">
                {response}
              </pre>
            </div>
          )}
        </div>
      </div>

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
