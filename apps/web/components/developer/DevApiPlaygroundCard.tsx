'use client'

import { useState } from 'react'
import { Card, CardContent, CardHeader, CardTitle, Button, Input, Textarea, cn } from '@sloughgpt/strui'
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
                  responseStatus && responseStatus >= 200 && responseStatus < 300 ? 'text-green-500' :
                  responseStatus && responseStatus >= 400 ? 'text-red-500' : 'text-muted-foreground',
                )}>
                  {responseStatus}
                </span>
                {responseTime != null && (
                  <span className="text-muted-foreground">{responseTime}ms</span>
                )}
              </div>
              <pre className="rounded-lg border border-border/50 bg-muted/30 p-3 text-xs font-mono overflow-auto max-h-96 whitespace-pre-wrap">
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
                    entry.status >= 200 && entry.status < 300 ? 'text-green-500' :
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
