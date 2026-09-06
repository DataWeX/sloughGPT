'use client'
export const dynamic = 'force-dynamic'

import { useState, useEffect, useCallback } from 'react'
import { PageContainer } from '@/components/PageContainer'
import {
  Card, CardContent, CardHeader, CardTitle, Button, Input,
  Textarea, Skeleton, cn,
} from '@sloughgpt/strui'
import { IconRefresh } from '@sloughgpt/strui'
import { TerminalPanel } from '@/components/shell/TerminalPanel'
import { FileStatsCard } from '@/components/files/FileStatsCard'
import { filesController, type FileEntry } from '@/lib/files-controller'
import { voiceController, type VoiceStatus } from '@/lib/voice-controller'

type DevTab = 'shell' | 'files' | 'voice' | 'api'

export default function DeveloperPage() {
  const [tab, setTab] = useState<DevTab>('shell')

  return (
    <PageContainer title="Developer" subtitle="Terminal, files & voice tools">
      <div className="flex gap-1 border-b border-border/30 mb-4" role="tablist" aria-label="Developer tools">
        {([
          { id: 'shell' as const, label: 'Terminal' },
          { id: 'files' as const, label: 'Files' },
          { id: 'voice' as const, label: 'Voice' },
          { id: 'api' as const, label: 'API' },
        ]).map(t => (
          <button
            type="button"
            role="tab"
            key={t.id}
            aria-selected={tab === t.id}
            aria-label={`${t.label} tab`}
            onClick={() => setTab(t.id)}
            className={cn(
              'px-3 py-1.5 text-xs font-medium rounded-t transition-colors',
              tab === t.id ? 'bg-primary/10 text-primary border-b-2 border-primary' : 'text-muted-foreground hover:text-foreground',
            )}
          >
            {t.label}
          </button>
        ))}
      </div>

      {tab === 'shell' && <ShellTab />}
      {tab === 'files' && <FilesTab />}
      {tab === 'voice' && <VoiceTab />}
      {tab === 'api' && <ApiTab />}
    </PageContainer>
  )
}

function ShellTab() {
  return (
    <Card className="h-[calc(100vh-8rem)]">
      <CardHeader>
        <CardTitle className="text-base">Dait Shell</CardTitle>
      </CardHeader>
      <CardContent className="h-[calc(100%-3rem)]">
        <TerminalPanel className="h-full" />
      </CardContent>
    </Card>
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
                    status.server_tts ? 'bg-green-500' : 'bg-red-500',
                  )} />
                  <span className="text-xs">{status.server_tts ? 'TTS Available' : 'TTS Unavailable'}</span>
                </div>
                <p className="text-xs text-muted-foreground">
                  Model: {status.model ?? 'none'}
                </p>
                {status.error && (
                  <p className="text-xs text-red-500">{status.error}</p>
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
              <p className="text-xs text-red-500">{ttsError}</p>
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

function ApiTab() {
  const [method, setMethod] = useState('GET')
  const [path, setPath] = useState('/health')
  const [body, setBody] = useState('')
  const [loading, setLoading] = useState(false)
  const [response, setResponse] = useState<string | null>(null)
  const [responseStatus, setResponseStatus] = useState<number | null>(null)
  const [responseTime, setResponseTime] = useState<number | null>(null)

  const METHODS = ['GET', 'POST', 'PUT', 'PATCH', 'DELETE'] as const

  const handleSend = async () => {
    setLoading(true)
    setResponse(null)
    setResponseStatus(null)
    setResponseTime(null)

    const start = Date.now()
    try {
      const baseUrl = window.location.origin.replace(':3001', ':8000')
      const opts: RequestInit = { method }
      if (body && method !== 'GET') {
        opts.body = body
        opts.headers = { 'Content-Type': 'application/json' }
      }
      const res = await fetch(`${baseUrl}${path}`, opts)
      const elapsed = Date.now() - start
      setResponseStatus(res.status)
      setResponseTime(elapsed)
      const text = await res.text()
      try {
        setResponse(JSON.stringify(JSON.parse(text), null, 2))
      } catch {
        setResponse(text)
      }
    } catch (err) {
      setResponse(err instanceof Error ? err.message : 'Request failed')
      setResponseStatus(0)
    } finally {
      setLoading(false)
    }
  }

  return (
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
  )
}
