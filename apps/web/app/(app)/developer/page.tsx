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

type DevTab = 'shell' | 'files' | 'voice'

export default function DeveloperPage() {
  const [tab, setTab] = useState<DevTab>('shell')

  return (
    <PageContainer title="Developer" subtitle="Terminal, files & voice tools">
      <div className="flex gap-1 border-b border-border/30 mb-4" role="tablist" aria-label="Developer tools">
        {([
          { id: 'shell' as const, label: 'Terminal' },
          { id: 'files' as const, label: 'Files' },
          { id: 'voice' as const, label: 'Voice' },
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
