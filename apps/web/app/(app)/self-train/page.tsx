'use client'
export const dynamic = 'force-dynamic'

import { useState, useCallback, useEffect, useRef } from 'react'
import { PageContainer } from '@/components/PageContainer'
import { Card, CardContent, CardHeader, CardTitle, Button, Input, Label } from '@sloughgpt/strui'
import { useToastStore } from '@/lib/toast-store'
import { apiGet, apiPost } from '@/lib/http-client'

interface SelfTrainStatus {
  status: 'not_started' | 'running' | 'exited'
  pid?: number
  returncode?: number
  history?: string[]
}

export default function SelfTrainPage() {
  const addToast = useToastStore(s => s.addToast)
  const [status, setStatus] = useState<SelfTrainStatus | null>(null)
  const [loading, setLoading] = useState(true)
  const [model, setModel] = useState('')
  const [temperature, setTemperature] = useState(0.7)
  const [forever, setForever] = useState(false)
  const [starting, setStarting] = useState(false)
  const historyEndRef = useRef<HTMLDivElement>(null)

  const fetchStatus = useCallback(async () => {
    try {
      const data = await apiGet<SelfTrainStatus>('/self-train/status')
      setStatus(data)
    } catch {
      addToast('Could not check status', 'error')
    } finally {
      setLoading(false)
    }
  }, [addToast])

  useEffect(() => { void fetchStatus() }, [fetchStatus])

  // Poll while running
  useEffect(() => {
    if (status?.status !== 'running') return
    const id = setInterval(() => void fetchStatus(), 3000)
    return () => clearInterval(id)
  }, [status?.status, fetchStatus])

  // Auto-scroll history
  useEffect(() => {
    historyEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [status?.history])

  const start = useCallback(async () => {
    setStarting(true)
    try {
      const body: Record<string, unknown> = {}
      if (model.trim()) body.model = model.trim()
      body.temperature = temperature
      body.forever = forever
      await apiPost('/self-train/start', body)
      addToast('Self-training started', 'success')
      void fetchStatus()
    } catch {
      addToast('Could not start', 'error')
    } finally {
      setStarting(false)
    }
  }, [model, temperature, forever, addToast, fetchStatus])

  const stop = useCallback(async () => {
    try {
      await apiPost('/self-train/stop')
      addToast('Self-training stopped', 'success')
      void fetchStatus()
    } catch {
      addToast('Could not stop', 'error')
    }
  }, [addToast, fetchStatus])

  const isRunning = status?.status === 'running'

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return
      if (e.key === 'r' && !e.metaKey && !e.ctrlKey) { e.preventDefault(); void fetchStatus() }
    }
    document.addEventListener('keydown', onKey)
    return () => document.removeEventListener('keydown', onKey)
  }, [fetchStatus])

  return (
    <PageContainer
      title="Self-train"
      subtitle="Autonomous self-training subprocess"
      headerRight={
        <Button size="sm" variant="ghost" onClick={() => void fetchStatus()}>Refresh</Button>
      }
    >
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <Card>
          <CardContent className="p-3">
            <p className="text-xs text-muted-foreground">Status</p>
            <p className="text-lg font-medium">
              {loading ? '...' : status?.status === 'running' ? 'Running' : status?.status === 'exited' ? 'Exited' : 'Not started'}
            </p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-3">
            <p className="text-xs text-muted-foreground">PID</p>
            <p className="text-lg font-medium font-mono">{status?.pid ?? '--'}</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-3">
            <p className="text-xs text-muted-foreground">Exit code</p>
            <p className="text-lg font-medium">{status?.returncode ?? '--'}</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-3">
            <p className="text-xs text-muted-foreground">History lines</p>
            <p className="text-lg font-medium">{status?.history?.length ?? 0}</p>
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-base">Controls</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
            <div className="flex flex-col gap-1">
              <Label htmlFor="st-model" variant="uppercase">Model (optional)</Label>
              <Input id="st-model" value={model} onChange={e => setModel(e.target.value)}
                placeholder="gpt2" className="h-8 text-xs font-mono" disabled={isRunning} />
            </div>
            <div className="flex flex-col gap-1">
              <Label htmlFor="st-temp" variant="uppercase">Temperature</Label>
              <Input id="st-temp" type="number" min={0} max={2} step={0.1} value={temperature}
                onChange={e => setTemperature(Number(e.target.value))}
                className="h-8 text-xs font-mono" disabled={isRunning} />
            </div>
            <div className="flex flex-col gap-1">
              <Label variant="uppercase">Mode</Label>
              <button
                type="button"
                onClick={() => setForever(!forever)}
                aria-pressed={forever}
                aria-label={forever ? 'Switch to single pass mode' : 'Switch to train forever mode'}
                disabled={isRunning}
                className={`h-8 rounded border px-2 text-xs transition-colors ${
                  forever ? 'border-primary bg-primary/10 text-primary' : 'border-border'
                }`}
              >
                {forever ? 'Train forever' : 'Single pass'}
              </button>
            </div>
          </div>
          <div className="flex items-center gap-2">
            {isRunning ? (
              <Button variant="destructive" size="sm" onClick={stop}>Stop</Button>
            ) : (
              <Button size="sm" onClick={start} disabled={starting}>
                {starting ? 'Starting...' : 'Start self-training'}
              </Button>
            )}
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-base">Training history</CardTitle>
        </CardHeader>
        <CardContent>
          {(!status?.history || status.history.length === 0) ? (
            <p className="text-xs text-muted-foreground">No history yet.</p>
          ) : (
            <div className="max-h-[400px] overflow-y-auto rounded bg-muted/30 p-3 font-mono text-xs">
              {status.history.map((line, i) => (
                <div key={i} className="whitespace-pre-wrap">{line}</div>
              ))}
              <div ref={historyEndRef} />
            </div>
          )}
        </CardContent>
      </Card>
    </PageContainer>
  )
}
