'use client'

import { useCallback, useEffect, useRef, useState } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@sloughgpt/strui'
import { Button } from '@sloughgpt/strui'
import { Skeleton } from '@sloughgpt/strui'
import { trainingController } from '@/lib/controllers'
import { useToastStore } from '@/lib/toast-store'
import { logger } from '@/lib/dev-log'
import type { AutoTrainStatus, ChatSession } from '@/lib/training-controller'

interface TrainProgress {
  step: number | null
  loss: number | null
  epoch: number | null
  progress_pct: number
  total_steps: number | null
  phase: string
  message: string
}

interface TrainSample {
  prompt: string
  response: string
}

export function TrainFromSessionsCard() {
  const addToast = useToastStore(s => s.addToast)
  const [status, setStatus] = useState<AutoTrainStatus | null>(null)
  const [sessions, setSessions] = useState<ChatSession[]>([])
  const [loading, setLoading] = useState(true)
  const [training, setTraining] = useState(false)
  const [showSessions, setShowSessions] = useState(false)
  const [selectedSessions, setSelectedSessions] = useState<Set<string>>(new Set())
  const [showConfig, setShowConfig] = useState(false)
  const [configThreshold, setConfigThreshold] = useState<string>('')
  const [configInterval, setConfigInterval] = useState<string>('')
  const [savingConfig, setSavingConfig] = useState(false)
  const [lastResult, setLastResult] = useState<{ loss: number; steps: number; elapsed_ms: number; checkpoint: string; perplexity?: number; samples?: TrainSample[] } | null>(null)
  const [progress, setProgress] = useState<TrainProgress | null>(null)
  const [elapsed, setElapsed] = useState(0)
  const startTimeRef = useRef<number>(0)
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const fetchStatus = useCallback(async () => {
    try {
      const [s, sess] = await Promise.all([
        trainingController.getAutoTrainStatus(),
        trainingController.listChatSessions(),
      ])
      setStatus(s)
      setSessions(sess)
    } catch {
      logger.warning('TrainFromSessionsCard: auto-train endpoint not available', {})
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => { void fetchStatus() }, [fetchStatus])

  // Poll every 30s when auto-train is enabled
  useEffect(() => {
    if (status?.enabled) {
      pollRef.current = setInterval(() => { void fetchStatus() }, 30000)
      return () => { if (pollRef.current) clearInterval(pollRef.current) }
    }
    return () => { if (pollRef.current) clearInterval(pollRef.current) }
  }, [status?.enabled, fetchStatus])

  const toggleSession = useCallback((id: string) => {
    setSelectedSessions(prev => {
      const next = new Set(prev)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })
  }, [])

  const handleTrainFromSessions = useCallback(async () => {
    setTraining(true)
    setLastResult(null)
    setProgress(null)
    startTimeRef.current = Date.now()
    setElapsed(0)
    timerRef.current = setInterval(() => {
      setElapsed(Math.floor((Date.now() - startTimeRef.current) / 1000))
    }, 1000)

    try {
      await trainingController.startFromSessionsSloNet({
        epochs: 5,
        min_pair_quality: 0.0,
        max_pairs: 500,
      })

      for await (const event of trainingController.streamFromSessionsSloNet()) {
        if (event.status === 'error') {
          const errorMsg = (event.data as { error?: string })?.error || event.message || 'Training failed'
          addToast(errorMsg, 'error')
          break
        }

        if (event.status === 'complete') {
          const d = event.data as { checkpoint?: string; final_loss?: number; num_pairs?: number; perplexity?: number; samples?: TrainSample[] }
          setLastResult({
            loss: d.final_loss ?? 0,
            steps: d.num_pairs ?? 0,
            elapsed_ms: Date.now() - startTimeRef.current,
            checkpoint: d.checkpoint ?? '',
            perplexity: d.perplexity,
            samples: d.samples,
          })
          addToast(
            `Training complete — ${(d.num_pairs ?? 0)} pairs, loss ${(d.final_loss ?? 0).toFixed(4)}`,
            'success'
          )
          void fetchStatus()
          break
        }

        if (event.phase === 'TRAIN' && event.status === 'working') {
          const d = event.data as { step?: number; loss?: number; epoch?: number }
          const m = event.meta as { epoch?: number; total_epochs?: number } | undefined
          setProgress({
            step: d.step ?? null,
            loss: d.loss ?? null,
            epoch: m?.epoch ?? d.epoch ?? null,
            progress_pct: m?.total_epochs && m.epoch ? Math.round(((m.epoch + 1) / m.total_epochs) * 100) : 0,
            total_steps: null,
            phase: event.phase,
            message: event.message,
          })
        }

        if (event.phase === 'PAIRS' && event.status === 'working') {
          setProgress(prev => ({
            ...(prev ?? { step: null, loss: null, epoch: null, progress_pct: 0, total_steps: null, phase: '', message: '' }),
            phase: 'PAIRS',
            message: event.message,
          }))
        }
      }
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Training failed'
      addToast(msg, 'error')
    } finally {
      setTraining(false)
      if (timerRef.current) { clearInterval(timerRef.current); timerRef.current = null }
    }
  }, [addToast, fetchStatus])

  const handleSaveConfig = useCallback(async () => {
    setSavingConfig(true)
    try {
      const params: { threshold?: number; interval_s?: number } = {}
      if (configThreshold) params.threshold = parseInt(configThreshold, 10)
      if (configInterval) params.interval_s = parseInt(configInterval, 10)
      if (Object.keys(params).length === 0) return
      await trainingController.updateAutoTrainConfig(params)
      addToast('Config updated', 'success')
      setConfigThreshold('')
      setConfigInterval('')
      void fetchStatus()
    } catch {
      addToast('Failed to update config', 'error')
    } finally {
      setSavingConfig(false)
    }
  }, [addToast, configThreshold, configInterval, fetchStatus])

  if (loading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="text-base">From conversations</CardTitle>
        </CardHeader>
        <CardContent>
          <Skeleton className="h-16 w-full" />
        </CardContent>
      </Card>
    )
  }

  if (!status) return null

  const progressPct = status.threshold > 0
    ? Math.min(100, Math.round((status.pending_conversations / status.threshold) * 100))
    : 0

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between">
        <CardTitle className="text-base">From conversations</CardTitle>
        {status.enabled && (
          <span className="flex items-center gap-1.5 text-[11px] text-muted-foreground">
            <span className="relative flex h-1.5 w-1.5 shrink-0">
              <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-green-500/60" />
              <span className="relative inline-flex h-1.5 w-1.5 rounded-full bg-green-500" />
            </span>
            Auto-training on
          </span>
        )}
      </CardHeader>
      <CardContent className="space-y-3">
        <p className="text-xs text-muted-foreground">
          Train from your server&apos;s own conversation logs — no data upload needed.
        </p>

        {/* Data source counts */}
        <div className="flex flex-wrap gap-x-4 gap-y-1 text-[11px] text-muted-foreground/70">
          <span>{status.session_count} conversations</span>
          <span>{status.response_log_count} log files</span>
        </div>

        {/* Config section */}
        <button
          className="text-[11px] text-muted-foreground hover:text-foreground transition-colors"
          onClick={() => setShowConfig(!showConfig)}
        >
          {showConfig ? 'Hide' : 'Configure'} auto-train
        </button>
        {showConfig && (
          <div className="flex flex-wrap items-end gap-2 rounded-lg border border-border/40 p-2">
            <label className="text-[11px] text-muted-foreground/70">
              Threshold
              <input
                type="number"
                min={1}
                max={100}
                value={configThreshold || status.threshold}
                onChange={e => setConfigThreshold(e.target.value)}
                className="ml-1 h-6 w-14 rounded border border-border/60 bg-background px-1.5 text-[11px] text-foreground"
              />
            </label>
            <label className="text-[11px] text-muted-foreground/70">
              Interval (s)
              <input
                type="number"
                min={30}
                max={3600}
                step={30}
                value={configInterval || status.interval_s}
                onChange={e => setConfigInterval(e.target.value)}
                className="ml-1 h-6 w-16 rounded border border-border/60 bg-background px-1.5 text-[11px] text-foreground"
              />
            </label>
            <Button
              size="sm"
              variant="ghost"
              className="h-6 text-[11px]"
              disabled={savingConfig || (!configThreshold && !configInterval)}
              onClick={() => void handleSaveConfig()}
            >
              {savingConfig ? 'Saving...' : 'Save'}
            </Button>
          </div>
        )}

        {/* Auto-train progress */}
        {status.enabled && (
          <div className="space-y-1.5">
            <div className="flex items-center justify-between text-[11px]">
              <span className="text-muted-foreground/70">
                Next train: {status.pending_conversations} / {status.threshold} conversations
              </span>
              <span className="text-muted-foreground/50">{progressPct}%</span>
            </div>
            <div className="h-1.5 w-full bg-muted rounded-full overflow-hidden">
              <div
                className="h-full bg-primary/60 transition-all duration-500 rounded-full"
                style={{ width: `${progressPct}%` }}
              />
            </div>
            {status.last_train && (
              <p className="text-[10px] text-muted-foreground/50">
                Last trained: {new Date(status.last_train).toLocaleTimeString()}
                {status.last_loss != null && <> · loss {status.last_loss.toFixed(4)}</>}
              </p>
            )}
          </div>
        )}

        {/* Training history (when not auto-train-enabled) */}
        {!status.enabled && status.total_trains > 0 && (
          <div className="text-[11px] text-muted-foreground/70">
            <span>{status.total_trains} trains completed</span>
            {status.last_loss != null && <> · last loss: {status.last_loss.toFixed(4)}</>}
            {status.last_checkpoint && <> · {status.last_checkpoint}</>}
          </div>
        )}

        {/* Session selector */}
        {sessions.length > 0 && (
          <div className="space-y-2">
            <button
              className="text-[11px] text-muted-foreground hover:text-foreground transition-colors"
              onClick={() => setShowSessions(!showSessions)}
            >
              {showSessions ? 'Hide' : 'Select'} sessions ({sessions.length})
              {selectedSessions.size > 0 && ` · ${selectedSessions.size} selected`}
            </button>
            {showSessions && (
              <div className="space-y-1 max-h-40 overflow-y-auto rounded-lg border border-border/40 p-2">
                {sessions.slice(0, 15).map(s => (
                  <label
                    key={s.id}
                    className="flex items-center gap-2 text-[11px] text-muted-foreground hover:text-foreground cursor-pointer py-0.5"
                  >
                    <input
                      type="checkbox"
                      checked={selectedSessions.has(s.id)}
                      onChange={() => toggleSession(s.id)}
                      className="h-3 w-3 rounded border-border"
                    />
                    <span className="truncate flex-1">{s.name}</span>
                    {s.messages && (
                      <span className="text-[10px] text-muted-foreground/50 shrink-0">
                        {s.messages.length} msgs
                      </span>
                    )}
                  </label>
                ))}
              </div>
            )}
          </div>
        )}

        {/* Live training progress (SSE) */}
        {training && (
          <div className="space-y-2 rounded-lg border border-primary/20 bg-primary/5 p-2">
            <div className="flex items-center gap-2 text-[11px]">
              <span className="relative flex h-2 w-2 shrink-0">
                <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-primary/60" />
                <span className="relative inline-flex h-2 w-2 rounded-full bg-primary" />
              </span>
              <span className="text-foreground font-medium">
                {progress?.phase === 'PAIRS' ? 'Extracting pairs...' : 'Training in progress'}
              </span>
              <span className="text-muted-foreground/50 ml-auto">{elapsed}s</span>
            </div>

            {/* Progress bar */}
            {progress && progress.progress_pct > 0 && (
              <div className="space-y-1">
                <div className="h-1.5 w-full bg-muted rounded-full overflow-hidden">
                  <div
                    className="h-full bg-primary/70 transition-all duration-300 rounded-full"
                    style={{ width: `${progress.progress_pct}%` }}
                  />
                </div>
                <div className="flex justify-between text-[10px] text-muted-foreground/60">
                  <span>
                    step {progress.step ?? '—'}
                    {progress.total_steps != null && <> / {progress.total_steps}</>}
                  </span>
                  <span>{progress.progress_pct}%</span>
                </div>
              </div>
            )}

            {/* Live loss */}
            {progress?.loss != null && (
              <div className="flex gap-4 text-[10px] text-muted-foreground/60">
                <span>loss {progress.loss.toFixed(4)}</span>
                {progress.epoch != null && <span>epoch {progress.epoch}</span>}
              </div>
            )}

            {/* Phase message */}
            {progress?.message && (
              <p className="text-[10px] text-muted-foreground/50 truncate">{progress.message}</p>
            )}
          </div>
        )}

        {/* Last training result */}
        {lastResult && !training && (
          <div className="rounded-lg border border-success/30 bg-success/5 p-2 text-[11px] space-y-1.5">
            <div className="flex items-center gap-1.5 text-success font-medium">
              <span className="inline-block w-1.5 h-1.5 rounded-full bg-success" />
              Training complete
            </div>
            <div className="text-muted-foreground/70">
              loss {lastResult.loss.toFixed(4)} · {lastResult.steps} pairs · {(lastResult.elapsed_ms / 1000).toFixed(1)}s
              {lastResult.perplexity != null && <> · perplexity {lastResult.perplexity.toFixed(2)}</>}
            </div>
            {lastResult.checkpoint && (
              <Button
                size="sm"
                variant="ghost"
                className="h-6 text-[11px]"
                onClick={() => {
                  const name = lastResult.checkpoint.split('/').pop()?.replace('.soul', '') || ''
                  trainingController.loadCheckpoint(name).then(() => {
                    addToast('Model loaded for chat', 'success')
                  }).catch((e: unknown) => {
                    addToast(`Load failed: ${e instanceof Error ? e.message : 'unknown'}`, 'error')
                  })
                }}
              >
                Load for chat
              </Button>
            )}
            {lastResult.samples && lastResult.samples.length > 0 && (
              <div className="space-y-1 mt-1.5 pt-1.5 border-t border-success/20">
                <p className="text-[10px] text-muted-foreground/50 font-medium">Samples:</p>
                {lastResult.samples.slice(0, 3).map((s, i) => (
                  <div key={i} className="text-[10px] text-muted-foreground/60">
                    <span className="text-foreground/70">{s.prompt}</span>
                    <span className="text-muted-foreground/40"> → </span>
                    <span>{s.response.slice(0, 60)}{s.response.length > 60 ? '...' : ''}</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        <Button
          size="sm"
          onClick={() => void handleTrainFromSessions()}
          disabled={training}
        >
          {training
            ? `Training... ${elapsed}s`
            : selectedSessions.size > 0
              ? `Train from ${selectedSessions.size} sessions`
              : 'Train from conversations'
          }
        </Button>
      </CardContent>
    </Card>
  )
}
