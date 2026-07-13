'use client'

import { useCallback, useEffect, useRef, useState } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@sloughgpt/strui'
import { Button } from '@sloughgpt/strui'
import { Skeleton } from '@sloughgpt/strui'
import { trainingController } from '@/lib/controllers'
import { useToastStore } from '@/lib/toast-store'
import type { AutoTrainStatus, ChatSession } from '@/lib/training-controller'

export function TrainFromSessionsCard() {
  const addToast = useToastStore(s => s.addToast)
  const [status, setStatus] = useState<AutoTrainStatus | null>(null)
  const [sessions, setSessions] = useState<ChatSession[]>([])
  const [loading, setLoading] = useState(true)
  const [training, setTraining] = useState(false)
  const [showSessions, setShowSessions] = useState(false)
  const [selectedSessions, setSelectedSessions] = useState<Set<string>>(new Set())
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
      // endpoint might not be available
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
    try {
      const sessionIds = selectedSessions.size > 0 ? Array.from(selectedSessions) : undefined
      const result = await trainingController.trainFromSessions({
        limit: 50,
        min_length: 5,
        session_ids: sessionIds,
      })
      const label = sessionIds ? `${sessionIds.length} sessions` : 'all sessions'
      addToast(
        `Trained from ${label} — loss ${result.loss.toFixed(4)}, ${result.steps} steps`,
        'success'
      )
      void fetchStatus()
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Training failed'
      addToast(msg, 'error')
    } finally {
      setTraining(false)
    }
  }, [addToast, fetchStatus, selectedSessions])

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

        <Button
          size="sm"
          onClick={() => void handleTrainFromSessions()}
          disabled={training}
        >
          {training
            ? 'Training...'
            : selectedSessions.size > 0
              ? `Train from ${selectedSessions.size} sessions`
              : 'Train from conversations'
          }
        </Button>
      </CardContent>
    </Card>
  )
}
