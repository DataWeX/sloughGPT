'use client'

import { useState, useCallback, useEffect, useRef } from 'react'
import { cn, Card, CardContent, CardHeader, CardTitle, Button, Checkbox, Skeleton } from '@sloughgpt/strui'
import { trainingJobsController, type ChatSession } from '@/lib/training-controller'
import { ConfirmDialog } from '@/components/ConfirmDialog'

interface Props {
  addToast: (msg: string, type?: 'success' | 'error' | 'info') => void
}

export function SessionTrainingCard({ addToast }: Props) {
  const [sessions, setSessions] = useState<ChatSession[]>([])
  const [loading, setLoading] = useState(true)
  const [selected, setSelected] = useState<Set<string>>(new Set())
  const [training, setTraining] = useState(false)
  const [pendingTrain, setPendingTrain] = useState(false)
  const [pairCounts, setPairCounts] = useState<Record<string, number>>({})

  const activeRef = useRef(true)

  const fetchSessions = useCallback(async () => {
    setLoading(true)
    try {
      const data = await trainingJobsController.listChatSessions()
      if (activeRef.current) setSessions(data)
    } catch {
      if (activeRef.current) addToast('Could not load chat sessions', 'error')
    } finally {
      if (activeRef.current) setLoading(false)
    }
  }, [addToast])

  useEffect(() => {
    activeRef.current = true
    void fetchSessions()
    return () => { activeRef.current = false }
  }, [fetchSessions])

  useEffect(() => {
    let active = true
    const fetchMissing = async () => {
      for (const id of selected) {
        if (pairCounts[id] == null) {
          try {
            const result = await trainingJobsController.getSessionPairs(id)
            if (active) setPairCounts(prev => ({ ...prev, [id]: result.count }))
          } catch {
            if (active) addToast('Could not load pair count', 'error')
          }
        }
      }
    }
    void fetchMissing()
    return () => { active = false }
  }, [selected])

  const toggleSelect = useCallback((id: string) => {
    setSelected(prev => {
      const next = new Set(prev)
      if (next.has(id)) next.delete(id); else next.add(id)
      return next
    })
  }, [])

  const handleTrain = useCallback(async () => {
    if (selected.size === 0) return
    setTraining(true)
    try {
      const result = await trainingJobsController.trainFromSessions({
        session_ids: Array.from(selected),
      })
      if (result.success) {
        addToast(`Trained from ${selected.size} sessions (loss: ${result.loss.toFixed(4)}, ${result.steps} steps)`, 'success')
      } else {
        addToast(result.message || 'Training completed with issues', 'info')
      }
    } catch (e) {
      addToast(`Session training failed: ${e instanceof Error ? e.message : 'Unknown error'}`, 'error')
    } finally {
      setTraining(false)
    }
  }, [selected, addToast])

  const totalPairs = Array.from(selected).reduce((sum, id) => sum + (pairCounts[id] ?? 0), 0)

  return (
    <Card>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="text-base">Train from Sessions</CardTitle>
          <div className="flex gap-1">
            {selected.size > 0 && (
              <Button size="sm" variant="ghost" className="text-destructive text-[10px]" onClick={() => setSelected(new Set())}>
                Clear ({selected.size})
              </Button>
            )}
            <Button size="sm" variant="ghost" className="text-[10px]" onClick={() => {
              if (selected.size === sessions.length) setSelected(new Set())
              else setSelected(new Set(sessions.map(s => s.id)))
            }}>
              {selected.size === sessions.length ? 'Deselect all' : 'Select all'}
            </Button>
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-2">
        {loading ? (
          <div className="space-y-1.5">
            <Skeleton className="h-12 w-full" />
            <Skeleton className="h-12 w-full" />
            <Skeleton className="h-12 w-full" />
          </div>
        ) : sessions.length === 0 ? (
          <p className="text-[10px] text-muted-foreground/60">No chat sessions found.</p>
        ) : (
          <div className="space-y-1 max-h-64 overflow-y-auto">
            {sessions.slice(0, 50).map(s => (
              <div
                key={s.id}
                className={cn('flex items-center gap-2 rounded-lg border px-2.5 py-1.5 text-xs transition-colors', selected.has(s.id) ? 'border-primary bg-primary/5' : 'border-border/40 hover:bg-muted/20')}
              >
                <Checkbox
                  checked={selected.has(s.id)}
                  onCheckedChange={() => toggleSelect(s.id)}
                  aria-label={`Select session ${s.name}`}
                  className="h-3.5 w-3.5 rounded border-border shrink-0"
                />
                <div className="min-w-0 flex-1">
                  <p className="truncate text-[11px] font-medium">{s.name}</p>
                  <div className="flex gap-1.5 text-[9px] text-muted-foreground/60">
                    <span className="tabular-nums">{new Date(s.updated_at).toLocaleDateString()}</span>
                    {pairCounts[s.id] != null && <span className="tabular-nums">{pairCounts[s.id]} pairs</span>}
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}

        {selected.size > 0 && (
          <div className="flex items-center justify-between rounded-lg bg-muted/20 px-2.5 py-1.5 text-[10px]">
            <span className="text-muted-foreground/60">
              {selected.size} sessions, ~{totalPairs} pairs
            </span>
            <Button size="sm" className="h-6 text-[10px]" onClick={() => setPendingTrain(true)} disabled={training}>
              {training ? 'Training...' : 'Train from Sessions'}
            </Button>
          </div>
        )}
      </CardContent>
      <ConfirmDialog
        open={pendingTrain}
        onOpenChange={setPendingTrain}
        title="Start training?"
        description={`This will train a model from ${selected.size} sessions (~${totalPairs} pairs). This may take a while.`}
        confirmLabel="Start training"
        destructive={false}
        onConfirm={() => { setPendingTrain(false); void handleTrain() }}
      />
    </Card>
  )
}
