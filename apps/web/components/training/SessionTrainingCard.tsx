'use client'

import { useState, useCallback, useEffect } from 'react'
import { cn, Card, CardContent, CardHeader, CardTitle, Button } from '@sloughgpt/strui'
import { trainingJobsController, type ChatSession } from '@/lib/training-controller'

interface Props {
  addToast: (msg: string, type?: 'success' | 'error' | 'info') => void
}

export function SessionTrainingCard({ addToast }: Props) {
  const [sessions, setSessions] = useState<ChatSession[]>([])
  const [loading, setLoading] = useState(true)
  const [selected, setSelected] = useState<Set<string>>(new Set())
  const [training, setTraining] = useState(false)
  const [pairCounts, setPairCounts] = useState<Record<string, number>>({})

  const fetchSessions = useCallback(async () => {
    setLoading(true)
    try {
      const data = await trainingJobsController.listChatSessions()
      setSessions(data)
    } catch {
      addToast('Could not load chat sessions', 'error')
    } finally {
      setLoading(false)
    }
  }, [addToast])

  useEffect(() => {
    let active = true
    const load = async () => {
      setLoading(true)
      try {
        const data = await trainingJobsController.listChatSessions()
        if (active) setSessions(data)
      } catch {
        if (active) addToast('Could not load chat sessions', 'error')
      } finally {
        if (active) setLoading(false)
      }
    }
    void load()
    return () => { active = false }
  }, [addToast])

  const fetchPairs = useCallback(async (sessionId: string) => {
    try {
      const result = await trainingJobsController.getSessionPairs(sessionId)
      setPairCounts(prev => ({ ...prev, [sessionId]: result.count }))
    } catch { /* silent */ }
  }, [])

  useEffect(() => {
    let active = true
    const fetchMissing = async () => {
      for (const id of selected) {
        if (pairCounts[id] == null) {
          try {
            const result = await trainingJobsController.getSessionPairs(id)
            if (active) setPairCounts(prev => ({ ...prev, [id]: result.count }))
          } catch { /* silent */ }
        }
      }
    }
    void fetchMissing()
    return () => { active = false }
  }, [selected, pairCounts, fetchPairs])

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
      <CardContent className="space-y-3">
        {loading ? (
          <p className="text-xs text-muted-foreground">Loading sessions...</p>
        ) : sessions.length === 0 ? (
          <p className="text-xs text-muted-foreground">No chat sessions found.</p>
        ) : (
          <div className="space-y-1.5 max-h-64 overflow-y-auto">
            {sessions.slice(0, 50).map(s => (
              <div
                key={s.id}
                className={cn('flex items-center gap-3 rounded border px-3 py-2 text-sm transition-colors', selected.has(s.id) ? 'border-primary bg-primary/5' : 'border-border/50 hover:bg-muted/30')}
              >
                <input
                  type="checkbox"
                  checked={selected.has(s.id)}
                  onChange={() => toggleSelect(s.id)}
                  aria-label={`Select session ${s.name}`}
                  className="h-3.5 w-3.5 rounded border-border shrink-0"
                />
                <div className="min-w-0 flex-1">
                  <p className="truncate text-xs font-medium">{s.name}</p>
                  <div className="flex gap-2 text-[10px] text-muted-foreground">
                    <span>{new Date(s.updated_at).toLocaleDateString()}</span>
                    {pairCounts[s.id] != null && <span>{pairCounts[s.id]} pairs</span>}
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}

        {selected.size > 0 && (
          <div className="flex items-center justify-between rounded bg-muted/30 px-3 py-2 text-xs">
            <span className="text-muted-foreground">
              {selected.size} sessions, ~{totalPairs} pairs
            </span>
            <Button size="sm" onClick={() => void handleTrain()} disabled={training}>
              {training ? 'Training...' : 'Train from Sessions'}
            </Button>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
