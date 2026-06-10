'use client'

import { useState, useEffect, useCallback } from 'react'
import { IconRefresh, IconCheck, IconX } from '@/components/ui'
import { Button } from '@/components/ui/button'
import { cn } from '@/lib/cn'
import { PUBLIC_API_URL } from '@/lib/config'

interface InspectData {
  session: {
    id: string
    message_count: number
    messages: { role: string; content: string }[]
  }
  knowledge: {
    total_facts: number
    topics: string[]
  }
  traits: {
    personality?: Record<string, number>
    cognition?: Record<string, number>
    emotion?: Record<string, number>
  }
  modes: {
    personality?: { label: string; confidence: number; scores: Record<string, number> }
    memory?: { label: string; confidence: number; capacity?: number; scores: Record<string, number> }
    style?: { label: string; confidence: number; scores: Record<string, number> }
    task?: { label: string; confidence: number; scores: Record<string, number> }
  }
  feedback: {
    total: number
    thumbs_up: number
    thumbs_down: number
  }
  workspace: {
    working_memory: unknown[]
    semantic_keys: string[]
    episodic_count: number
    sensory_buffer_size: number
    system_prompt: string
  }
}

interface Props {
  sessionId: string | null
}

function TraitBar({ label, value }: { label: string; value: number }) {
  const pct = Math.round(value * 100)
  return (
    <div className="flex items-center gap-2">
      <span className="w-20 truncate text-[10px] text-muted-foreground">{label}</span>
      <div className="flex-1 h-1.5 rounded-full bg-muted/50 overflow-hidden">
        <div
          className="h-full rounded-full bg-primary/60 transition-all"
          style={{ width: `${pct}%` }}
        />
      </div>
      <span className="w-6 text-right text-[10px] text-muted-foreground">{pct}</span>
    </div>
  )
}

function ModeCard({ label, confidence, extra }: { label: string; confidence: number; extra?: string }) {
  return (
    <div className="flex items-center justify-between rounded border border-border/50 bg-muted/20 px-2 py-1.5">
      <span className="text-[11px] font-medium">{label}</span>
      <div className="flex items-center gap-2">
        {extra && <span className="text-[10px] text-muted-foreground">{extra}</span>}
        <span className={cn(
          'text-[10px] font-medium',
          confidence > 0.6 ? 'text-success' : confidence > 0.4 ? 'text-warning' : 'text-muted-foreground',
        )}>
          {(confidence * 100).toFixed(0)}%
        </span>
      </div>
    </div>
  )
}

export function ContextInspector({ sessionId }: Props) {
  const [data, setData] = useState<InspectData | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const fetchData = useCallback(async () => {
    if (!sessionId) return
    setLoading(true)
    setError(null)
    try {
      const res = await fetch(`${PUBLIC_API_URL}/session/${sessionId}/inspector`)
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const json = await res.json()
      setData(json)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load')
    } finally {
      setLoading(false)
    }
  }, [sessionId])

  useEffect(() => {
    if (sessionId) fetchData()
  }, [sessionId, fetchData])

  if (!sessionId) {
    return (
      <div className="flex items-center justify-center py-8 text-xs text-muted-foreground">
        No active session
      </div>
    )
  }

  if (loading && !data) {
    return (
      <div className="space-y-3 animate-pulse">
        <div className="h-12 rounded bg-muted/30" />
        <div className="h-12 rounded bg-muted/30" />
        <div className="h-12 rounded bg-muted/30" />
      </div>
    )
  }

  if (error) {
    return (
      <div className="flex flex-col items-center gap-2 py-6">
        <span className="text-xs text-destructive">{error}</span>
        <Button variant="outline" size="sm" className="h-6 text-[10px]" onClick={fetchData}>
          <IconRefresh className="h-3 w-3 mr-1" /> Retry
        </Button>
      </div>
    )
  }

  if (!data) {
    return (
      <div className="flex flex-col items-center gap-2 py-6">
        <span className="text-xs text-muted-foreground">No data</span>
        <Button variant="outline" size="sm" className="h-6 text-[10px]" onClick={fetchData}>
          <IconRefresh className="h-3 w-3 mr-1" /> Load
        </Button>
      </div>
    )
  }

  const allModes: { manager: string; label: string; confidence: number; cap?: number }[] = [
    data.modes?.personality && { ...data.modes.personality, manager: 'Personality' },
    data.modes?.memory && { ...data.modes.memory, manager: 'Memory', cap: data.modes.memory.capacity },
    data.modes?.style && { ...data.modes.style, manager: 'Style' },
    data.modes?.task && { ...data.modes.task, manager: 'Task' },
  ].filter((m): m is NonNullable<typeof m> => m != null)

  const allTraits: Record<string, number> = {
    ...data.traits?.personality,
    ...data.traits?.cognition,
    ...data.traits?.emotion,
  }

  return (
    <div className="space-y-3">
      {/* Refresh */}
      <div className="flex items-center justify-between">
        <span className="text-[10px] text-muted-foreground">Live context state</span>
        <Button variant="ghost" size="sm" className="h-5 text-[10px] px-1" onClick={fetchData} disabled={loading}>
          <IconRefresh className={cn('h-3 w-3 mr-1', loading && 'animate-spin')} />
          Refresh
        </Button>
      </div>

      {/* Session summary */}
      <div className="rounded border border-border/50 bg-muted/10 p-2 space-y-1">
        <div className="text-[10px] font-medium text-muted-foreground">Session</div>
        <div className="flex gap-3 text-[11px]">
          <span>{data.session.message_count} messages</span>
        </div>
      </div>

      {/* Manager modes */}
      {allModes.length > 0 && (
        <div className="space-y-1">
          <div className="text-[10px] font-medium text-muted-foreground">Active Modes</div>
          <div className="space-y-1">
            {allModes.map(m => (
              <ModeCard
                key={m.manager}
                label={`${m.manager}: ${m.label}`}
                confidence={m.confidence}
                extra={m.cap != null ? `cap:${String(m.cap)}` : undefined}
              />
            ))}
          </div>
        </div>
      )}

      {/* Traits */}
      {Object.keys(allTraits).length > 0 && (
        <div className="space-y-1">
          <div className="text-[10px] font-medium text-muted-foreground">Trait Weights</div>
          <div className="space-y-0.5">
            {Object.entries(allTraits).slice(0, 10).map(([k, v]) => (
              <TraitBar key={k} label={k.replace(/_/g, ' ')} value={v as number} />
            ))}
          </div>
        </div>
      )}

      {/* Knowledge */}
      {data.knowledge.total_facts > 0 && (
        <div className="rounded border border-border/50 bg-muted/10 p-2 space-y-1">
          <div className="text-[10px] font-medium text-muted-foreground">Knowledge Base</div>
          <div className="text-[11px]">{data.knowledge.total_facts} facts</div>
          {data.knowledge.topics.length > 0 && (
            <div className="flex flex-wrap gap-1">
              {data.knowledge.topics.slice(0, 5).map(t => (
                <span key={t} className="px-1.5 py-0.5 rounded bg-muted/30 text-[10px] text-muted-foreground">{t}</span>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Feedback */}
      {data.feedback.total > 0 && (
        <div className="rounded border border-border/50 bg-muted/10 p-2 space-y-1">
          <div className="text-[10px] font-medium text-muted-foreground">Feedback</div>
          <div className="flex gap-3 text-[11px]">
            <span className="flex items-center gap-1"><IconCheck className="h-3 w-3 text-success" />{data.feedback.thumbs_up}</span>
            <span className="flex items-center gap-1"><IconX className="h-3 w-3 text-destructive" />{data.feedback.thumbs_down}</span>
            <span>{data.feedback.total} total</span>
          </div>
        </div>
      )}

      {/* Workspace */}
      {data.workspace.semantic_keys.length > 0 && (
        <div className="rounded border border-border/50 bg-muted/10 p-2 space-y-1">
          <div className="text-[10px] font-medium text-muted-foreground">Workspace Memory</div>
          <div className="text-[11px]">{data.workspace.episodic_count} episodes</div>
          {data.workspace.semantic_keys.length > 0 && (
            <div className="flex flex-wrap gap-1">
              {data.workspace.semantic_keys.slice(0, 8).map(k => (
                <span key={k} className="px-1.5 py-0.5 rounded bg-muted/30 text-[10px] text-muted-foreground font-mono">{k}</span>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  )
}
