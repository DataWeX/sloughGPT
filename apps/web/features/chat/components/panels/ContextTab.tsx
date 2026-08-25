'use client'

import { useEffect, useState } from 'react'
import { cn } from '@sloughgpt/strui'
import { IconRefresh } from '@sloughgpt/strui'
import { chatController, type ContextInspector } from '@/lib/chat-controller'
import { soulsController } from '@/lib/souls-controller'
import { feedbackController } from '@/lib/feedback-controller'
import { chatDB } from '@/lib/db'
import { logger } from '@/lib/dev-log'
import { useToastStore } from '@/lib/toast-store'

interface SteeringMode {
  label: string
  confidence: number
  capacity?: number
  scores?: Record<string, number>
}

const MODE_ORDER: Array<{ key: string; label: string; hint: string }> = [
  { key: 'personality', label: 'Personality', hint: 'tone & empathy' },
  { key: 'memory', label: 'Memory', hint: 'retention' },
  { key: 'style', label: 'Style', hint: 'formality' },
  { key: 'task', label: 'Task', hint: 'reasoning depth' },
]

const TRAIT_GROUPS: Array<{ key: string; label: string }> = [
  { key: 'personality', label: 'Personality' },
  { key: 'cognition', label: 'Cognition' },
  { key: 'emotion', label: 'Emotion' },
]

function confidencePct(value: number | undefined): number {
  return Math.round(Math.min(1, Math.max(0, value ?? 0)) * 100)
}

/**
 * Context Inspector — shows what the AI currently sees when it answers:
 * steering modes, trait weights, and workspace memory state. Read-only
 * introspection surface in the chat tools panel; all fetches fail soft so
 * a missing backend never breaks the panel.
 */
export function ContextTab() {
  const [inspector, setInspector] = useState<ContextInspector | null>(null)
  const [modes, setModes] = useState<Record<string, SteeringMode> | null>(null)
  const [traitWeights, setTraitWeights] = useState<Record<string, Record<string, number>> | null>(null)
  const [knowledgeCount, setKnowledgeCount] = useState<number | null>(null)
  const [feedback, setFeedback] = useState<{ up: number; down: number } | null>(null)
  const [loading, setLoading] = useState(true)
  const [showPrompt, setShowPrompt] = useState(false)

  const addToast = useToastStore(s => s.addToast)

  const fetchAll = async () => {
    setLoading(true)
    try {
      const [insp, modeData, traits, knowledge, fb] = await Promise.all([
        chatController.inspectContext(),
        soulsController.getModes().catch((e) => { logger.debug('Could not load modes', { e }); return null }),
        soulsController.getTraitWeights().catch((e) => { logger.debug('Could not load traits', { e }); return null }),
        chatDB.getKnowledge().catch((e) => { logger.debug('Could not load knowledge', { e }); return null }),
        feedbackController.getFeedbackStats().then(
          s => ({ up: s.db_stats?.thumbs_up ?? 0, down: s.db_stats?.thumbs_down ?? 0 }),
          (e) => { logger.debug('Could not load feedback', { e }); return null },
        ),
      ])
      setInspector(insp)
      setModes(modeData)
      setTraitWeights(traits)
      setKnowledgeCount(Array.isArray(knowledge) ? knowledge.length : null)
      setFeedback(fb)
    } catch (err) {
      addToast('Failed to load context inspector', 'error')
      logger.debug('Could not context inspector fetch', { exception: String(err) })
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { fetchAll() }, [])

  const hasAnything = Boolean(inspector) || Boolean(modes) || Boolean(traitWeights) || knowledgeCount != null || Boolean(feedback)
  const workingCount = inspector?.working_memory?.length ?? 0
  const memoryRows = [
    { label: 'Working', value: workingCount },
    { label: 'Semantic', value: inspector?.semantic_keys?.length ?? 0 },
    { label: 'Episodic', value: inspector?.episodic_count ?? 0 },
    { label: 'Sensory', value: inspector?.sensory_buffer_size ?? 0 },
  ]
  const activeModes = modes ? Object.entries(modes).filter(([, m]) => m && typeof m.label === 'string') : []

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <span className="text-xs text-muted-foreground">What the model sees</span>
        <button
          type="button"
          onClick={fetchAll}
          disabled={loading}
          className="h-6 w-6 flex items-center justify-center rounded text-muted-foreground hover:text-foreground hover:bg-muted/60 transition-colors disabled:opacity-40"
          aria-label="Refresh context"
        >
          <IconRefresh className={cn('h-3 w-3', loading && 'animate-spin')} />
        </button>
      </div>

      {loading ? (
        <p className="text-[10px] text-muted-foreground text-center py-3" aria-busy="true">Loading context…</p>
      ) : !hasAnything ? (
        <p className="text-xs text-muted-foreground text-center py-4">
          Context unavailable right now. Please try again.
        </p>
      ) : (
        <>
          {activeModes.length > 0 && (
            <div className="space-y-1.5">
              <span className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wider">Steering modes</span>
              {MODE_ORDER.map(({ key, label, hint }) => {
                const mode = modes?.[key]
                if (!mode) return null
                return (
                  <div key={key} className="flex items-center gap-2">
                    <span className="w-16 shrink-0 text-[10px] text-muted-foreground">{label}</span>
                    <span className="flex-1 h-1.5 rounded-full bg-muted overflow-hidden">
                      <span
                        className="block h-full rounded-full bg-primary/70 transition-all"
                        style={{ width: `${confidencePct(mode.confidence)}%` }}
                      />
                    </span>
                    <span className="w-16 shrink-0 text-right text-[10px] text-muted-foreground truncate" title={`${label} — ${hint}`}>
                      {mode.label}
                    </span>
                  </div>
                )
              })}
            </div>
          )}

          {traitWeights && (Object.keys(traitWeights).length > 0) && (
            <div className="space-y-1.5">
              <span className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wider">Trait weights</span>
              {TRAIT_GROUPS.map(({ key, label }) => {
                const group = traitWeights[key]
                if (!group || Object.keys(group).length === 0) return null
                const traits = Object.entries(group).slice(0, 3)
                return (
                  <div key={key} className="space-y-0.5">
                    <span className="text-[9px] text-muted-foreground/70 uppercase">{label}</span>
                    {traits.map(([trait, value]) => (
                      <div key={trait} className="flex items-center gap-2">
                        <span className="w-16 shrink-0 text-[10px] text-muted-foreground truncate capitalize">{trait}</span>
                        <span className="flex-1 h-1 rounded-full bg-muted overflow-hidden">
                          <span className="block h-full rounded-full bg-accent/70 transition-all" style={{ width: `${confidencePct(value)}%` }} />
                        </span>
                        <span className="w-6 shrink-0 text-right text-[10px] text-muted-foreground">{confidencePct(value)}%</span>
                      </div>
                    ))}
                  </div>
                )
              })}
            </div>
          )}

          {inspector && (
            <div className="space-y-1.5">
              <span className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wider">Workspace memory</span>
              <div className="grid grid-cols-2 gap-1">
                {memoryRows.map(({ label, value }) => (
                  <div key={label} className="flex items-center justify-between px-2 py-1 rounded bg-muted/30 border border-border/40">
                    <span className="text-[10px] text-muted-foreground">{label}</span>
                    <span className="text-[10px] font-medium">{value}</span>
                  </div>
                ))}
              </div>
              {inspector.frame_history_size != null && (
                <p className="text-[9px] text-muted-foreground/60">Context frames: {inspector.frame_history_size}</p>
              )}
              {inspector.system_prompt && (
                <div className="pt-1 border-t border-border/30">
                  <button
                    type="button"
                    onClick={() => setShowPrompt(s => !s)}
                    className="text-[10px] text-muted-foreground hover:text-foreground transition-colors"
                    aria-expanded={showPrompt}
                  >
                    {showPrompt ? 'Hide' : 'Show'} system prompt
                  </button>
                  {showPrompt && (
                    <p className="mt-1 text-[10px] text-muted-foreground/80 leading-relaxed whitespace-pre-wrap max-h-32 overflow-y-auto">
                      {inspector.system_prompt}
                    </p>
                  )}
                </div>
              )}
            </div>
          )}

          {(knowledgeCount != null || feedback) && (
            <div className="space-y-1.5">
              <span className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wider">Inputs</span>
              <div className="grid grid-cols-2 gap-1">
                {knowledgeCount != null && (
                  <div className="flex items-center justify-between px-2 py-1 rounded bg-muted/30 border border-border/40">
                    <span className="text-[10px] text-muted-foreground">Knowledge</span>
                    <span className="text-[10px] font-medium">{knowledgeCount} snippet{knowledgeCount !== 1 ? 's' : ''}</span>
                  </div>
                )}
                {feedback && (
                  <div className="flex items-center justify-between px-2 py-1 rounded bg-muted/30 border border-border/40" title="Feedback recorded on your responses">
                    <span className="text-[10px] text-muted-foreground">Feedback</span>
                    <span className="text-[10px] font-medium">{feedback.up} up · {feedback.down} down</span>
                  </div>
                )}
              </div>
            </div>
          )}
        </>
      )}
    </div>
  )
}
