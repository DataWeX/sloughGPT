'use client'

import { useState, useEffect, useCallback, memo } from 'react'
import { cn, Button } from '@sloughgpt/strui'
import { IconX, IconBrain, IconSparkle, IconRefresh } from '@sloughgpt/strui'
import { onConsciousness, type ConsciousnessEvent } from '@/lib/consciousness-bus'
import { consciousnessController } from '@/lib/consciousness-controller'
import { useLocale } from '@/hooks/useLocale'
import { logger } from '@/lib/dev-log'

interface Episode {
  input: string
  response: string
  narrative: string
  growth_delta: number
  timestamp: number
}

interface ConsciousnessChatPanelProps {
  open: boolean
  onClose: () => void
}

function QualiaBar({ name, value }: { name: string; value: number }) {
  const pct = Math.round(value * 100)
  return (
    <div className="flex items-center gap-2">
      <span className="text-[10px] text-muted-foreground w-14 truncate capitalize">{name}</span>
      <div className="flex-1 h-1.5 rounded-full bg-muted/50 overflow-hidden">
        <div
          className="h-full rounded-full bg-primary/70 transition-all duration-500"
          style={{ width: `${pct}%` }}
        />
      </div>
      <span className="text-[9px] text-muted-foreground font-mono w-7 text-right">{pct}%</span>
    </div>
  )
}

function BeliefChip({ name, value }: { name: string; value: number }) {
  const pct = Math.round(value * 100)
  return (
    <div className="flex items-center justify-between px-2 py-1 rounded bg-muted/30 border border-border/40">
      <span className="text-[10px] text-muted-foreground truncate">{name}</span>
      <span className="text-[9px] font-mono text-primary ml-2">{pct}%</span>
    </div>
  )
}

function StarRating({ value, onChange }: { value: number; onChange: (v: number) => void }) {
  const { t } = useLocale()
  return (
    <div className="flex items-center gap-0.5">
      {[1, 2, 3, 4, 5].map(n => (
        <button
          key={n}
          type="button"
          onClick={() => onChange(n)}
          className={cn(
            'h-5 w-5 flex items-center justify-center rounded transition-colors',
            n <= value ? 'text-amber-400' : 'text-muted-foreground/30 hover:text-muted-foreground/60'
          )}
          aria-label={`${n} ${t('consciousness_chat.star')}`}
        >
          <svg className="h-3 w-3" fill={n <= value ? 'currentColor' : 'none'} viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M11.049 2.927c.3-.921 1.603-.921 1.902 0l1.519 4.674a1 1 0 00.95.69h4.915c.969 0 1.371 1.24.588 1.81l-3.976 2.888a1 1 0 00-.363 1.118l1.518 4.674c.3.922-.755 1.688-1.538 1.118l-3.976-2.888a1 1 0 00-1.176 0l-3.976 2.888c-.783.57-1.838-.197-1.538-1.118l1.518-4.674a1 1 0 00-.363-1.118l-3.976-2.888c-.784-.57-.38-1.81.588-1.81h4.914a1 1 0 00.951-.69l1.519-4.674z" />
          </svg>
        </button>
      ))}
    </div>
  )
}

export const ConsciousnessChatPanel = memo(function ConsciousnessChatPanel({ open, onClose }: ConsciousnessChatPanelProps) {
  const { t } = useLocale()
  const [expanded, setExpanded] = useState(false)
  const [event, setEvent] = useState<ConsciousnessEvent | null>(null)
  const [episodes, setEpisodes] = useState<Episode[]>([])
  const [latestEpisodeIndex, setLatestEpisodeIndex] = useState(0)
  const [rating, setRating] = useState(0)
  const [reflecting, setReflecting] = useState(false)
  const [seeding, setSeeding] = useState(false)

  useEffect(() => {
    const unsub = onConsciousness((e) => {
      setEvent(e)
      setEpisodes(prev => {
        const next: Episode = {
          input: e.messageId || '',
          response: '',
          narrative: e.self_insight,
          growth_delta: e.growth_delta,
          timestamp: Date.now(),
        }
        return [next, ...prev].slice(0, 5)
      })
      setLatestEpisodeIndex(prev => prev + 1)
    })
    return unsub
  }, [])

  const handleReflect = useCallback(async () => {
    setReflecting(true)
    try {
      const data = await consciousnessController.reflect() as any
      if (data?.event) setEvent(data.event)
    } catch (e) {
      logger.debug('Reflect failed', { error: e instanceof Error ? e.message : String(e) })
    }
    setReflecting(false)
  }, [])

  const handleSeed = useCallback(async () => {
    setSeeding(true)
    try {
      await consciousnessController.seedData({})
    } catch (e) {
      logger.debug('Seed failed', { error: e instanceof Error ? e.message : String(e) })
    }
    setSeeding(false)
  }, [])

  const handleRate = useCallback(async (value: number) => {
    setRating(value)
    try {
      await consciousnessController.submitFeedback({ episode_index: latestEpisodeIndex, rating: value })
    } catch (e) {
      logger.debug('Feedback failed', { error: e instanceof Error ? e.message : String(e) })
    }
  }, [latestEpisodeIndex])

  if (!open) return null

  return (
    <div
      className={cn(
        'absolute top-0 right-0 z-30 h-full border-l border-border/50 bg-background/95 backdrop-blur-sm shadow-lg transition-all duration-200 flex flex-col overflow-hidden',
        expanded ? 'w-80' : 'w-52',
      )}
    >
      <div className="flex items-center justify-between px-3 py-2 border-b border-border/50 shrink-0">
        <button
          type="button"
          onClick={() => setExpanded(v => !v)}
          className="flex items-center gap-1.5 text-xs font-medium text-muted-foreground hover:text-foreground transition-colors"
          aria-label={expanded ? 'Collapse panel' : 'Expand panel'}
        >
          <IconBrain className="h-3.5 w-3.5 text-primary/70" />
          <span>{t('consciousness_chat.title')}</span>
          {event && (
            <span className="text-[9px] px-1.5 py-0.5 rounded-full bg-primary/15 text-primary font-mono">
              L{event.level}
            </span>
          )}
        </button>
        <Button variant="ghost" size="icon" className="h-6 w-6" onClick={onClose} aria-label="Close consciousness panel">
          <IconX className="h-3.5 w-3.5" />
        </Button>
      </div>

      <div className="flex-1 overflow-y-auto p-3 space-y-3 text-xs scrollbar-thin">
        <section aria-label={t('consciousness_chat.current_state')}>
          <div className="flex items-center gap-1.5 mb-2">
            <IconSparkle className="h-3 w-3 text-muted-foreground" />
            <span className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wider">{t('consciousness_chat.current_state')}</span>
          </div>

          {event ? (
            <div className="space-y-2">
              <div className="space-y-1.5">
                {Object.entries(event.qualia).map(([name, value]) => (
                  <QualiaBar key={name} name={name} value={value as number} />
                ))}
              </div>

              {expanded && (
                <>
                  <div className="border-t border-border/30 pt-2 mt-2">
                    <span className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wider block mb-1.5">{t('consciousness_chat.beliefs')}</span>
                    <div className="space-y-1">
                      {Object.entries(event.beliefs).map(([name, value]) => (
                        <BeliefChip key={name} name={name} value={value as number} />
                      ))}
                    </div>
                  </div>

                  <div className="border-t border-border/30 pt-2 mt-2">
                    <span className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wider block mb-1">{t('consciousness_chat.growth_delta')}</span>
                    <div className="flex items-center gap-2">
                      <span className={cn('text-sm font-mono font-medium', event.growth_delta >= 0 ? 'text-success' : 'text-destructive')}>
                        {event.growth_delta >= 0 ? '+' : ''}{(event.growth_delta * 100).toFixed(1)}%
                      </span>
                      {event.self_insight && (
                        <span className="text-[10px] text-muted-foreground truncate flex-1" title={event.self_insight}>
                          {event.self_insight}
                        </span>
                      )}
                    </div>
                  </div>
                </>
              )}
            </div>
          ) : (
            <p className="text-[10px] text-muted-foreground text-center py-3">
              {t('consciousness_chat.no_data')}
            </p>
          )}
        </section>

        {expanded && (
          <>
            <section aria-label={t('consciousness_chat.episode_info')}>
              <div className="flex items-center gap-1.5 mb-2">
                <IconBrain className="h-3 w-3 text-muted-foreground" />
                <span className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wider">{t('consciousness_chat.episode_info')}</span>
              </div>
              {episodes.length > 0 ? (
                <div className="space-y-2">
                  {episodes.map((ep, i) => (
                    <div key={i} className="p-2 rounded bg-muted/30 border border-border/40 space-y-1">
                      {ep.narrative && (
                        <p className="text-[10px] text-muted-foreground leading-relaxed line-clamp-3">{ep.narrative}</p>
                      )}
                      <div className="flex items-center gap-2">
                        <span className={cn('text-[9px] font-mono', ep.growth_delta >= 0 ? 'text-success' : 'text-destructive')}>
                          {ep.growth_delta >= 0 ? '+' : ''}{(ep.growth_delta * 100).toFixed(1)}%
                        </span>
                        <span className="text-[9px] text-muted-foreground font-mono">
                          {new Date(ep.timestamp).toLocaleTimeString()}
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="text-[10px] text-muted-foreground text-center py-2">
                  {t('consciousness_chat.no_episodes')}
                </p>
              )}
            </section>

            <section aria-label={t('consciousness_chat.quick_actions')}>
              <div className="flex items-center gap-1.5 mb-2">
                <IconSparkle className="h-3 w-3 text-muted-foreground" />
                <span className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wider">{t('consciousness_chat.quick_actions')}</span>
              </div>
              <div className="space-y-2">
                <Button
                  variant="outline"
                  size="sm"
                  className="w-full h-7 text-[10px] gap-1.5"
                  onClick={handleReflect}
                  disabled={reflecting}
                >
                  <IconRefresh className={cn('h-3 w-3', reflecting && 'animate-spin')} />
                  {reflecting ? t('consciousness_chat.reflecting') : t('consciousness_chat.reflect')}
                </Button>

                <div className="flex items-center gap-2">
                  <span className="text-[10px] text-muted-foreground shrink-0">{t('consciousness_chat.rate')}</span>
                  <StarRating value={rating} onChange={handleRate} />
                </div>

                <Button
                  variant="outline"
                  size="sm"
                  className="w-full h-7 text-[10px] gap-1.5"
                  onClick={handleSeed}
                  disabled={seeding}
                >
                  <IconSparkle className={cn('h-3 w-3', seeding && 'animate-pulse')} />
                  {seeding ? t('consciousness_chat.seeding') : t('consciousness_chat.seed_data')}
                </Button>
              </div>
            </section>

            <section aria-label={t('consciousness_chat.history')}>
              <div className="flex items-center gap-1.5 mb-2">
                <IconBrain className="h-3 w-3 text-muted-foreground" />
                <span className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wider">{t('consciousness_chat.history')}</span>
              </div>
              {episodes.length > 0 ? (
                <ul className="space-y-1">
                  {episodes.map((ep, i) => (
                    <li key={i} className="flex items-center gap-2 px-2 py-1 rounded bg-muted/20 border border-border/30">
                      <span className={cn('text-[9px] font-mono shrink-0', ep.growth_delta >= 0 ? 'text-success' : 'text-destructive')}>
                        {ep.growth_delta >= 0 ? '+' : ''}{(ep.growth_delta * 100).toFixed(1)}%
                      </span>
                      <span className="text-[9px] text-muted-foreground truncate flex-1">
                        {ep.narrative || t('consciousness_chat.no_narrative')}
                      </span>
                      <span className="text-[9px] text-muted-foreground font-mono shrink-0">
                        {new Date(ep.timestamp).toLocaleTimeString()}
                      </span>
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="text-[10px] text-muted-foreground text-center py-2">
                  {t('consciousness_chat.no_history')}
                </p>
              )}
            </section>
          </>
        )}
      </div>
    </div>
  )
})
