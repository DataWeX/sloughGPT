'use client'

/**
 * ConsciousnessProvider — context layer that makes consciousness state
 * available to the entire chat component tree.
 *
 * Subscribes to the consciousness bus (SSE events during streaming)
 * and exposes qualia, beliefs, growth, episodes, and actions to all children.
 *
 * This replaces per-component bus subscriptions with a single provider
 * at the chat page level. Any child can access consciousness via useConsciousness().
 */

import {
  createContext,
  useContext,
  useState,
  useEffect,
  useCallback,
  useRef,
  type ReactNode,
} from 'react'
import { onConsciousness, type ConsciousnessEvent } from '@/lib/consciousness-bus'
import { consciousnessController } from '@/lib/consciousness-controller'
import { logger } from '@/lib/dev-log'

const _log = logger.child('consciousness-provider')

// ── Types ────────────────────────────────────────────────────────────────────

export interface ConsciousnessEpisode {
  input: string
  response: string
  narrative: string
  growth_delta: number
  timestamp: number
  messageId?: string
}

export interface ConsciousnessState {
  /** Latest consciousness event from the SSE stream */
  event: ConsciousnessEvent | null
  /** Current consciousness level (0=off, 1=basic, 2=full, 3=deep) */
  level: number
  /** Current qualia state (valence, arousal, dominance, novelty, coherence, beauty) */
  qualia: Record<string, number>
  /** Current beliefs (competence, helpfulfulness, creativity, accuracy, empathy) */
  beliefs: Record<string, number>
  /** Latest growth delta (-0.15 to +0.15) */
  growthDelta: number
  /** Latest self-insight text */
  selfInsight: string
  /** Last N episodes (most recent first) */
  episodes: ConsciousnessEpisode[]
  /** Whether consciousness is enabled */
  enabled: boolean
  /** Mood derived from qualia */
  mood: string
}

export interface ConsciousnessActions {
  /** Trigger a reflection cycle */
  reflect: () => Promise<void>
  /** Seed 30 synthetic training episodes */
  seed: () => Promise<void>
  /** Submit rating for the latest episode */
  rate: (value: number) => Promise<void>
  /** Whether an action is in progress */
  reflecting: boolean
  seeding: boolean
}

export interface ConsciousnessContextValue extends ConsciousnessState, ConsciousnessActions {}

// ── Context ──────────────────────────────────────────────────────────────────

const ConsciousnessContext = createContext<ConsciousnessContextValue | null>(null)

export function useConsciousness(): ConsciousnessContextValue {
  const ctx = useContext(ConsciousnessContext)
  if (!ctx) throw new Error('useConsciousness must be used within ConsciousnessProvider')
  return ctx
}

/** Optional consciousness access — returns null if no provider */
export function useConsciousnessOptional(): ConsciousnessContextValue | null {
  return useContext(ConsciousnessContext)
}

// ── Mood derivation from qualia ──────────────────────────────────────────────

function deriveMood(qualia: Record<string, number>): string {
  const valence = qualia.valence ?? 0
  const arousal = qualia.arousal ?? 0.5
  const novelty = qualia.novelty ?? 0.5
  const coherence = qualia.coherence ?? 0.5

  if (novelty > 0.7) return 'curious'
  if (valence > 0.3 && arousal > 0.6) return 'engaged'
  if (valence > 0.3 && arousal < 0.4) return 'calm'
  if (valence < -0.3) return 'uneasy'
  if (coherence < 0.3) return 'confused'
  if (arousal > 0.7) return 'excited'
  return 'neutral'
}

// ── Provider ─────────────────────────────────────────────────────────────────

const MAX_EPISODES = 10

interface ConsciousnessProviderProps {
  children: ReactNode
  /** Whether consciousness is enabled (default: true) */
  enabled?: boolean
}

export function ConsciousnessProvider({ children, enabled = true }: ConsciousnessProviderProps) {
  const [event, setEvent] = useState<ConsciousnessEvent | null>(null)
  const [episodes, setEpisodes] = useState<ConsciousnessEpisode[]>([])
  const [reflecting, setReflecting] = useState(false)
  const [seeding, setSeeding] = useState(false)
  const latestEpisodeIndexRef = useRef(0)

  // Subscribe to consciousness bus — single subscription for the entire tree
  useEffect(() => {
    if (!enabled) return
    const unsub = onConsciousness((e) => {
      setEvent(e)
      setEpisodes((prev) => {
        const next: ConsciousnessEpisode = {
          input: e.messageId || '',
          response: '',
          narrative: e.self_insight,
          growth_delta: e.growth_delta,
          timestamp: Date.now(),
          messageId: e.messageId,
        }
        return [next, ...prev].slice(0, MAX_EPISODES)
      })
      latestEpisodeIndexRef.current += 1
    })
    return unsub
  }, [enabled])

  // Derive state from latest event
  const level = event?.level ?? 0
  const qualia = event?.qualia ?? {}
  const beliefs = event?.beliefs ?? {}
  const growthDelta = event?.growth_delta ?? 0
  const selfInsight = event?.self_insight ?? ''
  const mood = deriveMood(qualia)

  // Actions
  const reflect = useCallback(async () => {
    setReflecting(true)
    try {
      const data = (await consciousnessController.reflect()) as any
      if (data?.event) setEvent(data.event)
    } catch (e) {
      _log.debug('Reflect failed', { error: e instanceof Error ? e.message : String(e) })
    }
    setReflecting(false)
  }, [])

  const seed = useCallback(async () => {
    setSeeding(true)
    try {
      await consciousnessController.seedData({})
    } catch (e) {
      _log.debug('Seed failed', { error: e instanceof Error ? e.message : String(e) })
    }
    setSeeding(false)
  }, [])

  const rate = useCallback(async (value: number) => {
    try {
      await consciousnessController.submitFeedback({
        episode_index: latestEpisodeIndexRef.current,
        rating: value,
      })
    } catch (e) {
      _log.debug('Feedback failed', { error: e instanceof Error ? e.message : String(e) })
    }
  }, [])

  const value: ConsciousnessContextValue = {
    event,
    level,
    qualia,
    beliefs,
    growthDelta,
    selfInsight,
    episodes,
    enabled,
    mood,
    reflect,
    seed,
    rate,
    reflecting,
    seeding,
  }

  return <ConsciousnessContext.Provider value={value}>{children}</ConsciousnessContext.Provider>
}
