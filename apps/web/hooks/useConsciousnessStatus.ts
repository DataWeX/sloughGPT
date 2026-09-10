'use client'

import { useEffect, useState, useCallback } from 'react'
import { PUBLIC_API_URL } from '@/lib/config'

export interface ConsciousnessStatus {
  enabled: boolean
  level: number
  episodes: number
  beliefs: Record<string, number>
  current_qualia: {
    valence: number
    arousal: number
    novelty: number
    coherence: number
    salience: number
  }
}

const LEVEL_LABELS = ['Off', 'Basic', 'Full', 'Deep'] as const
const POLL_INTERVAL = 30_000

export function getConsciousnessLevelLabel(level: number): string {
  return LEVEL_LABELS[level] ?? 'Off'
}

export function getQualiaMood(qualia: ConsciousnessStatus['current_qualia'] | undefined): string {
  if (!qualia) return ''
  const { valence, arousal, novelty } = qualia
  if (valence > 0.3 && arousal > 0.5) return 'engaged'
  if (valence > 0.3) return 'positive'
  if (valence < -0.3) return 'uneasy'
  if (novelty > 0.6) return 'curious'
  if (arousal < 0.2) return 'calm'
  return 'neutral'
}

export function useConsciousnessStatus() {
  const [status, setStatus] = useState<ConsciousnessStatus | null>(null)
  const [loading, setLoading] = useState(true)

  const fetchStatus = useCallback(async () => {
    try {
      const res = await fetch(`${PUBLIC_API_URL}/consciousness/status`)
      if (!res.ok) return
      const json = await res.json()
      setStatus(json.data ?? json)
    } catch {
      // Consciousness endpoint may not exist yet — degrade silently
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchStatus()
    const id = setInterval(fetchStatus, POLL_INTERVAL)
    return () => clearInterval(id)
  }, [fetchStatus])

  return { status, loading, levelLabel: getConsciousnessLevelLabel(status?.level ?? 0) }
}
