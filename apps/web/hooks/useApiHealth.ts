'use client'

import { useCallback, useEffect, useState } from 'react'

import { modelController } from '@/lib/model-controller'
import type { HealthStatus as ApiHealth } from '@/lib/model-controller'

const POLL_MS = 28_000

let _sharedState: ApiHealthSnapshot = null
let _pollId: ReturnType<typeof setInterval> | null = null
let _subscribers: Set<(s: ApiHealthSnapshot) => void> = new Set()

function _notify() {
  _subscribers.forEach(fn => fn(_sharedState))
}

async function _fetchAndNotify() {
  const h = await modelController.getHealth()
  _sharedState = h ?? 'offline'
  _notify()
}

/** Start the poll interval with an immediate first fetch. */
function _startPolling() {
  if (_pollId) return
  _fetchAndNotify().catch(() => {})
  _pollId = setInterval(_fetchAndNotify, POLL_MS)
}

// ── Eager pre-fetch at module init ──────────────────────────────────────
if (typeof window !== 'undefined' && _sharedState === null) {
  _fetchAndNotify().catch(() => {
    _sharedState = 'offline'
    _notify()
  })
}

export type ApiHealthSnapshot = ApiHealth | 'offline' | null

/** One-line status for headers (Models page, tooltips). */
export function inferenceHealthLabel(state: ApiHealthSnapshot): string {
  if (state === null) return 'checking...'
  if (state === 'offline') return 'disconnected'
  if (state.model_loaded) return `inference ready · ${state.model_type}`
  return `connected · no weights (${state.model_type})`
}

/**
 * Polls `GET /health` on an interval and when the tab becomes visible again.
 * Use for chat, models, and anywhere inference readiness matters.
 * Uses shared singleton polling to avoid redundant requests.
 */
export function useApiHealth() {
  const [state, setState] = useState<ApiHealthSnapshot>(_sharedState)

  useEffect(() => {
    _startPolling()
    const fn = (s: ApiHealthSnapshot) => setState(s)
    _subscribers.add(fn)
    return () => { _subscribers.delete(fn) }
  }, [])

  const refresh = useCallback(async () => {
    const h = await modelController.getHealth()
    const s = h ?? 'offline'
    _sharedState = s
    _notify()
  }, [])

  return { state, refresh }
}
