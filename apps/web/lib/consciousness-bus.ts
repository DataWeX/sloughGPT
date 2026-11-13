'use client'

export interface ConsciousnessEvent {
  level: number
  qualia: Record<string, number>
  beliefs: Record<string, number>
  growth_delta: number
  self_insight: string
  messageId?: string
}

type Listener = (event: ConsciousnessEvent) => void

const listeners = new Set<Listener>()

export function emitConsciousness(event: ConsciousnessEvent) {
  for (const fn of listeners) fn(event)
}

export function onConsciousness(fn: Listener): () => void {
  listeners.add(fn)
  return () => { listeners.delete(fn) }
}
