'use client'

import { useState, useEffect } from 'react'

/**
 * Shared tick hook for components that need periodic re-renders
 * (e.g. for relative time labels). Uses a single global interval
 * so multiple components share one timer instead of each creating their own.
 */
const subscribers = new Set<() => void>()
let globalInterval: ReturnType<typeof setInterval> | null = null

function ensureInterval(ms: number) {
  if (globalInterval) return
  globalInterval = setInterval(() => {
    for (const fn of subscribers) fn()
  }, ms)
}

export function useTick(intervalMs: number = 10_000) {
  const [, setTick] = useState(0)

  useEffect(() => {
    const bump = () => setTick(t => t + 1)
    subscribers.add(bump)
    ensureInterval(intervalMs)
    return () => {
      subscribers.delete(bump)
      if (subscribers.size === 0 && globalInterval) {
        clearInterval(globalInterval)
        globalInterval = null
      }
    }
  }, [intervalMs])
}
