'use client'

import { useState, useEffect } from 'react'

/**
 * Shared tick hook for components that need periodic re-renders
 * (e.g. for relative time labels). Uses a single global interval
 * so multiple components share one timer instead of each creating their own.
 * 
 * If subscribers request different intervals, the fastest interval wins
 * to ensure all subscribers get timely updates.
 */
const subscribers = new Map<() => void, number>()
let globalInterval: ReturnType<typeof setInterval> | null = null
let currentMs: number = 0

function updateInterval() {
  const vals = Array.from(subscribers.values())
  const newMs = vals.length > 0 ? Math.min(...vals) : currentMs
  if (newMs === currentMs && globalInterval) return

  if (globalInterval) clearInterval(globalInterval)
  currentMs = newMs
  globalInterval = setInterval(() => {
    Array.from(subscribers.keys()).forEach(fn => fn())
  }, currentMs)
}

export function useTick(intervalMs: number = 10_000) {
  const [, setTick] = useState(0)

  useEffect(() => {
    const bump = () => setTick(t => t + 1)
    subscribers.set(bump, intervalMs)
    updateInterval()
    return () => {
      subscribers.delete(bump)
      if (subscribers.size === 0 && globalInterval) {
        clearInterval(globalInterval)
        globalInterval = null
        currentMs = 0
      } else {
        updateInterval()
      }
    }
  }, [intervalMs])
}
