'use client'

import { useEffect, useRef, useCallback } from 'react'
import { onConsciousness, type ConsciousnessEvent } from '@/lib/consciousness-bus'

interface UseConsciousnessStreamOptions {
  messageId?: string
  onConsciousnessUpdate?: (event: ConsciousnessEvent) => void
}

export function useConsciousnessStream({ messageId, onConsciousnessUpdate }: UseConsciousnessStreamOptions = {}) {
  const latestRef = useRef<ConsciousnessEvent | null>(null)
  const callbackRef = useRef(onConsciousnessUpdate)
  callbackRef.current = onConsciousnessUpdate

  useEffect(() => {
    if (!messageId) return
    const unsub = onConsciousness((event) => {
      latestRef.current = event
      callbackRef.current?.(event)
    })
    return unsub
  }, [messageId])

  const getLatest = useCallback(() => latestRef.current, [])

  return { latest: latestRef.current, getLatest }
}
