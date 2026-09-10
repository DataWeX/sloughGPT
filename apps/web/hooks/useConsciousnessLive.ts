'use client'

import { useEffect, useState, useCallback, useRef } from 'react'
import { onConsciousness, type ConsciousnessEvent } from '@/lib/consciousness-bus'

export function useConsciousnessLive() {
  const [isLive, setIsLive] = useState(true)
  const [lastUpdate, setLastUpdate] = useState<number | null>(null)
  const [latestEvent, setLatestEvent] = useState<ConsciousnessEvent | null>(null)

  const isLiveRef = useRef(isLive)
  isLiveRef.current = isLive

  useEffect(() => {
    const unsub = onConsciousness((event) => {
      if (!isLiveRef.current) return
      setLatestEvent(event)
      setLastUpdate(Date.now())
    })
    return unsub
  }, [])

  const toggleLive = useCallback(() => {
    setIsLive(prev => !prev)
  }, [])

  return { isLive, lastUpdate, toggleLive, latestEvent }
}
