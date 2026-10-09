'use client'

import { useCallback, useEffect, useRef, useState } from 'react'
import { memoryController, type MemoryItem } from '@/lib/memory-controller'
import { subscribeMemoryEvents } from '@/lib/memory-events'
import { logger } from '@/lib/dev-log'

const HIGHLIGHT_MS = 4000
const COPY_BADGE_MS = 1500
const CONSOLIDATE_MSG_MS = 3500

export interface UseChatMemoryReturn {
  highlightedId: string | null
  copiedId: string | null
  consolidateMsg: string | null
  consolidating: boolean
  handleCopy: (content: string, id: string) => void
  handleConsolidate: () => Promise<void>
  highlightItem: (content: string, items: MemoryItem[]) => void
  pendingSseFact: string | null
  consumePendingSseFact: () => string | null
}

export function useChatMemory(fetchData: () => Promise<void>): UseChatMemoryReturn {
  const highlightTimerRef = useRef<number | null>(null)
  const copyTimerRef = useRef<number | null>(null)
  const consolidateTimerRef = useRef<number | null>(null)
  const [highlightedId, setHighlightedId] = useState<string | null>(null)
  const [copiedId, setCopiedId] = useState<string | null>(null)
  const [consolidateMsg, setConsolidateMsg] = useState<string | null>(null)
  const [consolidating, setConsolidating] = useState(false)
  const [pendingSseFact, setPendingSseFact] = useState<string | null>(null)

  useEffect(() => {
    return () => {
      if (highlightTimerRef.current) window.clearTimeout(highlightTimerRef.current)
      if (copyTimerRef.current) window.clearTimeout(copyTimerRef.current)
      if (consolidateTimerRef.current) window.clearTimeout(consolidateTimerRef.current)
    }
  }, [])

  const highlightItem = useCallback((content: string, items: MemoryItem[]) => {
    const match = items.find(i => i.content === content)
    if (match) {
      setHighlightedId(match.id)
      if (highlightTimerRef.current) window.clearTimeout(highlightTimerRef.current)
      highlightTimerRef.current = window.setTimeout(() => setHighlightedId(null), HIGHLIGHT_MS)
    }
  }, [])

  const consumePendingSseFact = useCallback(() => {
    const fact = pendingSseFact
    setPendingSseFact(null)
    return fact
  }, [pendingSseFact])

  const handleCopy = useCallback(async (content: string, id: string) => {
    try {
      await navigator.clipboard.writeText(content)
      setCopiedId(id)
      if (copyTimerRef.current) window.clearTimeout(copyTimerRef.current)
      copyTimerRef.current = window.setTimeout(() => setCopiedId(null), COPY_BADGE_MS)
    } catch (err) {
      logger.debug('Could not memory copy', { exception: String(err) })
    }
  }, [])

  const handleConsolidate = useCallback(async () => {
    setConsolidating(true)
    setConsolidateMsg(null)
    try {
      const result = await memoryController.consolidate()
      setConsolidateMsg(
        result.removed > 0
          ? `Consolidated ${result.removed} duplicate fact(s), kept ${result.kept}`
          : 'No near-duplicate facts found',
      )
    } catch {
      setConsolidateMsg('Could not consolidate memory')
    } finally {
      setConsolidating(false)
      if (consolidateTimerRef.current) window.clearTimeout(consolidateTimerRef.current)
      consolidateTimerRef.current = window.setTimeout(() => setConsolidateMsg(null), CONSOLIDATE_MSG_MS)
    }
    fetchData()
  }, [fetchData])

  useEffect(() => {
    const unsubscribe = subscribeMemoryEvents((info) => {
      if (info.stored) {
        const fact = info.facts?.[0] ?? info.fact ?? null
        if (fact) setPendingSseFact(fact)
        fetchData()
      }
    })
    return unsubscribe
  }, [fetchData])

  return {
    highlightedId,
    copiedId,
    consolidateMsg,
    consolidating,
    handleCopy,
    handleConsolidate,
    highlightItem,
    pendingSseFact,
    consumePendingSseFact,
  }
}
