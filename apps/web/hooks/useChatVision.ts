'use client'
import { logger } from '@/lib/dev-log'

import { useState, useEffect, useCallback } from 'react'
import { multimodalController, type MultimodalCapabilities } from '@/lib/multimodal-controller'

export function useChatVision() {
  const [visionCaps, setVisionCaps] = useState<MultimodalCapabilities | null>(null)
  const [visionCaptionHistory, setVisionCaptionHistory] = useState<string[]>([])
  const [visionVocabSize, setVisionVocabSize] = useState<number | undefined>(undefined)

  const refreshVision = useCallback(async () => {
    try {
      const caps = await multimodalController.getCapabilities()
      setVisionCaps(caps)
      const report = await multimodalController.getTrainingReport()
      setVisionCaptionHistory(report.caption_history || [])
      setVisionVocabSize(report.vocab_size)
    } catch {}
  }, [])

  useEffect(() => {
    multimodalController.getCapabilities().then(setVisionCaps).catch((e) => logger.debug('Vision capabilities load failed', e))
    multimodalController.getTrainingReport().then(r => {
      setVisionCaptionHistory(r.caption_history || [])
      setVisionVocabSize(r.vocab_size)
    }).catch(() => {})
  }, [])

  useEffect(() => {
    const handler = () => { refreshVision() }
    window.addEventListener('refresh-vision', handler)
    return () => window.removeEventListener('refresh-vision', handler)
  }, [refreshVision])

  return {
    visionCaps, setVisionCaps,
    visionCaptionHistory, setVisionCaptionHistory,
    visionVocabSize, setVisionVocabSize,
    refreshVision,
  }
}
