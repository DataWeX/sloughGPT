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
    } catch (e) { logger.debug('Could not vision refresh', { exception: String(e) }) }
  }, [])

  useEffect(() => {
    let ignore = false
    multimodalController.getCapabilities().then(caps => { if (!ignore) setVisionCaps(caps) }).catch((e) => logger.debug('Could not vision capabilities load', { exception: String(e) }))
    multimodalController.getTrainingReport().then(r => {
      if (ignore) return
      setVisionCaptionHistory(r.caption_history || [])
      setVisionVocabSize(r.vocab_size)
    }).catch(e => logger.debug('Could not vision training report', { exception: String(e) }))
    return () => { ignore = true }
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
