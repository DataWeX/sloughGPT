'use client'

import { useState, useEffect, useCallback, useRef } from 'react'
import { apiGet } from '@/lib/http-client'

interface ConversionStatus {
  model_id: string
  stage: 'idle' | 'downloading' | 'converting' | 'protecting' | 'loading' | 'ready' | 'error'
  progress: number
  message: string
  error?: string
  elapsed_s: number
}

export function useConversionStatus(modelId: string | null) {
  const [status, setStatus] = useState<ConversionStatus | null>(null)
  const intervalRef = useRef<NodeJS.Timeout | null>(null)
  const hiddenRef = useRef(false)

  const fetchStatus = useCallback(async () => {
    if (!modelId || hiddenRef.current) { return }
    try {
      const data = await apiGet<{ data?: ConversionStatus }>(`/models/conversion-status?model_id=${encodeURIComponent(modelId)}`, undefined, { silent: true })
      const s = data?.data || data as ConversionStatus
      setStatus(s)
      if (s.stage === 'ready' || s.stage === 'error' || s.stage === 'idle') {
        if (intervalRef.current) clearInterval(intervalRef.current)
      }
    } catch { /* polling errors are expected during conversion */ }
  }, [modelId])

  useEffect(() => {
    if (!modelId) { setStatus(null); return }

    fetchStatus()

    intervalRef.current = setInterval(fetchStatus, 500)

    const onVisibility = () => {
      hiddenRef.current = document.hidden
      if (!document.hidden) fetchStatus()
    }
    document.addEventListener('visibilitychange', onVisibility)

    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current)
      document.removeEventListener('visibilitychange', onVisibility)
    }
  }, [modelId, fetchStatus])

  const startTracking = useCallback((id: string) => {
    apiGet<{ data?: ConversionStatus }>(`/models/conversion-status?model_id=${encodeURIComponent(id)}`, undefined, { silent: true })
      .then(data => { if (data?.data) setStatus(data.data) })
      .catch(() => {})
  }, [])

  return { status, startTracking }
}

export function formatStage(stage: string): string {
  const labels: Record<string, string> = {
    idle: 'Preparing',
    downloading: 'Downloading',
    converting: 'Converting to .slnc',
    protecting: 'Protecting files',
    loading: 'Loading into memory',
    ready: 'Ready',
    error: 'Error',
  }
  return labels[stage] || stage
}
