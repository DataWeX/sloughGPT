'use client'

import { useState, useEffect, useCallback, useRef } from 'react'
import { PUBLIC_API_URL } from '@/lib/config'

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
      const base = PUBLIC_API_URL
      const res = await fetch(`${base}/models/conversion-status?model_id=${encodeURIComponent(modelId)}`)
      if (res.ok) {
        const data = await res.json()
        const s = data?.data || data
        setStatus(s)
        // Stop polling when done
        if (s.stage === 'ready' || s.stage === 'error' || s.stage === 'idle') {
          if (intervalRef.current) clearInterval(intervalRef.current)
        }
      }
    } catch { /* polling errors are expected during conversion */ }
  }, [modelId])

  useEffect(() => {
    if (!modelId) { setStatus(null); return }

    // Initial fetch
    fetchStatus()

    // Poll every 500ms while converting
    intervalRef.current = setInterval(fetchStatus, 500)

    // Pause when tab hidden
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
    // Force an immediate fetch for a new model
    const base = PUBLIC_API_URL
    fetch(`${base}/models/conversion-status?model_id=${encodeURIComponent(id)}`)
      .then(r => r.ok ? r.json() : null)
      .then(data => { if (data?.data) setStatus(data.data) })
      .catch(() => /* initial status fetch failed — will retry on poll */ {})
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
