'use client'

import { useState, useEffect, useCallback, useRef } from 'react'

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

  const fetchStatus = useCallback(async () => {
    if (!modelId) { setStatus(null); return }
    try {
      const base = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'
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
    } catch {
      // Ignore poll errors
    }
  }, [modelId])

  useEffect(() => {
    if (!modelId) { setStatus(null); return }

    // Initial fetch
    fetchStatus()

    // Poll every 500ms while converting
    intervalRef.current = setInterval(fetchStatus, 500)

    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current)
    }
  }, [modelId, fetchStatus])

  const startTracking = useCallback((id: string) => {
    // Force an immediate fetch for a new model
    const base = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'
    fetch(`${base}/models/conversion-status?model_id=${encodeURIComponent(id)}`)
      .then(r => r.ok ? r.json() : null)
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
