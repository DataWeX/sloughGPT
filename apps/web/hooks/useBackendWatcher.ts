'use client'
import { useEffect, useRef } from 'react'
import { useApiMonitor } from '@/lib/api-monitor-store'
import { PUBLIC_API_URL } from '@/lib/config'

const POLL_INTERVAL = 3000
const REQUEST_TIMEOUT = 3000

interface HealthSummary {
  score: number
  status: string
  summary: string
  model_loaded: boolean
}

export function useBackendWatcher() {
  const setStatus = useApiMonitor((s) => s.setStatus)
  const wasOffline = useRef(false)
  const lastScoreStatus = useRef<string | null>(null)

  useEffect(() => {
    let cancelled = false
    let timer: ReturnType<typeof setTimeout>

    const check = async () => {
      if (cancelled) return
      const controller = new AbortController()
      const timeout = setTimeout(() => controller.abort(), REQUEST_TIMEOUT)

      try {
        const res = await fetch(`${PUBLIC_API_URL}/health/summary`, {
          signal: controller.signal,
          cache: 'no-store',
        })
        clearTimeout(timeout)
        if (!res.ok) throw new Error(String(res.status))

        const data: HealthSummary = await res.json()

        if (wasOffline.current) {
          wasOffline.current = false
          setStatus('connected')
        } else {
          setStatus('connected')
        }

        // Health threshold alerts — only on status transitions
        if (data.status && data.status !== lastScoreStatus.current) {
          if (lastScoreStatus.current !== null) {
            const { useToastStore } = await import('@/lib/toast-store')
            const summary = data.summary || `Score: ${data.score}`
            if (data.status === 'degraded') {
              useToastStore.getState().addToast(summary, 'info')
            } else if (data.status === 'unhealthy') {
              useToastStore.getState().addToast(summary, 'error')
            }
          }
          lastScoreStatus.current = data.status
        }
      } catch {
        clearTimeout(timeout)
        wasOffline.current = true
        setStatus('reloading')
      }

      if (!cancelled) timer = setTimeout(check, POLL_INTERVAL)
    }

    check()
    return () => {
      cancelled = true
      clearTimeout(timer)
    }
  }, [setStatus])
}
