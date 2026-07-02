'use client'
import { useEffect, useRef } from 'react'
import { useApiMonitor } from '@/lib/api-monitor-store'
import { apiGet } from '@/lib/http-client'

const POLL_INTERVAL = 8000
const REQUEST_TIMEOUT = 10000
const MAX_FAILURES = 3
const RELOAD_DELAY_MS = 1500

interface HealthSummary {
  score: number
  status: string
  summary: string
  model_loaded: boolean
}

export function useBackendWatcher() {
  const setStatus = useApiMonitor((s) => s.setStatus)
  const setHealthSummary = useApiMonitor((s) => s.setHealthSummary)
  const wasOffline = useRef(false)
  const lastScoreStatus = useRef<string | null>(null)
  const failureCount = useRef(0)

  useEffect(() => {
    let cancelled = false
    let timer: ReturnType<typeof setTimeout>

    const check = async () => {
      if (cancelled) return

      try {
        const data = await apiGet<HealthSummary>('/health/summary', undefined, {
          timeout: REQUEST_TIMEOUT,
          silent: true,
        })
        if (cancelled) return
        failureCount.current = 0
        setHealthSummary(data as any)

        if (wasOffline.current) {
          wasOffline.current = false
          setStatus('connected')
          const { useToastStore } = await import('@/lib/toast-store')
          useToastStore.getState().addToast('Server reconnected', 'success')
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
        if (cancelled) return
        failureCount.current += 1
        wasOffline.current = true
        if (failureCount.current >= MAX_FAILURES) {
          setStatus('reloading')
          // Auto-reload after a brief delay to recover from server restart
          setTimeout(() => {
            if (!cancelled) window.location.reload()
          }, RELOAD_DELAY_MS)
          return // stop polling — reload will restart it
        }
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
