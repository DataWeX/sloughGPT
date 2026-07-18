'use client'
import { useEffect, useRef } from 'react'
import { useApiMonitor, type HealthSummaryData } from '@/lib/api-monitor-store'
import { apiGet } from '@/lib/http-client'
import { logger } from '@/lib/dev-log'

const _log = logger.child('backend-watcher')

const BASE_POLL_MS = 8000
const REQUEST_TIMEOUT = 20000
const MAX_FAILURES_BEFORE_RELOAD = 4
const RELOAD_DELAY_MS = 2000
const MIN_BACKOFF_MS = 2000
const MAX_BACKOFF_MS = 30000

interface HealthSummary {
  score: number
  status: string
  summary: string
  model_loaded: boolean
  model_loading: boolean
}

export function useBackendWatcher() {
  const setStatus = useApiMonitor((s) => s.setStatus)
  const setHealthSummary = useApiMonitor((s) => s.setHealthSummary)
  const clearFailures = useApiMonitor((s) => s.clearFailures)
  const wasOffline = useRef(false)
  const lastScoreStatus = useRef<string | null>(null)
  const failureCount = useRef(0)
  const consecutiveRateLimits = useRef(0)
  const startTime = useRef(Date.now())

  useEffect(() => {
    let cancelled = false
    let timer: ReturnType<typeof setTimeout>

    const getPollInterval = () => {
      // Exponential backoff when failing, capped at MAX_BACKOFF_MS
      const backoff = failureCount.current > 0
        ? Math.min(MAX_BACKOFF_MS, MIN_BACKOFF_MS * Math.pow(2, failureCount.current - 1))
        : 0
      return BASE_POLL_MS + backoff
    }

    const check = async () => {
      if (cancelled) return

      try {
        const data = await apiGet<HealthSummary>('/health/summary', undefined, {
          timeout: REQUEST_TIMEOUT,
          silent: true,
        })
        if (cancelled) return

        // Server responded — reset all failure tracking
        failureCount.current = 0
        consecutiveRateLimits.current = 0
        clearFailures()
        setHealthSummary(data as HealthSummaryData)

        if (wasOffline.current) {
          wasOffline.current = false
          setStatus('connected')
          const uptime = Math.round((Date.now() - startTime.current) / 1000)
          _log.info(`Server reconnected after ${uptime}s`)
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
      } catch (err: unknown) {
        if (cancelled) return

        const errStatus = err instanceof Error ? (err as Error & { status?: number }).status : undefined
        const errMsg = err instanceof Error ? err.message : undefined
        // Check if this is a rate limit (429) vs actual server down
        const isRateLimit = errStatus === 429
        const isConnectionDown = errStatus === 0 || errMsg === 'Connection unavailable'
        const elapsed = Math.round((Date.now() - startTime.current) / 1000)

        if (isRateLimit) {
          // Rate limited: back off but don't count as server failure
          consecutiveRateLimits.current++
          failureCount.current = Math.max(0, failureCount.current - 1) // decay toward 0
          setStatus('connected') // server IS responding, just throttled
        } else if (isConnectionDown) {
          failureCount.current += 1
          wasOffline.current = true
          const reason = err?.message || 'Connection unavailable'
          _log.warning(`Connection failed (${failureCount.current}/${MAX_FAILURES_BEFORE_RELOAD}): ${reason} — server startup: ${elapsed}s`)
          setStatus(failureCount.current >= MAX_FAILURES_BEFORE_RELOAD ? 'reloading' : 'connecting')
        } else {
          // 4xx/5xx — server is up but returning errors
          failureCount.current += 1
          wasOffline.current = true
          _log.warning(`HTTP error (${failureCount.current}/${MAX_FAILURES_BEFORE_RELOAD}): ${err?.status} ${err?.message}`)
          setStatus(failureCount.current >= MAX_FAILURES_BEFORE_RELOAD ? 'reloading' : 'connecting')
        }

        // Only reload after sustained failures — never on rate limit
        if (failureCount.current >= MAX_FAILURES_BEFORE_RELOAD && !isRateLimit) {
          _log.warning(`Too many failures, reloading page...`)
          setTimeout(() => {
            if (!cancelled) window.location.reload()
          }, RELOAD_DELAY_MS)
          return // stop polling — reload will restart it
        }
      }

      if (!cancelled) {
        timer = setTimeout(check, getPollInterval())
      }
    }

    check()
    return () => {
      cancelled = true
      clearTimeout(timer)
    }
  }, [setStatus])
}
