'use client'
import { useEffect, useRef } from 'react'
import { useApiMonitor } from '@/lib/api-monitor-store'
import { PUBLIC_API_URL } from '@/lib/config'

const POLL_INTERVAL = 3000
const REQUEST_TIMEOUT = 3000

export function useBackendWatcher() {
  const setStatus = useApiMonitor((s) => s.setStatus)
  const wasOffline = useRef(false)

  useEffect(() => {
    let cancelled = false
    let timer: ReturnType<typeof setTimeout>

    const check = async () => {
      if (cancelled) return
      const controller = new AbortController()
      const timeout = setTimeout(() => controller.abort(), REQUEST_TIMEOUT)

      try {
        const res = await fetch(`${PUBLIC_API_URL}/health`, {
          signal: controller.signal,
          cache: 'no-store',
        })
        clearTimeout(timeout)
        if (!res.ok) throw new Error(String(res.status))

        if (wasOffline.current) {
          wasOffline.current = false
          setStatus('connected')
        } else {
          setStatus('connected')
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
