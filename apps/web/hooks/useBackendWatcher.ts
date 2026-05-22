'use client'

import { useEffect, useRef } from 'react'
import { modelController } from '@/lib/model-controller'
import { useToastStore } from '@/lib/toast-store'

const POLL_INTERVAL = 2000

/**
 * Polls the backend health endpoint.
 * When the server is detected as back online after a disconnection,
 * the page is automatically reloaded so the UI picks up any API changes.
 * Shows a notification before reloading.
 */
export function useBackendWatcher() {
  const wasOffline = useRef(false)

  useEffect(() => {
    let cancelled = false
    let timeout: ReturnType<typeof setTimeout>

    const check = async () => {
      if (cancelled) return
      try {
        const health = await modelController.getHealth()
        // Health returned successfully -> server is up
        if (wasOffline.current) {
          // Server just came back after being offline – reload
          useToastStore.getState().addToast(
            'Backend reconnected — reloading page…',
            'info',
          )
          setTimeout(() => {
            if (!cancelled) window.location.reload()
          }, 1500)
          return
        }
        wasOffline.current = false
      } catch {
        // Server is offline / unreachable
        wasOffline.current = true
      }
      if (!cancelled) {
        timeout = setTimeout(check, POLL_INTERVAL)
      }
    }

    // First check immediately, then poll
    check()
    return () => {
      cancelled = true
      clearTimeout(timeout)
    }
  }, [])
}
