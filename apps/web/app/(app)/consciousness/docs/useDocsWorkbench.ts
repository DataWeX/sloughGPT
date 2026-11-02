'use client'

/**
 * Stateful core of the Consciousness API docs page.
 *
 * Owns the active category, the per-endpoint try-it response/loading maps,
 * and the request runner (authFetch, timing, JSON formatting, scroll-into-
 * view). The sidebar and endpoint list own render.
 */
import { useState, useCallback, useRef } from 'react'
import { useToastStore } from '@/lib/toast-store'
import { authFetch } from '@/lib/http-client'
import { extractErrorMessage } from '@/lib/error-utils'
import { useLocale } from '@/hooks/useLocale'
import type { Endpoint, Category } from './docs-endpoints'

export function useDocsWorkbench() {
  const addToast = useToastStore((state) => state.addToast)
  const { t } = useLocale()
  const [activeCategory, setActiveCategory] = useState<Category>('core')
  const [responses, setResponses] = useState<
    Record<string, { status: number; body: string; timeMs: number } | null>
  >({})
  const [loadingMap, setLoadingMap] = useState<Record<string, boolean>>({})
  const responseRefs = useRef<Record<string, HTMLDivElement | null>>({})

  const endpointKey = (ep: Endpoint) => `${ep.method}:${ep.path}`

  const handleTryIt = useCallback(async (ep: Endpoint) => {
    const key = endpointKey(ep)
    setLoadingMap((prev) => ({ ...prev, [key]: true }))
    setResponses((prev) => ({ ...prev, [key]: null }))

    const start = Date.now()
    try {
      const init: RequestInit = {
        method: ep.method,
        headers: { 'Content-Type': 'application/json' },
      }
      if (ep.requestBody && ep.method !== 'GET') {
        init.body = ep.requestBody
      }
      const res = await authFetch(ep.path, init)
      const elapsed = Date.now() - start
      const text = await res.text()
      let formatted = text
      try {
        formatted = JSON.stringify(JSON.parse(text), null, 2)
      } catch {
        /* raw text */
      }
      setResponses((prev) => ({
        ...prev,
        [key]: { status: res.status, body: formatted, timeMs: elapsed },
      }))
    } catch (e) {
      setResponses((prev) => ({
        ...prev,
        [key]: { status: 0, body: extractErrorMessage(e), timeMs: Date.now() - start },
      }))
    } finally {
      setLoadingMap((prev) => ({ ...prev, [key]: false }))
      setTimeout(() => {
        responseRefs.current[key]?.scrollIntoView({ behavior: 'smooth', block: 'nearest' })
      }, 50)
    }
  }, [])

  return {
    activeCategory,
    addToast,
    endpointKey,
    t,
    handleTryIt,
    loadingMap,
    responseRefs,
    responses,
    setActiveCategory,
    setLoadingMap,
    setResponses,
  }
}

/** The contract the docs panels render against. */
export type DocsWorkbench = ReturnType<typeof useDocsWorkbench>
