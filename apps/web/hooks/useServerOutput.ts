/**
 * useServerOutput — hook for streaming server output in any component.
 *
 * Usage:
 *   const { lines, streaming, clear, paused, togglePause, exportLines } = useServerOutput({ tail: 50 })
 */

'use client'

import { useState, useEffect, useRef, useCallback } from 'react'
import { systemController, type OutputLine } from '@/lib/system-controller'

interface UseServerOutputOptions {
  tail?: number
  maxLines?: number
}

interface UseServerOutputReturn {
  lines: OutputLine[]
  streaming: boolean
  clear: () => void
  scrollRef: React.RefObject<HTMLDivElement>
  paused: boolean
  togglePause: () => void
  exportLines: (format?: 'text' | 'json') => void
}

export function useServerOutput(opts: UseServerOutputOptions = {}): UseServerOutputReturn {
  const { tail = 50, maxLines = 200 } = opts
  const [lines, setLines] = useState<OutputLine[]>([])
  const [streaming, setStreaming] = useState(false)
  const [paused, setPaused] = useState(false)
  const scrollRef = useRef<HTMLDivElement>(null!)
  const mountedRef = useRef(true)
  const pausedRef = useRef(false)

  useEffect(() => {
    pausedRef.current = paused
  }, [paused])

  useEffect(() => {
    mountedRef.current = true
    let cancelled = false

    const start = async () => {
      setStreaming(true)
      try {
        for await (const line of systemController.streamOutput(tail)) {
          if (cancelled || !mountedRef.current) break
          if (!pausedRef.current) {
            setLines(prev => {
              const next = [...prev, line]
              return next.length > maxLines ? next.slice(-maxLines) : next
            })
          }
        }
      } catch {}
      if (mountedRef.current) setStreaming(false)
    }

    start()
    return () => {
      cancelled = true
      mountedRef.current = false
    }
  }, [tail, maxLines])

  useEffect(() => {
    if (scrollRef.current && !paused) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight
    }
  }, [lines, paused])

  const clear = useCallback(() => setLines([]), [])
  const togglePause = useCallback(() => setPaused(p => !p), [])

  const exportLines = useCallback((format: 'text' | 'json' = 'text') => {
    let content: string
    if (format === 'json') {
      content = JSON.stringify(lines, null, 2)
    } else {
      content = lines.map(l => {
        const tag = l.tag ? ` [${l.tag}]` : ''
        const src = l.source ? ` ${l.source}` : ''
        return `${l.level}${tag}${src}  ${l.text}`
      }).join('\n')
    }
    const blob = new Blob([content], { type: format === 'json' ? 'application/json' : 'text/plain' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `server-output.${format === 'json' ? 'json' : 'log'}`
    a.click()
    URL.revokeObjectURL(url)
  }, [lines])

  return { lines, streaming, clear, scrollRef, paused, togglePause, exportLines }
}
