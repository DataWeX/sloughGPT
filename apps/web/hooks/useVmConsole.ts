'use client'

import { useCallback, useEffect, useRef, useState } from 'react'
import { apiPost, apiDelete } from '@/lib/http-client'
import { PUBLIC_API_URL } from '@/lib/config'

export type VmConsolePhase = 'idle' | 'connecting' | 'live' | 'complete' | 'error' | 'closed'

interface ConsoleEnvelope {
  stream?: string
  phase?: string
  status?: string
  data?: {
    text?: string
    status?: string
    error?: string
    exit_code?: number
    backlog?: boolean
  }
}

export interface UseVmConsoleResult {
  sessionId: string | null
  phase: VmConsolePhase
  output: string
  error: string | null
  start: (role?: string) => Promise<void>
  sendInput: (text: string) => Promise<void>
  disconnect: () => Promise<void>
}

const ANSI_RE = /\x1b\[[0-9;]*[A-Za-z]/g

/**
 * Sanitize one output chunk against the accumulated buffer.
 *
 * - `ESC[2J` (kernel `clear` command) wipes the buffer; only the
 *   remainder of the chunk follows on the cleared screen.
 * - Other ANSI sequences are stripped for plain `<pre>` display.
 * - `\b` erases the previous character of the merged buffer, giving
 *   terminal-style backspace handling for kernel echo (`\b \b`).
 */
export function sanitizeVmOutput(prev: string, text: string): string {
  const parts = text.split('\x1b[2J')
  const cleared = parts.length > 1
  const body = cleared ? parts[parts.length - 1] : text
  const clean = body.replace(ANSI_RE, '')
  const merged = (cleared ? '' : prev) + clean
  if (!clean.includes('\b')) return merged
  const out: string[] = []
  for (const ch of merged) {
    if (ch === '\b') out.pop()
    else out.push(ch)
  }
  return out.join('')
}

/**
 * Interactive VM console session over SSE.
 *
 * Creates a /vm/session, streams output events, and delivers keyboard
 * input. The kernel echoes typing, so output is only appended from the
 * stream (never locally).
 */
export function useVmConsole(): UseVmConsoleResult {
  const [sessionId, setSessionId] = useState<string | null>(null)
  const [phase, setPhase] = useState<VmConsolePhase>('idle')
  const [output, setOutput] = useState('')
  const [error, setError] = useState<string | null>(null)
  const esRef = useRef<EventSource | null>(null)
  const sessionRef = useRef<string | null>(null)

  const openStream = useCallback((sid: string) => {
    esRef.current?.close()
    const es = new EventSource(`${PUBLIC_API_URL}/vm/session/${sid}/stream`)
    esRef.current = es
    es.onmessage = (e: MessageEvent) => {
      try {
        const env = JSON.parse(String(e.data)) as ConsoleEnvelope
        if (env.stream !== 'vm_console') return
        if (env.phase === 'output' && typeof env.data?.text === 'string') {
          const text = env.data.text
          setOutput((prev) => sanitizeVmOutput(prev, text))
        } else if (env.phase === 'start') {
          setPhase('live')
        } else if (env.phase === 'status') {
          const status = env.data?.status ?? env.status
          if (status === 'complete') {
            setPhase('complete')
            es.close()
          } else if (status === 'closed') {
            setPhase('closed')
            es.close()
          } else if (status === 'error') {
            setPhase('error')
            setError(env.data?.error ?? 'VM session error')
            es.close()
          }
        }
      } catch {
        /* malformed event — ignore */
      }
    }
    es.onerror = () => {
      if (es.readyState === EventSource.CLOSED) {
        setPhase((p) => {
          if (p === 'complete' || p === 'closed' || p === 'error') return p
          return 'error'
        })
      }
    }
  }, [])

  const start = useCallback(
    async (role: string = 'user') => {
      esRef.current?.close()
      esRef.current = null
      setPhase('connecting')
      setError(null)
      setOutput('')
      try {
        const res = await apiPost<{ session_id: string }>('/vm/session', { role })
        const sid = res.session_id
        sessionRef.current = sid
        setSessionId(sid)
        openStream(sid)
      } catch (e) {
        setPhase('error')
        setError(e instanceof Error ? e.message : 'Failed to start VM session')
      }
    },
    [openStream],
  )

  const sendInput = useCallback(async (text: string) => {
    const sid = sessionRef.current
    if (!sid) return
    try {
      await apiPost(`/vm/session/${sid}/input`, { text })
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to send input')
    }
  }, [])

  const disconnect = useCallback(async () => {
    esRef.current?.close()
    esRef.current = null
    const sid = sessionRef.current
    sessionRef.current = null
    setSessionId(null)
    setPhase('closed')
    if (sid) {
      try {
        await apiDelete(`/vm/session/${sid}`)
      } catch {
        /* session already gone */
      }
    }
  }, [])

  useEffect(() => {
    return () => {
      esRef.current?.close()
    }
  }, [])

  return { sessionId, phase, output, error, start, sendInput, disconnect }
}
