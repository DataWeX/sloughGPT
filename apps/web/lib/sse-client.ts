/**
 * Unified SSE Client — shared infrastructure for all server-sent event streams.
 *
 * Provides:
 *   - Envelope parsing (standard {stream, phase, status, data, meta, message})
 *   - Auto-reconnection with exponential backoff
 *   - AbortController support for cleanup
 *   - Typed event callbacks
 *
 * Usage:
 *   const stream = createSSEStream({ url: '/health/stream', onEvent: (env) => {...} })
 *   stream.start()
 *   stream.stop()  // later
 */

import { fetchSSE } from './http-client'

// The wire-frame parser moved down into the transport (it is pure decode with
// no app dependency). Imported locally for connect()'s decode loop, and
// re-exported so app-layer callers keep their import path.
import { consumeSSEChunk, consumeSSEDrain } from '../../lib/sse-parse'
export type { SSEParseResult } from '../../lib/sse-parse'
export { consumeSSEChunk, consumeSSEDrain }

export interface SSEEnvelope {
  stream: string
  phase: string
  status: 'working' | 'success' | 'error' | 'complete'
  data: Record<string, unknown>
  meta: Record<string, unknown>
  message: string
}

export interface SSEStreamOptions {
  /** Endpoint path (appended to PUBLIC_API_URL) */
  url: string
  /** Called for every parsed envelope */
  onEvent: (envelope: SSEEnvelope) => void
  /** Called on connection open */
  onOpen?: () => void
  /** Called on connection close (after all retries exhausted) */
  onClose?: () => void
  /** Called on error (fetch failure, parse error) */
  onError?: (error: Error) => void
  /** Auto-reconnect on failure (default: true) */
  reconnect?: boolean
  /** Max reconnect attempts before giving up (default: Infinity) */
  maxReconnects?: number
  /** Base reconnect delay in ms (default: 2000, doubles each retry) */
  baseReconnectMs?: number
  /** Max reconnect delay in ms (default: 30000) */
  maxReconnectMs?: number
  /** Additional headers */
  headers?: Record<string, string>
}

export interface SSEStream {
  /** Start the stream (or reconnect if stopped) */
  start: () => void
  /** Stop the stream and abort any in-flight request */
  stop: () => void
  /** Whether the stream is currently connected */
  readonly connected: boolean
}

/**
 * Create a managed SSE stream with auto-reconnect.
 *
 * Each call creates an independent connection. Multiple streams can coexist.
 * The stream automatically reconnects on failure unless explicitly stopped.
 */
export function createSSEStream(options: SSEStreamOptions): SSEStream {
  const {
    url,
    onEvent,
    onOpen,
    onClose,
    onError,
    reconnect = true,
    maxReconnects = Infinity,
    baseReconnectMs = 2000,
    maxReconnectMs = 30_000,
    headers: extraHeaders,
  } = options

  let controller: AbortController | null = null
  let reconnectCount = 0
  let reconnectTimer: ReturnType<typeof setTimeout> | null = null
  let _connected = false
  let _stopped = false

  async function connect() {
    if (_stopped) return

    controller = new AbortController()

    const headers: Record<string, string> = {
      Accept: 'text/event-stream',
      'Cache-Control': 'no-cache',
      ...extraHeaders,
    }

    try {
      // Base URL resolution and the bearer token now come from the transport
      // (fetchSSE) instead of a second hand-rolled fetch in this file.
      const res = await fetchSSE(url, {
        headers,
        signal: controller.signal,
      })

      if (!res.ok) {
        throw new Error(`SSE ${res.status}: ${res.statusText}`)
      }

      _connected = true
      reconnectCount = 0
      onOpen?.()

      const reader = res.body?.getReader()
      if (!reader) throw new Error('No ReadableStream')

      const decoder = new TextDecoder()
      let buffer = ''

      for (;;) {
        const { done, value } = await reader.read()
        if (done) break

        const { events, buffer: nextBuffer } = consumeSSEChunk(buffer, decoder, value)
        buffer = nextBuffer
        for (const event of events) {
          onEvent(event as SSEEnvelope)
        }
      }

      for (const event of consumeSSEDrain(buffer)) {
        onEvent(event as SSEEnvelope)
      }

      // Stream ended normally
      _connected = false
      onClose?.()
    } catch (err: unknown) {
      _connected = false

      if (err instanceof Error && (err.name === 'AbortError' || _stopped)) {
        // Intentional stop — don't reconnect
        onClose?.()
        return
      }

      onError?.(err instanceof Error ? err : new Error(String(err)))

      if (reconnect && reconnectCount < maxReconnects) {
        reconnectCount++
        // Detect connection refused / fetch failures during cold start.
        // Use a longer initial delay to avoid hammering a dead server.
        const isConnRefused =
          err instanceof Error &&
          (err.message.includes('Failed to fetch') ||
            err.message.includes('ECONNREFUSED') ||
            err.message.includes('NetworkError'))
        const effectiveBase =
          isConnRefused && reconnectCount <= 3 ? Math.max(baseReconnectMs, 5000) : baseReconnectMs
        const delay = Math.min(maxReconnectMs, effectiveBase * Math.pow(2, reconnectCount - 1))
        reconnectTimer = setTimeout(connect, delay)
      } else {
        onClose?.()
      }
    }
  }

  return {
    start() {
      _stopped = false
      reconnectCount = 0
      connect()
    },
    stop() {
      _stopped = true
      if (reconnectTimer) {
        clearTimeout(reconnectTimer)
        reconnectTimer = null
      }
      if (controller) {
        controller.abort()
        controller = null
      }
      _connected = false
    },
    get connected() {
      return _connected
    },
  }
}
