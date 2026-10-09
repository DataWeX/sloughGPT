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

import { PUBLIC_API_URL } from './config'
import { useAuthStore } from './auth'

// ── Shared SSE wire-frame parser ────────────────────────────────────────────
// Single implementation of the chunk → line → data: → JSON.parse state machine
// used by both createSSEStream (callback + reconnect) and streamSSE in
// http-client (async-generator + retry). Keeps the two stream control flows
// distinct while unifying the actual wire decoding.

export interface SSEParseResult {
  /** Parsed payloads extracted from this chunk (already JSON-decoded). */
  events: unknown[]
  /** Bytes that didn't yet form a complete line; carry into next chunk. */
  buffer: string
}

export function consumeSSEChunk(
  buffer: string,
  decoder: TextDecoder,
  chunk: Uint8Array,
  onMalformed?: (payload: string) => void,
): SSEParseResult {
  let nextBuffer = buffer + decoder.decode(chunk, { stream: true })
  const lines = nextBuffer.split('\n')
  nextBuffer = lines.pop() || ''
  const events: unknown[] = []
  for (const line of lines) {
    const trimmed = line.trimEnd()
    if (!trimmed.startsWith('data:')) continue
    const payload = trimmed.slice(5).trim()
    if (!payload || payload === '[DONE]') continue
    try {
      events.push(JSON.parse(payload))
    } catch (e) {
      onMalformed?.(payload.slice(0, 80))
    }
  }
  return { events, buffer: nextBuffer }
}

export function consumeSSEDrain(
  buffer: string,
  onMalformed?: (payload: string) => void,
): unknown[] {
  const trimmed = buffer.trimEnd()
  if (!trimmed.startsWith('data:')) return []
  const payload = trimmed.slice(5).trim()
  if (!payload || payload === '[DONE]') return []
  try {
    return [JSON.parse(payload)]
  } catch (e) {
    onMalformed?.(payload.slice(0, 80))
    return []
  }
}

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
    const apiUrl = `${PUBLIC_API_URL}${url}`

    const headers: Record<string, string> = {
      Accept: 'text/event-stream',
      'Cache-Control': 'no-cache',
      ...extraHeaders,
    }
    const token = useAuthStore.getState().token
    if (token) headers['Authorization'] = `Bearer ${token}`

    try {
      const res = await fetch(apiUrl, {
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
