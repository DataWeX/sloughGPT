'use client'

/**
 * Error Stream — real-time error events from the backend via SSE.
 *
 * Subscribes to /errors/stream and maintains a timeline of error events.
 * Falls back to polling /errors/recent if SSE fails.
 *
 * Usage:
 *   const { errors, connected, clear } = useErrorStream()
 */

import { createStore } from 'zustand/vanilla'
import { useStore } from 'zustand'
import { createSSEStream, type SSEEnvelope } from '@/lib/sse-client'

export interface ErrorEvent {
  id: string
  message: string
  level: 'error' | 'critical' | 'warning' | 'info'
  source: string
  phase: string
  tag?: string
  stack?: string | null
  url?: string | null
  line?: number | null
  col?: number | null
  fingerprint?: string
  count?: number
  correlationId?: string
  httpMethod?: string
  httpPath?: string
  httpStatus?: number
  durationMs?: number
  timestamp: number
  context?: Record<string, unknown>
}

interface ErrorStreamState {
  errors: ErrorEvent[]
  connected: boolean
  totalReceived: number
  addError: (event: ErrorEvent) => void
  clearErrors: () => void
  setConnected: (connected: boolean) => void
}

const MAX_ERRORS = 100

const errorStreamStore = createStore<ErrorStreamState>((set) => ({
  errors: [],
  connected: false,
  totalReceived: 0,

  addError: (event) =>
    set((s) => ({
      errors: [event, ...s.errors].slice(0, MAX_ERRORS),
      totalReceived: s.totalReceived + 1,
    })),

  clearErrors: () => set({ errors: [], totalReceived: 0 }),

  setConnected: (connected) => set({ connected }),
}))

let _stream: ReturnType<typeof createSSEStream> | null = null
let _started = false
let _fallbackTimer: ReturnType<typeof setInterval> | null = null

function onEvent(envelope: SSEEnvelope) {
  if (envelope.stream !== 'errors') return
  const d = envelope.data
  const ctx = d.context && typeof d.context === 'object' ? d.context as Record<string, unknown> : undefined

  // Extract correlation ID from multiple possible key names
  const correlationId = ctx
    ? String(ctx.corrId || ctx.correlation_id || ctx.correlationId || ctx.traceId || '')
    : undefined

  const event: ErrorEvent = {
    id: String(d.id || `evt_${Date.now()}_${Math.random().toString(36).slice(2, 6)}`),
    message: String(d.message || ''),
    level: (d.level as ErrorEvent['level']) || 'error',
    source: String(d.source || ''),
    phase: envelope.phase || 'ERROR',
    tag: d.tag ? String(d.tag) : undefined,
    stack: d.stack ? String(d.stack) : null,
    url: d.url ? String(d.url) : null,
    line: typeof d.line === 'number' ? d.line : null,
    col: typeof d.col === 'number' ? d.col : null,
    fingerprint: d.fingerprint ? String(d.fingerprint) : undefined,
    count: typeof d.count === 'number' ? d.count : undefined,
    correlationId,
    httpMethod: d.http_method ? String(d.http_method) : ctx?.method ? String(ctx.method) : undefined,
    httpPath: d.http_path ? String(d.http_path) : ctx?.path ? String(ctx.path) : undefined,
    httpStatus: typeof d.http_status === 'number' ? d.http_status : typeof ctx?.status === 'number' ? Number(ctx.status) : undefined,
    durationMs: typeof d.duration_ms === 'number' ? d.duration_ms : typeof ctx?.dur_ms === 'number' ? Number(ctx.dur_ms) : undefined,
    timestamp: typeof d.ts === 'number' ? d.ts * 1000 : Date.now(),
    context: ctx,
  }
  errorStreamStore.getState().addError(event)
}

function onError() {
  errorStreamStore.getState().setConnected(false)
}

function onOpen() {
  errorStreamStore.getState().setConnected(true)
}

function onClose() {
  errorStreamStore.getState().setConnected(false)
  startFallbackPoll()
}

async function fallbackPoll() {
  try {
    const res = await fetch('/errors/recent?limit=20', { signal: AbortSignal.timeout(5000) })
    if (!res.ok) return
    const json = await res.json()
    const errors = json?.data?.errors || []
    const state = errorStreamStore.getState()
    // Only add if we have no errors yet (initial load)
    if (state.errors.length === 0 && Array.isArray(errors)) {
      for (const err of errors.reverse()) {
        state.addError({
          id: String(err.id || `poll_${Date.now()}`),
          message: String(err.message || ''),
          level: 'error',
          source: String(err.source || 'backend'),
          phase: 'CLIENT_ERROR',
          stack: err.stack,
          url: err.url,
          line: err.line,
          col: err.col,
          fingerprint: err.fingerprint,
          count: err.count,
          timestamp: err.timestamp ? new Date(err.timestamp).getTime() : Date.now(),
        })
      }
    }
  } catch {
    // silent
  }
}

function startFallbackPoll() {
  if (_fallbackTimer) return
  _fallbackTimer = setInterval(fallbackPoll, 10000)
  fallbackPoll()
}

function stopFallbackPoll() {
  if (_fallbackTimer) {
    clearInterval(_fallbackTimer)
    _fallbackTimer = null
  }
}

/**
 * Initialize the error stream. Call once at app root.
 * Returns a cleanup function.
 */
export function initErrorStream(): () => void {
  if (_started) return () => {}
  _started = true

  _stream = createSSEStream({
    url: '/errors/stream',
    onEvent,
    onOpen,
    onClose,
    onError,
    reconnect: true,
    maxReconnects: Infinity,
    baseReconnectMs: 3000,
    maxReconnectMs: 15_000,
  })

  _stream.start()

  return () => {
    _started = false
    _stream?.stop()
    _stream = null
    stopFallbackPoll()
  }
}

/**
 * React hook — returns real-time error events.
 */
export function useErrorStream() {
  const errors = useErrorStreamStore((s) => s.errors)
  const connected = useErrorStreamStore((s) => s.connected)
  const totalReceived = useErrorStreamStore((s) => s.totalReceived)
  const clearErrors = useErrorStreamStore((s) => s.clearErrors)

  return { errors, connected, totalReceived, clearErrors }
}

function useErrorStreamStore<T>(selector: (s: ErrorStreamState) => T): T {
  return useStore(errorStreamStore, selector)
}

export { useErrorStreamStore }
