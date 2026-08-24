'use client'

import { useEffect, useCallback, useRef } from 'react'
import { useShallow } from 'zustand/react/shallow'
import { createSSEStream, type SSEEnvelope } from '@/lib/sse-client'
import {
  operationsStore,
  useOperationsStore,
  type Operation,
  type OpType,
} from '@/lib/operations-store'

export type { Operation, OpType }

// ── Singleton SSE connection ────────────────────────────────────────

let _stream: ReturnType<typeof createSSEStream> | null = null
let _fallbackTimer: ReturnType<typeof setInterval> | null = null
let _receivedInit = false
let _stopped = true
let _refCount = 0

function startFallback() {
  if (_fallbackTimer) return
  operationsStore.getState().fetch()
  _fallbackTimer = setInterval(() => operationsStore.getState().fetch(), 15000)
}

function stopFallback() {
  if (_fallbackTimer) {
    clearInterval(_fallbackTimer)
    _fallbackTimer = null
  }
}

function onEvent(envelope: SSEEnvelope) {
  if (envelope.stream !== 'operations') return
  _receivedInit = true
  stopFallback()

  const action = envelope.phase?.toLowerCase()
  const data = envelope.data as Record<string, unknown>

  if (action === 'init') {
    const ops = data.operations as Operation[] | undefined
    const cnts = data.counts as Record<string, number> | undefined
    if (ops) {
      operationsStore.setState({ operations: ops, counts: cnts || {}, loading: false, error: null })
    }
    return
  }

  if (action === 'registered' || action === 'started' || action === 'finished' || action === 'cancelled') {
    operationsStore.getState().fetch()
  }
}

function onOpen() {
  operationsStore.setState({ loading: false, error: null })
}

function onClose() {
  if (!_stopped) startFallback()
}

function onError() {
  if (!_stopped) startFallback()
}

/**
 * Initialize the singleton operations SSE stream.
 * Safe to call multiple times — idempotent.
 */
export function initOperationsStream(): () => void {
  _refCount++
  if (_stream) return () => decrementRef()

  _stopped = false
  _receivedInit = false

  _stream = createSSEStream({
    url: '/operations/stream',
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

  // Grace period
  const graceTimer = setTimeout(() => {
    if (!_receivedInit && !_stopped) startFallback()
  }, 5000)

  return () => {
    clearTimeout(graceTimer)
    decrementRef()
  }
}

function decrementRef() {
  _refCount--
  if (_refCount <= 0) {
    _refCount = 0
    _stopped = true
    _stream?.stop()
    _stream = null
    stopFallback()
  }
}

// ── Hook ────────────────────────────────────────────────────────────

/**
 * SSE-backed operations hook — replaces HTTP polling with real-time push.
 *
 * Uses a singleton SSE connection to `/operations/stream`. Multiple
 * components can use this hook simultaneously without extra connections.
 * Falls back to HTTP polling if SSE fails.
 *
 * Usage:
 *   const { operations, isActive, cancel, cancelAll } = useLiveOperations('training')
 */
export function useLiveOperations(type?: OpType) {
  const { operations, counts, loading, error, fetchOps, cancel, cancelAll, activeOps, isActive, hasTraining, hasInference } = useOperationsStore(
    useShallow((s) => {
      const active = s.operations.filter(
        (op) => ['registered', 'running', 'cancelling'].includes(op.status) && (!type || op.type === type)
      )
      return {
        operations: s.operations,
        counts: s.counts,
        loading: s.loading,
        error: s.error,
        fetchOps: s.fetch,
        cancel: s.cancel,
        cancelAll: s.cancelAll,
        activeOps: active,
        isActive: active.length > 0,
        hasTraining: s.operations.some((op) => ['registered', 'running', 'cancelling'].includes(op.status) && op.type === 'training'),
        hasInference: s.operations.some((op) => ['registered', 'running', 'cancelling'].includes(op.status) && op.type === 'inference'),
      }
    })
  )

  // Manage singleton lifecycle
  const cleanupRef = useRef<(() => void) | null>(null)
  useEffect(() => {
    cleanupRef.current = initOperationsStream()
    return () => { cleanupRef.current?.(); cleanupRef.current = null }
  }, [])

  const cancelOp = useCallback(
    async (opId: string) => {
      const ok = await cancel(opId)
      await fetchOps()
      return ok
    },
    [cancel, fetchOps]
  )

  const cancelAllByType = useCallback(
    async (cancelType?: OpType) => {
      const n = await cancelAll(cancelType ?? type)
      await fetchOps()
      return n
    },
    [cancelAll, fetchOps, type]
  )

  const refresh = useCallback(() => fetchOps(), [fetchOps])

  return {
    operations,
    activeOps,
    counts,
    loading,
    error,
    isActive,
    hasTraining,
    hasInference,
    cancel: cancelOp,
    cancelAll: cancelAllByType,
    refresh,
    connected: _stream?.connected ?? false,
  }
}
