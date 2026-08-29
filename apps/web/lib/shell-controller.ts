/**
 * Shell Controller — Dait shell command execution.
 *
 * Wraps POST /shell/exec (sync) and POST /shell/exec/stream (SSE streaming).
 */

'use client'

import { apiPost, streamSSE, type SSEEvent } from '@/lib/http-client'
import { logger } from '@/lib/dev-log'

export interface ShellExecResult {
  output: string
  exit_code: number
  elapsed_ms: number
}

export interface ShellStreamEvent {
  stream: string
  phase: string
  status: string
  data: { line?: string; index?: number; exit_code?: number; lines?: number; error?: string }
  meta?: { elapsed_ms?: number }
  message: string
}

export interface ShellStreamCallbacks {
  onLine?: (line: string, index: number) => void
  onComplete?: (exitCode: number, totalLines: number, elapsedMs: number) => void
  onError?: (error: string) => void
}

/** Execute a shell command synchronously. */
export async function shellExec(
  command: string,
  timeoutMs = 30000,
  signal?: AbortSignal,
): Promise<ShellExecResult> {
  return apiPost<ShellExecResult>('/shell/exec', { command, timeout_ms: timeoutMs }, { signal })
}

/** Execute a shell command with SSE streaming. */
export async function shellExecStream(
  command: string,
  callbacks: ShellStreamCallbacks,
  signal?: AbortSignal,
  timeoutMs = 30000,
): Promise<void> {
  const gen = streamSSE('/shell/exec/stream', {
    method: 'POST',
    body: { command, timeout_ms: timeoutMs },
    signal,
  })

  let completed = false
  let errored = false

  try {
    for await (const event of gen) {
      const d = (event as unknown as ShellStreamEvent).data ?? {}
      const phase = (event as unknown as ShellStreamEvent).phase
      const status = (event as unknown as ShellStreamEvent).status

      if (phase === 'STREAMING' && status === 'working' && d.line !== undefined) {
        callbacks.onLine?.(d.line, d.index ?? 0)
      } else if (phase === 'STREAMING' && status === 'complete') {
        completed = true
        const meta = (event as unknown as ShellStreamEvent).meta
        callbacks.onComplete?.(d.exit_code ?? 1, d.lines ?? 0, meta?.elapsed_ms ?? 0)
      } else if (phase === 'STREAMING' && status === 'error') {
        completed = true
        errored = true
        const msg = d.error || 'Could not command'
        callbacks.onError?.(msg) ?? logger.error('shell command error', { exception: msg })
      } else if (event.status === 'error') {
        errored = true
        const msg = event.message || 'Unknown error'
        callbacks.onError?.(msg) ?? logger.error('shell event error', { exception: msg })
      }
    }
  } catch (err) {
    errored = true
    const msg = err instanceof Error ? err.message : 'Connection error'
    callbacks.onError?.(msg) ?? logger.error('shell connection error', { exception: msg })
  }

  if (!completed && !errored) {
    callbacks.onComplete?.(1, 0, 0)
  }
}

export const shellController = {
  exec: shellExec,
  execStream: shellExecStream,
}
