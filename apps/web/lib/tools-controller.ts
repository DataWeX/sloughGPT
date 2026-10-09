/**
 * Tools Controller — everyday tools API.
 *
 * Matches the tools domain + router:
 *   GET  /tools                    → list tool profiles with UI metadata
 *   POST /tools/{tool_id}/generate → SSE streaming through the inference provider
 *
 * All calls go through http-client (apiGet / streamSSE).
 */

'use client'

import { apiGet, streamSSE, type SSEEvent } from '@/lib/http-client'
import { logger } from '@/lib/dev-log'

const _log = logger.child('tools-controller')

export interface ToolOption {
  id: string
  label: string
  description: string
}

export interface ToolParam {
  id: string
  label: string
  placeholder: string
  multiline?: boolean
  optional?: boolean
}

export interface ToolProfile {
  id: string
  name: string
  description: string
  icon: string
  params: ToolParam[]
  options: Record<string, ToolOption[]>
  default_options: Record<string, string>
  system_prompt: string
  max_tokens: number
}

export interface GenerateCallbacks {
  onToken?: (token: string) => void
  onDone?: () => void
  onError?: (message: string) => void
}

/** Map tool id → human-readable built-in options for quick client-side hints. */
export const TOOL_IDS = [
  'writing',
  'translate',
  'rewrite',
  'brainstorm',
  'decide',
  'explain',
  'wellness',
] as const

export type ToolId = (typeof TOOL_IDS)[number]

export async function listTools(): Promise<ToolProfile[]> {
  try {
    const data = await apiGet<{ tools?: ToolProfile[] }>('/tools')
    return data?.tools ?? []
  } catch (err) {
    _log.debug('Failed to fetch tools', { error: err instanceof Error ? err.message : String(err) })
    return []
  }
}

/**
 * Stream a tool generation through the backend. Batches tokens and calls
 * `onToken` with appended deltas; resolves when the stream completes or
 * rejects on a hard error.
 */
export async function generateTool(
  toolId: ToolId | string,
  payload: Record<string, unknown>,
  callbacks: GenerateCallbacks,
  options?: { max_tokens?: number; temperature?: number },
): Promise<void> {
  const body: Record<string, unknown> = { payload }
  if (options?.max_tokens) body.max_tokens = options.max_tokens
  if (options?.temperature != null) body.temperature = options.temperature

  const gen = streamSSE(`/tools/${toolId}/generate`, { body })

  let acc = ''
  const flush = (token: string) => {
    acc += token
    callbacks.onToken?.(acc)
  }

  try {
    for await (const event of gen) {
      const e = event as SSEEvent
      if (e.status === 'error') {
        callbacks.onError?.(e.message || (e.data?.error as string) || 'Generation failed')
        return
      }
      if (e.data?.token) flush(e.data.token as string)
      if (e.status === 'complete') {
        callbacks.onDone?.()
        return
      }
    }
    callbacks.onDone?.()
  } catch (err) {
    callbacks.onError?.(`Connection error: ${err instanceof Error ? err.message : 'unknown'}`)
  }
}