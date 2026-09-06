'use client'

import { streamSSE } from '@/lib/http-client'
import { useErrorStore } from '@/lib/error-store'
import { logger } from '@/lib/dev-log'

const _log = logger.child('stream-chat-response')

const RETRYABLE_STATUSES = new Set([408, 429, 502, 503, 504])
const MAX_RETRIES = 2
const BASE_DELAY_MS = 500
const MODEL_LOADING_DELAY_MS = 3000  // Longer delay for model loading (503)

export interface ToolCallEvent {
  tool: string
  status: 'executing' | 'success' | 'error'
  output?: string
  error?: string
  duration_ms?: number
  args?: Record<string, unknown>
}

interface StreamChatParams {
  messages: { role: string; content: string }[]
  model: string
  systemPrompt: string
  maxTokens: number
  temperature: number
  userId: string
  sessionId: string
  images?: string[]
  signal?: AbortSignal
  agentId?: string
  knowledge?: string[]
  onToken: (token: string) => void
  onComplete: () => void
  onError: (status: number, text?: string, opts?: { correlationId?: string; backendError?: string; httpMethod?: string; httpPath?: string; durationMs?: number }) => void
  onKnowledge?: (source: string, count: number) => void
  onThinking?: () => void
  onToolCall?: (event: ToolCallEvent) => void
  onMemory?: (info: { stored: boolean; fact?: string; facts?: string[] }) => void
  onRagVerification?: (info: {
    confidence: number
    is_verified: boolean
    hallucination_rate: number
    citations: string
    grounded_claims: number
    hallucinated_claims: number
  }) => void
  onControl?: (event: { action: string; tool?: string; approved?: boolean; context?: string }) => void
}

function buildBody(params: StreamChatParams) {
  return {
    messages: params.messages,
    model: params.model,
    system_prompt: params.systemPrompt,
    max_new_tokens: params.maxTokens,
    temperature: params.temperature,
    user_id: params.userId,
    session_id: params.sessionId,
    images: params.images,
    knowledge: params.knowledge,
    agent_id: params.agentId || undefined,
    use_rag: true,
  }
}

export async function streamChatResponse(params: StreamChatParams): Promise<void> {
  const { signal, onError } = params
  const body = buildBody(params)
  let retries = 0
  let lastEventId: string | undefined
  let tokenCount = 0
  let startTime = Date.now()

  _log.debug('streamChatResponse START', {
    msgCount: params.messages.length,
    maxTokens: params.maxTokens,
    sessionId: params.sessionId,
    hasImages: !!params.images?.length,
  })

  while (true) {
    let hasContent = false
    let completed = false
    let shouldRetry = false

    try {
      for await (const event of streamSSE('/chat/stream', { body, signal, lastEventId })) {
        const d = event.data ?? {}

        // Track last event ID for reconnection
        if (event.id) {
          lastEventId = event.id as string
        }

        if (event.status === 'thinking') {
          params.onThinking?.()
          continue
        }

        if (event.phase === 'MEMORY') {
          params.onMemory?.({
            stored: d.stored === true,
            fact: typeof d.fact === 'string' ? d.fact : undefined,
            facts: Array.isArray(d.facts)
              ? d.facts.filter((f): f is string => typeof f === 'string')
              : undefined,
          })
          continue
        }

        if (event.phase === 'RAG_VERIFICATION') {
          params.onRagVerification?.({
            confidence: typeof d.confidence === 'number' ? d.confidence : 0,
            is_verified: d.is_verified === true,
            hallucination_rate: typeof d.hallucination_rate === 'number' ? d.hallucination_rate : 0,
            citations: typeof d.citations === 'string' ? d.citations : '',
            grounded_claims: typeof d.grounded_claims === 'number' ? d.grounded_claims : 0,
            hallucinated_claims: typeof d.hallucinated_claims === 'number' ? d.hallucinated_claims : 0,
          })
          continue
        }

        if (event.phase === 'TOOL') {
          if (d.tool && typeof d.tool === 'string') {
            const toolEvent: ToolCallEvent = {
              tool: d.tool as string,
              status: (event.status === 'complete' ? 'success' : event.status === 'error' ? 'error' : 'executing') as ToolCallEvent['status'],
              output: d.output as string | undefined,
              error: d.error as string | undefined,
              duration_ms: d.duration_ms as number | undefined,
              args: d.args as Record<string, unknown> | undefined,
            }
            params.onToolCall?.(toolEvent)
          }
          continue
        }

        if (event.phase === 'CONTROL') {
          params.onControl?.({
            action: event.status || 'unknown',
            tool: d.tool as string | undefined,
            approved: d.approved as boolean | undefined,
            context: d.context as string | undefined,
          })
          continue
        }

        if (event.status === 'error') {
          const hasHttpStatus = d != null && typeof d.http_status === 'number'
          const httpStatus = hasHttpStatus ? d.http_status as number : 0
          const errStr = typeof d.error === 'string' ? d.error : undefined
          const message = event.message || errStr || 'Stream error'

          if (signal?.aborted) {
            _log.debug('Stream aborted by user')
            return
          }

          if (RETRYABLE_STATUSES.has(httpStatus) && retries < MAX_RETRIES) {
            retries++
            // Use longer delay for model loading (503 with MODEL_LOADING code)
            const isModelLoading = httpStatus === 503 && (
              message.includes('loading') || message.includes('Loading') ||
              d.code === 'MODEL_LOADING'
            )
            const delay = isModelLoading
              ? MODEL_LOADING_DELAY_MS * retries
              : BASE_DELAY_MS * Math.pow(2, retries - 1)
            _log.debug('Retrying chat stream after transient error', {
              status: httpStatus, retries, delay, isModelLoading,
            })
            shouldRetry = true
            break
          }

          useErrorStore.getState().addError(
            new Error(message),
            { source: 'chat/stream', title: `Chat stream error (${httpStatus})` },
          )
          onError(httpStatus, message, {
            correlationId: typeof d.correlation_id === 'string' ? d.correlation_id : undefined,
            backendError: typeof d.error === 'string' ? d.error : undefined,
            httpMethod: typeof d.http_method === 'string' ? d.http_method : undefined,
            httpPath: typeof d.http_path === 'string' ? d.http_path : undefined,
            durationMs: typeof d.duration_ms === 'number' ? d.duration_ms : undefined,
          })
          return
        }

        if (d.source && typeof d.source === 'string') {
          const fc = typeof d.fact_count === 'number' ? d.fact_count : 0
          params.onKnowledge?.(d.source, fc)
        }

        const token = d.token as string | undefined
        if (token) {
          if (tokenCount === 0) {
            _log.debug('First token received', { elapsed: Date.now() - startTime })
          }
          tokenCount++
          hasContent = true
          params.onToken(token)
        }

        if (event.status === 'complete') {
          _log.debug('Stream complete', { tokens: tokenCount, elapsed: Date.now() - startTime })
          if (!hasContent) params.onToken('')
          params.onComplete()
          completed = true
          continue
        }
      }
    } catch (err) {
      if (err instanceof Error && err.name === 'AbortError') {
        _log.debug('Stream aborted by user')
        return
      }

      const message = err instanceof Error ? err.message : 'Network error'

      if (retries < MAX_RETRIES) {
        retries++
        const delay = BASE_DELAY_MS * Math.pow(2, retries - 1)
        _log.debug('Retrying chat stream after network error', { retries, delay })
        await new Promise(r => setTimeout(r, delay))
        continue
      }

      _log.error('Stream network error after retries', { retries, message })
      useErrorStore.getState().addError(
        err instanceof Error ? err : new Error(message),
        { source: 'chat/stream', title: 'Connection Error' },
      )
      onError(0, message, undefined)
      return
    }

    if (shouldRetry) {
      await new Promise(r => setTimeout(r, BASE_DELAY_MS * Math.pow(2, retries - 1)))
      continue
    }

    if (!hasContent && !completed) {
      onError(0, 'Stream ended without response')
      return
    }
    if (!completed) params.onComplete()
    return
  }
}
