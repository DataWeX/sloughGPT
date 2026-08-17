'use client'

import { PUBLIC_API_URL } from '@/lib/config'

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
  onError: (status: number, text?: string) => void
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
}

export async function streamChatResponse(params: StreamChatParams): Promise<void> {
  const {
    messages, model, systemPrompt, maxTokens, temperature,
    userId, sessionId, images, signal,
    onToken, onComplete, onError, onKnowledge, onThinking, onToolCall, onMemory, onRagVerification,
  } = params

  const response = await fetch(`${PUBLIC_API_URL}/chat/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      messages,
      model,
      system_prompt: systemPrompt,
      max_new_tokens: maxTokens,
      temperature,
      user_id: userId,
      session_id: sessionId,
      images,
      knowledge: params.knowledge,
      agent_id: params.agentId || undefined,
      use_rag: true,
    }),
    signal,
  })

  if (!response.ok) {
    const errorText = await response.text().catch(() => '')
    onError(response.status, errorText)
    return
  }

  const reader = response.body?.getReader()
  const decoder = new TextDecoder()
  let hasContent = false
  let completed = false
  let buffer = ''

  if (!reader) return

  while (true) {
    const { done, value } = await reader.read()
    if (done) break

    buffer += decoder.decode(value, { stream: true })
    const lines = buffer.split('\n')
    buffer = lines.pop() || ''

    for (const line of lines) {
      const trimmed = line.trimEnd()
      if (!trimmed.startsWith('data:')) continue
      const payload = trimmed.slice(5).trim()
      if (!payload || payload === '[DONE]') continue

      try {
        const envelope = JSON.parse(payload) as {
          stream?: string; phase?: string; status?: string
          data?: Record<string, unknown>
          error?: string; message?: string
        }

        if (envelope.status === 'thinking') {
          onThinking?.()
          continue
        }

        const d = envelope.data ?? {}

        // ── Post-complete memory event (arrives after status=complete) ──
        if (envelope.phase === 'MEMORY') {
          onMemory?.({
            stored: d.stored === true,
            fact: typeof d.fact === 'string' ? d.fact : undefined,
            facts: Array.isArray(d.facts)
              ? d.facts.filter((f): f is string => typeof f === 'string')
              : undefined,
          })
          continue
        }

        // ── RAG grounding verification event ──
        if (envelope.phase === 'RAG_VERIFICATION') {
          onRagVerification?.({
            confidence: typeof d.confidence === 'number' ? d.confidence : 0,
            is_verified: d.is_verified === true,
            hallucination_rate: typeof d.hallucination_rate === 'number' ? d.hallucination_rate : 0,
            citations: typeof d.citations === 'string' ? d.citations : '',
            grounded_claims: typeof d.grounded_claims === 'number' ? d.grounded_claims : 0,
            hallucinated_claims: typeof d.hallucinated_claims === 'number' ? d.hallucinated_claims : 0,
          })
          continue
        }

        // ── Tool call events (must come before generic error check) ──
        if (envelope.phase === 'TOOL') {
          if (d.tool && typeof d.tool === 'string') {
            const toolEvent: ToolCallEvent = {
              tool: d.tool as string,
              status: (envelope.status === 'complete' ? 'success' : envelope.status === 'error' ? 'error' : 'executing') as ToolCallEvent['status'],
              output: d.output as string | undefined,
              error: d.error as string | undefined,
              duration_ms: d.duration_ms as number | undefined,
              args: d.args as Record<string, unknown> | undefined,
            }
            onToolCall?.(toolEvent)
          }
          continue
        }

        if (envelope.status === 'error') {
          const errStr = typeof envelope.data?.error === 'string' ? envelope.data.error : undefined
          onError(500, envelope.message || errStr || 'Stream error')
          return
        }

        if (d.source && typeof d.source === 'string') {
          const fc = typeof d.fact_count === 'number' ? d.fact_count : 0
          onKnowledge?.(d.source, fc)
        }

        const token = d.token as string | undefined
        if (token) {
          hasContent = true
          onToken(token)
        }

        if (envelope.status === 'complete') {
          if (!hasContent) onToken('')
          onComplete()
          completed = true
          continue
        }
      } catch {
        // skip malformed lines
      }
    }
  }

  if (!hasContent) onToken('')
  if (!completed) onComplete()
}
