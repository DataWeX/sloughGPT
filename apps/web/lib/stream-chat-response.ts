'use client'

import { API_CHAT_ENDPOINT } from '@/lib/config'

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
}

export async function streamChatResponse(params: StreamChatParams): Promise<void> {
  const {
    messages, model, systemPrompt, maxTokens, temperature,
    userId, sessionId, images, signal,
    onToken, onComplete, onError, onKnowledge, onThinking,
  } = params

  const response = await fetch(API_CHAT_ENDPOINT, {
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

        if (envelope.status === 'error') {
          const errStr = typeof envelope.data?.error === 'string' ? envelope.data.error : undefined
          onError(500, envelope.message || errStr || 'Stream error')
          return
        }

        const d = envelope.data ?? {}
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
          return
        }
      } catch {
        // skip malformed lines
      }
    }
  }

  if (!hasContent) onToken('')
  onComplete()
}
