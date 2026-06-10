/**
 * Chat Controller — axios-based API for chat operations.
 *
 * Uses apiClient for REST calls, fetch for SSE streaming.
 */

import { apiPost, apiGet } from './http-client'
import { PUBLIC_API_URL } from './config'
import { useAuthStore } from './auth'
import { modelController, type ModelStatus } from './model-controller'

export interface ChatMessage {
  role: 'user' | 'assistant' | 'system'
  content: string
}

interface ChatRequest {
  messages: ChatMessage[]
  max_tokens?: number
  temperature?: number
}

interface ChatResponse {
  message: string
  session_id: string
  done: boolean
}

function authHeaders(): Record<string, string> {
  const token = useAuthStore.getState().token
  return token ? { Authorization: `Bearer ${token}` } : {}
}

export const chatController = {
  async send(message: string, options?: {
    max_tokens?: number
    temperature?: number
    session_id?: string
  }): Promise<ChatResponse> {
    const modelStatus = await modelController.status()
    if (!modelStatus.loaded) throw new Error('No model loaded. Load a model first.')

    try {
      const data = await apiPost<{ message?: string; text?: string; session_id?: string }>(
        '/chat',
        {
          messages: [{ role: 'user', content: message }],
          max_tokens: options?.max_tokens ?? 100,
          temperature: options?.temperature ?? 0.8,
        } as ChatRequest,
      )
      return {
        message: data.message || data.text || '',
        session_id: data.session_id || options?.session_id || 'default',
        done: true,
      }
    } catch {
      const fallback = await apiPost<{ text?: string }>(
        '/inference/generate',
        {
          prompt: `User: ${message}\nAssistant:`,
          max_new_tokens: options?.max_tokens ?? 100,
          temperature: options?.temperature ?? 0.8,
        },
      )
      return {
        message: fallback.text || '',
        session_id: options?.session_id || 'default',
        done: true,
      }
    }
  },

  async *stream(message: string, options?: {
    max_tokens?: number
    temperature?: number
  }): AsyncGenerator<string> {
    const modelStatus = await modelController.status()
    if (!modelStatus.loaded) { yield '[No model loaded]'; return }

    const res = await fetch(`${PUBLIC_API_URL}/chat/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', ...authHeaders() },
      body: JSON.stringify({
        messages: [{ role: 'user', content: message }],
        max_new_tokens: options?.max_tokens ?? 100,
        temperature: options?.temperature ?? 0.8,
      }),
    })

    const reader = res.body?.getReader()
    if (!reader) { yield '[Stream error]'; return }
    const decoder = new TextDecoder()
    try {
      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        for (const line of decoder.decode(value).split('\n')) {
          if (line.startsWith('data: ')) {
            try {
              const envelope = JSON.parse(line.slice(6)) as {
                stream?: string; phase?: string; status?: string
                data?: { token?: string; error?: string }
                error?: string; message?: string
              }
              if (envelope.status === 'error') {
                yield `[${envelope.message || envelope.data?.error || 'Stream error'}]`
                return
              }
              if (envelope.data?.token) yield envelope.data.token
              if (envelope.status === 'complete') return
            } catch { /* skip */ }
          }
        }
      }
    } finally { reader.releaseLock() }
  },

  async checkReady(): Promise<ModelStatus> {
    return modelController.status()
  },

  async *regenerateStream(sessionId: string, messages: ChatMessage[]): AsyncGenerator<{ token?: string; done?: boolean; error?: string }> {
    const res = await fetch(`${PUBLIC_API_URL}/session/${sessionId}/regenerate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', ...authHeaders() },
      body: JSON.stringify({ session_id: sessionId, messages, regenerate: true }),
    })
    const reader = res.body?.getReader()
    if (!reader) { yield { error: 'Stream error' }; return }
    const decoder = new TextDecoder()
    try {
      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        for (const line of decoder.decode(value).split('\n')) {
          const trimmed = line.trimEnd()
          if (!trimmed.startsWith('data:')) continue
          const payload = trimmed.slice(5).trim()
          if (!payload || payload === '[DONE]') continue
          try {
            const envelope = JSON.parse(payload) as {
              stream?: string; phase?: string; status?: string
              data?: { token?: string; error?: string }
              error?: string; message?: string
            }
            if (envelope.status === 'error') {
              yield { error: envelope.message || envelope.data?.error || 'Stream error' }
              return
            }
            if (envelope.data?.token) yield { token: envelope.data.token }
            if (envelope.status === 'complete') { yield { done: true }; return }
          } catch { /* skip */ }
        }
      }
    } finally { reader.releaseLock() }
  },

  async saveSessionContext(sessionId: string, messages: ChatMessage[]): Promise<void> {
    await apiPost(`/session/${sessionId}/context`, { messages })
  },

  formatMessages(messages: ChatMessage[]): string {
    let prompt = ''
    for (const m of messages) {
      const role = m.role === 'system' ? 'System' : m.role === 'user' ? 'User' : 'Assistant'
      prompt += `${role}: ${m.content}\n`
    }
    return prompt + 'Assistant:'
  },
}

export type { ChatResponse }
