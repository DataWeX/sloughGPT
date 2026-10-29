/**
 * Chat Controller — axios-based API for chat operations.
 *
 * Uses apiClient for REST calls, fetch for SSE streaming.
 */

import { apiPost, apiGet, streamSSE } from './http-client'
import { modelController, type ModelStatus } from './model-controller'
import { logger } from './dev-log'
import { PUBLIC_API_URL } from './config'

const _log = logger.child('chat-controller')

export interface ChatMessage {
  role: 'user' | 'assistant' | 'system'
  content: string
}

export const chatController = {
  async *stream(
    message: string,
    options?: {
      max_tokens?: number
      temperature?: number
      waitForModel?: boolean
    },
  ): AsyncGenerator<string> {
    let modelStatus = await modelController.status()
    if (!modelStatus.loaded) {
      if (options?.waitForModel) {
        modelStatus = await modelController.waitForReady()
        if (!modelStatus.loaded) {
          yield '[Model still loading — please wait.]'
          return
        }
      } else {
        yield '[No model loaded]'
        return
      }
    }

    const body = {
      messages: [{ role: 'user', content: message }],
      max_new_tokens: options?.max_tokens ?? 100,
      temperature: options?.temperature ?? 0.8,
    }

    try {
      for await (const event of streamSSE('/chat', { body })) {
        if (event.status === 'error') {
          yield `[${event.message || (event.data?.error as string) || 'Stream error'}]`
          return
        }
        if (event.data?.token) yield event.data.token as string
        if (
          event.status === 'finalized' ||
          event.status === 'complete' ||
          event.status === 'cancelled'
        )
          return
      }
    } catch (err) {
      yield `[Connection error: ${err instanceof Error ? err.message : 'unknown'}]`
    }
  },

  async checkReady(): Promise<ModelStatus> {
    return modelController.status()
  },

  async *regenerateStream(
    sessionId: string,
    messages: ChatMessage[],
  ): AsyncGenerator<{ token?: string; done?: boolean; error?: string }> {
    const body = { session_id: sessionId, messages, regenerate: true }
    try {
      for await (const event of streamSSE(`/chat/${sessionId}/regenerate`, { body })) {
        if (event.status === 'error') {
          yield { error: event.message || (event.data?.error as string) || 'Stream error' }
          return
        }
        if (event.data?.token) yield { token: event.data.token as string }
        if (
          event.status === 'finalized' ||
          event.status === 'complete' ||
          event.status === 'cancelled'
        ) {
          yield { done: true }
          return
        }
      }
    } catch (err) {
      yield { error: `Connection error: ${err instanceof Error ? err.message : 'unknown'}` }
    }
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

  async getSuggestions(): Promise<{ text: string; icon: string }[]> {
    try {
      const data = await apiGet<{ suggestions?: { text: string; icon: string }[] }>(
        '/chat/suggestions',
      )
      return data?.suggestions ?? []
    } catch (err) {
      _log.debug('Failed to fetch suggestions', {
        error: err instanceof Error ? err.message : String(err),
      })
      return []
    }
  },

  async inspectContext(): Promise<ContextInspector | null> {
    try {
      return await apiGet<ContextInspector>('/context/inspect')
    } catch (err) {
      _log.debug('Failed to inspect context', {
        error: err instanceof Error ? err.message : String(err),
      })
      return null
    }
  },

  async sendVoiceMessage(
    sessionId: string,
    audioBlob: Blob,
    language = 'en',
  ): Promise<{
    audio_path: string
    audio_duration_ms: number
    transcript?: string
  }> {
    const formData = new FormData()
    formData.append('file', audioBlob, `voice-${Date.now()}.webm`)
    formData.append('language', language)
    return apiPost(`/chat/voice/${sessionId}`, formData, { raw: true })
  },

  getVoiceAudioUrl(sessionId: string, messageId: string): string {
    return `${PUBLIC_API_URL}/chat/audio/${sessionId}/${messageId}`
  },

  async cancelStream(sessionId: string): Promise<void> {
    await apiPost('/chat/control', { session_id: sessionId, action: 'cancel' })
  },

  async approveTool(sessionId: string, toolName: string, approved: boolean): Promise<void> {
    await apiPost('/chat/control', {
      session_id: sessionId,
      action: 'approve',
      tool_name: toolName,
      approved,
    })
  },

  async injectContext(sessionId: string, context: string): Promise<void> {
    await apiPost('/chat/control', {
      session_id: sessionId,
      action: 'context',
      context,
    })
  },
}

export interface ContextInspector {
  system_prompt?: string
  session_messages?: Array<{ role?: string; content?: string }>
  working_memory?: unknown[]
  semantic_keys?: string[]
  episodic_count?: number
  sensory_buffer_size?: number
  frame_history_size?: number
  last_frame?: Record<string, unknown> | null
  error?: string
}
