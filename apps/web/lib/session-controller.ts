/**
 * Session Controller — axios-based API for conversation management.
 */

import { apiGet, apiPost, apiPut, apiDelete } from './http-client'
import { logger } from './dev-log'

const _log = logger.child('session-controller')

export interface SessionInspector {
  session: {
    id: string
    message_count: number
    messages: Array<{ role: string; content: string; ts?: number }>
  }
  knowledge: {
    total_facts: number
    topics: string[]
  }
  traits: Record<string, unknown>
  modes: Record<string, string>
  feedback: {
    total: number
    thumbs_up: number
    thumbs_down: number
  }
  workspace: {
    working_memory: string[]
    semantic_keys: string[]
    episodic_count: number
    sensory_buffer_size: number
    system_prompt: string
  }
  elapsed_ms: number
}

export interface Conversation {
  id: string
  name: string
  session_id: string
  created_at: string
  updated_at: string
  pinned: boolean
  starred: boolean
  archived?: boolean
  unread?: boolean
  message_count: number
  messages?: Array<{ id: string; role: string; content: string; timestamp?: string | number }>
  user_id?: string
  feedback_count?: number
  synced?: boolean
  createdAt?: string
  updatedAt?: string
}

export interface SearchMatch {
  role: string
  content: string
  timestamp: string
}

export interface SearchResult {
  id: string
  name: string
  created_at: string
  updated_at: string
  match_count: number
  matches: SearchMatch[]
}

export interface Session {
  id: string
  name: string
  created_at: string
  updated_at: string
  starred?: boolean
  pinned?: boolean
  archived?: boolean
  messages?: Array<{ id?: string; role: string; content: string; timestamp?: string }>
  source?: string
}

export const sessionController = {
  async list(archived?: boolean): Promise<Session[]> {
    const suffix = archived !== undefined ? `?archived=${archived}` : ''
    const data = await apiGet<Session[] | { sessions: Session[] }>(`/chat/sessions${suffix}`)
    return Array.isArray(data) ? data : (data.sessions ?? [])
  },

  async listArchived(): Promise<Session[]> {
    return sessionController.list(true)
  },

  async getCurrent(): Promise<Session | null> {
    try {
      return await apiGet<Session>('/chat/sessions/current')
    } catch (err) {
      _log.debug('Failed to get current session', { error: err instanceof Error ? err.message : String(err) })
      return null
    }
  },

  async getSoul(): Promise<{ name: string; traits: string[]; personality?: Record<string, number> } | null> {
    try {
      return await apiGet<{ name: string; traits: string[]; personality?: Record<string, number> }>('/souls/current')
    } catch (err) {
      _log.debug('Failed to get soul for session', { error: err instanceof Error ? err.message : String(err) })
      return null
    }
  },

  async create(name: string, id?: string): Promise<Session> {
    return apiPost<Session>('/chat/sessions', { name, session_id: id })
  },

  async update(id: string, data: { name?: string; starred?: boolean; pinned?: boolean; archived?: boolean; unread?: boolean }): Promise<void> {
    if (!id) return
    await apiPut(`/chat/sessions/${encodeURIComponent(id)}`, data)
  },

  async delete(id: string): Promise<void> {
    await apiDelete(`/chat/sessions/${id}`)
  },

  async search(q: string, limit = 20): Promise<SearchResult[]> {
    if (!q.trim()) return []
    const data = await apiGet<SearchResult[] | { results: SearchResult[] }>(`/chat/sessions/search?q=${encodeURIComponent(q)}&limit=${limit}`)
    return Array.isArray(data) ? data : (data.results ?? [])
  },

  async saveContext(id: string, messages: { role: string; content: string }[]): Promise<void> {
    await apiPost(`/session/${id}/context`, { messages })
  },

  async fetchMessages(id: string, opts?: { signal?: AbortSignal; silent?: boolean }): Promise<Array<{ role: string; content: string }>> {
    const data = opts
      ? await apiGet<{ messages: Array<{ role: string; content: string }> }>(`/session/${id}/messages`, undefined, { signal: opts.signal, silent: opts.silent })
      : await apiGet<{ messages: Array<{ role: string; content: string }> }>(`/session/${id}/messages`)
    return data?.messages ?? []
  },

  async getInspector(sessionId: string): Promise<SessionInspector> {
    return apiGet<SessionInspector>(`/session/${sessionId}/inspector`)
  },

  async regenerate(sessionId: string): Promise<{ status: string }> {
    return apiPost<{ status: string }>(`/session/${sessionId}/regenerate`)
  },

  async forwardMessage(targetSessionId: string, content: string, role: string = 'user'): Promise<void> {
    await apiPost(`/session/${targetSessionId}/context`, { messages: [{ role, content }] })
  },
}
