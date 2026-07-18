/**
 * Session Controller — axios-based API for conversation management.
 */

import { apiGet, apiPost, apiPut, apiDelete } from './http-client'

export interface Conversation {
  id: string
  name: string
  session_id: string
  created_at: string
  updated_at: string
  pinned: boolean
  starred: boolean
  archived?: boolean
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
    } catch {
      return null
    }
  },

  async getSoul(): Promise<{ name: string; traits: string[]; personality?: Record<string, number> } | null> {
    try {
      return await apiGet<{ name: string; traits: string[]; personality?: Record<string, number> }>('/souls/current')
    } catch {
      return null
    }
  },

  async create(name: string, id?: string): Promise<Session> {
    return apiPost<Session>('/chat/sessions', { name, session_id: id })
  },

  async update(id: string, data: { name?: string; starred?: boolean; pinned?: boolean; archived?: boolean }): Promise<void> {
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

  async fetchMessages(id: string): Promise<Array<{ role: string; content: string }>> {
    const data = await apiGet<{ messages: Array<{ role: string; content: string }> }>(`/session/${id}/messages`)
    return data.messages || []
  },
}
