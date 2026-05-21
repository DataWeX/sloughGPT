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
  message_count: number
  messages?: Array<{ id: string; role: string; content: string; timestamp?: string | number }>
  user_id?: string
  feedback_count?: number
  synced?: boolean
  createdAt?: string
  updatedAt?: string
}

export interface Session {
  id: string
  name: string
  created_at: string
  updated_at: string
  starred?: boolean
  pinned?: boolean
  messages?: Array<{ id?: string; role: string; content: string; timestamp?: string }>
  source?: string
}

export const sessionController = {
  async list(): Promise<Session[]> {
    const data = await apiGet<{ sessions: Session[] }>('/chat/sessions')
    return data.sessions || []
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

  async update(id: string, data: { name?: string; starred?: boolean; pinned?: boolean }): Promise<void> {
    await apiPut(`/chat/sessions/${id}`, data)
  },

  async delete(id: string): Promise<void> {
    await apiDelete(`/chat/sessions/${id}`)
  },

  async saveContext(id: string, messages: { role: string; content: string }[]): Promise<void> {
    await apiPost(`/session/${id}/context`, { messages })
  },

  async fetchMessages(id: string): Promise<Array<{ role: string; content: string }>> {
    const data = await apiGet<{ messages: Array<{ role: string; content: string }> }>(`/session/${id}/messages`)
    return data.messages || []
  },
}
