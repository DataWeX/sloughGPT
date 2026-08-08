import { apiGet, apiPost, apiPatch, apiDelete } from './http-client'

export interface Conversation {
  id: string
  name: string
  session_id: string
  created_at: string
  updated_at: string
  pinned: boolean
  starred: boolean
  message_count: number
}

export const feedbackConversationsController = {
  async list(): Promise<Conversation[]> {
    const data = await apiGet<Conversation[] | { conversations: Conversation[] }>('/feedback/conversations')
    return Array.isArray(data) ? data : data?.conversations ?? []
  },

  async create(name: string): Promise<Conversation> {
    return apiPost<Conversation>('/feedback/conversations', { name })
  },

  async delete(id: string): Promise<void> {
    await apiDelete(`/feedback/conversations/${id}`)
  },

  async togglePin(id: string, pinned: boolean): Promise<Conversation> {
    return apiPatch<Conversation>(`/feedback/conversations/${id}`, { pinned })
  },

  async toggleStar(id: string, starred: boolean): Promise<Conversation> {
    return apiPatch<Conversation>(`/feedback/conversations/${id}`, { starred })
  },
}
