/**
 * Agents Controller — axios-based API for agent management.
 */

import { apiGet, apiPost, apiPut, apiDelete } from './http-client'

export interface Agent {
  id: string
  name: string
  description: string
  instructions: string
  tools: string[]
  avatar: string
}

export const agentsController = {
  async list(): Promise<Agent[]> {
    return apiGet<Agent[]>('/agents')
  },

  async create(data: { name: string; description?: string; instructions?: string; tools?: string[]; avatar?: string }): Promise<Agent> {
    return apiPost<Agent>('/agents', data)
  },

  async update(id: string, data: { name?: string; description?: string; instructions?: string; tools?: string[]; avatar?: string }): Promise<Agent> {
    return apiPut<Agent>(`/agents/${encodeURIComponent(id)}`, data)
  },

  async delete(id: string): Promise<void> {
    await apiDelete(`/agents/${encodeURIComponent(id)}`)
  },

  async execute(id: string, request: string, sessionId?: string): Promise<{ response: string; tools_used: Array<{ tool: string; result: unknown }> }> {
    return apiPost<{ response: string; tools_used: Array<{ tool: string; result: unknown }> }>(`/agents/${encodeURIComponent(id)}/execute`, { request, session_id: sessionId || '' })
  },
}
