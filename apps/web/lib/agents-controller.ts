/**
 * Agents Controller — axios-based API for agent management.
 */

import { apiGet, apiPost, apiPut, apiDelete } from './http-client'
import { PUBLIC_API_URL } from './config'

export interface Agent {
  id: string
  name: string
  description: string
  instructions: string
  tools: string[]
  avatar: string
}

export interface OrchestrateTask {
  id: string
  description: string
  agent: string
  status: string
  result_preview: string
  depends_on: string[]
}

export interface OrchestrateCallbacks {
  onPlan?: (tasks: OrchestrateTask[]) => void
  onTaskStatus?: (taskId: string, status: string, agent: string, description: string, resultPreview?: string) => void
  onTaskComplete?: (taskId: string, result: string) => void
  onLevelChange?: (level: number, totalLevels: number, taskIds: string[]) => void
  onCompose?: () => void
  onComplete: (response: string, tasks: OrchestrateTask[]) => void
  onError: (error: string) => void
}

const API_ORCHESTRATE_ENDPOINT = `${PUBLIC_API_URL}/agents/orchestrate`

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

  async orchestrate(goal: string, context: string, callbacks: OrchestrateCallbacks, signal?: AbortSignal): Promise<void> {
    const response = await fetch(API_ORCHESTRATE_ENDPOINT, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ goal, context }),
      signal,
    })

    if (!response.ok) {
      callbacks.onError(`HTTP ${response.status}: ${response.statusText}`)
      return
    }

    const reader = response.body?.getReader()
    const decoder = new TextDecoder()
    if (!reader) {
      callbacks.onError('No response body')
      return
    }

    let buffer = ''
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
        if (!payload) continue

        try {
          const envelope = JSON.parse(payload) as {
            stream?: string
            phase?: string
            status?: string
            data?: Record<string, unknown>
            message?: string
          }

          const phase = envelope.phase
          const status = envelope.status
          const d = envelope.data ?? {}

          if (status === 'error') {
            const errStr = typeof d.error === 'string' ? d.error : envelope.message || 'Orchestration failed'
            callbacks.onError(errStr)
            return
          }

          if (phase === 'PLAN' && status === 'success') {
            const tasks = (d.tasks as OrchestrateTask[]) ?? []
            callbacks.onPlan?.(tasks)
            continue
          }

          if (phase === 'EXECUTE' && status === 'working') {
            if (d.level !== undefined) {
              callbacks.onLevelChange?.(d.level as number, (d.levels as number) ?? 1, (d.tasks as string[]) ?? [])
            }
            continue
          }

          if (phase === 'EXECUTE') {
            const taskId = d.task_id as string | undefined
            if (taskId) {
              callbacks.onTaskStatus?.(
                taskId,
                status ?? 'unknown',
                (d.agent as string) ?? '',
                (d.description as string) ?? '',
                d.result_preview as string | undefined,
              )
            }
            continue
          }

          if (phase === 'COMPOSE' && status === 'working') {
            callbacks.onCompose?.()
            continue
          }

          if (status === 'complete') {
            const resp = (d.response as string) ?? ''
            const tasks = (d.tasks as OrchestrateTask[]) ?? []
            callbacks.onComplete(resp, tasks)
            return
          }
        } catch {
          // skip malformed lines
        }
      }
    }

    callbacks.onError('Stream ended unexpectedly')
  },
}
