/**
 * Agents Controller — axios-based API for agent management.
 */

import { apiGet, apiPost, apiPut, apiDelete, streamSSE } from './http-client'

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

export interface AgentRun {
  id: string
  goal: string
  context: string
  status: 'running' | 'completed' | 'failed'
  started_at: string | null
  finished_at: string | null
  tasks: OrchestrateTask[]
  completed_count: number
  failed_count: number
  response: string
  error: string
  logs: string[]
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

  async listRuns(limit = 20): Promise<{ runs: AgentRun[]; count: number }> {
    return apiGet<{ runs: AgentRun[]; count: number }>(`/agents/runs?limit=${limit}`)
  },

  async getRun(runId: string): Promise<AgentRun> {
    return apiGet<AgentRun>(`/agents/runs/${encodeURIComponent(runId)}`)
  },

  async orchestrate(goal: string, context: string, callbacks: OrchestrateCallbacks, signal?: AbortSignal): Promise<void> {
    try {
      for await (const event of streamSSE('/agents/orchestrate', { body: { goal, context }, signal })) {
        if (event.status === 'error') {
          callbacks.onError(event.message || (event.data?.error as string) || 'Orchestration failed')
          return
        }

        const phase = event.phase
        const d = event.data ?? {}

        if (phase === 'PLAN' && event.status === 'success') {
          callbacks.onPlan?.((d.tasks as OrchestrateTask[]) ?? [])
          continue
        }

        if (phase === 'EXECUTE' && event.status === 'working') {
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
              event.status ?? 'unknown',
              (d.agent as string) ?? '',
              (d.description as string) ?? '',
              d.result_preview as string | undefined,
            )
          }
          continue
        }

        if (phase === 'COMPOSE' && event.status === 'working') {
          callbacks.onCompose?.()
          continue
        }

        if (event.status === 'complete') {
          callbacks.onComplete((d.response as string) ?? '', (d.tasks as OrchestrateTask[]) ?? [])
          return
        }
      }
      callbacks.onError('Stream ended unexpectedly')
    } catch (err) {
      callbacks.onError(err instanceof Error ? err.message : 'Connection error')
    }
  },
}
