import { apiGet, apiPost } from './http-client'
import type { WorkflowStatus } from './types'

export type { WorkflowStatus, WorkflowConfig, WorkflowStats, WorkflowLastRuns } from './types'

export const workflowController = {
  async status(): Promise<WorkflowStatus> {
    const data = await apiGet<WorkflowStatus>('/workflow/status')
    return data ?? { running: false } as WorkflowStatus
  },

  async start(): Promise<{ status: string }> {
    return apiPost<{ status: string }>('/workflow/start', {})
  },

  async stop(): Promise<{ status: string }> {
    return apiPost<{ status: string }>('/workflow/stop', {})
  },

  async trigger(action: string): Promise<{ status: string }> {
    return apiPost<{ status: string }>(`/workflow/trigger/${action}`, {})
  },
}
