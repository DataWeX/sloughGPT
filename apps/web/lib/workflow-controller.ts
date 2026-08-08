import { apiGet, apiPost } from './http-client'

export interface WorkflowConfig {
  aggregate_interval_minutes: number
  prune_interval_minutes: number
  export_interval_hours: number
  health_check_interval_seconds: number
}

export interface WorkflowStats {
  feedback_records: number
  adapters_count: number
  last_aggregate?: string
  last_prune?: string
  last_export?: string
}

export interface WorkflowStatus {
  running: boolean
  config?: WorkflowConfig
  stats?: WorkflowStats
}

export const workflowController = {
  async status(): Promise<WorkflowStatus> {
    const data = await apiGet<WorkflowStatus>('/workflow/status')
    return data ?? { running: false }
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
