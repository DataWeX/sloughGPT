import { apiGet, apiPost, apiDelete } from './http-client'

export interface Experiment {
  id: string
  name?: string
  created?: string
  runs?: number
  status?: string
}

export const experimentsController = {
  async list(): Promise<Experiment[]> {
    const data = await apiGet<{ experiments: string[] }>('/experiments')
    return (data?.experiments ?? []).map((id: string) => ({ id }))
  },

  async create(name: string): Promise<{ id: string; name: string; created: boolean }> {
    return apiPost<{ id: string; name: string; created: boolean }>('/experiments', { name })
  },

  async delete(experimentId: string): Promise<{ id: string; deleted: boolean }> {
    return apiDelete<{ id: string; deleted: boolean }>(
      `/experiments/${encodeURIComponent(experimentId)}`,
    )
  },

  async logMetric(experimentId: string, metricName: string, value: number): Promise<{ status: string }> {
    const data = await apiPost<{ status: string }>(
      `/experiments/${encodeURIComponent(experimentId)}/log_metric`,
      { metric_name: metricName, value },
    )
    return data
  },

  async logParam(experimentId: string, paramName: string, value: string): Promise<{ status: string }> {
    const data = await apiPost<{ status: string }>(
      `/experiments/${encodeURIComponent(experimentId)}/log_param`,
      { param_name: paramName, value },
    )
    return data
  },

  async complete(experimentId: string): Promise<{ status: string }> {
    const data = await apiPost<{ status: string }>(
      `/experiments/${encodeURIComponent(experimentId)}/complete`,
      {},
    )
    return data
  },

  async getExperimentData(experimentId: string): Promise<{
    id: string
    metrics: Array<{ metric: string; value: number; step: number; timestamp: string }>
    params: Array<{ param: string; value: string; timestamp: string }>
    status: { status: string; completed_at?: string } | null
  }> {
    const data = await apiGet<{
      id: string
      metrics: Array<{ metric: string; value: number; step: number; timestamp: string }>
      params: Array<{ param: string; value: string; timestamp: string }>
      status: { status: string; completed_at?: string } | null
    }>(`/experiments/${encodeURIComponent(experimentId)}/data`)
    return data
  },
}
