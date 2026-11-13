import { apiGet, apiDelete } from './http-client'

export interface GroupedError {
  fingerprint: string
  message: string
  source: string
  count: number
  latest: string
  sample_id: string
  sample_url: string
  sample_line: number | null
}

export interface TrendPoint {
  hour: string
  count: number
}

export interface RecentError {
  id: string
  message: string
  source: string
  url?: string
  line?: number
  timestamp: string
  fingerprint: string
}

export const errorsController = {
  async getGrouped(): Promise<GroupedError[]> {
    const data = await apiGet<{ groups: GroupedError[] }>('/errors/grouped')
    return data?.groups ?? []
  },

  async getRecent(limit = 30): Promise<{ errors: RecentError[]; total: number }> {
    const data = await apiGet<{ errors: RecentError[]; total: number }>(`/errors/recent?limit=${limit}`)
    return { errors: data?.errors ?? [], total: data?.total ?? 0 }
  },

  async getTrends(hours = 24): Promise<TrendPoint[]> {
    const data = await apiGet<{ trends: TrendPoint[] }>(`/errors/trends?hours=${hours}`)
    return data?.trends ?? []
  },

  async clear(): Promise<void> {
    await apiDelete('/errors/clear')
  },

  async export(): Promise<unknown> {
    return apiGet<unknown>('/errors/export')
  },
}
