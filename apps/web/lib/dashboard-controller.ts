/**
 * Dashboard Controller — quick system summary for the CLI monitor and web UI.
 */

import { apiGet } from './http-client'

export interface DashboardSummary {
  health: {
    model_loaded: boolean
    model_type: string
    uptime_seconds: number
    request_count: number
    error_count: number
    tokens_per_sec: number
    avg_latency_ms: number
    cpu_percent: number
    memory_percent: number
    memory_used_mb: number
  }
  active_processes: number
  processes: Record<string, {
    type: string
    status: string
    label: string
    detail: string
    progress: number
  }>
  services: { total: number; healthy: number }
}

export interface DashboardEvent {
  ts: number
  level: string
  message: string
  source?: string
}

export const dashboardController = {
  async getSummary(): Promise<DashboardSummary> {
    return apiGet<DashboardSummary>('/dashboard/summary')
  },

  async getEvents(n: number = 20): Promise<{ events: DashboardEvent[]; count: number }> {
    return apiGet(`/dashboard/events?n=${n}`)
  },
}
