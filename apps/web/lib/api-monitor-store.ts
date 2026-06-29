'use client'
import { create } from 'zustand'

export type ApiStatus = 'connected' | 'connecting' | 'offline' | 'reloading'

export interface HealthSummaryData {
  score: number
  status: string
  summary: string
  model_loaded: boolean
  model_type: string | null
  soul: string | null
  uptime_seconds: number
  request_count: number
  error_count: number
  tokens_per_sec: number
  cpu_percent: number | null
  memory_percent: number | null
}

interface ApiMonitorState {
  status: ApiStatus
  lastOnline: number | null
  healthSummary: HealthSummaryData | null
  setStatus: (status: ApiStatus) => void
  setHealthSummary: (data: HealthSummaryData | null) => void
}

export const useApiMonitor = create<ApiMonitorState>((set) => ({
  status: 'connecting',
  lastOnline: null,
  healthSummary: null,
  setStatus: (status) =>
    set((s) => ({
      status,
      lastOnline: status === 'connected' ? Date.now() : s.lastOnline,
    })),
  setHealthSummary: (data) => set({ healthSummary: data }),
}))
