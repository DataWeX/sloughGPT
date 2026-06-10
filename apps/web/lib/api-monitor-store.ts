'use client'
import { create } from 'zustand'

export type ApiStatus = 'connected' | 'connecting' | 'offline' | 'reloading'

interface ApiMonitorState {
  status: ApiStatus
  lastOnline: number | null
  setStatus: (status: ApiStatus) => void
}

export const useApiMonitor = create<ApiMonitorState>((set) => ({
  status: 'connecting',
  lastOnline: null,
  setStatus: (status) =>
    set((s) => ({
      status,
      lastOnline: status === 'connected' ? Date.now() : s.lastOnline,
    })),
}))
