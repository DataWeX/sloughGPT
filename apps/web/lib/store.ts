'use client'

import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import { PUBLIC_API_URL } from '@/lib/config'

export interface AppSettings {
  apiUrl: string
  hfToken: string
  defaultTemp: number
  defaultMaxTokens: number
  defaultTopP: number
  defaultTopK: number
  theme: 'dark' | 'light' | 'system'
  streaming: boolean
  customContext: string
  collapsibleMessageLength: number
}

export interface InjectedKnowledge {
  id: string
  content: string
  timestamp: number
}

interface AppStore {
  settings: AppSettings
  injectedKnowledge: InjectedKnowledge[]
  updateSettings: (partial: Partial<AppSettings>) => void
  addKnowledge: (content: string) => void
  removeKnowledge: (id: string) => void
  clearKnowledge: () => void
}

export const DEFAULT_SETTINGS: AppSettings = {
  apiUrl: PUBLIC_API_URL,
  hfToken: '',
  defaultTemp: 0.7,
  defaultMaxTokens: 300,
  defaultTopP: 0.85,
  defaultTopK: 40,
  theme: 'light',
  streaming: true,
  customContext: '',
  collapsibleMessageLength: 500,
}

export const useAppStore = create<AppStore>()(
  persist(
    (set, get) => ({
      settings: DEFAULT_SETTINGS,
      injectedKnowledge: [],

      updateSettings: (partial) =>
        set((state) => ({
          settings: { ...state.settings, ...partial },
        })),

      addKnowledge: (content) =>
        set((state) => ({
          injectedKnowledge: [
            ...state.injectedKnowledge,
            {
              id: `know_${Date.now()}`,
              content,
              timestamp: Date.now(),
            },
          ],
        })),

      removeKnowledge: (id) =>
        set((state) => ({
          injectedKnowledge: state.injectedKnowledge.filter((k) => k.id !== id),
        })),

      clearKnowledge: () =>
        set({ injectedKnowledge: [] }),
    }),
    {
      name: 'man-store',
      merge: (persisted, current) => {
        const p = (persisted ?? {}) as Partial<AppStore>
        return {
          ...current,
          ...p,
          settings: {
            ...DEFAULT_SETTINGS,
            ...(p.settings ?? {}),
          },
        }
      },
    }
  )
)

export function useSettings() {
  return useAppStore((state) => state.settings)
}

export function useUpdateSettings() {
  return useAppStore((state) => state.updateSettings)
}

export function getKnowledgeContext(): string {
  const state = useAppStore.getState()
  const { customContext } = state.settings
  const { injectedKnowledge } = state
  const allKnowledge = [
    ...customContext ? [{ content: customContext }] : [],
    ...injectedKnowledge,
  ]
  if (allKnowledge.length === 0) return ''
  return `\n\n[IMPORTANT KNOWLEDGE - Use this information when responding:]\n${allKnowledge.map((k) => `• ${k.content}`).join('\n')}\n[/IMPORTANT KNOWLEDGE]`
}
