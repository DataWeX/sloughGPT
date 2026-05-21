'use client'

import { create } from 'zustand'
import { persist } from 'zustand/middleware'

export interface AppSettings {
  apiUrl: string
  hfToken: string
  defaultModel: string
  defaultTemp: number
  defaultMaxTokens: number
  theme: 'dark' | 'light' | 'system'
  streaming: boolean
  customContext: string
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

const DEFAULT_SETTINGS: AppSettings = {
  apiUrl: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000',
  hfToken: '',
  defaultModel: 'gpt2',
  defaultTemp: 0.8,
  defaultMaxTokens: 200,
  theme: 'light',
  streaming: true,
  customContext: '',
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
      name: 'sloughgpt-store',
    }
  )
)

export function useSettings() {
  return useAppStore((state) => state.settings)
}

export function useUpdateSettings() {
  return useAppStore((state) => state.updateSettings)
}

export function useKnowledge() {
  return useAppStore((state) => ({
    items: state.injectedKnowledge,
    add: state.addKnowledge,
    remove: state.removeKnowledge,
    clear: state.clearKnowledge,
  }))
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