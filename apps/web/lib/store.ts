'use client'

import { create } from 'zustand'
import { PUBLIC_API_URL } from '@/lib/config'
import { chatDB } from '@/lib/db'

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

const SETTINGS_KEY = 'app-settings'
const KNOWLEDGE_KEY = 'app-knowledge'

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

export const useAppStore = create<AppStore>()((set, get) => ({
  settings: DEFAULT_SETTINGS,
  injectedKnowledge: [],

  updateSettings: (partial) => {
    const next = { ...get().settings, ...partial }
    set({ settings: next })
    chatDB.setKV(SETTINGS_KEY, next).catch(() => {})
  },

  addKnowledge: (content) => {
    const item = { id: `know_${Date.now()}`, content, timestamp: Date.now() }
    const next = [...get().injectedKnowledge, item]
    set({ injectedKnowledge: next })
    chatDB.setKV(KNOWLEDGE_KEY, next).catch(() => {})
  },

  removeKnowledge: (id) => {
    const next = get().injectedKnowledge.filter((k) => k.id !== id)
    set({ injectedKnowledge: next })
    chatDB.setKV(KNOWLEDGE_KEY, next).catch(() => {})
  },

  clearKnowledge: () => {
    set({ injectedKnowledge: [] })
    chatDB.deleteKV(KNOWLEDGE_KEY).catch(() => {})
  },
}))

export async function initStore() {
  try {
    const [storedSettings, storedKnowledge] = await Promise.all([
      chatDB.getKV<AppSettings>(SETTINGS_KEY),
      chatDB.getKV<InjectedKnowledge[]>(KNOWLEDGE_KEY),
    ])
    if (storedSettings) {
      useAppStore.setState({ settings: { ...DEFAULT_SETTINGS, ...storedSettings } })
    }
    if (storedKnowledge) {
      useAppStore.setState({ injectedKnowledge: storedKnowledge })
    }
  } catch {
    // ignore init errors — store stays at defaults
  }
}

if (typeof window !== 'undefined') {
  initStore()
}

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
