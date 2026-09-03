'use client'

import { create } from 'zustand'
import { PUBLIC_API_URL } from '@/lib/config'
import { chatDB } from '@/lib/db'
import { trackEvent } from '@/lib/dev-log'

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
  autoApproveTools: boolean
}

export interface InjectedKnowledge {
  id: string
  content: string
  timestamp: number
}

export interface ModelReadiness {
  ready: boolean
  phase: string
  step: number
  total: number
  message: string
}

interface AppStore {
  settings: AppSettings
  injectedKnowledge: InjectedKnowledge[]
  modelReadiness: ModelReadiness
  updateSettings: (partial: Partial<AppSettings>) => void
  addKnowledge: (content: string) => void
  removeKnowledge: (id: string) => void
  clearKnowledge: () => void
  setModelReadiness: (readiness: Partial<ModelReadiness>) => void
}

const SETTINGS_KEY = 'app-settings'
const KNOWLEDGE_KEY = 'app-knowledge'
const DEBOUNCE_MS = 300

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
  autoApproveTools: false,
}

let _settingsTimer: ReturnType<typeof setTimeout> | null = null
let _pendingSettings: Partial<AppSettings> | null = null

function _flushSettings(get: () => AppStore) {
  _settingsTimer = null
  if (!_pendingSettings) return
  const partial = _pendingSettings
  _pendingSettings = null
  const full = { ...get().settings, ...partial }
  chatDB.setKV(SETTINGS_KEY, full).catch(() => {})
}

export const useAppStore = create<AppStore>()((set, get) => ({
  settings: DEFAULT_SETTINGS,
  injectedKnowledge: [],
  modelReadiness: { ready: false, phase: 'initializing', step: 0, total: 9, message: 'Starting...' },

  updateSettings: (partial) => {
    trackEvent('settings_changed', { keys: Object.keys(partial) })
    const next = { ...get().settings, ...partial }
    set({ settings: next })
    _pendingSettings = { ...(_pendingSettings ?? {}), ...partial }
    if (_settingsTimer) clearTimeout(_settingsTimer)
    _settingsTimer = setTimeout(() => _flushSettings(get), DEBOUNCE_MS)
  },

  setModelReadiness: (readiness) => {
    set((state) => ({ modelReadiness: { ...state.modelReadiness, ...readiness } }))
  },

  addKnowledge: (content) => {
    trackEvent('knowledge_added')
    const item = { id: `know_${Date.now()}`, content, timestamp: Date.now() }
    const next = [...get().injectedKnowledge, item]
    set({ injectedKnowledge: next })
    chatDB.setKV(KNOWLEDGE_KEY, next).catch(() => {})
  },

  removeKnowledge: (id) => {
    trackEvent('knowledge_removed', { id })
    const next = get().injectedKnowledge.filter((k) => k.id !== id)
    set({ injectedKnowledge: next })
    chatDB.setKV(KNOWLEDGE_KEY, next).catch(() => {})
  },

  clearKnowledge: () => {
    trackEvent('knowledge_cleared')
    set({ injectedKnowledge: [] })
    chatDB.deleteKV(KNOWLEDGE_KEY).catch(() => {})
  },
}))

let _initPromise: Promise<void> | null = null

export async function initStore() {
  if (_initPromise) return _initPromise
  _initPromise = (async () => {
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
  })()
  return _initPromise
}

/** Reset the init guard — for tests only. */
export function _resetInitGuard() {
  _initPromise = null
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

export function useModelReadiness() {
  return useAppStore((state) => state.modelReadiness)
}

export function setModelReadiness(readiness: Partial<ModelReadiness>) {
  useAppStore.getState().setModelReadiness(readiness)
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
