import { create } from 'zustand'
import { persist, createJSONStorage } from 'zustand/middleware'
import AsyncStorage from '@react-native-async-storage/async-storage'

type Theme = 'light' | 'dark' | 'system'

interface SettingsState {
  theme: Theme
  temperature: number
  maxTokens: number
  memoryContext: string
  apiUrl: string
  setTheme: (theme: Theme) => void
  update: (partial: Partial<Omit<SettingsState, 'setTheme' | 'update' | 'reset'>>) => void
  reset: () => void
}

const DEFAULTS = {
  theme: 'system' as Theme,
  temperature: 0.8,
  maxTokens: 256,
  memoryContext: '',
  apiUrl: '',
}

export const useSettingsStore = create<SettingsState>()(
  persist(
    (set) => ({
      ...DEFAULTS,

      setTheme: (theme) => set({ theme }),

      update: (partial) => set(partial),

      reset: () => set(DEFAULTS),
    }),
    {
      name: 'sloughgpt-settings',
      storage: createJSONStorage(() => AsyncStorage),
    }
  )
)
