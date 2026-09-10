'use client'

import { createStore } from 'zustand/vanilla'
import { useStore } from 'zustand'
import type { PhonemeScoreResult } from './phoneme-controller'

export interface HistoryEntry {
  id: string
  timestamp: number
  target: string
  spoken: string
  score: number
  language: string
  targetPhonemes: string[]
  spokenPhonemes: string[]
}

interface PhonemeStore {
  history: HistoryEntry[]
  quizScore: number
  quizTotal: number
  quizStreak: number
  quizBestStreak: number
  randomWordTrigger: number
  addToHistory: (entry: Omit<HistoryEntry, 'id' | 'timestamp'>) => void
  clearHistory: () => void
  exportHistory: () => string
  importHistory: (json: string) => void
  incrementQuizScore: (correct: boolean) => void
  resetQuiz: () => void
  triggerRandomWord: () => void
}

const STORAGE_KEY = 'sloughgpt-phoneme-history'

function loadHistory(): HistoryEntry[] {
  if (typeof window === 'undefined') return []
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    return raw ? JSON.parse(raw) : []
  } catch {
    return []
  }
}

function saveHistory(history: HistoryEntry[]) {
  if (typeof window === 'undefined') return
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(history))
  } catch {
    // storage full or unavailable
  }
}

const phonemeStore = createStore<PhonemeStore>((set, get) => ({
  history: loadHistory(),
  quizScore: 0,
  quizTotal: 0,
  quizStreak: 0,
  quizBestStreak: 0,
  randomWordTrigger: 0,

  addToHistory: (entry) => {
    const id = `hist_${Date.now()}_${Math.random().toString(36).slice(2, 6)}`
    const newEntry: HistoryEntry = { ...entry, id, timestamp: Date.now() }
    set(prev => {
      const history = [newEntry, ...prev.history].slice(0, 200)
      saveHistory(history)
      return { history }
    })
  },

  clearHistory: () => {
    set({ history: [] })
    saveHistory([])
  },

  exportHistory: () => {
    return JSON.stringify(get().history, null, 2)
  },

  importHistory: (json: string) => {
    try {
      const parsed = JSON.parse(json) as HistoryEntry[]
      if (Array.isArray(parsed)) {
        set({ history: parsed })
        saveHistory(parsed)
      }
    } catch {
      // invalid JSON
    }
  },

  incrementQuizScore: (correct: boolean) => {
    set(prev => {
      const newStreak = correct ? prev.quizStreak + 1 : 0
      return {
        quizScore: prev.quizScore + (correct ? 1 : 0),
        quizTotal: prev.quizTotal + 1,
        quizStreak: newStreak,
        quizBestStreak: Math.max(prev.quizBestStreak, newStreak),
      }
    })
  },

  resetQuiz: () => {
    set({ quizScore: 0, quizTotal: 0, quizStreak: 0, quizBestStreak: 0 })
  },

  triggerRandomWord: () => {
    set(prev => ({ randomWordTrigger: prev.randomWordTrigger + 1 }))
  },
}))

export function usePhonemeStore<T>(selector: (state: PhonemeStore) => T): T {
  return useStore(phonemeStore, selector)
}

export { phonemeStore }
