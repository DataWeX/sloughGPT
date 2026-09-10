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

export interface FlashcardSRS {
  interval: number
  ease: number
  nextReview: number
  reviews: number
}

interface PhonemeStore {
  history: HistoryEntry[]
  quizScore: number
  quizTotal: number
  quizStreak: number
  quizBestStreak: number
  randomWordTrigger: number
  flashcardSRS: Record<string, FlashcardSRS>
  addToHistory: (entry: Omit<HistoryEntry, 'id' | 'timestamp'>) => void
  clearHistory: () => void
  exportHistory: () => string
  importHistory: (json: string) => void
  incrementQuizScore: (correct: boolean) => void
  resetQuiz: () => void
  triggerRandomWord: () => void
  updateFlashcardSRS: (cardKey: string, known: boolean) => void
  resetFlashcardSRS: () => void
}

const STORAGE_KEY = 'sloughgpt-phoneme-history'
const SRS_KEY = 'sloughgpt-phoneme-srs'

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

function loadSRS(): Record<string, FlashcardSRS> {
  if (typeof window === 'undefined') return {}
  try {
    const raw = localStorage.getItem(SRS_KEY)
    return raw ? JSON.parse(raw) : {}
  } catch {
    return {}
  }
}

function saveSRS(srs: Record<string, FlashcardSRS>) {
  if (typeof window === 'undefined') return
  try {
    localStorage.setItem(SRS_KEY, JSON.stringify(srs))
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
  flashcardSRS: loadSRS(),

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

  updateFlashcardSRS: (cardKey: string, known: boolean) => {
    set(prev => {
      const srs = { ...prev.flashcardSRS }
      const existing = srs[cardKey] || { interval: 0, ease: 2.5, nextReview: Date.now(), reviews: 0 }

      if (known) {
        const newInterval = Math.max(1, Math.ceil(existing.interval * existing.ease))
        const newEase = Math.min(3.0, existing.ease + 0.1)
        srs[cardKey] = {
          interval: newInterval,
          ease: newEase,
          nextReview: Date.now() + newInterval * 86400000,
          reviews: existing.reviews + 1,
        }
      } else {
        srs[cardKey] = {
          interval: 1,
          ease: Math.max(1.3, existing.ease - 0.2),
          nextReview: Date.now() + 86400000,
          reviews: existing.reviews + 1,
        }
      }

      saveSRS(srs)
      return { flashcardSRS: srs }
    })
  },

  resetFlashcardSRS: () => {
    set({ flashcardSRS: {} })
    saveSRS({})
  },
}))

export function usePhonemeStore<T>(selector: (state: PhonemeStore) => T): T {
  return useStore(phonemeStore, selector)
}

export { phonemeStore }
