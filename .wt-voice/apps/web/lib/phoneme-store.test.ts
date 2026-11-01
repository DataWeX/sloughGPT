import { describe, it, expect, beforeEach } from 'vitest'
import { phonemeStore } from './phoneme-store'

describe('phonemeStore', () => {
  beforeEach(() => {
    phonemeStore.getState().clearHistory()
    phonemeStore.getState().resetQuiz()
  })

  describe('history', () => {
    it('adds entry to history', () => {
      const { addToHistory } = phonemeStore.getState()
      addToHistory({
        target: 'hello',
        spoken: 'helo',
        score: 0.75,
        language: 'en',
        targetPhonemes: ['HH', 'EH', 'L', 'OW'],
        spokenPhonemes: ['HH', 'EH', 'L'],
      })

      const { history } = phonemeStore.getState()
      expect(history).toHaveLength(1)
      expect(history[0].target).toBe('hello')
      expect(history[0].score).toBe(0.75)
    })

    it('clears history', () => {
      const { addToHistory, clearHistory } = phonemeStore.getState()
      addToHistory({
        target: 'hello',
        spoken: 'helo',
        score: 0.75,
        language: 'en',
        targetPhonemes: ['HH', 'EH', 'L', 'OW'],
        spokenPhonemes: ['HH', 'EH', 'L'],
      })

      clearHistory()
      expect(phonemeStore.getState().history).toHaveLength(0)
    })

    it('exports history as JSON', () => {
      const { addToHistory, exportHistory } = phonemeStore.getState()
      addToHistory({
        target: 'hello',
        spoken: 'helo',
        score: 0.75,
        language: 'en',
        targetPhonemes: ['HH', 'EH', 'L', 'OW'],
        spokenPhonemes: ['HH', 'EH', 'L'],
      })

      const json = exportHistory()
      const parsed = JSON.parse(json)
      expect(parsed).toHaveLength(1)
      expect(parsed[0].target).toBe('hello')
    })

    it('imports history from JSON', () => {
      const { importHistory } = phonemeStore.getState()
      const json = JSON.stringify([
        {
          id: 'test_1',
          timestamp: Date.now(),
          target: 'world',
          spoken: 'worl',
          score: 0.8,
          language: 'en',
          targetPhonemes: ['W', 'ER', 'L', 'D'],
          spokenPhonemes: ['W', 'ER', 'L'],
        },
      ])

      importHistory(json)
      expect(phonemeStore.getState().history).toHaveLength(1)
      expect(phonemeStore.getState().history[0].target).toBe('world')
    })
  })

  describe('quiz', () => {
    it('increments quiz score on correct', () => {
      const { incrementQuizScore } = phonemeStore.getState()
      incrementQuizScore(true)
      incrementQuizScore(true)
      incrementQuizScore(false)

      const { quizScore, quizTotal } = phonemeStore.getState()
      expect(quizScore).toBe(2)
      expect(quizTotal).toBe(3)
    })

    it('resets quiz', () => {
      const { incrementQuizScore, resetQuiz } = phonemeStore.getState()
      incrementQuizScore(true)
      incrementQuizScore(true)

      resetQuiz()
      expect(phonemeStore.getState().quizScore).toBe(0)
      expect(phonemeStore.getState().quizTotal).toBe(0)
    })
  })
})
